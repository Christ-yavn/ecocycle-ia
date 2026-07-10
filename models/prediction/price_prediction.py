"""
Modèle de prédiction des prix des matériaux recyclables - EcoLoop AI.

Ce module implémente un système de prédiction des prix des matériaux
recyclables basé sur Facebook Prophet. Il permet de prévoir l'évolution
des prix pour le plastique, le métal, le verre et le papier.

Auteur : EcoLoop AI Team
"""

import os
import logging
import pickle
from typing import Dict, List, Optional

import pandas as pd
from prophet import Prophet

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Répertoire de sauvegarde des modèles
SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'saved_models'
)


class PricePredictor:
    """
    Prédicteur de prix des matériaux recyclables basé sur Prophet.

    Utilise Facebook Prophet pour modéliser les tendances saisonnières
    et prévoir les prix futurs des matériaux recyclables avec des
    intervalles de confiance.

    Attributes:
        MATERIALS (list): Liste des matériaux supportés.
        models (dict): Dictionnaire des modèles Prophet par matériau.
    """

    # Matériaux recyclables supportés
    MATERIALS = ['plastique', 'metal', 'verre', 'papier']

    def __init__(self):
        """
        Initialise le prédicteur de prix.

        Tente de charger les modèles pré-entraînés depuis le répertoire
        de sauvegarde. Si aucun modèle n'est trouvé, initialise un
        dictionnaire vide.
        """
        self.models: Dict[str, Prophet] = {}
        self._load_models()
        logger.info(
            f"PricePredictor initialisé. "
            f"Modèles chargés : {list(self.models.keys()) or 'aucun'}"
        )

    def _load_models(self) -> None:
        """
        Charge les modèles Prophet sauvegardés depuis le disque.

        Recherche les fichiers .pkl dans le répertoire saved_models/
        avec le format de nommage : price_model_{material}.pkl

        Les modèles qui ne sont pas trouvés sont simplement ignorés
        (ils devront être entraînés avant utilisation).
        """
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

        for material in self.MATERIALS:
            model_path = os.path.join(
                SAVED_MODELS_DIR,
                f'price_model_{material}.pkl'
            )
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[material] = pickle.load(f)
                    logger.info(
                        f"Modèle Prophet chargé pour '{material}' "
                        f"depuis {model_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Impossible de charger le modèle pour '{material}' : "
                        f"{str(e)}"
                    )
            else:
                logger.debug(
                    f"Aucun modèle trouvé pour '{material}' "
                    f"(chemin attendu : {model_path})"
                )

    def train(self, data: pd.DataFrame, material: str) -> Prophet:
        """
        Entraîne un modèle Prophet pour un matériau donné.

        Le modèle est configuré avec :
            - Saisonnalité annuelle activée
            - Saisonnalité hebdomadaire activée
            - changepoint_prior_scale = 0.05 (flexibilité modérée)

        Les données doivent contenir deux colonnes :
            - 'ds' : dates (format datetime ou string parsable)
            - 'y' : prix du matériau

        Args:
            data (pd.DataFrame): DataFrame avec colonnes 'ds' (date) et 'y' (prix).
            material (str): Nom du matériau à entraîner.

        Returns:
            Prophet: Modèle Prophet entraîné.

        Raises:
            ValueError: Si le matériau n'est pas dans la liste des matériaux
                supportés ou si les données sont invalides.
        """
        # Validation du matériau
        material = material.lower().strip()
        if material not in self.MATERIALS:
            raise ValueError(
                f"Matériau non supporté : '{material}'. "
                f"Matériaux disponibles : {', '.join(self.MATERIALS)}"
            )

        # Validation des données
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Les données doivent être un DataFrame pandas.")

        required_cols = ['ds', 'y']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Colonnes manquantes dans les données : {missing_cols}. "
                f"Le DataFrame doit contenir les colonnes 'ds' (date) et 'y' (prix)."
            )

        logger.info(
            f"Entraînement du modèle Prophet pour '{material}' "
            f"({len(data)} observations)..."
        )

        # Préparer les données
        train_data = data[['ds', 'y']].copy()
        train_data['ds'] = pd.to_datetime(train_data['ds'])

        # Créer et configurer le modèle Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )

        # Entraîner le modèle
        model.fit(train_data)

        # Sauvegarder le modèle
        self.models[material] = model
        self._save_model(model, material)

        logger.info(
            f"Modèle Prophet entraîné et sauvegardé pour '{material}'."
        )

        return model

    def predict(
        self,
        material: str,
        periods: int = 30
    ) -> dict:
        """
        Prédit les prix futurs d'un matériau recyclable.

        Génère des prévisions pour le nombre de périodes (jours)
        spécifié, avec les intervalles de confiance.

        Args:
            material (str): Nom du matériau à prévoir.
            periods (int): Nombre de jours à prévoir. Par défaut : 30.

        Returns:
            dict: Dictionnaire contenant :
                - 'material' (str): Nom du matériau.
                - 'predictions' (list[dict]): Liste de prédictions, chacune avec :
                    - 'date' (str): Date de la prédiction (format YYYY-MM-DD).
                    - 'price' (float): Prix prédit.
                    - 'price_min' (float): Borne inférieure de l'intervalle de confiance.
                    - 'price_max' (float): Borne supérieure de l'intervalle de confiance.

        Raises:
            ValueError: Si le matériau n'est pas supporté ou si aucun
                modèle n'a été entraîné pour ce matériau.
        """
        material = material.lower().strip()

        # Vérifier que le matériau est supporté
        if material not in self.MATERIALS:
            raise ValueError(
                f"Matériau non supporté : '{material}'. "
                f"Matériaux disponibles : {', '.join(self.MATERIALS)}"
            )

        # Vérifier qu'un modèle existe pour ce matériau
        if material not in self.models:
            raise ValueError(
                f"Aucun modèle entraîné pour '{material}'. "
                "Veuillez d'abord entraîner le modèle avec la méthode train()."
            )

        logger.info(
            f"Prédiction des prix pour '{material}' "
            f"sur {periods} jours..."
        )

        model = self.models[material]

        # Créer un DataFrame de dates futures
        future = model.make_future_dataframe(periods=periods)

        # Effectuer la prédiction
        forecast = model.predict(future)

        # Extraire uniquement les prédictions futures
        forecast_future = forecast.tail(periods)

        # Construire la liste de prédictions
        predictions = []
        for _, row in forecast_future.iterrows():
            predictions.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'price': round(float(row['yhat']), 2),
                'price_min': round(float(row['yhat_lower']), 2),
                'price_max': round(float(row['yhat_upper']), 2),
            })

        result = {
            'material': material,
            'predictions': predictions,
        }

        logger.info(
            f"Prédictions générées pour '{material}' : "
            f"{len(predictions)} jours, "
            f"prix moyen prédit : {sum(p['price'] for p in predictions) / len(predictions):.2f}"
        )

        return result

    def _save_model(self, model: Prophet, material: str) -> None:
        """
        Sauvegarde un modèle Prophet sur le disque.

        Args:
            model (Prophet): Modèle Prophet à sauvegarder.
            material (str): Nom du matériau (utilisé pour le nommage du fichier).
        """
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(
            SAVED_MODELS_DIR,
            f'price_model_{material}.pkl'
        )

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Modèle sauvegardé : {model_path}")

    def get_available_models(self) -> list:
        """
        Retourne la liste des matériaux pour lesquels un modèle est disponible.

        Returns:
            list: Liste des noms de matériaux avec un modèle entraîné.
        """
        return list(self.models.keys())

    def get_supported_materials(self) -> list:
        """
        Retourne la liste des matériaux supportés.

        Returns:
            list: Liste des noms de matériaux supportés.
        """
        return self.MATERIALS.copy()
