"""
Modèle de prédiction des volumes de collecte de déchets - EcoLoop AI.

Ce module implémente un système de prédiction des volumes de déchets
collectés par zone géographique, basé sur XGBoost. Il utilise des
features temporelles pour anticiper les volumes de collecte.

Auteur : EcoLoop AI Team
"""

import os
import logging
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Répertoire de sauvegarde des modèles
SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'saved_models'
)


class VolumePredictor:
    """
    Prédicteur de volumes de collecte de déchets basé sur XGBoost.

    Utilise un modèle XGBRegressor pour prédire les volumes de déchets
    collectés par zone géographique, en se basant sur des features
    temporelles (jour de la semaine, mois, week-end, etc.).

    Attributes:
        model (XGBRegressor): Modèle XGBoost entraîné.
        feature_columns (list): Liste des colonnes de features utilisées.
    """

    def __init__(self):
        """
        Initialise le prédicteur de volumes.

        Tente de charger un modèle pré-entraîné depuis le disque.
        Si aucun modèle n'est trouvé, initialise le modèle à None.
        """
        self.model: Optional[XGBRegressor] = None
        self.feature_columns = [
            'zone_id', 'day_of_week', 'month', 'day_of_month', 'is_weekend'
        ]
        self._load_model()

    def _load_model(self) -> None:
        """
        Charge le modèle XGBoost sauvegardé depuis le disque.

        Recherche le fichier volume_model.pkl dans le répertoire
        saved_models/. Si le fichier n'existe pas, le modèle
        reste à None et devra être entraîné.
        """
        model_path = os.path.join(SAVED_MODELS_DIR, 'volume_model.pkl')

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Modèle XGBoost chargé depuis : {model_path}")
            except Exception as e:
                logger.warning(
                    f"Impossible de charger le modèle XGBoost : {str(e)}"
                )
                self.model = None
        else:
            logger.info(
                f"Aucun modèle trouvé à {model_path}. "
                "Le modèle devra être entraîné."
            )

    def train(self, data: pd.DataFrame) -> dict:
        """
        Entraîne le modèle XGBoost de prédiction de volumes.

        Configuration du modèle :
            - n_estimators : 100
            - max_depth : 6
            - learning_rate : 0.1
            - subsample : 0.8
            - colsample_bytree : 0.8

        Le jeu de données est divisé en 80% entraînement / 20% test.
        L'erreur absolue moyenne (MAE) est calculée sur le jeu de test.

        Args:
            data (pd.DataFrame): DataFrame contenant les colonnes nécessaires :
                - 'zone_id' (int): Identifiant de la zone de collecte.
                - 'date' (str/datetime): Date de la collecte.
                - 'volume_kg' (float): Volume collecté en kilogrammes (cible).

        Returns:
            dict: Dictionnaire contenant :
                - 'mae' (float): Erreur absolue moyenne sur le jeu de test.
                - 'model_path' (str): Chemin du modèle sauvegardé.
                - 'n_samples' (int): Nombre total d'échantillons.
                - 'n_features' (int): Nombre de features utilisées.

        Raises:
            ValueError: Si les données sont invalides ou insuffisantes.
        """
        logger.info("Début de l'entraînement du modèle de prédiction de volumes...")

        # Validation des données
        required_cols = ['zone_id', 'date', 'volume_kg']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Colonnes manquantes : {missing_cols}. "
                f"Colonnes requises : {required_cols}"
            )

        # Créer les features temporelles
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = self._create_features(df)

        # Séparer features et cible
        X = df[self.feature_columns]
        y = df['volume_kg']

        logger.info(
            f"Données préparées : {len(X)} échantillons, "
            f"{len(self.feature_columns)} features"
        )

        # Division train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(
            f"Division : {len(X_train)} entraînement, "
            f"{len(X_test)} test"
        )

        # Créer et entraîner le modèle XGBoost
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Évaluation
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"MAE sur le jeu de test : {mae:.2f} kg")

        # Sauvegarde du modèle
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(SAVED_MODELS_DIR, 'volume_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Modèle sauvegardé : {model_path}")

        return {
            'mae': round(mae, 2),
            'model_path': model_path,
            'n_samples': len(data),
            'n_features': len(self.feature_columns),
        }

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features temporelles à partir de la colonne 'date'.

        Features générées :
            - day_of_week (int): Jour de la semaine (0=lundi, 6=dimanche).
            - month (int): Mois de l'année (1-12).
            - day_of_month (int): Jour du mois (1-31).
            - is_weekend (int): 1 si samedi/dimanche, 0 sinon.

        Args:
            df (pd.DataFrame): DataFrame avec une colonne 'date' en datetime.

        Returns:
            pd.DataFrame: DataFrame enrichi avec les features temporelles.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Extraction des features temporelles
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        logger.debug(
            f"Features temporelles créées : "
            f"day_of_week, month, day_of_month, is_weekend"
        )

        return df

    def predict(
        self,
        zone_id: int,
        date: str,
        features: Optional[Dict] = None
    ) -> dict:
        """
        Prédit le volume de déchets collectés pour une zone et une date données.

        Args:
            zone_id (int): Identifiant de la zone de collecte.
            date (str): Date de la prédiction au format 'YYYY-MM-DD'.
            features (dict, optional): Features supplémentaires à inclure.
                Si None, seules les features temporelles sont utilisées.

        Returns:
            dict: Dictionnaire contenant :
                - 'zone_id' (int): Identifiant de la zone.
                - 'date' (str): Date de la prédiction.
                - 'predicted_volume_kg' (float): Volume prédit en kg.
                - 'confidence' (str): Niveau de confiance ('haute', 'moyenne', 'basse').

        Raises:
            ValueError: Si le modèle n'a pas été entraîné ou si les
                paramètres sont invalides.
        """
        if self.model is None:
            raise ValueError(
                "Aucun modèle disponible. Veuillez d'abord entraîner "
                "le modèle avec la méthode train()."
            )

        # Préparer les données d'entrée
        input_df = self._prepare_input(zone_id, date, features)

        # Effectuer la prédiction
        prediction = self.model.predict(input_df)[0]

        # Déterminer le niveau de confiance basé sur la profondeur de l'arbre
        # et le nombre d'estimateurs ayant contribué
        confidence = self._estimate_confidence(input_df)

        result = {
            'zone_id': zone_id,
            'date': date,
            'predicted_volume_kg': round(float(prediction), 2),
            'confidence': confidence,
        }

        logger.info(
            f"Prédiction pour zone {zone_id} le {date} : "
            f"{result['predicted_volume_kg']} kg "
            f"(confiance : {confidence})"
        )

        return result

    def _prepare_input(
        self,
        zone_id: int,
        date: str,
        features: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Prépare un DataFrame d'entrée pour la prédiction.

        Convertit les paramètres d'entrée en un DataFrame avec les
        features temporelles nécessaires au modèle.

        Args:
            zone_id (int): Identifiant de la zone de collecte.
            date (str): Date au format 'YYYY-MM-DD'.
            features (dict, optional): Features supplémentaires.

        Returns:
            pd.DataFrame: DataFrame prêt pour la prédiction avec les
                colonnes correspondant aux features du modèle.

        Raises:
            ValueError: Si le format de la date est invalide.
        """
        try:
            date_dt = pd.to_datetime(date)
        except (ValueError, TypeError):
            raise ValueError(
                f"Format de date invalide : '{date}'. "
                "Utilisez le format 'YYYY-MM-DD'."
            )

        # Construire le dictionnaire de features
        input_data = {
            'zone_id': [zone_id],
            'day_of_week': [date_dt.dayofweek],
            'month': [date_dt.month],
            'day_of_month': [date_dt.day],
            'is_weekend': [1 if date_dt.dayofweek >= 5 else 0],
        }

        # Ajouter les features supplémentaires si fournies
        if features:
            for key, value in features.items():
                if key in self.feature_columns:
                    input_data[key] = [value]

        # Créer le DataFrame
        input_df = pd.DataFrame(input_data)

        # S'assurer que l'ordre des colonnes est correct
        input_df = input_df[self.feature_columns]

        return input_df

    def _estimate_confidence(self, input_df: pd.DataFrame) -> str:
        """
        Estime le niveau de confiance de la prédiction.

        Le niveau de confiance est déterminé en analysant la variance
        des prédictions individuelles des arbres du modèle.

        Args:
            input_df (pd.DataFrame): DataFrame d'entrée pour la prédiction.

        Returns:
            str: Niveau de confiance parmi 'haute', 'moyenne', 'basse'.
        """
        try:
            # Obtenir les prédictions de chaque arbre
            booster = self.model.get_booster()
            import xgboost as xgb
            dmatrix = xgb.DMatrix(input_df)

            # Calculer la variance des prédictions par arbre
            tree_predictions = []
            for i in range(self.model.n_estimators):
                pred = booster.predict(
                    dmatrix,
                    iteration_range=(0, i + 1)
                )
                tree_predictions.append(pred[0])

            variance = np.var(tree_predictions[-10:])  # Variance des 10 derniers

            # Déterminer le niveau de confiance
            if variance < 50:
                return 'haute'
            elif variance < 200:
                return 'moyenne'
            else:
                return 'basse'

        except Exception:
            # En cas d'erreur, retourner une confiance moyenne par défaut
            return 'moyenne'
