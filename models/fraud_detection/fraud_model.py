"""Module de détection de fraude dans les transactions.

Utilise Isolation Forest (Scikit-learn) pour détecter les comportements
anormaux dans les transactions de recyclage.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class FraudDetector:
    """Détecteur de fraude basé sur Isolation Forest.
    
    Analyse les transactions pour détecter les anomalies
    (pesées falsifiées, volumes incohérents, transactions suspectes).
    """
    
    RISK_LEVELS = {
        'normal': (0, 30),
        'suspect': (30, 70),
        'fraude': (70, 100)
    }
    
    def __init__(self, models_dir='saved_models'):
        """Initialise le détecteur et charge le modèle existant.
        
        Args:
            models_dir: Répertoire contenant les modèles sauvés
        """
        self.model = None
        self.scaler = StandardScaler()
        self.models_dir = models_dir
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle et le scaler depuis les fichiers sauvés."""
        model_path = os.path.join(self.models_dir, 'fraud_model.pkl')
        scaler_path = os.path.join(self.models_dir, 'fraud_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("[OK] Modèle fraude chargé")
            except Exception as e:
                print(f"[ERREUR] Impossible de charger le modèle fraude : {e}")
    
    def train(self, data):
        """Entraîne le modèle de détection de fraude.
        
        Args:
            data: Liste de dicts ou DataFrame avec les colonnes de transactions
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Extraire les features
        features = self._extract_features(df)
        
        # Normalisation
        features_scaled = self.scaler.fit_transform(features)
        
        # Entraîner Isolation Forest
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            max_samples='auto'
        )
        self.model.fit(features_scaled)
        
        # Sauvegarder
        os.makedirs(self.models_dir, exist_ok=True)
        with open(os.path.join(self.models_dir, 'fraud_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.models_dir, 'fraud_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("[OK] Modèle de fraude entraîné et sauvegardé")
    
    def check_transaction(self, transaction):
        """Vérifie si une transaction est frauduleuse.
        
        Args:
            transaction: Dict avec les clés 'poids', 'prix', et optionnellement
                        'heure', 'jour_semaine', 'id'
        
        Returns:
            Dict avec risk_score (0-100), risk_level, is_fraud, reasons, transaction_id
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Entraînez ou chargez un modèle d'abord.")
        
        features = self._transaction_to_features(transaction)
        features_scaled = self.scaler.transform(features)
        
        # Score d'anomalie (-1 = anomalie, 1 = normal)
        raw_score = self.model.decision_function(features_scaled)[0]
        
        # Convertir en score de risque 0-100
        risk_score = self._normalize_score(raw_score)
        
        # Déterminer le niveau de risque
        risk_level = self._get_risk_level(risk_score)
        
        # Identifier les raisons
        reasons = self._identify_reasons(transaction, risk_score)
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'is_fraud': risk_level == 'fraude',
            'reasons': reasons,
            'transaction_id': transaction.get('id', 'N/A')
        }
    
    def _extract_features(self, df):
        """Extrait les features pour l'entraînement.
        
        Args:
            df: DataFrame de transactions
        
        Returns:
            DataFrame de features
        """
        features = pd.DataFrame()
        features['poids'] = pd.to_numeric(df['poids'], errors='coerce').fillna(0)
        features['prix'] = pd.to_numeric(df['prix'], errors='coerce').fillna(0)
        features['heure'] = pd.to_numeric(df.get('heure', pd.Series([12]*len(df))), errors='coerce').fillna(12)
        features['jour_semaine'] = pd.to_numeric(df.get('jour_semaine', pd.Series([0]*len(df))), errors='coerce').fillna(0)
        features['prix_par_kg'] = features['prix'] / features['poids'].replace(0, 0.01)
        features['ecart_prix_moyen'] = abs(features['prix'] - features['prix'].mean()) / max(features['prix'].std(), 0.01)
        return features
    
    def _transaction_to_features(self, transaction):
        """Convertit une transaction en features pour la prédiction.
        
        Args:
            transaction: Dict de transaction
        
        Returns:
            DataFrame avec une seule ligne de features
        """
        poids = float(transaction.get('poids', 0))
        prix = float(transaction.get('prix', 0))
        data = {
            'poids': [poids],
            'prix': [prix],
            'heure': [int(transaction.get('heure', 12))],
            'jour_semaine': [int(transaction.get('jour_semaine', 0))],
            'prix_par_kg': [prix / max(poids, 0.01)],
            'ecart_prix_moyen': [0]  # Calculé dynamiquement en production
        }
        return pd.DataFrame(data)
    
    def _normalize_score(self, raw_score):
        """Normalise le score brut d'Isolation Forest en score 0-100.
        
        Plus le score est négatif, plus la transaction est suspecte.
        
        Args:
            raw_score: Score brut d'Isolation Forest
        
        Returns:
            Score de risque normalisé entre 0 et 100
        """
        score = max(0, min(100, 50 - raw_score * 50))
        return score
    
    def _get_risk_level(self, score):
        """Retourne le niveau de risque basé sur le score.
        
        Args:
            score: Score de risque (0-100)
        
        Returns:
            Niveau de risque : 'normal', 'suspect', ou 'fraude'
        """
        for level, (low, high) in self.RISK_LEVELS.items():
            if low <= score < high:
                return level
        return 'fraude'
    
    def _identify_reasons(self, transaction, risk_score):
        """Identifie les raisons potentielles d'une fraude.
        
        Args:
            transaction: Dict de transaction
            risk_score: Score de risque calculé
        
        Returns:
            Liste de raisons détectées (str)
        """
        reasons = []
        poids = float(transaction.get('poids', 0))
        prix = float(transaction.get('prix', 0))
        heure = int(transaction.get('heure', 12))
        
        if poids > 1000:
            reasons.append("Poids anormalement élevé")
        if poids > 0 and prix / poids > 10:
            reasons.append("Prix par kg anormalement élevé")
        if heure < 6 or heure > 22:
            reasons.append("Transaction à une heure inhabituelle")
        if poids <= 0:
            reasons.append("Poids nul ou négatif")
        if prix <= 0:
            reasons.append("Prix nul ou négatif")
        if not reasons and risk_score > 50:
            reasons.append("Combinaison de facteurs inhabituels détectée par le modèle")
        
        return reasons
