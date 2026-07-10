"""
Module de détection de fraudes pour EcoLoop AI.

Ce sous-package contient le modèle de détection de fraudes basé sur
Isolation Forest, ainsi que les utilitaires d'ingénierie de features
pour l'analyse des transactions suspectes.
"""

from .fraud_model import FraudDetector
from .feature_engineering import engineer_fraud_features
