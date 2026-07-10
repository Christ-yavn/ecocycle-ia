"""
Module de prédiction pour EcoLoop AI.

Ce sous-package contient les modèles de prédiction des prix des matériaux
recyclables et des volumes de collecte de déchets, ainsi que les utilitaires
de traitement de données associés.
"""

from .price_prediction import PricePredictor
from .volume_prediction import VolumePredictor
from .data_utils import load_csv_data, clean_data, split_by_material, validate_date_format
