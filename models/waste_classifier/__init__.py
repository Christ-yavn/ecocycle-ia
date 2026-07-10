"""
Module de classification des déchets pour EcoLoop AI.

Ce sous-package contient le modèle de classification d'images de déchets
basé sur MobileNetV2 avec transfer learning, ainsi que les utilitaires
de prétraitement et d'entraînement associés.
"""

from .model import WasteClassifier
from .preprocess import preprocess_image, get_recycling_tips
