#!/usr/bin/env python3
"""
Script de réentraînement des modèles IA pour EcoLoop AI.

Ce script réentraîne trois types de modèles en utilisant les classes du projet :
- Modèles de prédiction de prix (Prophet) pour chaque matériau
- Modèle de prédiction de volumes (XGBoost)
- Modèle de détection de fraudes (Isolation Forest)

Utilisation :
    python scripts/retrain_models.py
"""

import os
import sys
from datetime import datetime

# Ajouter le répertoire racine au path pour les imports
sys.path.append('.')

import pandas as pd

from models.prediction.price_prediction import PricePredictor
from models.prediction.volume_prediction import VolumePredictor
from models.fraud_detection.fraud_model import FraudDetector

def retrain_price_models() -> dict:
    """Réentraîne les modèles Prophet via la classe PricePredictor."""
    predictor = PricePredictor()
    materiaux = predictor.get_supported_materials()
    resultats = {}

    for mat in materiaux:
        try:
            chemin_csv = f'data/synthetic/prix_{mat}.csv'
            print(f"    Chargement de {chemin_csv}...")
            df = pd.read_csv(chemin_csv)
            # La classe attend des colonnes 'ds' et 'y'
            df = df.rename(columns={'date': 'ds', 'price': 'y'})
            
            predictor.train(df, mat)
            print(f"    ✓ Modèle Prophet ({mat}) entraîné et sauvegardé.")
            resultats[mat] = True
        except Exception as e:
            print(f"    ✗ Erreur pour {mat} : {e}")
            resultats[mat] = False

    return resultats


def retrain_volume_model() -> bool:
    """Réentraîne le modèle XGBoost via la classe VolumePredictor."""
    try:
        chemin_csv = 'data/synthetic/volumes_collecte.csv'
        print(f"    Chargement de {chemin_csv}...")
        df = pd.read_csv(chemin_csv)
        
        # La classe attend 'zone_id', 'date', 'volume_kg'
        df = df.rename(columns={'volume': 'volume_kg'})
        
        predictor = VolumePredictor()
        predictor.train(df)
        print(f"    ✓ Modèle XGBoost (volumes) entraîné et sauvegardé.")
        return True
    except Exception as e:
        print(f"    ✗ Erreur : {e}")
        return False


def retrain_fraud_model() -> bool:
    """Réentraîne le modèle Isolation Forest via la classe FraudDetector."""
    try:
        chemin_csv = 'data/synthetic/transactions.csv'
        print(f"    Chargement de {chemin_csv}...")
        df = pd.read_csv(chemin_csv)
        
        predictor = FraudDetector()
        predictor.train(df)
        print(f"    ✓ Modèle Isolation Forest (fraudes) entraîné et sauvegardé.")
        return True
    except Exception as e:
        print(f"    ✗ Erreur : {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("  EcoLoop AI - Réentraînement des modèles")
    print("=" * 60)
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs('saved_models', exist_ok=True)

    print("\n📈 Réentraînement des modèles de prix (Prophet)...")
    resultats_prix = retrain_price_models()

    print("\n📦 Réentraînement du modèle de volumes (XGBoost)...")
    succes_volumes = retrain_volume_model()

    print("\n🔍 Réentraînement du modèle de fraudes (Isolation Forest)...")
    succes_fraudes = retrain_fraud_model()

    print("\n" + "=" * 60)
    print("  Résumé du réentraînement")
    print("=" * 60)

    for mat, succes in resultats_prix.items():
        symbole = "✓" if succes else "✗"
        print(f"  {symbole} Prophet ({mat})")

    symbole_vol = "✓" if succes_volumes else "✗"
    print(f"  {symbole_vol} XGBoost (volumes)")

    symbole_fraud = "✓" if succes_fraudes else "✗"
    print(f"  {symbole_fraud} Isolation Forest (fraudes)")

    tout_ok = all(resultats_prix.values()) and succes_volumes and succes_fraudes
    if tout_ok:
        print("\n  ✅ Tous les modèles ont été réentraînés avec succès !")
    else:
        print("\n  ⚠️  Certains modèles n'ont pas pu être réentraînés.")
    print("=" * 60)
