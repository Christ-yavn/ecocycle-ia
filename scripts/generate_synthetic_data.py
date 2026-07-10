#!/usr/bin/env python3
"""
Script de génération de données synthétiques pour EcoLoop AI.

Ce script génère trois types de données synthétiques :
- Données de prix des matériaux recyclables (plastique, métal, verre, papier)
- Données de volumes de collecte par zone géographique
- Données de transactions avec injection de fraudes

Utilisation :
    python scripts/generate_synthetic_data.py

Les fichiers CSV générés sont sauvegardés dans data/synthetic/.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_price_data(material: str, days: int = 365) -> pd.DataFrame:
    """
    Génère des données synthétiques de prix pour un matériau recyclable.

    Le prix simulé combine trois composantes :
    - Une tendance linéaire légère (hausse progressive)
    - Une saisonnalité sinusoïdale (cycle annuel)
    - Un bruit gaussien aléatoire

    Args:
        material: Nom du matériau ('plastique', 'metal', 'verre', 'papier')
        days: Nombre de jours à simuler (défaut: 365)

    Returns:
        DataFrame avec colonnes 'date' et 'price'
    """
    # Prix de base par matériau (€/kg)
    prix_base = {
        'plastique': 0.30,
        'metal': 1.50,
        'verre': 0.10,
        'papier': 0.15
    }

    base = prix_base.get(material, 0.20)
    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    # Composante tendance linéaire (légère hausse)
    tendance = np.linspace(0, base * 0.1, days)

    # Composante saisonnalité sinusoïdale (cycle annuel)
    saisonnalite = base * 0.15 * np.sin(2 * np.pi * np.arange(days) / 365)

    # Composante bruit gaussien
    bruit = np.random.normal(0, base * 0.05, days)

    # Prix final = base + tendance + saisonnalité + bruit
    prix = base + tendance + saisonnalite + bruit
    # S'assurer que les prix restent positifs
    prix = np.maximum(prix, 0.01)

    df = pd.DataFrame({
        'date': dates,
        'price': np.round(prix, 4)
    })

    return df


def generate_volume_data(zones: int = 5, days: int = 365) -> pd.DataFrame:
    """
    Génère des données synthétiques de volumes de collecte par zone.

    Chaque zone a un volume de base aléatoire. Les volumes sont modulés par :
    - Un facteur weekend (réduction à 30% le samedi/dimanche)
    - Une saisonnalité mensuelle
    - Un bruit gaussien

    Args:
        zones: Nombre de zones géographiques (défaut: 5)
        days: Nombre de jours à simuler (défaut: 365)

    Returns:
        DataFrame avec colonnes 'zone_id', 'date', 'volume', 'day_of_week'
    """
    donnees = []

    for zone_id in range(1, zones + 1):
        # Volume de base aléatoire entre 100 et 500 kg par zone
        volume_base = np.random.uniform(100, 500)
        dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

        for i, date in enumerate(dates):
            jour_semaine = date.weekday()  # 0=lundi, 6=dimanche

            # Facteur weekend : réduction à 30%
            facteur_weekend = 0.3 if jour_semaine >= 5 else 1.0

            # Saisonnalité sinusoïdale
            saisonnalite = 1 + 0.2 * np.sin(2 * np.pi * i / 365)

            # Bruit gaussien
            bruit = np.random.normal(1, 0.1)

            # Volume final
            volume = volume_base * facteur_weekend * saisonnalite * bruit
            volume = max(volume, 0)

            donnees.append({
                'zone_id': zone_id,
                'date': date,
                'volume': round(volume, 2),
                'day_of_week': jour_semaine
            })

    df = pd.DataFrame(donnees)
    return df


def generate_transaction_data(n: int = 1000) -> pd.DataFrame:
    """
    Génère des données synthétiques de transactions avec injection de fraudes.

    5% des transactions sont frauduleuses avec des caractéristiques distinctes :
    - Poids anormalement élevé (500-5000 kg)
    - Prix par kg suspects (5-20 €/kg)
    - Heures nocturnes (0h-4h et 23h)

    Les 95% restants sont des transactions normales :
    - Poids standard (10-200 kg)
    - Prix par kg normal (0.5-3 €/kg)
    - Heures de bureau (6h-20h)

    Args:
        n: Nombre total de transactions à générer (défaut: 1000)

    Returns:
        DataFrame avec colonnes : id, user_id, poids, prix, date, heure,
        jour_semaine, is_fraud
    """
    donnees = []
    nb_fraudes = int(n * 0.05)  # 5% de fraudes
    nb_normales = n - nb_fraudes

    # Heures nocturnes pour les fraudes
    heures_nocturnes = [0, 1, 2, 3, 4, 23]

    for i in range(n):
        is_fraud = i < nb_fraudes  # Les premières entrées sont des fraudes

        date = datetime.now() - timedelta(days=np.random.randint(0, 365))

        if is_fraud:
            # Transaction frauduleuse
            poids = np.random.uniform(500, 5000)
            prix_kg = np.random.uniform(5, 20)
            heure = np.random.choice(heures_nocturnes)
        else:
            # Transaction normale
            poids = np.random.uniform(10, 200)
            prix_kg = np.random.uniform(0.5, 3)
            heure = np.random.randint(6, 21)  # 6h à 20h

        prix_total = round(poids * prix_kg, 2)

        donnees.append({
            'id': i + 1,
            'user_id': f'USR_{np.random.randint(1000, 9999)}',
            'poids': round(poids, 2),
            'prix': prix_total,
            'date': date.strftime('%Y-%m-%d'),
            'heure': heure,
            'jour_semaine': date.weekday(),
            'is_fraud': int(is_fraud)
        })

    # Mélanger les données pour ne pas avoir les fraudes en premier
    df = pd.DataFrame(donnees)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


if __name__ == '__main__':
    """Point d'entrée principal pour la génération de données synthétiques."""

    print("=" * 60)
    print("  EcoLoop AI - Génération de données synthétiques")
    print("=" * 60)

    # Créer le répertoire de sortie
    os.makedirs('data/synthetic', exist_ok=True)

    # --- Génération des données de prix ---
    print("\n📊 Génération des données de prix par matériau...")
    materiaux = ['plastique', 'metal', 'verre', 'papier']
    for mat in materiaux:
        df_prix = generate_price_data(mat)
        chemin = f'data/synthetic/prix_{mat}.csv'
        df_prix.to_csv(chemin, index=False)
        print(f"  ✓ {mat.capitalize()} : {len(df_prix)} entrées → {chemin}")

    # --- Génération des données de volumes ---
    print("\n📦 Génération des données de volumes de collecte...")
    df_volumes = generate_volume_data(zones=5, days=365)
    chemin_volumes = 'data/synthetic/volumes_collecte.csv'
    df_volumes.to_csv(chemin_volumes, index=False)
    print(f"  ✓ Volumes : {len(df_volumes)} entrées ({df_volumes['zone_id'].nunique()} zones) → {chemin_volumes}")

    # --- Génération des données de transactions ---
    print("\n💳 Génération des données de transactions...")
    df_transactions = generate_transaction_data(n=1000)
    chemin_transactions = 'data/synthetic/transactions.csv'
    df_transactions.to_csv(chemin_transactions, index=False)
    nb_fraudes = df_transactions['is_fraud'].sum()
    print(f"  ✓ Transactions : {len(df_transactions)} entrées ({nb_fraudes} fraudes, {len(df_transactions) - nb_fraudes} normales) → {chemin_transactions}")

    # --- Résumé final ---
    print("\n" + "=" * 60)
    print("  ✅ Génération terminée avec succès !")
    print("=" * 60)
    print(f"\n  Fichiers générés dans : data/synthetic/")
    print(f"  - 4 fichiers de prix (1 par matériau)")
    print(f"  - 1 fichier de volumes de collecte")
    print(f"  - 1 fichier de transactions (avec {nb_fraudes} fraudes)")
    print(f"\n  Total : 6 fichiers CSV")
