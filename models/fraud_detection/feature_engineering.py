"""Module d'ingénierie de features pour la détection de fraude.

Contient des fonctions pour créer des features avancées
à partir des données de transactions brutes.
"""

import pandas as pd
import numpy as np


def engineer_fraud_features(transactions_df):
    """Crée des features avancées pour la détection de fraude.
    
    Args:
        transactions_df: DataFrame de transactions brutes
    
    Returns:
        DataFrame enrichi avec les features d'ingénierie
    """
    df = transactions_df.copy()
    
    # ---- Features temporelles ----
    df['date'] = pd.to_datetime(df['date'])
    df['heure'] = df['date'].dt.hour
    df['jour_semaine'] = df['date'].dt.dayofweek
    df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
    df['mois'] = df['date'].dt.month
    
    # ---- Features de prix ----
    df['prix_par_kg'] = df['prix'] / df['poids'].replace(0, 0.01)
    
    # ---- Features par utilisateur ----
    if 'user_id' in df.columns:
        user_stats = df.groupby('user_id').agg({
            'poids': ['mean', 'std'],
            'prix': ['mean', 'std'],
            'prix_par_kg': ['mean']
        }).reset_index()
        
        user_stats.columns = [
            'user_id', 'poids_moyen', 'poids_std',
            'prix_moyen', 'prix_std', 'prix_kg_moyen'
        ]
        
        # Remplir les std NaN (si un seul enregistrement)
        user_stats['poids_std'] = user_stats['poids_std'].fillna(0)
        user_stats['prix_std'] = user_stats['prix_std'].fillna(0)
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Écarts par rapport à la moyenne de l'utilisateur
        df['ecart_poids'] = abs(df['poids'] - df['poids_moyen']) / df['poids_std'].replace(0, 1)
        df['ecart_prix'] = abs(df['prix'] - df['prix_moyen']) / df['prix_std'].replace(0, 1)
        
        # Fréquence de transactions par jour
        df['nb_transactions_jour'] = df.groupby(
            ['user_id', df['date'].dt.date]
        )['prix'].transform('count')
    
    return df


def get_feature_importance(df, target_col='is_fraud'):
    """Calcule l'importance relative des features.
    
    Args:
        df: DataFrame avec les features et la colonne cible
        target_col: Nom de la colonne cible
    
    Returns:
        Dict {feature_name: importance_score}
    """
    if target_col not in df.columns:
        return {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    importance = {}
    for col in numeric_cols:
        try:
            correlation = abs(df[col].corr(df[target_col]))
            importance[col] = round(correlation, 4)
        except Exception:
            importance[col] = 0.0
    
    # Trier par importance décroissante
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance
