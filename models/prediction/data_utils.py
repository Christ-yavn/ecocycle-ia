"""
Utilitaires de données pour les modèles de prédiction - EcoLoop AI.

Ce module fournit des fonctions utilitaires pour le chargement,
le nettoyage et la préparation des données utilisées par les modèles
de prédiction de prix et de volumes de déchets.

Auteur : EcoLoop AI Team
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV en DataFrame pandas.

    Tente de détecter automatiquement le séparateur et l'encodage
    du fichier CSV.

    Args:
        filepath (str): Chemin vers le fichier CSV à charger.

    Returns:
        pd.DataFrame: DataFrame contenant les données du CSV.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        pd.errors.EmptyDataError: Si le fichier est vide.
        pd.errors.ParserError: Si le fichier ne peut pas être parsé.

    Exemple:
        >>> df = load_csv_data('data/prix_materiaux.csv')
        >>> print(df.shape)
        (1000, 5)
    """
    try:
        # Tenter le chargement avec UTF-8 d'abord
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(
            f"Fichier CSV chargé avec succès : {filepath} "
            f"({df.shape[0]} lignes, {df.shape[1]} colonnes)"
        )
        return df

    except UnicodeDecodeError:
        # Fallback sur l'encodage latin-1
        logger.warning(
            f"Encodage UTF-8 échoué pour {filepath}, "
            "tentative avec latin-1..."
        )
        df = pd.read_csv(filepath, encoding='latin-1')
        logger.info(
            f"Fichier CSV chargé avec encodage latin-1 : {filepath} "
            f"({df.shape[0]} lignes, {df.shape[1]} colonnes)"
        )
        return df

    except FileNotFoundError:
        logger.error(f"Fichier introuvable : {filepath}")
        raise FileNotFoundError(
            f"Le fichier CSV n'existe pas : {filepath}"
        )

    except pd.errors.EmptyDataError:
        logger.error(f"Fichier CSV vide : {filepath}")
        raise pd.errors.EmptyDataError(
            f"Le fichier CSV est vide : {filepath}"
        )

    except Exception as e:
        logger.error(f"Erreur lors du chargement du CSV : {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame en supprimant les valeurs manquantes et les doublons.

    Étapes de nettoyage :
        1. Suppression des lignes entièrement vides.
        2. Suppression des lignes avec des valeurs manquantes (NaN).
        3. Suppression des doublons exacts.
        4. Réinitialisation de l'index.

    Args:
        df (pd.DataFrame): DataFrame à nettoyer.

    Returns:
        pd.DataFrame: DataFrame nettoyé.

    Exemple:
        >>> df_brut = pd.DataFrame({'a': [1, None, 1], 'b': [2, None, 2]})
        >>> df_propre = clean_data(df_brut)
        >>> print(df_propre.shape)
        (1, 2)
    """
    taille_initiale = len(df)

    # Suppression des lignes entièrement vides
    df = df.dropna(how='all')
    apres_vides = len(df)

    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna()
    apres_nan = len(df)

    # Suppression des doublons
    df = df.drop_duplicates()
    apres_doublons = len(df)

    # Réinitialisation de l'index
    df = df.reset_index(drop=True)

    # Log des résultats du nettoyage
    logger.info(
        f"Nettoyage terminé : {taille_initiale} → {apres_doublons} lignes. "
        f"Supprimé : {taille_initiale - apres_vides} vides, "
        f"{apres_vides - apres_nan} NaN, "
        f"{apres_nan - apres_doublons} doublons."
    )

    return df


def split_by_material(
    df: pd.DataFrame,
    material_col: str = 'material'
) -> Dict[str, pd.DataFrame]:
    """
    Sépare un DataFrame en sous-DataFrames par type de matériau.

    Args:
        df (pd.DataFrame): DataFrame contenant une colonne de matériaux.
        material_col (str): Nom de la colonne contenant le type de matériau.
            Par défaut : 'material'.

    Returns:
        Dict[str, pd.DataFrame]: Dictionnaire où les clés sont les noms
            des matériaux et les valeurs sont les DataFrames correspondants.

    Raises:
        KeyError: Si la colonne spécifiée n'existe pas dans le DataFrame.

    Exemple:
        >>> df = pd.DataFrame({
        ...     'material': ['plastique', 'metal', 'plastique'],
        ...     'prix': [1.2, 3.5, 1.3]
        ... })
        >>> splits = split_by_material(df)
        >>> print(list(splits.keys()))
        ['plastique', 'metal']
    """
    if material_col not in df.columns:
        colonnes_disponibles = ', '.join(df.columns.tolist())
        raise KeyError(
            f"Colonne '{material_col}' introuvable dans le DataFrame. "
            f"Colonnes disponibles : {colonnes_disponibles}"
        )

    # Grouper par matériau et convertir en dictionnaire
    grouped = {
        material: group.reset_index(drop=True)
        for material, group in df.groupby(material_col)
    }

    # Log des résultats
    for material, group_df in grouped.items():
        logger.info(
            f"Matériau '{material}' : {len(group_df)} enregistrements"
        )

    logger.info(
        f"Données séparées en {len(grouped)} matériaux : "
        f"{', '.join(grouped.keys())}"
    )

    return grouped


def validate_date_format(
    date_str: str,
    expected_format: str = '%Y-%m-%d'
) -> bool:
    """
    Vérifie si une chaîne de caractères est dans le format de date attendu.

    Args:
        date_str (str): Chaîne de caractères représentant la date à valider.
        expected_format (str): Format de date attendu (notation strftime).
            Par défaut : '%Y-%m-%d' (ex: '2024-01-15').

    Returns:
        bool: True si la date est dans le format attendu, False sinon.

    Exemple:
        >>> validate_date_format('2024-01-15')
        True
        >>> validate_date_format('15/01/2024')
        False
        >>> validate_date_format('15/01/2024', '%d/%m/%Y')
        True
    """
    try:
        datetime.strptime(date_str, expected_format)
        logger.debug(f"Date '{date_str}' validée (format : {expected_format})")
        return True
    except ValueError:
        logger.warning(
            f"Date '{date_str}' invalide pour le format '{expected_format}'"
        )
        return False
