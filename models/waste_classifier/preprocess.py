"""
Prétraitement des images de déchets - EcoLoop AI.

Ce module fournit les fonctions utilitaires pour le prétraitement des images
de déchets avant classification, ainsi que les conseils de recyclage
associés à chaque catégorie de déchet.

Auteur : EcoLoop AI Team
"""

import io
import logging

import numpy as np
from PIL import Image

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Taille d'entrée du modèle
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Prétraite une image à partir de bytes pour la classification.

    Convertit des données brutes d'image (bytes) en un tableau numpy
    normalisé et prêt pour l'entrée du modèle MobileNetV2.

    Étapes :
        1. Décodage des bytes en image PIL
        2. Conversion en mode RGB
        3. Redimensionnement à 224x224
        4. Conversion en array numpy float32
        5. Normalisation MobileNetV2 (plage [-1, 1])
        6. Ajout de la dimension batch (expand_dims)

    Args:
        image_bytes (bytes): Données brutes de l'image (JPEG, PNG, etc.).

    Returns:
        np.ndarray: Image prétraitée de forme (1, 224, 224, 3),
            normalisée pour MobileNetV2.

    Raises:
        ValueError: Si les bytes ne peuvent pas être décodés en image valide.

    Exemple:
        >>> with open('photo_dechet.jpg', 'rb') as f:
        ...     image_bytes = f.read()
        >>> processed = preprocess_image(image_bytes)
        >>> print(processed.shape)
        (1, 224, 224, 3)
    """
    try:
        # Import paresseux : TensorFlow n'est requis que pour ce prétraitement
        # MobileNetV2 (V1). Le runtime YOLO (V2) ne l'utilise pas.
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        # Décoder les bytes en image PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Convertir en RGB (gérer les images RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Image convertie de {image.mode} vers RGB.")

        # Redimensionner à la taille d'entrée du modèle
        image = image.resize(IMG_SIZE, Image.LANCZOS)

        # Convertir en array numpy
        image_array = np.array(image, dtype=np.float32)

        # Normalisation spécifique à MobileNetV2 (plage [-1, 1])
        image_array = preprocess_input(image_array)

        # Ajouter la dimension batch
        image_array = np.expand_dims(image_array, axis=0)

        logger.info(
            f"Image prétraitée avec succès. "
            f"Forme : {image_array.shape}, "
            f"Plage : [{image_array.min():.2f}, {image_array.max():.2f}]"
        )

        return image_array

    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image : {str(e)}")
        raise ValueError(
            f"Impossible de prétraiter l'image : {str(e)}. "
            "Assurez-vous que les données sont une image valide (JPEG, PNG, etc.)."
        )


def get_recycling_tips(waste_type: str) -> dict:
    """
    Retourne les conseils de recyclage pour un type de déchet donné.

    Fournit des informations détaillées en français sur comment recycler
    ou traiter correctement chaque type de déchet.

    Args:
        waste_type (str): Type de déchet parmi les catégories supportées :
            'plastique', 'metal', 'verre', 'papier', 'organique', 'dangereux'.

    Returns:
        dict: Dictionnaire contenant :
            - 'categorie' (str): Nom de la catégorie.
            - 'poubelle' (str): Couleur/type de poubelle à utiliser.
            - 'conseils' (list[str]): Liste de conseils de tri.
            - 'exemples' (list[str]): Exemples d'objets de cette catégorie.
            - 'erreurs_courantes' (list[str]): Erreurs fréquentes à éviter.
            - 'impact_environnemental' (str): Impact du bon recyclage.

    Raises:
        ValueError: Si le type de déchet n'est pas reconnu.

    Exemple:
        >>> tips = get_recycling_tips('plastique')
        >>> print(tips['poubelle'])
        'Poubelle jaune (tri sélectif)'
    """
    # Dictionnaire complet des conseils de recyclage en français
    recycling_tips = {
        'plastique': {
            'categorie': 'Plastique',
            'poubelle': 'Poubelle jaune (tri sélectif)',
            'conseils': [
                'Videz et rincez les emballages plastiques avant de les jeter.',
                'Ne pas imbriquer les emballages les uns dans les autres.',
                'Écrasez les bouteilles dans le sens de la longueur, bouchon vissé.',
                'Les films plastiques et sacs sont désormais acceptés dans le tri.',
                'Inutile de retirer les étiquettes des bouteilles.',
            ],
            'exemples': [
                'Bouteilles d\'eau et de soda',
                'Flacons de produits ménagers',
                'Pots de yaourt',
                'Barquettes alimentaires',
                'Films et sacs plastiques',
            ],
            'erreurs_courantes': [
                'Jeter les jouets en plastique dans le tri (à déposer en déchèterie).',
                'Mettre les emballages souillés (huile, sauce) sans les rincer.',
                'Confondre polystyrène expansé avec plastique recyclable.',
            ],
            'impact_environnemental': (
                'Le recyclage d\'une tonne de plastique permet d\'économiser '
                '800 kg de pétrole brut et réduit les émissions de CO2 de 1,5 tonne.'
            ),
        },
        'metal': {
            'categorie': 'Métal',
            'poubelle': 'Poubelle jaune (tri sélectif)',
            'conseils': [
                'Videz les contenants métalliques avant de les jeter.',
                'Inutile de laver les boîtes de conserve, les vider suffit.',
                'Les aérosols vides vont dans le tri sélectif.',
                'Laissez le couvercle des boîtes de conserve attaché ou placez-le à l\'intérieur.',
                'Le papier aluminium peut être recyclé s\'il est propre et en boule.',
            ],
            'exemples': [
                'Boîtes de conserve',
                'Canettes de boissons',
                'Aérosols (déodorant, laque, etc.)',
                'Barquettes en aluminium',
                'Couvercles et capsules métalliques',
            ],
            'erreurs_courantes': [
                'Jeter les casseroles et poêles dans le tri (à déposer en déchèterie).',
                'Mettre le papier aluminium souillé de nourriture.',
                'Confondre les emballages composites (type Tetra Pak) avec du métal pur.',
            ],
            'impact_environnemental': (
                'L\'aluminium est recyclable à l\'infini. Recycler une canette '
                'économise 95% de l\'énergie nécessaire à sa fabrication initiale.'
            ),
        },
        'verre': {
            'categorie': 'Verre',
            'poubelle': 'Conteneur à verre (vert)',
            'conseils': [
                'Retirez les bouchons et couvercles (à mettre dans le tri sélectif).',
                'Inutile de rincer les bouteilles en verre.',
                'Ne pas casser le verre intentionnellement avant de le déposer.',
                'Seuls les emballages en verre sont acceptés (pas la vaisselle).',
                'Le verre se recycle à l\'infini sans perte de qualité.',
            ],
            'exemples': [
                'Bouteilles de vin, bière, jus',
                'Pots de confiture et conserves',
                'Bocaux en verre',
                'Flacons de parfum vides',
                'Petits pots pour bébé',
            ],
            'erreurs_courantes': [
                'Jeter la vaisselle en verre (pyrex, cristal) dans le conteneur.',
                'Mettre les ampoules et néons (à déposer en magasin ou déchèterie).',
                'Confondre verre et céramique/porcelaine.',
                'Jeter les miroirs dans le conteneur à verre.',
            ],
            'impact_environnemental': (
                'Le recyclage du verre économise 30% d\'énergie par rapport à '
                'la fabrication à partir de matières premières. Une bouteille '
                'recyclée redevient une bouteille en 30 jours.'
            ),
        },
        'papier': {
            'categorie': 'Papier / Carton',
            'poubelle': 'Poubelle jaune (tri sélectif) ou conteneur papier',
            'conseils': [
                'Aplatissez les cartons pour gagner de la place.',
                'Retirez le film plastique des magazines et prospectus.',
                'Les enveloppes à fenêtre peuvent être recyclées telles quelles.',
                'Le papier peut être recyclé jusqu\'à 7 fois.',
                'Les briques alimentaires (Tetra Pak) vont dans le tri sélectif.',
            ],
            'exemples': [
                'Journaux et magazines',
                'Cartons d\'emballage',
                'Enveloppes et courrier',
                'Cahiers et feuilles de papier',
                'Briques alimentaires (lait, jus)',
            ],
            'erreurs_courantes': [
                'Jeter le papier peint dans le tri (non recyclable).',
                'Mettre les mouchoirs et essuie-tout usagés (poubelle classique).',
                'Recycler du papier ou carton très souillé (pizza grasse).',
                'Confondre papier photo et papier classique.',
            ],
            'impact_environnemental': (
                'Recycler une tonne de papier sauve 17 arbres et '
                'économise 26 000 litres d\'eau. Le carton recyclé réduit '
                'les émissions de gaz à effet de serre de 60%.'
            ),
        },
        'organique': {
            'categorie': 'Déchets Organiques',
            'poubelle': 'Composteur ou poubelle marron (biodéchets)',
            'conseils': [
                'Utilisez un composteur individuel ou collectif si possible.',
                'Alternez déchets verts (épluchures) et bruns (feuilles sèches).',
                'Coupez les gros morceaux pour accélérer le compostage.',
                'Depuis 2024, le tri des biodéchets est obligatoire en France.',
                'Le compost mûr est utilisable comme engrais naturel après 3-6 mois.',
            ],
            'exemples': [
                'Épluchures de fruits et légumes',
                'Marc de café et sachets de thé',
                'Restes de repas (sans viande ni poisson idéalement)',
                'Coquilles d\'œufs écrasées',
                'Fleurs fanées et petits déchets de jardin',
            ],
            'erreurs_courantes': [
                'Composter les agrumes en grande quantité (acidifie le compost).',
                'Mettre de la viande ou du poisson (attire les nuisibles).',
                'Ajouter des plantes malades dans le composteur.',
                'Oublier d\'aérer et retourner le compost régulièrement.',
            ],
            'impact_environnemental': (
                'Les biodéchets représentent 30% de nos poubelles. '
                'Le compostage réduit les émissions de méthane des décharges '
                'et produit un engrais naturel de qualité.'
            ),
        },
        'dangereux': {
            'categorie': 'Déchets Dangereux',
            'poubelle': 'Points de collecte spécialisés ou déchèterie',
            'conseils': [
                'Ne JAMAIS jeter les déchets dangereux dans la poubelle classique.',
                'Rapportez les médicaments en pharmacie (dispositif Cyclamed).',
                'Les piles et batteries vont dans les collecteurs en magasin.',
                'Les peintures et solvants doivent être déposés en déchèterie.',
                'Conservez les produits dans leur emballage d\'origine si possible.',
            ],
            'exemples': [
                'Piles et batteries',
                'Peintures, vernis et solvants',
                'Produits phytosanitaires (pesticides)',
                'Ampoules fluocompactes et néons',
                'Huiles de vidange et liquides de frein',
            ],
            'erreurs_courantes': [
                'Jeter les piles dans la poubelle ordinaire (pollution des sols).',
                'Vider les produits chimiques dans l\'évier ou les toilettes.',
                'Mélanger différents produits chimiques (risque de réaction).',
                'Brûler les déchets dangereux en plein air.',
            ],
            'impact_environnemental': (
                'Une seule pile au mercure peut polluer 400 litres d\'eau et '
                '1 m³ de terre pendant 50 ans. Le traitement approprié des '
                'déchets dangereux prévient la contamination de l\'environnement.'
            ),
        },
    }

    # Normaliser le type de déchet (minuscules, sans accents superflus)
    waste_type_lower = waste_type.lower().strip()

    if waste_type_lower not in recycling_tips:
        categories_disponibles = ', '.join(recycling_tips.keys())
        raise ValueError(
            f"Type de déchet non reconnu : '{waste_type}'. "
            f"Catégories disponibles : {categories_disponibles}"
        )

    logger.info(f"Conseils de recyclage récupérés pour : {waste_type_lower}")
    return recycling_tips[waste_type_lower]
