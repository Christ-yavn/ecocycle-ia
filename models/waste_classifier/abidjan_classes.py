"""
Catalogue des déchets adapté au contexte d'Abidjan (Côte d'Ivoire).

Ce module centralise :
1. Les 9 catégories « métier » EcoLoop (alignées avec WasteCategory du backend).
2. Le poids moyen estimé par item (kg) — calibré sur déchets observés à Abidjan.
3. Le seuil de collectabilité par catégorie (kg minimum rentable).
4. Le facteur de qualité (propreté attendue) par catégorie.
5. Le mapping COCO → EcoLoop enrichi (gère sacs, bidons, canettes…).
6. Les marques-signaux d'Abidjan (sacs noirs, bidons, pneus…) pour heuristique.

Auteur : EcoLoop AI Team
"""

from __future__ import annotations

# =============================================================================
# 1. CATÉGORIES ÉCOLOOP (alignées backend WasteCategory)
# =============================================================================
ECOLOOP_CATEGORIES = [
    "plastique",
    "metal",
    "verre",
    "papier",
    "organique",
    "textile",
    "electronique",
    "dangereux",
    "residuel",
]

# =============================================================================
# 2. POIDS MOYEN ESTIMÉ PAR ITEM (kg) — calibré Abidjan
# =============================================================================
# Sources : observation terrain Abidjan (Cocody, Yopougon, Adjamé, Treichville),
# études Banque Mondiale « What a Waste 2.0 », et fiches ADEME-CI.
# Ces valeurs servent d'estimation heuristique quand on ne dispose pas d'une balance.
ITEM_WEIGHT_KG = {
    "plastique":     0.045,   # bouteille PET 1.5L ≈ 35-50 g
    "metal":         0.015,   # canette alu 33cL ≈ 13-18 g
    "verre":         0.350,   # bouteille 75cL ≈ 300-400 g
    "papier":        0.060,   # journal/carton moyen ≈ 40-80 g
    "organique":     0.150,   # épluchure/reste estimé
    "textile":       0.250,   # vêtement usagé moyen
    "electronique":  0.500,   # petit DEEE (téléphone, câble…)
    "dangereux":     0.200,   # pile/petit flacon chimique
    "residuel":      0.080,   # sac usuel d'ordures
}

# =============================================================================
# 3. SEUIL DE COLLECTABILITÉ (kg minimum rentable pour un collecteur informel)
# =============================================================================
# Calibré sur coût carburant Abidjan + valeur marchande matière en FCFA/kg.
SEUIL_COLLECTABILITE_KG = {
    "plastique":     10.0,
    "metal":          5.0,
    "verre":         20.0,
    "papier":         8.0,
    "organique":    100.0,    # non rentable seul — compost
    "textile":        8.0,
    "electronique":   2.0,    # forte valeur unitaire
    "dangereux":      5.0,
    "residuel":     200.0,    # jamais rentable seul
}

# =============================================================================
# 4. FACTEUR QUALITÉ PAR DÉFAUT (0-1) — propreté attendue de la matière
# =============================================================================
# Sert de base au score qualité avant analyse visuelle (voir quality_estimator).
FACTEUR_PROPRETE_DEFAUT = {
    "plastique":     0.55,    # souvent souillé (restes, huile)
    "metal":         0.70,    # robuste à la saleté
    "verre":         0.65,
    "papier":        0.45,    # vite mouillé/souillé
    "organique":     0.30,    # intrinsèquement sale
    "textile":       0.40,
    "electronique":  0.80,
    "dangereux":     0.60,
    "residuel":      0.10,
}

# =============================================================================
# 5. MAPPING COCO → ECOLOOP (enrichi)
# =============================================================================
# YOLO pré-entraîné COCO ne connaît que 80 classes génériques. On mappe chaque
# classe COCO susceptible d'apparaître sur photo de déchets Abidjan vers la
# catégorie métier EcoLoop.
COCO_TO_ECOLOOP = {
    # Plastique
    "bottle": "plastique",
    "wine glass": "verre",
    "cup": "plastique",
    # Organique
    "apple": "organique",
    "orange": "organique",
    "banana": "organique",
    "broccoli": "organique",
    "carrot": "organique",
    "cake": "organique",
    "donut": "organique",
    "sandwich": "organique",
    # Papier / carton
    "book": "papier",
    "paper": "papier",
    # Métal (canettes, boîtes, couverts en inox)
    "fork": "metal",
    "knife": "metal",
    "spoon": "metal",
    "scissors": "metal",
    # Textile
    "tie": "textile",
    # Dangereux / DEEE
    "laptop": "electronique",
    "mouse": "electronique",
    "keyboard": "electronique",
    "cell phone": "electronique",
    "remote": "electronique",
    "clock": "electronique",
    # Bouteille verre (COCO « wine glass » déjà mappé verre)
    # Résiduel (lorsque rien ne colle, on range en résiduel)
}

# =============================================================================
# 6. SIGNAUX VISUELS D'ABIDJAN — heuristique complémentaire
# =============================================================================
# Ces mots-clés (en anglais, issus de COCO names) signalent une photo
# typiquement « Abidjan » : sac noir, déchet lagune, caniveau. La présence
# d'un de ces mots dans la détection brute abaisse le score qualité global.
SIGNAUX_ABIDJAN_DEGRADANTS = {"person", "surfboard", "boat"}

# =============================================================================
# 7. ALIAS / SYNONYMES (français local Abidjan)
# =============================================================================
# Aide à la robustesse du mapping côté backend (libellés compris par les users).
ALIAS_ABIDJAN = {
    "sac plastique": "plastique",
    "sac noir": "residuel",
    "bidon": "plastique",
    "bouteille": "plastique",
    "canette": "metal",
    "boîte conserve": "metal",
    "carton": "papier",
    "journal": "papier",
    "verre": "verre",
    "bouteille verre": "verre",
    "pneu": "dangereux",
    "pile": "dangereux",
    "épluchure": "organique",
    "reste aliment": "organique",
    "tissu": "textile",
    "vêtement": "textile",
    "téléphone": "electronique",
}


_KEYWORDS_ECOLOOP = [
    (["plastic", "bottle", "cup", "bag", "wrapper", "styrofoam", "styofoam", "bidon", "film", "blister"], "plastique"),
    (["can", "metal", "tin", "aluminium", "aluminum", "aerosol", "steel", "scrap"], "metal"),
    (["glass", "jar"], "verre"),
    (["paper", "cardboard", "carton", "magazine", "book", "box", "newspaper"], "papier"),
    (["food", "apple", "orange", "banana", "organic", "fruit", "vegetable", "peel", "biodegradable", "compost"], "organique"),
    (["cloth", "clothes", "fabric", "textile", "shoe", "garment", "textiles"], "textile"),
    (["laptop", "phone", "mouse", "keyboard", "remote", "cable", "electronic", "appliance", "battery", "device"], "electronique"),
    (["chemical", "paint", "solvent", "spray", "hazardous", "syringe", "tire", "pneu", "toxic"], "dangereux"),
]


def coco_to_ecoloop(class_name: str, default: str = "residuel") -> str:
    """Mappe une classe COCO/TACO brute vers la catégorie métier EcoLoop."""
    looked = COCO_TO_ECOLOOP.get(class_name)
    if looked is not None:
        return looked
    name_lower = class_name.lower()
    for keywords, ecoloop_class in _KEYWORDS_ECOLOOP:
        if any(k in name_lower for k in keywords):
            return ecoloop_class
    return default


def poids_moyen(categorie: str) -> float:
    """Retourne le poids moyen estimé (kg) pour une catégorie EcoLoop."""
    return ITEM_WEIGHT_KG.get(categorie, 0.1)


def seuil_collectabilite(categorie: str) -> float:
    """Retourne le seuil de rentabilité collecte (kg) pour une catégorie."""
    return SEUIL_COLLECTABILITE_KG.get(categorie, 50.0)


def facteur_proprete(categorie: str) -> float:
    """Retourne le facteur de propreté par défaut (0-1) pour une catégorie."""
    return FACTEUR_PROPRETE_DEFAUT.get(categorie, 0.5)