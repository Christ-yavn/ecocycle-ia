"""
EcoLoop AI - Routes de Classification des Déchets
===================================================

Routeur FastAPI dédié à la classification des déchets par image.
Fournit les endpoints pour :
- Classifier un déchet à partir d'une image uploadée
- Lister les catégories de déchets disponibles

Préfixe : /api/classify
"""

import io
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from PIL import Image

# --- Import des modules de classification ---
from models.waste_classifier.model import WasteClassifier
from models.waste_classifier.preprocess import preprocess_image, get_recycling_tips


def _tips_safe(type_dominant: str) -> dict | list:
    """
    Appel robuste à get_recycling_tips : renvoie {} si le type n'est pas
    reconnu (ex: 'inconnu' quand YOLO ne détecte rien), au lieu de crasher.
    """
    try:
        return get_recycling_tips(type_dominant)
    except (ValueError, KeyError):
        return {}

# --- Configuration du logging ---
logger = logging.getLogger("ecoloop_ai.classify")

# --- Création du routeur ---
router = APIRouter(
    prefix="/api/classify",
    tags=["Classification"],
    responses={
        400: {"description": "Requête invalide"},
        413: {"description": "Fichier trop volumineux"},
        500: {"description": "Erreur interne du serveur"},
        503: {"description": "Modèle non disponible"},
    }
)

# --- Constantes ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 Mo en octets
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}

# Catégories de déchets supportées par le classificateur
CATEGORIES_DECHETS = [
    {
        "id": "plastique",
        "nom": "Plastique",
        "description": "Bouteilles, emballages, sacs plastiques",
        "couleur_poubelle": "Jaune",
        "recyclable": True
    },
    {
        "id": "verre",
        "nom": "Verre",
        "description": "Bouteilles, pots, bocaux en verre",
        "couleur_poubelle": "Vert",
        "recyclable": True
    },
    {
        "id": "papier",
        "nom": "Papier / Carton",
        "description": "Journaux, magazines, cartons d'emballage",
        "couleur_poubelle": "Bleu",
        "recyclable": True
    },
    {
        "id": "metal",
        "nom": "Métal",
        "description": "Canettes, conserves, aluminium",
        "couleur_poubelle": "Jaune",
        "recyclable": True
    },
    {
        "id": "organique",
        "nom": "Organique",
        "description": "Restes alimentaires, épluchures, déchets verts",
        "couleur_poubelle": "Marron",
        "recyclable": False
    },
    {
        "id": "textile",
        "nom": "Textile",
        "description": "Vêtements, tissus, chaussures",
        "couleur_poubelle": "Conteneur spécifique",
        "recyclable": True
    },
    {
        "id": "electronique",
        "nom": "Électronique (DEEE)",
        "description": "Appareils électriques, piles, batteries",
        "couleur_poubelle": "Déchèterie",
        "recyclable": True
    },
    {
        "id": "dangereux",
        "nom": "Dangereux",
        "description": "Produits chimiques, peintures, solvants",
        "couleur_poubelle": "Déchèterie",
        "recyclable": False
    },
    {
        "id": "residuel",
        "nom": "Résiduel",
        "description": "Déchets non recyclables, ordures ménagères",
        "couleur_poubelle": "Gris/Noir",
        "recyclable": False
    }
]


# =============================================================================
# Modèles Pydantic
# =============================================================================

class ItemTrouve(BaseModel):
    type: str
    classe_brute: str
    confidence: float
    box_xywh: list

class ClassificationResult(BaseModel):
    """Résultat de classification d'un déchet par YOLO."""
    total_items: int = Field(..., description="Nombre total d'objets détectés")
    type_dominant: str = Field(..., description="Catégorie majoritaire dans l'image")
    resume_quantite: dict = Field(..., description="Résumé des quantités par type")
    items_trouves: list[ItemTrouve] = Field(..., description="Détail des objets détectés")
    tips: list = Field(default=[], description="Conseils de recyclage pour la catégorie dominante")
    fallback_used: bool = Field(default=False, description="True si le modèle fine-tuné est absent (mode secours COCO)")
    nom_fichier: Optional[str] = Field(None, description="Nom du fichier original")
    taille_fichier: Optional[int] = Field(None, description="Taille du fichier en octets")
    timestamp: str = Field(..., description="Horodatage de la classification")


class AnalyzeItem(BaseModel):
    type: str
    classe_brute: str
    confidence: float
    box_xywh: list

class DetailCollectabilite(BaseModel):
    poids_estime_kg: float
    seuil_rentabilite_kg: float
    rentable: bool

class AnalyzeResult(BaseModel):
    """
    Analyse complète métier produite pour une photo de déchets.
    C'est la réponse used par l'app mobile Flutter et le web, conforme à la
    vision produit EcoLoop (état + qualité + poids + collectabilité + reco).
    """
    total_items: int
    type_dominant: str
    resume_quantite: dict
    items_trouves: list[AnalyzeItem]
    etat: str = Field(..., description="propre | sale | melange | trie | inconnu")
    score_qualite: int = Field(..., description="Score de qualité 0-100")
    poids_estime_kg: float = Field(..., description="Poids total estimé (kg)")
    poids_par_categorie_kg: dict = Field(..., description="Poids estimé par catégorie (kg)")
    collectable: bool = Field(..., description="True si la collecte est rentable")
    raison_collectabilite: str
    details_collectabilite: dict
    recommandations: list[str] = Field(..., description="Actions simples conseillées au producteur")
    tips: list = Field(default=[], description="Conseils de recyclage pour la catégorie dominante")
    fallback_used: bool = Field(default=False)
    nom_fichier: Optional[str] = Field(None)
    taille_fichier: Optional[int] = Field(None)
    timestamp: str = Field(...)

class CategoriesResponse(BaseModel):
    """Réponse contenant la liste des catégories de déchets."""
    categories: list = Field(..., description="Liste des catégories de déchets supportées")
    total: int = Field(..., description="Nombre total de catégories")

# =============================================================================
# Variable globale pour le modèle (initialisé dans ai_server.py)
# =============================================================================

_classifier: Optional[WasteClassifier] = None

def get_classifier() -> Optional[WasteClassifier]:
    return _classifier

def set_classifier(classifier: WasteClassifier) -> None:
    global _classifier
    _classifier = classifier

# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/",
    response_model=ClassificationResult,
    summary="Détecter les déchets avec YOLO",
    description="Analyse une image de déchets avec YOLOv8 pour détecter de multiples objets."
)
async def classify_image(
    file: UploadFile = File(...)
):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Veuillez envoyer une image valide.")

    try:
        # Vérification précoce de la taille (avant chargement en mémoire)
        if file.size is not None and file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

        contents = await file.read()
        file_size = len(contents)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

        classifier = get_classifier()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible.")

        # YOLO lit directement l'image binaire via PIL
        image = Image.open(io.BytesIO(contents))

        # --- Classification YOLO (exécutée dans un thread pool pour ne pas bloquer l'event loop) ---
        result = await run_in_threadpool(classifier.predict, image)

        # --- Récupération des conseils de recyclage pour le type dominant ---
        type_dominant = result.get("type_dominant", "inconnu")
        tips = _tips_safe(type_dominant)

        return ClassificationResult(
            total_items=result.get("total_items", 0),
            type_dominant=type_dominant,
            resume_quantite=result.get("resume_quantite", {}),
            items_trouves=result.get("items_trouves", []),
            tips=tips,
            fallback_used=result.get("fallback_used", False),
            nom_fichier=file.filename,
            taille_fichier=file_size,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        # Re-lever les HTTPException sans modification
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la classification : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "erreur": "Erreur interne",
                "message": f"Une erreur est survenue lors de la classification : {str(e)}"
            }
        )


@router.post(
    "/analyze",
    response_model=AnalyzeResult,
    summary="Analyse complète d'une photo de déchets (défaut produit)",
    description=(
        "Détecte les déchets (YOLO), évalue l'état (propre/sale/mélangé/trié), "
        "calcule un score de qualité 0-100, estime le poids total et par catégorie, "
        "détermine la collectabilité (rentabilité logistique) et recommande des "
        "actions simples au producteur. C'est la route'utilisée par l'app mobile "
        "lorsqu'un restaurateur/hôtelier prend une photo de ses déchets."
    )
)
async def analyze_image(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Veuillez envoyer une image valide.")

    try:
        # Vérification précoce de la taille (avant chargement en mémoire)
        if file.size is not None and file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

        contents = await file.read()
        file_size = len(contents)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

        classifier = get_classifier()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible.")

        image = Image.open(io.BytesIO(contents))
        result = await run_in_threadpool(classifier.analyze, image)

        type_dominant = result.get("type_dominant", "inconnu")
        tips = _tips_safe(type_dominant)

        # tips est un dict détaillé, on envoie la liste des conseils texte pour l'UI mobile.
        tips_list = []
        if isinstance(tips, dict):
            tips_list = tips.get("conseils", [])

        return AnalyzeResult(
            total_items=result["total_items"],
            type_dominant=type_dominant,
            resume_quantite=result["resume_quantite"],
            items_trouves=result["items_trouves"],
            etat=result["etat"],
            score_qualite=result["score_qualite"],
            poids_estime_kg=result["poids_estime_kg"],
            poids_par_categorie_kg=result["poids_par_categorie_kg"],
            collectable=result["collectable"],
            raison_collectabilite=result["raison_collectabilite"],
            details_collectabilite=result["details_collectabilite"],
            recommandations=result["recommandations"],
            tips=tips_list,
            fallback_used=getattr(classifier, "use_fallback", False),
            nom_fichier=file.filename,
            taille_fichier=file_size,
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erreur inattendue lors de l'analyse : %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"erreur": "Erreur interne", "message": str(e)}
        )


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="Lister les catégories de déchets",
    description="Retourne la liste complète des catégories de déchets supportées par le classificateur."
)
async def get_categories():
    """
    Retourne la liste de toutes les catégories de déchets supportées.
    
    Chaque catégorie contient :
    - id : identifiant unique
    - nom : nom affiché
    - description : description de la catégorie
    - couleur_poubelle : couleur de la poubelle associée
    - recyclable : indique si la catégorie est recyclable
    """
    return CategoriesResponse(
        categories=CATEGORIES_DECHETS,
        total=len(CATEGORIES_DECHETS)
    )
