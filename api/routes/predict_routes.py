"""
EcoLoop AI - Routes de Prédiction
===================================

Routeur FastAPI dédié aux prédictions par intelligence artificielle.
Fournit les endpoints pour :
- Prédiction des prix des matériaux recyclables
- Prédiction des volumes de collecte par zone géographique

Préfixe : /api/predict
"""

import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# --- Import des modèles de prédiction ---
from models.prediction.price_prediction import PricePredictor
from models.prediction.volume_prediction import VolumePredictor

# --- Configuration du logging ---
logger = logging.getLogger("ecoloop_ai.predict")

# --- Création du routeur ---
router = APIRouter(
    prefix="/api/predict",
    tags=["Prédiction"],
    responses={
        400: {"description": "Requête invalide"},
        500: {"description": "Erreur interne du serveur"},
        503: {"description": "Modèle non disponible"},
    }
)

# --- Liste des matériaux supportés ---
MATERIAUX_SUPPORTES = [
    "plastique", "verre", "papier", "carton", "aluminium",
    "acier", "cuivre", "bois", "textile", "caoutchouc"
]


# =============================================================================
# Modèles Pydantic
# =============================================================================

class PriceRequest(BaseModel):
    """Requête de prédiction de prix pour un matériau recyclable."""
    material: str = Field(
        ...,
        description="Nom du matériau recyclable (ex: 'plastique', 'aluminium')",
        examples=["plastique", "aluminium", "verre"]
    )
    periods: Optional[int] = Field(
        default=30,
        description="Nombre de jours de prédiction (1 à 365, défaut : 30)",
        ge=1,
        le=365
    )

    @field_validator("material")
    @classmethod
    def valider_material(cls, v: str) -> str:
        """Valide et normalise le nom du matériau."""
        return v.strip().lower()


class VolumeRequest(BaseModel):
    """Requête de prédiction de volume de collecte."""
    zone_id: int = Field(
        ...,
        description="Identifiant unique de la zone de collecte",
        ge=1
    )
    date: str = Field(
        ...,
        description="Date cible de la prédiction au format YYYY-MM-DD",
        examples=["2026-07-15"]
    )

    @field_validator("date")
    @classmethod
    def valider_date(cls, v: str) -> str:
        """Valide le format de la date (YYYY-MM-DD)."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Format de date invalide : '{v}'. "
                f"Utilisez le format YYYY-MM-DD (ex: 2026-07-15)."
            )
        return v


class PricePredictionItem(BaseModel):
    """Élément individuel d'une prédiction de prix."""
    date: str = Field(..., description="Date de la prédiction")
    prix_predit: float = Field(..., description="Prix prédit en euros par tonne")
    intervalle_bas: Optional[float] = Field(
        None, description="Borne basse de l'intervalle de confiance"
    )
    intervalle_haut: Optional[float] = Field(
        None, description="Borne haute de l'intervalle de confiance"
    )


class PricePredictionResponse(BaseModel):
    """Réponse complète de prédiction de prix."""
    material: str = Field(..., description="Matériau analysé")
    predictions: list = Field(..., description="Liste des prédictions jour par jour")
    periods: int = Field(..., description="Nombre de périodes prédites")
    unite: str = Field(default="EUR/tonne", description="Unité des prix prédits")
    timestamp: str = Field(..., description="Horodatage de la prédiction")


class VolumePredictionResponse(BaseModel):
    """Réponse de prédiction de volume de collecte."""
    zone_id: int = Field(..., description="Identifiant de la zone")
    date: str = Field(..., description="Date cible de la prédiction")
    predicted_volume: float = Field(..., description="Volume prédit en mètres cubes (m³)")
    unite: str = Field(default="m³", description="Unité du volume prédit")
    confiance: Optional[float] = Field(
        None, description="Score de confiance de la prédiction"
    )
    timestamp: str = Field(..., description="Horodatage de la prédiction")


# =============================================================================
# Variables globales pour les modèles (initialisés dans ai_server.py)
# =============================================================================

_price_predictor: Optional[PricePredictor] = None
_volume_predictor: Optional[VolumePredictor] = None


def get_price_predictor() -> Optional[PricePredictor]:
    """Retourne l'instance du prédicteur de prix."""
    return _price_predictor


def set_price_predictor(predictor: PricePredictor) -> None:
    """Définit l'instance du prédicteur de prix."""
    global _price_predictor
    _price_predictor = predictor


def get_volume_predictor() -> Optional[VolumePredictor]:
    """Retourne l'instance du prédicteur de volume."""
    return _volume_predictor


def set_volume_predictor(predictor: VolumePredictor) -> None:
    """Définit l'instance du prédicteur de volume."""
    global _volume_predictor
    _volume_predictor = predictor


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/price",
    response_model=PricePredictionResponse,
    summary="Prédire le prix d'un matériau",
    description=(
        "Prédit l'évolution du prix d'un matériau recyclable sur une période donnée. "
        "Utilise un modèle de séries temporelles entraîné sur les données historiques du marché."
    )
)
async def predict_price(request: PriceRequest):
    """
    Prédit le prix d'un matériau recyclable.
    
    Le modèle analyse les tendances historiques des prix pour générer
    des prédictions jour par jour sur la période demandée.
    
    Paramètres :
    - material : nom du matériau (sera normalisé en minuscules)
    - periods : nombre de jours de prédiction (défaut : 30)
    
    Retourne :
    - Liste de prédictions avec date et prix prédit
    - Intervalle de confiance (si disponible)
    """
    # --- Vérification du modèle ---
    predictor = get_price_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "erreur": "Modèle non disponible",
                "message": "Le modèle de prédiction de prix n'est pas chargé. Réessayez plus tard."
            }
        )

    # --- Vérification du matériau ---
    if request.material not in MATERIAUX_SUPPORTES:
        logger.warning(f"Matériau inconnu demandé : {request.material}")
        # On ne bloque pas, le modèle peut gérer des matériaux non listés

    try:
        logger.info(
            f"Prédiction de prix demandée : matériau={request.material}, "
            f"périodes={request.periods}"
        )

        # --- Lancer la prédiction ---
        result = predictor.predict(
            material=request.material,
            periods=request.periods
        )
        predictions = result.get('predictions', [])

        logger.info(
            f"Prédiction de prix réussie : {len(predictions)} points générés "
            f"pour {request.material}"
        )

        return PricePredictionResponse(
            material=request.material,
            predictions=predictions,
            periods=request.periods,
            unite="EUR/tonne",
            timestamp=datetime.utcnow().isoformat()
        )

    except ValueError as e:
        logger.error(f"Erreur de validation pour la prédiction de prix : {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "erreur": "Données invalides",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction de prix : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "erreur": "Erreur interne",
                "message": f"Erreur lors de la prédiction de prix : {str(e)}"
            }
        )


@router.post(
    "/volume",
    response_model=VolumePredictionResponse,
    summary="Prédire le volume de collecte",
    description=(
        "Prédit le volume de déchets à collecter pour une zone géographique "
        "et une date données. Utile pour optimiser la logistique de collecte."
    )
)
async def predict_volume(request: VolumeRequest):
    """
    Prédit le volume de collecte pour une zone et une date données.
    
    Le modèle prend en compte les données historiques de collecte,
    les caractéristiques de la zone et les facteurs saisonniers
    pour estimer le volume de déchets attendu.
    
    Paramètres :
    - zone_id : identifiant de la zone de collecte
    - date : date cible au format YYYY-MM-DD
    
    Retourne :
    - Volume prédit en mètres cubes
    - Score de confiance (si disponible)
    """
    # --- Vérification du modèle ---
    predictor = get_volume_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "erreur": "Modèle non disponible",
                "message": "Le modèle de prédiction de volume n'est pas chargé. Réessayez plus tard."
            }
        )

    try:
        logger.info(
            f"Prédiction de volume demandée : zone_id={request.zone_id}, "
            f"date={request.date}"
        )

        # --- Lancer la prédiction ---
        result = predictor.predict(
            zone_id=request.zone_id,
            date=request.date
        )

        predicted_volume = result.get("predicted_volume", 0.0)
        confiance = result.get("confidence", None)

        logger.info(
            f"Prédiction de volume réussie : zone {request.zone_id} → "
            f"{predicted_volume:.2f} m³ le {request.date}"
        )

        return VolumePredictionResponse(
            zone_id=request.zone_id,
            date=request.date,
            predicted_volume=predicted_volume,
            unite="m³",
            confiance=confiance,
            timestamp=datetime.utcnow().isoformat()
        )

    except ValueError as e:
        logger.error(f"Erreur de validation pour la prédiction de volume : {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "erreur": "Données invalides",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction de volume : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "erreur": "Erreur interne",
                "message": f"Erreur lors de la prédiction de volume : {str(e)}"
            }
        )
