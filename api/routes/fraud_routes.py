"""
EcoLoop AI - Routes de Détection de Fraude
============================================

Routeur FastAPI dédié à la détection de fraude sur les transactions.
Fournit les endpoints pour :
- Vérifier si une transaction est potentiellement frauduleuse
- Analyser les caractéristiques suspectes d'une transaction

Préfixe : /api/fraud
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

# --- Import du modèle de détection de fraude ---
from models.fraud_detection.fraud_model import FraudDetector

# --- Configuration du logging ---
logger = logging.getLogger("ecoloop_ai.fraud")

# --- Création du routeur ---
router = APIRouter(
    prefix="/api/fraud",
    tags=["Fraude"],
    responses={
        400: {"description": "Requête invalide"},
        500: {"description": "Erreur interne du serveur"},
        503: {"description": "Modèle non disponible"},
    }
)


# =============================================================================
# Modèles Pydantic
# =============================================================================

class TransactionRequest(BaseModel):
    """
    Requête de vérification de fraude sur une transaction.
    
    Contient toutes les informations nécessaires pour évaluer
    le risque de fraude d'une transaction de recyclage.
    """
    id: Optional[str] = Field(
        default=None,
        description="Identifiant unique de la transaction (auto-généré si absent)"
    )
    poids: float = Field(
        ...,
        description="Poids de la transaction en kilogrammes",
        gt=0,
        le=100000,
        examples=[25.5]
    )
    prix: float = Field(
        ...,
        description="Prix de la transaction en euros",
        gt=0,
        le=1000000,
        examples=[150.0]
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Identifiant de l'utilisateur effectuant la transaction"
    )
    heure: Optional[int] = Field(
        default=12,
        description="Heure de la transaction (0 à 23)",
        ge=0,
        le=23
    )
    jour_semaine: Optional[int] = Field(
        default=0,
        description="Jour de la semaine (0=Lundi, 1=Mardi, ..., 6=Dimanche)",
        ge=0,
        le=6
    )

    @field_validator("poids")
    @classmethod
    def valider_poids(cls, v: float) -> float:
        """Valide que le poids est dans une plage réaliste."""
        if v <= 0:
            raise ValueError("Le poids doit être strictement positif.")
        if v > 100000:
            raise ValueError(
                f"Le poids ({v} kg) dépasse la limite autorisée (100 000 kg)."
            )
        return round(v, 2)

    @field_validator("prix")
    @classmethod
    def valider_prix(cls, v: float) -> float:
        """Valide que le prix est dans une plage réaliste."""
        if v <= 0:
            raise ValueError("Le prix doit être strictement positif.")
        if v > 1000000:
            raise ValueError(
                f"Le prix ({v} €) dépasse la limite autorisée (1 000 000 €)."
            )
        return round(v, 2)


class FraudCheckResponse(BaseModel):
    """Réponse de vérification de fraude sur une transaction."""
    transaction_id: str = Field(
        ...,
        description="Identifiant unique de la transaction vérifiée"
    )
    is_fraudulent: bool = Field(
        ...,
        description="Indique si la transaction est suspectée d'être frauduleuse"
    )
    fraud_score: float = Field(
        ...,
        description="Score de risque de fraude (0.0 = sûr, 1.0 = très suspect)",
        ge=0.0,
        le=1.0
    )
    niveau_risque: str = Field(
        ...,
        description="Niveau de risque textuel (faible, moyen, élevé, critique)"
    )
    details: Dict[str, Any] = Field(
        default={},
        description="Détails supplémentaires de l'analyse de fraude"
    )
    recommandation: str = Field(
        ...,
        description="Recommandation d'action à entreprendre"
    )
    timestamp: str = Field(
        ...,
        description="Horodatage de la vérification"
    )


# =============================================================================
# Variable globale pour le modèle (initialisé dans ai_server.py)
# =============================================================================

_fraud_detector: Optional[FraudDetector] = None


def get_fraud_detector() -> Optional[FraudDetector]:
    """Retourne l'instance du détecteur de fraude."""
    return _fraud_detector


def set_fraud_detector(detector: FraudDetector) -> None:
    """Définit l'instance du détecteur de fraude."""
    global _fraud_detector
    _fraud_detector = detector


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def determiner_niveau_risque(score: float) -> str:
    """
    Détermine le niveau de risque textuel à partir du score de fraude.
    
    Paramètres :
    - score : score de fraude entre 0.0 et 1.0
    
    Retourne :
    - Niveau de risque : 'faible', 'moyen', 'élevé' ou 'critique'
    """
    if score < 0.25:
        return "faible"
    elif score < 0.50:
        return "moyen"
    elif score < 0.75:
        return "élevé"
    else:
        return "critique"


def generer_recommandation(score: float, is_fraudulent: bool) -> str:
    """
    Génère une recommandation d'action basée sur le résultat de l'analyse.
    
    Paramètres :
    - score : score de fraude
    - is_fraudulent : résultat de la détection
    
    Retourne :
    - Recommandation textuelle
    """
    if not is_fraudulent and score < 0.25:
        return "Transaction approuvée. Aucune action requise."
    elif not is_fraudulent and score < 0.50:
        return "Transaction approuvée avec vigilance. Surveillance recommandée."
    elif is_fraudulent and score < 0.75:
        return "Transaction suspecte. Vérification manuelle recommandée avant validation."
    else:
        return "Transaction hautement suspecte. Blocage et investigation immédiate recommandés."


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/check",
    response_model=FraudCheckResponse,
    summary="Vérifier une transaction pour fraude",
    description=(
        "Analyse les caractéristiques d'une transaction de recyclage "
        "pour détecter un potentiel comportement frauduleux. "
        "Retourne un score de risque et une recommandation d'action."
    )
)
async def check_fraud(request: TransactionRequest):
    """
    Vérifie si une transaction est potentiellement frauduleuse.
    
    Le modèle analyse plusieurs facteurs :
    - Rapport poids/prix (ratio anormal)
    - Heure de la transaction (horaires inhabituels)
    - Jour de la semaine (patterns suspects)
    - Historique de l'utilisateur (si disponible)
    
    Processus :
    1. Validation des données de la transaction
    2. Préparation des caractéristiques pour le modèle
    3. Analyse par le modèle de détection de fraude
    4. Calcul du niveau de risque et génération de recommandation
    
    Retourne :
    - transaction_id : identifiant de la transaction
    - is_fraudulent : indicateur de fraude
    - fraud_score : score de risque (0.0 à 1.0)
    - niveau_risque : niveau textuel (faible, moyen, élevé, critique)
    - details : détails de l'analyse
    - recommandation : action recommandée
    """
    # --- Vérification du modèle ---
    detector = get_fraud_detector()
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "erreur": "Modèle non disponible",
                "message": (
                    "Le modèle de détection de fraude n'est pas chargé. "
                    "Réessayez plus tard."
                )
            }
        )

    try:
        # --- Générer un ID si non fourni ---
        transaction_id = request.id or str(uuid.uuid4())

        # --- Préparer les données de la transaction ---
        transaction_data = {
            "id": transaction_id,
            "poids": request.poids,
            "prix": request.prix,
            "user_id": request.user_id,
            "heure": request.heure,
            "jour_semaine": request.jour_semaine,
        }

        logger.info(
            f"Vérification de fraude demandée : transaction={transaction_id}, "
            f"poids={request.poids} kg, prix={request.prix} €"
        )

        # --- Lancer l'analyse de fraude ---
        result = detector.check_transaction(transaction_data)

        # --- Extraire les résultats ---
        is_fraudulent = result.get("is_fraud", False)
        fraud_score = result.get("risk_score", 0.0) / 100.0  # Normalize to 0-1
        details = {
            "reasons": result.get("reasons", []),
            "risk_level": result.get("risk_level", "inconnu")
        }

        # --- Déterminer le niveau de risque ---
        niveau_risque = determiner_niveau_risque(fraud_score)

        # --- Générer la recommandation ---
        recommandation = generer_recommandation(fraud_score, is_fraudulent)

        # --- Log du résultat ---
        log_level = logging.WARNING if is_fraudulent else logging.INFO
        logger.log(
            log_level,
            f"Résultat fraude : transaction={transaction_id}, "
            f"frauduleuse={is_fraudulent}, score={fraud_score:.2f}, "
            f"niveau={niveau_risque}"
        )

        return FraudCheckResponse(
            transaction_id=transaction_id,
            is_fraudulent=is_fraudulent,
            fraud_score=fraud_score,
            niveau_risque=niveau_risque,
            details=details,
            recommandation=recommandation,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        # Re-lever les HTTPException sans modification
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation pour la transaction : {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "erreur": "Données invalides",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(
            f"Erreur lors de la vérification de fraude : {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "erreur": "Erreur interne",
                "message": f"Erreur lors de la vérification de fraude : {str(e)}"
            }
        )
