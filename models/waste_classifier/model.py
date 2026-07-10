"""
Modèle de Détection des Déchets (Object Detection) - EcoLoop AI.

V2 : YOLOv8 (You Only Look Once) fine-tuné sur TACO.
Contrairement à la V1 (MobileNetV2, classification mono-objet), YOLO détecte,
encadre et compte plusieurs déchets mélangés présents dans une seule image —
cas réel d'un producteur Abidjanais prenant en photo un tas.

V2.1 (mise à jour Abidjan) :
    - Mapping COCO → EcoLoop enrichi via `abidjan_classes`.
    - Filtrage des détections absurdes (confiance < CONFIANCE_MIN).
    - Nouvelle méthode `analyze()` qui produit l'analyse complète attendue par
      la vision produit (état, score qualité, poids estimé, collectabilité,
      recommandations) en enchaînant `quality_estimator`.

Auteur : EcoLoop AI Team
"""

import os
import logging
from typing import Any

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from models.waste_classifier.abidjan_classes import coco_to_ecoloop, ECOLOOP_CATEGORIES
from models.waste_classifier.quality_estimator import analyser as analyser_qualite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Seuil minimal de confiance pour conserver une détection.
# En dessous, la prédiction est trop incertaine → on l'écarte pour éviter
# d'afficher des catégories absurdes (ex: "person" sur un tas de déchets).
CONFIANCE_MIN = 0.35


class WasteClassifier:
    """
    Détecteur de déchets YOLOv8 prêt pour le terrain Abidjanais.

    Deux modes d'appel :
    - predict(image)    : détection brute (legacy, conserve la signature ancienne).
    - analyze(image)    : analyse complète métier (état + qualité + poids +
                         collectabilité + recommandations). À préférer pour
                         la route d'API « photo du producteur ».
    """

    CATEGORIES_ECOLOOP = ECOLOOP_CATEGORIES
    IMG_SIZE = (224, 224)  # conservé pour compat avec old train.py

    def __init__(self, model_path: str | None = None):
        if YOLO is None:
            raise RuntimeError(
                "Librairie 'ultralytics' non installée. "
                "Installez-la : pip install ultralytics"
            )

        if model_path is None:
            model_path = os.path.join("saved_models", "ecoloop_yolo.pt")

        self._fallback_used = False
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self._fallback_used = False
            logger.info("✅ Modèle YOLO EcoLoop chargé depuis : %s", model_path)
        else:
            logger.warning(
                "⚠️ Fichier %s introuvable — chargement du modèle de secours "
                "yolov8n.pt (COCO non fine-tuné). Qualité Abidjan dégradée.",
                model_path,
            )
            self.model = YOLO("yolov8n.pt")
            self._fallback_used = True
            logger.info("✅ Modèle de secours yolov8n.pt chargé.")

    @property
    def use_fallback(self) -> bool:
        """Indique si le modèle fine-tuné est absent (mode secours COCO)."""
        return self._fallback_used

    def predict(self, image, conf_threshold: float = CONFIANCE_MIN) -> dict:
        """
        Détection multi-objet brute.

        Args :
            image : chemin de fichier, objet PIL.Image ou numpy array.
            conf_threshold : confiance minimale pour conserver une détection.

        Returns :
            dict {
                total_items, type_dominant, resume_quantite, items_trouves,
                fallback_used
            }
        """
        try:
            results = self.model(image, imgsz=640, verbose=False, conf=conf_threshold)

            items_trouves: list[dict] = []
            resume_quantite: dict[str, int] = {}

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = str(self.model.names[class_id])
                    ecoloop_class = coco_to_ecoloop(class_name, default="residuel")

                    confidence = float(box.conf[0])
                    xywh = box.xywh[0].tolist()

                    items_trouves.append({
                        "type": ecoloop_class,
                        "classe_brute": class_name,
                        "confidence": round(confidence, 2),
                        "box_xywh": [round(c, 2) for c in xywh],
                    })
                    resume_quantite[ecoloop_class] = (
                        resume_quantite.get(ecoloop_class, 0) + 1
                    )

            type_dominant = (
                max(resume_quantite, key=resume_quantite.get)
                if resume_quantite else "inconnu"
            )

            logger.info(
                "Prédiction YOLO : %d objets (%d filtrés). Dominant : %s",
                len(items_trouves),
                sum(len(b.boxes) for b in results) - len(items_trouves),
                type_dominant,
            )

            return {
                "total_items": len(items_trouves),
                "type_dominant": type_dominant,
                "resume_quantite": resume_quantite,
                "items_trouves": items_trouves,
                "fallback_used": self._fallback_used,
            }

        except Exception as e:
            logger.error("Erreur lors de la prédiction YOLO : %s", e, exc_info=True)
            raise ValueError(f"Impossible d'analyser l'image avec YOLO : {e}")

    def analyze(self, image, conf_threshold: float = CONFIANCE_MIN) -> dict:
        """
        Analyse complète métier (raccourci predict() + quality_estimator).

        Retourne TOUT ce que la vision produit exige pour 1 photo :
        détection + état + score qualité + poids estimé + collectabilité
        + recommandations d'actions.
        """
        brut = self.predict(image, conf_threshold=conf_threshold)
        return analyser_qualite(brut)

    def get_categories(self) -> list[str]:
        return self.CATEGORIES_ECOLOOP.copy()