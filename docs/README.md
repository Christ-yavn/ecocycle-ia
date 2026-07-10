# Dossier IA ECOLOOP — Vue d'ensemble

> _Emplacement : `C:\Users\CHRIST\Desktop\ANTIGRAVITY\ia`_
> _Source originale : `Ecoloop/ecoloop_ai/` (repo GitHub EPDU35/Ecoloop)_
> _Derniere mise a jour : 10/07/2026_

Ce dossier centralise **tout ce qui concerne le modele d'Intelligence Artificielle**
d'EcoLoop, code source + modeles entraines + documentation authentique.

---

## Structure du dossier

```
ia/
├── model/                          ← Code source des modeles IA
│   ├── waste_classifier/           ← Detecteur de dechets (YOLOv8)
│   │   ├── model.py                ← Classe WasteClassifier (predict + analyze)
│   │   ├── abidjan_classes.py      ← Catalogue calibre Abidjan
│   │   ├── quality_estimator.py    ← Score qualite + etat + collectabilite
│   │   ├── preprocess.py           ← Pretraitement image + conseils tri
│   │   └── train.py                ← Script entrainement MobileNetV2 (orphelin)
│   ├── prediction/                ← Prediction prix + volumes + data_utils
│   └── fraud_detection/            ← Detection fraude (Isolation Forest)
├── routes/                         ← API FastAPI
│   ├── classify_routes.py          ← Endpoint /classify + /analyze
│   ├── predict_routes.py           ← Endpoint /predict/price + /predict/volume
│   ├── fraud_routes.py             ← Endpoint /fraud/check
│   ├── ai_server.py                ← Serveur principal (startup + health)
│   └── middleware/                 ← Auth middleware
├── config/                         ← Settings (env, CORS, origines)
├── saved_models/                   ← Modeles entraines (.pt, .h5, .tflite, .pkl)
├── scripts/                        ← Generation donnees synthetiques + scrapers
├── docs/                           ← Documentation
│   ├── DOC_FONCTIONNALITES_REELLES.md   ← Ce qui marche VRAIMENT
│   ├── CE_QUI_RESTE_A_FAIRE.md          ← TODO precis
│   ├── PROMPT_REENTRAINEMENT_ABIDJAN.md ← Prompt Colab pour fine-tuning
│   ├── RAPPORT_IA_ECOLOOP.md           ← Rapport technique complet
│   └── Entrainement_YOLO_Colab.md       ← Guide Colab original (TACO)
├── requirements.txt                ← Dependances Python
├── Dockerfile                      ← Container Docker
├── docker-compose.yml              ← Orchestration
├── .env.example                    ← Variables d'environnement
├── test_live.py                    ← Script de test API
└── test_bottle.jpg                 ← Image de test
```

---

## Modeles entraines (saved_models/)

| Fichier | Algo | Taille | Source | Utilise en prod ? |
|---|---|---|---|---|
| `ecoloop_yolo.pt` | YOLOv8n fine-tune TACO | 6,2 Mo | Colab 50 epochs | OUI (model.py) |
| `waste_classifier.h5` | MobileNetV2 transfer learning | 19,9 Mo | Local 700 images | NON (orphelin) |
| `waste_classifier.tflite` | export TFLite | 2,7 Mo | idem | NON (orphelin) |
| `prophet_*.pkl` (×4) | Prophet prix materiaux | ~5 Mo | donnees synthetiques | OUI |
| `price_model_*.pkl` (×4) | XGBoost prix materiaux | ~3 Mo | donnees synthetiques | OUI |
| `volume_model.pkl` | XGBoost volumes | ~2 Mo | donnees synthetiques | OUI |
| `fraud_model.pkl` | Isolation Forest fraude | ~1 Mo | donnees synthetiques | OUI |
| `fraud_scaler.pkl` | Scaler fraude | <1 Mo | idem | OUI |

---

## Pour demarrer

```bash
pip install -r requirements.txt
cp .env.example .env  # remplir les variables
uvicorn routes.ai_server:app --reload --port 8000
```

API docs : `http://localhost:8000/docs`

---

## Documentation prioritaire

| Document | A lire en premier ? |
|---|---|
| `docs/DOC_FONCTIONNALITES_REELLES.md` | **OUI** — ce qui marche vraiment |
| `docs/CE_QUI_RESTE_A_FAIRE.md` | **OUI** — les TODO prioritaires |
| `docs/PROMPT_REENTRAINEMENT_ABIDJAN.md` | Pour fine-tuning YOLO sur Abidjan |
| `docs/RAPPORT_IA_ECOLOOP.md` | Rapport technique complet |
| `docs/Entrainement_YOLO_Colab.md` | Guide original TACO (Colab) |