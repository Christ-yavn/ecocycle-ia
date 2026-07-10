# Ce qui reste a faire — IA EcoLoop

> _Priorise par impact jury + faisabilite_
> _09/07/2026_

---

## P1 — CRITIQUE (a faire avant le pitch)

### P1.1 Reentrainer YOLO sur dechets Abidjan
- **Pourquoi** : le modele actuel est fine-tune sur TACO (Europe/USA). Sur de vraies
  photos Abidjan (sacs noirs, bidons, canettes locales, pneus, dechets lagune),
  la generalisation est faible (~40% mAP@50 estime vs ~85% sur TACO val).
- **Comment** : suivre exactement `docs/PROMPT_REENTRAINEMENT_ABIDJAN.md`
  (collecter 200-400 photos Abidjan → annoter sur Roboflow → fine-tuning Colab
  en 2 phases → exporter `ecoloop_yolo_v2_abidjan.pt`).
- **Temps** : 1 jour (dont 4h collecte + 4h annotation + 2h Colab)
- **Gain** : mAP@50 0.40 → 0.72 attendu → Innovation 15% + Qualite 25% booster
- **Critere acceptation** : mAP@50 ≥ 0.70 sur test Abidjan + taille ≤ 25 Mo

### P1.2 Publier la matrice de confusion + courbes
- **Pourquoi** : Dr. BAYOH (Chief IA Officer, jury) et M. MANOUAN (Consultant IA)
  demanderont les metriques. Sans preuve, le mot "IA" devient du blabla.
- **Comment** : apres fine-tuning (P1.1), recuperer les fichiers generes par
  `model.val()` dans `EcoLoop/v2_abidjan_eval/` :
  - `confusion_matrix.png`
  - `PR_curve.png`
  - `F1_curve.png`
  - imprimer les valeurs `mAP@50`, `mAP@50-95`, `precision`, `recall`
- **Ou** : creer `docs/metrics/` dans le repo et les y copier
- **Temps** : 10 minutes apres P1.1
- **Gain** : neutralise les objections IA des 2 jurés les plus exigeants

### P1.3 Preenregistrer la demo IA (ne JAMAIS faire du live)
- **Pourquoi** : meme avec le reentrainement, YOLO peut rater ou renvoyer des
  classes absurdes sur une photo improvisée. Un echec live = catastrophe.
- **Comment** : filmer 8-10 photos de vrais dechets Abidjan ou l'IA reussit
  → montrer la chaine complete (`/api/classify/analyze` → etat + score + poids +
  collectabilite + recommandations) → si on demande "pourquoi preenregistre ?"
  → "fiabilite reseau du jury + charge GPU Render free tier"
- **Temps** : 1 heure
- **Gain** : prototype 20% securise

---

## P2 — IMPORTANT (fort impact jury)

### P2.1 Brancher l'app mobile sur `/api/classify/analyze`
- **Pourquoi** : le nouvel endpoint renvoie l'analyse complete (etat + score +
  poids + collectabilite + recommandations). Si l'app mobile consomme encore
  l'ancien `/api/classify/` (detection simple), elle n'affiche pas les nouveaux
  indicateurs metier.
- **Comment** : dans l'app Flutter, trouver l'appel API de classification photo,
  changer l'URL `/api/classify/` → `/api/classify/analyze`, parser les nouveaux
  champs (`etat`, `score_qualite`, `poids_estime_kg`, `collectable`,
  `recommandations`) et les afficher dans l'UI.
- **Temps** : 2-3 heures
- **Gain** : demo produit coherente (photo → verdict complet affiche)

### P2.2 Nettoyer les orphelins IA visibles sur le repo GitHub
- **Pourquoi** : un jure qui visite le repo verra des fichiers contradictoires
  (MobileNetV2 dans le README, YOLO dans le code, .h5 orphelin…).
- **A ranger / supprimer du repo public** :
  - `saved_models/waste_classifier.h5` (19,9 Mo, non charge)
  - `saved_models/waste_classifier.tflite` (2,7 Mo, non charge)
  - `saved_models/prophet_*.pkl` doublons avec `price_model_*.pkl`
  - `saved_models/isolation_forest_fraudes.pkl` doublon avec `fraud_model.pkl`
  - `train.py` (MobileNetV2) si non utilise → deplacer dans `archive/`
  - `test_bing.py`, `test_playwright.py`, `scripts/agent_scraper_drive.py`
  - C++/CMake visible dans GitHub languages (bruit sans valeur)
  - Licence "prive et confidentiel" sur un repo public → mettre MIT
- **Temps** : 30 minutes
- **Gain** : +credibilite technique, evite qu'un jure detecte des incoherences

### P2.3 Obtenir une lettre d'interet d'un industriel recycleur
- **Pourquoi** : sans traction terrain, le critere Impact 20% plafone a 12/20.
  Christ LOKONDA (Barka FUND) et Mame Sokhna Sarr (TBI) veulent voir une preuve
  d'adoption reelle.
- **Comment** : le seed contient "SOCIETE PLASTIQUE CI" → les contacter vraiment,
  demander une lettre d'interet (meme 1 page) pour "flux de matiere triee
  hebdomadaire". En parallel, une mairie (Cocody/Yopougon) pour un pilote verbal.
- **Temps** : quelques jours (demarrer MAINTENANT)
- **Gain** : impact 20% +5 a +7 points

---

## P3 — AMELIORATION (si le temps le permet)

### P3.1 Recuperer de vraies donnees pour Prophet/XGBoost
- **Pourquoi** : les modeles de prediction prix/volumes sont entraines sur des
  donnees synthetiques gaussiennes. Pour la production, il faut des donnees
  reelles Abidjan (prix materiaux en FCFA/kg par jour, volumes collectes).
- **Comment** : contacter un recycleur pour historique de prix + demarrer le
  pilote pour accumuler des donnees de collecte reelles.
- **Temps** : 2-3 mois (post-competition)
- **Gain** : prediction credible pour production

### P3.2 Calibrer le score de confiance (volume_prediction)
- **Pourquoi** : `_estimate_confidence()` utilise des seuils magiques (50, 200)
  non calibres. La confiance affichee n'est pas statistiquement valide.
- **Comment** : utiliser `predict_proba` ou calibration isotomique apres
  avoir des donnees reelles.
- **Temps** : 1 jour avec donnees reelles

### P3.3 Tester les API endpoints en conditions reelles
- **Pourquoi** : le code a ete valide en syntaxe + tests unitaires purs, mais
  l'API complete (FastAPI + ultralytics + PIL + Pydantic) n'a pas ete lancee
  end-to-end sur ce poste. Il faut verifier que le serveur demarre et que
  `/api/classify/analyze` repond correctement sur une vraie image.
- **Comment** :
  ```bash
  pip install -r requirements.txt
  uvicorn routes.ai_server:app --reload --port 8000
  # puis curl avec une image de test
  curl -X POST http://localhost:8000/api/classify/analyze -F "file=@test_bottle.jpg"
  ```
- **Temps** : 30 minutes
- **Gain** : securise la demo live (si besoin) + detecte bugs d'integration

### P3.4 Connecter l'export TFLite dans l'app mobile Flutter
- **Pourquoi** : `waste_classifier.tflite` (2,7 Mo) existe et est pret pour
  edge/mobile → argument eco-conception imparable pour Dr. BAYOH.
- **Comment** : ajouter `tflite_flutter` package dans Flutter, charger le .tflite
  au demarrage, classifier on-device en fallback hors-ligne.
- **Temps** : 1 jour
- **Gain** : Innovation 15% (+eco-conception), argument anti-greenwashing IA

### P3.5 Implementer l'agregateur multi-lots par zone
- **Pourquoi** : c'est le VRAI differenciateur "couche de coordination" promis
  au pitch, mais pas code dans le backend. Quand plusieurs producteurs voisins
  declarent de petits lots, la plateforme devrait les regrouper en une
  "collecte groupee candidate" rentable.
- **Comment** : endpoint `GET /collections/grouped?zone=cocody&min_total_kg=100`
  + service Python (grid 500m × 500m + SUM(weight) ≥ seuil).
- **Temps** : 1 jour
- **Gain** : +3 a +4 points au total (Innovation + Solution + Impact)

---

## ORDRE D'EXECUTION RECOMMANDE

```
P1.1 (reentrainement YOLO)  ──┐
P2.2 (nettoyage repo)        ──┤── JOUR 1 (parallel)
P3.3 (test API reel)         ──┘
         │
P1.2 (metrics)               ──┐
P1.3 (demo preenregistree)   ──┤── JOUR 2
P2.1 (mobile /analyze)       ──┘
         │
P2.3 (LOI industriel)        ── des maintenant (en parallel)
P3.5 (agregateur)            ── si temps JOUR 2-3
```

## CE QU'IL NE FAUT SURTOUT PAS FAIRE

- ❌ Demo IA live sur photo impre_vue (roulette russe)
- ❌ Mentionner la detection de fraude comme feature live (pas de transactions reelles)
- ❌ Afficher le web Vercel s'il est encore vide (placeholder)
- ❌ Pretendre un "score qualite" sans montrer l'output concret
- ❌ Dire "IA generative" si on n'utilise pas de LLM/generation
- ❌ Laisser le repo public avec licence "prive et confidentiel"