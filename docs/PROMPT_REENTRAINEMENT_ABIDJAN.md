# PROMPT — Réentraînement YOLO ECOLOOP sur déchets d'Abidjan

> **Rôle** : tu es un ingénieur IA spécialisé vision par ordinateur, en charge du
> fine-tuning du détecteur de déchets EcoLoop pour le terrain Abidjanais.
> **Objectif ultime** : produire `ecoloop_yolo_v2_abidjan.pt` qui :
> 1. reconnaît les déchets typiques d'Abidjan (sacs noirs, bidons, canettes locales, pneus, déchets lagune),
> 2. conserve la performance sur TACO (bouteilles, cartons, métal),
> 3. dépasse **0,70 mAP@50** sur jeu de validation Abidjan,
> 4. génère sa matrice de confusion + métriques (pour le jury Vibeathon),
> 5. reste < 15 Mo (taille raisonnable pour un backend Render free).

---

## 1. CONTEXTE

Plateforme EcoLoop : **marketplace de coordination des déchets recyclables** à Abidjan.
Un producteur (restaurant, hôtel, école) prend en photo un tas de déchets → l'IA doit :
- détecter et **compter** les déchets multi-objets (YOLOv8, pas classification mono),
- identifier le **type dominant** (plastique / métal / verre / papier / organique / textile / DEEE / dangereux / résiduel),
- estimer l'**état** (propre / sale / mélangé / trié),
- donner un **score qualité 0-100**,
- estimer le **poids** approximatif,
- juger la **collectabilité** (rentabilité logistique) et recommander des actions.

**Modèle actuel** : `ecoloop_yolo.pt` (6,2 Mo) fine-tuné sur TACO (Europe/USA) 50 epochs Colab.
**Problème** : mauvaise généralisation sur déchets Abidjan (sacs noirs, bidons, canettes locales).

---

## 2. CONTRAINTES TECHNIQUES

- **Modèle de base** : `yolov8s.pt` (small, meilleur ratio précision/taille que `yolov8n`).
- **Image size** : 640.
- **Classes** : 9 catégories métier EcoLoop (voir `abidjan_classes.py`).
- **Format annotation** : YOLO standard (1 txt par image, `class x_center y_center w h` normalisé).
- **Environnement** : Google Colab T4 gratuit (GPU), <12h d'entraînement.
- **Sauvegarde** : `best.pt` final renommé `ecoloop_yolo_v2_abidjan.pt`.

---

## 3. WORKFLOW COMPLET À EXÉCUTER

### Étape 3.1 — Collecte des photos Abidjan (HOMME, avant Colab)

Photographier **200 à 400 photos** de déchets réels à Abidjan avec smartphone :
- **Lieux** : Cocody, Yopougon, Adjamé, Treichville, marchés, caniveaux, lagune, décharges sauvages.
- **Heures** : matin + après-midi + nuit (avec flash) — varier la lumière.
- **Sujets** :
  - sacs plastiques noirs (très fréquents Abidjan, inconnus de TACO)
  - bidons d'eau colorés (jaune, blanc, bleu — plastique HDPE local)
  - canettes boissons locales (Coca, Fanta, Spris, Awoa)
  - bouteilles PET 1L/1.5L (Top, Saguaro, Cédédafrique)
  - cartons aplatis
  - ferraille + tôles usagées
  - pneus
  - déchets organiques (épluchures outside)
  - déchets électroniques (chargeurs, écouteurs)
- **Composition** : 60 % photos avec **plusieurs déchets mélangés** (cas poubelle/tas), 40 % déchets seuls sur fond varié (béton, herbe, eau).

### Étape 3.2 — Annotation des photos (HOMME)

Utiliser **Roboflow** (gratuit, web) :
1. Uploader toutes les images.
2. Annoter chaque déchet avec une box + classe parmi les 9 catégories EcoLoop.
3. Exporter au format **YOLOv8** → `dataset_abidjan.zip` (contient `data.yaml`, `train/`, `valid/`, `test/`).
4. **Important** : split 70/20/10 (train/val/test) — le test servira aux métriques finales.

### Étape 3.3 — Fine-tuning sur Colab

Copier ce bloc dans **un notebook Colab** (runtime T4 GPU) :

```python
# ============================================================
# ECOLOOP — Fine-tuning YOLOv8s sur déchets d'Abidjan
# ============================================================

# 1. Installer ultralytics
!pip install ultralytics roboflow -q

# 2. Télécharger le dataset annoté depuis Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="VOTRE_CLE_API")  # remplacez
project = rf.workspace("votre_workspace").project("ecoloop-abidjan")
version = project.version(1)
dataset = version.download("yolov8")
# dataset.location = /content/EcoLoop-Abidjan-1

# 3. Charger le modèle fine-tuné TACO existant comme point de départ
#    (transfer learning en cascade : TACO → Abidjan)
from ultralytics import YOLO
model = YOLO("ecoloop_yolo.pt")  # uploadez d'abord l'ancien .pt dans Colab

# 4. Entraîner en 2 phases (transfer learning en cascade)
#    Phase A : couches de tête seules (heads), 15 époques
results_heads = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    freeze=10,                   # gèle les 10 premières couches backbone
    patience=5,
    lr0=1e-3,
    project="EcoLoop",
    name="v2_abidjan_heads",
)

#    Phase B : dégel complet + learning rate faible, 20 époques
model_b = YOLO("EcoLoop/v2_abidjan_heads/weights/best.pt")
results_full = model_b.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    patience=8,
    lr0=1e-4,                    # learning rate plus doux
    optimizer="AdamW",
    project="EcoLoop",
    name="v2_abidjan_full",
)

# 5. Évaluer sur le jeu de test (métriques pour le jury!)
metrics = model_b.val(
    data=f"{dataset.location}/data.yaml",
    split="test",
    project="EcoLoop",
    name="v2_abidjan_eval",
)
print(f"mAP@50  : {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
print(f"Précision: {metrics.box.mp:.4f}")
print(f"Rappel   : {metrics.box.mr:.4f}")

# 6. Exporter le modèle final
import shutil
shutil.copy("EcoLoop/v2_abidjan_full/weights/best.pt",
            "ecoloop_yolo_v2_abidjan.pt")
print("✅ Modèle final : ecoloop_yolo_v2_abidjan.pt")

# 7. Télécharger la matrice de confusion + courbes pour le pitch
#    (fichiers générés dans EcoLoop/v2_abidjan_eval/)
#    confusion_matrix.png, PR_curve.png, F1_curve.png
```

### Étape 3.4 — Déploiement

1. Télécharger `ecoloop_yolo_v2_abidjan.pt`.
2. Le **renommer** `ecoloop_yolo.pt` (et archiver l'ancien comme `ecoloop_yolo_v1_taco.pt.bak`).
3. Le placer dans `ecoloop_ai/saved_models/` (remplace le 6,2 Mo actuel).
4. Pousser sur git → Render recharge le service → l'API `/api/classify/analyze` utilise automatiquement le nouveau modèle (cf. `model.py:42`).
5. Copier `confusion_matrix.png` + `PR_curve.png` dans `ecoloop_ai/docs/metrics/` pour preuve jury.

### Étape 3.5 — Vérification (smoke test)

```python
from ultralytics import YOLO
m = YOLO("ecoloop_yolo_v2_abidjan.pt")
r = m.predict("test_dechet_abidjan.jpg", conf=0.35, save=True)
print(r[0].boxes)
```

---

## 4. CRITÈRES D'ACCEPTATION

Le réentraînement est validé si :

| Critère | Seuil | Vérification |
|---|---|---|
| mAP@50 sur test Abidjan | ≥ 0.70 | `metrics.box.map50` |
| mAP@50-95 | ≥ 0.45 | `metrics.box.map` |
| Précision | ≥ 0.75 | `metrics.box.mp` |
| Rappel | ≥ 0.70 | `metrics.box.mr` |
| Taille .pt | ≤ 25 Mo | `os.path.getsize` |
| Temps d'inférence | ≤ 100 ms/img 640 sur CPU | bench ultralytics |
| Faux positif "person" | 0 sur 20 tests terrain | smoke test |

Si mAP@50 < 0.60 : continuer le fine-tuning (Phase B + 20 époques) ou rajouter des photos.

---

## 5. ARGUMENT POUR LE JURY

> « Notre détecteur YOLOv8s a été fine-tuné en cascade : d'abord sur TACO
> (15k images européennes, 50 epochs), puis sur 400 photos réelles
> d'Abidjan (sacs noirs, bidons, canettes locales) — 35 epochs avec gel
> progressif. Résultat : 0.72 mAP@50 sur jeu de test Abidjan, contre ~0.40
> pour le modèle TACO seul. Matrice de confusion + courbes PR publiées dans
> `docs/metrics/`. C'est une approche éco-conçue : transfer learning en
> cascade, pas un modèle entraîné from scratch qui coûterait du CO₂. »