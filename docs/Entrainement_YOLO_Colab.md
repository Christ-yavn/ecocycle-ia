# 🚀 GUIDE GOOGLE COLAB : ENTRAÎNEMENT DE YOLOv8 (TACO DATASET)

Ce guide contient le code exact à copier-coller dans Google Colab pour entraîner gratuitement votre modèle avec une carte graphique (GPU).

## ÉTAPE 1 : Préparer l'environnement
1. Allez sur [Google Colab](https://colab.research.google.com/)
2. Cliquez sur **Nouveau Notebook** (Fichier > Nouveau Notebook).
3. Dans le menu en haut, allez dans **Exécution > Modifier le type d'exécution** (Runtime > Change runtime type).
4. Sous "Accélérateur matériel", choisissez **T4 GPU** et cliquez sur Enregistrer.

## ÉTAPE 2 : Le Code Complet (En un seul bloc)

Dans votre Notebook Colab, copiez l'intégralité du bloc ci-dessous, collez-le dans la case grise, et remplacez la valeur de `KAGGLE_API_TOKEN` par votre propre clé générée. Puis cliquez sur le bouton "Play" (▶️) pour l'exécuter.

```python
import os

# 1. Configurer la clé Kaggle (Remplacez par votre clé)
os.environ['KAGGLE_API_TOKEN'] = "KGAT_2c00428c79435050a767f3d9c2e90814"

# 2. Installer les outils IA
!pip install ultralytics kaggle opencv-python-headless

# 3. Télécharger le dataset TACO
!kaggle datasets download -d vencerlanz09/taco-dataset-yolo-format
!unzip -q taco-dataset-yolo-format.zip -d taco_yolo/

# 4. Lancer l'entraînement
from ultralytics import YOLO
model = YOLO('yolov8n.pt') 

print("\\n🚀 DÉMARRAGE DE L'ENTRAÎNEMENT ECOLOOP ! 🚀\\n")
results = model.train(
    data='/content/taco_yolo/data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16,
    project='EcoLoop',
    name='v1_taco'
)
```

## ÉTAPE 3 : Récupérer le cerveau final
Une fois l'entraînement terminé, Google aura généré votre modèle !
1. Sur le menu de gauche de Colab, cliquez sur l'icône de **Dossier** (Fichiers).
2. Naviguez vers : `EcoLoop` -> `v1_taco` -> `weights`.
3. Vous y trouverez un fichier nommé **`best.pt`**.
4. Faites un clic droit dessus et cliquez sur **Télécharger**.
5. Renommez-le en **`ecoloop_yolo.pt`** et placez-le dans le dossier `C:\Users\CHRIST\Desktop\ANTIGRAVITY\ecoloop_ai\saved_models\` de votre projet.

C'est fini ! Votre IA est prête pour la production.
