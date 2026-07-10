import os
import subprocess
import zipfile
import shutil
import yaml

def main():
    print("🤖 AGENT ANTIGRAVITY : Prise de contrôle de l'entraînement...")
    
    # 1. Configurer Kaggle API (définir KAGGLE_API_TOKEN dans les variables d'environnement)
    if 'KAGGLE_API_TOKEN' not in os.environ:
        print("⚠️ KAGGLE_API_TOKEN non défini. Définissez-le dans vos variables d'environnement.")
        return
    
    # 2. Télécharger avec Kaggle CLI
    import sys
    print("📡 Téléchargement du Dataset TACO (Plusieurs centaines de Mo)...")
    try:
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", "karthikeyanb4u/taco-trash-dataset-yolov8-format"], check=True)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement Kaggle : {e}")
        return
        
    # 3. Extraction
    print("📦 Extraction des données en cours...")
    extract_dir = os.path.join("data", "taco_yolo")
    os.makedirs(extract_dir, exist_ok=True)
    
    zip_path = "taco-trash-dataset-yolov8-format.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path) # Nettoyage
    
    # 4. Ajustement du fichier data.yaml pour Windows (chemins absolus)
    yaml_path = os.path.join(extract_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print("🔧 Configuration des chemins absolus pour YOLO...")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        abs_base = os.path.abspath(extract_dir)
        data['path'] = abs_base
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
            
    # 5. Lancement de l'Entraînement YOLOv8
    print("🧠 Démarrage de l'entraînement YOLOv8 sur le processeur (CPU)...")
    from ultralytics import YOLO
    
    # On commence à partir d'un modèle "nano" pour la rapidité
    model = YOLO('yolov8n.pt')
    
    # Entraînement : 10 epochs (pour que ça finisse en un temps raisonnable sur PC)
    results = model.train(
        data=yaml_path, 
        epochs=10, 
        imgsz=416,  # Réduction de la taille de l'image pour accélérer le CPU
        batch=8, 
        project='saved_models', 
        name='ecoloop_v1'
    )
    
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ! Les poids sont dans saved_models/ecoloop_v1/weights/best.pt")

if __name__ == "__main__":
    main()
