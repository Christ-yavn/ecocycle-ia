import os
import urllib.request
import zipfile
import subprocess

def download_taco_dataset():
    """
    Télécharge le dataset TACO (Trash Annotations in Context)
    Il s'agit de la référence mondiale scientifique pour les déchets.
    Contient des images HD de déchets dans la nature/rue avec bounding boxes.
    """
    print("🚀 Initialisation du téléchargement du TACO Dataset...")
    
    # Création du dossier cible
    target_dir = os.path.join("data", "taco_dataset")
    os.makedirs(target_dir, exist_ok=True)
    
    print("Installation de la librairie Kaggle...")
    subprocess.run(["pip", "install", "kaggle"], check=True)
    
    print("\n⚠️ ATTENTION ⚠️")
    print("Pour que ce téléchargement fonctionne, vous devez :")
    print("1. Créer un compte gratuit sur kaggle.com")
    print("2. Aller dans vos 'Settings' > 'Create New API Token'")
    print("3. Placer le fichier 'kaggle.json' téléchargé dans : C:\\Users\\CHRIST\\.kaggle\\kaggle.json")
    print("=========================================================\n")
    
    try:
        # Téléchargement depuis Kaggle (le dataset officiel hébergé par Knerc)
        print("Téléchargement des données (Plusieurs Go, veuillez patienter...)")
        subprocess.run(["kaggle", "datasets", "download", "-d", "knerc/taco-trash-annotations-in-context", "-p", target_dir], check=True)
        
        # Extraction
        zip_path = os.path.join(target_dir, "taco-trash-annotations-in-context.zip")
        print("Extraction des images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        # Nettoyage
        os.remove(zip_path)
        print("✅ TACO Dataset installé avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement. Avez-vous bien configuré kaggle.json ? Erreur : {e}")

if __name__ == "__main__":
    download_taco_dataset()
