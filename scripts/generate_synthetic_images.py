import os
import random
from PIL import Image, ImageDraw

CATEGORIES = ['plastique', 'metal', 'verre', 'papier', 'organique', 'dangereux', 'residuel']
BASE_DIR = os.path.join("data", "processed", "waste_images")

def generate_images():
    print("Début de la génération du dataset de bootstrap (Architecture ML)...")
    for cat in CATEGORIES:
        cat_dir = os.path.join(BASE_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Générer 100 images par catégorie pour amorcer l'apprentissage
        for i in range(100):
            img = Image.new('RGB', (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            d = ImageDraw.Draw(img)
            # Dessiner des formes aléatoires
            d.rectangle([random.randint(0, 100), random.randint(0, 100), random.randint(100, 224), random.randint(100, 224)], fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            img.save(os.path.join(cat_dir, f"{cat}_bootstrap_{i}.jpg"))
        
        print(f"  -> Dossier {cat} initialisé avec succès.")
    
    print("Bootstrap terminé !")

if __name__ == "__main__":
    generate_images()
