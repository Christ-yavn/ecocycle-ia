import os
import sys
sys.path.append(r'C:\Users\CHRIST\AppData\Roaming\Python\Python312\site-packages')
import requests
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor
import time

CATEGORIES = {
    "plastique": [
        "bouteille plastique déchet", "sachet eau plastique poubelle afrique", 
        "déchets plastiques abidjan", "emballage plastique vide",
        "bouteille plastique écrasée"
    ],
    "metal": [
        "canette métal déchet", "boite de conserve vide poubelle", 
        "ferraille déchet afrique", "canette aluminium écrasée"
    ],
    "verre": [
        "bouteille en verre vide déchet", "bris de verre poubelle", 
        "bouteille bière vide afrique"
    ],
    "papier": [
        "carton déchet poubelle", "papier froissé", 
        "vieux journaux déchets", "emballage carton vide"
    ],
    "organique": [
        "déchets alimentaires organiques", "peau de banane poubelle", 
        "restes de nourriture compost", "épluchures légumes"
    ],
    "dangereux": [
        "piles usagées déchets", "batterie voiture déchet", 
        "bidon huile moteur vide", "produits chimiques déchets"
    ]
}

BASE_DIR = os.path.join("data", "processed", "waste_images")

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def fetch_category_images(category, queries, max_images_per_query=50):
    cat_dir = os.path.join(BASE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    
    count = 0
    with DDGS() as ddgs:
        for query in queries:
            print(f"[{category}] Recherche : {query}")
            try:
                results = list(ddgs.images(query, max_results=max_images_per_query))
                urls = [r.get('image') for r in results if r.get('image')]
                
                for url in urls:
                    if count >= 300:  # Limite par catégorie pour aller vite
                        break
                    
                    ext = url.split('.')[-1][:3].lower()
                    if ext not in ['jpg', 'png', 'jpe']:
                        ext = 'jpg'
                    
                    save_path = os.path.join(cat_dir, f"{category}_{count}.{ext}")
                    if download_image(url, save_path):
                        count += 1
                        if count % 10 == 0:
                            print(f"  -> {count} images téléchargées pour {category}...")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Erreur recherche {query}: {e}")

if __name__ == "__main__":
    print("Début du téléchargement du Dataset d'Images (Focus Afrique/Côte d'Ivoire)...")
    for category, queries in CATEGORIES.items():
        fetch_category_images(category, queries, max_images_per_query=60)
    print("Téléchargement terminé !")
