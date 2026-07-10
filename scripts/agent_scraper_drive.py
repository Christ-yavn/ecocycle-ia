"""
Agent Scraper Avancé - Collecte d'images de déchets pour EcoLoop AI
Cible : Google Images -> Google Drive
"""
import os
import time
import requests
from io import BytesIO
from urllib.parse import urlparse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Si vous utilisez Playwright pour naviguer comme un humain
# pip install playwright
# playwright install chromium
from playwright.sync_api import sync_playwright

# --- CONFIGURATION GOOGLE DRIVE ---
SCOPES = ['https://www.googleapis.com/auth/drive.file']
DRIVE_FOLDER_ID = '1_Uz_FNb1s9u4pfh2-0KFmidhrPEP_h__'

# --- MOTS-CLÉS HYPER-CIBLÉS (Afrique / EcoLoop) ---
SEARCH_QUERIES = {
    "plastique": [
        "déchets bouteille PET afrique", "sachet plastique PEHD poubelle", 
        "déchets film plastique emballage", "bidons plastiques usagés", 
        "tas de déchets plastiques mélangés"
    ],
    "papier_carton": [
        "cartons ondulés déchets", "papier bureautique poubelle", 
        "vieux journaux recyclage", "emballages carton écrasés"
    ],
    "verre": [
        "bouteilles en verre vides décharge", "bocaux en verre déchets", 
        "bris de verre recyclage"
    ],
    "metal": [
        "ferraille rouillée déchet", "canettes aluminium écrasées", 
        "câbles cuivre déchets", "déchets inox usine"
    ],
    "organique": [
        "restes alimentaires pourris", "déchets de marché afrique", 
        "déchets verts compost", "épluchures légumes pourries"
    ],
    "speciaux_industriels": [
        "palettes bois cassées", "pneus usagés décharge", 
        "vêtements usagés déchets", "DEEE déchets électroniques", 
        "bidons huiles usées"
    ],
    "melanges_complexes": [
        "décharge publique afrique", "poubelle de rue débordante",
        "déchets mélangés sol", "ordures ménagères sac déchiré"
    ]
}

def authenticate_gdrive():
    """Authentification à l'API Google Drive."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_to_drive(service, file_content, file_name, folder_id):
    """Envoie une image directement sur Google Drive."""
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaIoBaseUpload(BytesIO(file_content), mimetype='image/jpeg', resumable=True)
    try:
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"Erreur upload Drive : {e}")
        return None

def scrape_images():
    print("Initialisation de la connexion Google Drive...")
    try:
        drive_service = authenticate_gdrive()
        print("✅ Connexion Google Drive réussie !")
    except Exception as e:
        print(f"❌ Erreur d'authentification Google Drive. Avez-vous placé le fichier 'credentials.json' ? ({e})")
        return

    print("Démarrage du navigateur (Agent Humain)...")
    with sync_playwright() as p:
        # Lancer Chrome en mode visible pour éviter les blocages
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()

        for category, queries in SEARCH_QUERIES.items():
            print(f"\n📂 Catégorie : {category.upper()}")
            
            # Vérifier si le dossier existe déjà dans Drive
            query_folder = f"name='{category}' and '{DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = drive_service.files().list(q=query_folder, spaces='drive', fields='files(id, name)').execute()
            items = results.get('files', [])
            
            if not items:
                # Créer un sous-dossier dans Drive s'il n'existe pas
                folder_metadata = {
                    'name': category,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [DRIVE_FOLDER_ID]
                }
                cat_folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
                cat_folder_id = cat_folder.get('id')
                print(f"  [Nouveau dossier créé sur Drive]")
            else:
                cat_folder_id = items[0]['id']
                print(f"  [Dossier existant trouvé sur Drive, on va le remplir]")

            for query in queries:
                print(f"  🔍 Recherche : {query}")
                # Utiliser BING IMAGES (beaucoup plus fiable pour le scraping)
                page.goto(f"https://www.bing.com/images/search?q={query}&form=HDRSC3")
                time.sleep(3) # Attendre le chargement

                # Scroller pour charger plus d'images
                for _ in range(5):
                    page.keyboard.press("PageDown")
                    time.sleep(1)

                # Récupérer les URLs des images sur Bing (la classe est souvent mimg)
                images = page.locator("img.mimg, img").element_handles()
                
                count = 0
                for img in images[:100]: # Prendre jusqu'à 100 images par requête
                    try:
                        img_url = img.get_attribute("src")
                        # Bing stocke parfois les vraies urls dans 'data-src' pour le lazy loading
                        data_src = img.get_attribute("data-src")
                        
                        final_url = data_src if data_src else img_url
                        
                        if not final_url:
                            continue
                            
                        # Gérer les images Base64 (très fréquent sur Bing)
                        if final_url.startswith("data:image"):
                            import base64
                            try:
                                header, encoded = final_url.split(",", 1)
                                image_data = base64.b64decode(encoded)
                                if len(image_data) > 5000:
                                    file_name = f"{category}_{int(time.time())}_{count}.jpg"
                                    upload_to_drive(drive_service, image_data, file_name, cat_folder_id)
                                    count += 1
                                    if count % 10 == 0:
                                        print(f"    -> {count} images envoyées sur Drive...")
                            except Exception as e:
                                print(f"    [Erreur décodage Base64: {str(e)[:50]}]")
                            continue
                            
                        if not final_url.startswith("http"):
                            continue
                            
                        # Télécharger l'image classique avec un faux User-Agent
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                        response = requests.get(final_url, headers=headers, timeout=5)
                        if response.status_code == 200:
                            if len(response.content) > 5000: # Ignorer les minuscules icônes
                                file_name = f"{category}_{int(time.time())}_{count}.jpg"
                                upload_to_drive(drive_service, response.content, file_name, cat_folder_id)
                                count += 1
                                if count % 10 == 0:
                                    print(f"    -> {count} images envoyées sur Drive...")
                            else:
                                pass # Trop petite
                        else:
                            print(f"    [Erreur téléchargement URL : Statut {response.status_code}]")
                    except Exception as e:
                        print(f"    [Exception sur l'image : {str(e)[:50]}]")

        browser.close()
    print("\n🎉 Collecte terminée ! Vérifiez votre Google Drive.")

if __name__ == "__main__":
    scrape_images()
