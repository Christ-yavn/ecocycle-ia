"""
Script de test de la route /api/classify/analyze.

Environnement : le serveur FastAPI doit tourner sur http://localhost:8000
(l'utilisateur le lance manuellement, PyTorch étant lourd à installer ici).

Lancement :
    python scripts/test_analyze.py

Le script :
    1. Envoie l'image test_bottle.jpg vers POST /api/classify/analyze.
    2. Vérifie l'absence (total_items == 0) au lieu de crasher.
    3. Vérifie la présence des clés du contrat : score_qualite, type_dominant,
       poids_estime_kg.
    4. Affiche un resulté formaté lisible.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "http://localhost:8000/api/classify/analyze"
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
IMAGE_PATH = PROJECT_ROOT / "test_bottle.jpg"

CONTRAT_KEYS = ("score_qualite", "type_dominant", "poids_estime_kg")


def build_multipart(file_path: Path, field_name: str = "file") -> tuple[bytes, str]:
    boundary = "----EcoLoopTestBoundary0x42424242"
    with file_path.open("rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{file_path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    body += file_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def main() -> int:
    if not IMAGE_PATH.exists():
        print(f"[KO] Image de test introuvable : {IMAGE_PATH}")
        return 2

    body, content_type = build_multipart(IMAGE_PATH)
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )

    print(f"[..] Envoi de {IMAGE_PATH.name} vers {API_URL} ...")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"[KO] HTTP {e.code} renvoyé par l'API IA.")
        print(e.read().decode("utf-8", errors="replace"))
        return 1
    except urllib.error.URLError as e:
        print(f"[KO] Serveur IA injoignable sur {API_URL}.")
        print(f"    -> {e.reason}")
        print("    Démarre le serveur : uvicorn api.ai_server:app --port 8000")
        return 1

    print(f"[OK] Statut HTTP {status}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("[KO] La réponse n'est pas du JSON valide :")
        print(raw[:500])
        return 1

    missing = [k for k in CONTRAT_KEYS if k not in data]
    if missing:
        print(f"[KO] Clés du contrat manquantes : {missing}")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 1
    print("[OK] Contrat de données respecté (score_qualite, type_dominant, poids_estime_kg).")

    print("\n===== RÉSULTAT DE L'ANALYSE =====")
    print(f"  Items détectés      : {data.get('total_items')}")
    print(f"  Type dominant       : {data.get('type_dominant')}")
    print(f"  État du tas         : {data.get('etat')}")
    print(f"  Score qualité       : {data.get('score_qualite')} / 100")
    print(f"  Poids estimé        : {data.get('poids_estime_kg')} kg")
    print(f"  Poids par catégorie : {data.get('poids_par_categorie_kg')}")
    print(f"  Collectable         : {data.get('collectable')}")
    print(f"  Raison collectab.   : {data.get('raison_collectabilite')}")
    print(f"  Fallback COCO       : {data.get('fallback_used')}")
    recos = data.get("recommandations") or []
    if isinstance(recos, list) and recos:
        print("  Recommandations     :")
        for i, r in enumerate(recos, 1):
            print(f"    {i}. {r}")
    items = data.get("items_trouves") or []
    if isinstance(items, list) and items:
        print("  Items trouvés       :")
        for it in items:
            print(
                f"    - {it.get('type')} ({it.get('classe_brute')}) "
                f"conf={it.get('confidence')}"
            )
    print("=================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())