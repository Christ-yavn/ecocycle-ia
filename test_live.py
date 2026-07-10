import requests
import json

print("Test Prédiction des prix :")
try:
    r1 = requests.post("http://localhost:8000/api/predict/price", json={"material": "plastique", "periods": 3})
    print(json.dumps(r1.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")

print("\n--------------------------\n")
print("Test Détection de fraudes :")
try:
    r2 = requests.post("http://localhost:8000/api/fraud/check", json={"poids": 50, "prix": 100, "heure": 14, "jour_semaine": 2})
    print(json.dumps(r2.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Erreur : {e}")
