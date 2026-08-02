"""
Estimateur de qualité / état / collectabilité des déchets détectés.

À partir des résultats bruts de YOLO (liste d'objets détectés + confiance + boxes),
ce module dérive les 5 indicateurs métier attendus par la vision produit EcoLoop :

1. score_qualite   : 0-100 (propreté + tri + confiance détection)
2. etat            : "propre" | "sale" | "melange" | "trie"
3. poids_estime_kg : estimation heuristique du poids total
4. collectable     : bool + raison (rentabilité logistique vs seuil par catégorie)
5. recommandations : liste d'actions simples (séparer / nettoyer / regrouper)

Auteur : EcoLoop AI Team
"""

from __future__ import annotations

from typing import Any

from models.waste_classifier.abidjan_classes import (
    poids_moyen,
    seuil_collectabilite,
    facteur_proprete,
    SIGNAUX_ABIDJAN_DEGRADANTS,
)


def estimer_poids(items: list[dict]) -> dict[str, float]:
    """
    Estime le poids total par catégorie en kg à partir des items détectés.
    Utilise l'aire de la bounding box (box_xywh) pour estimer le volume d'un tas,
    car un simple comptage sous-estime massivement les sacs ou les piles de déchets.

    Args :
        items : liste de dicts au format YOLO (`type`, `confidence`, `box_xywh`)

    Returns :
        dict {categorie: poids_kg}
    """
    res: dict[str, float] = {}
    for it in items:
        cat = it.get("type") or it.get("classe_brute") or "residuel"
        
        # Récupération de l'aire relative (les images sont redimensionnées à 640x640 par YOLO)
        box = it.get("box_xywh", [0, 0, 0, 0])
        w, h = box[2], box[3]
        area_ratio = (w * h) / (640.0 * 640.0) if (w * h) > 0 else 0.01
        
        # Si le déchet occupe tout l'écran, on estime que c'est un gros sac/tas (ex: 35 kg max)
        # On prend le max entre le poids unitaire standard et le poids volumétrique estimé
        poids_volumetrique = area_ratio * 35.0
        poids_item = max(poids_moyen(cat), poids_volumetrique)
        
        res[cat] = res.get(cat, 0.0) + poids_item
        
    return {k: round(v, 2) for k, v in res.items()}


def evaluer_etat(items: list[dict], resume_quantite: dict) -> str:
    """
    Détermine l'état global du tas : propre / sale / melange / trie.

    Heuristique :
    - ≥2 catégories distinctes ET 1 catégorie dominante > 80% → « trie »
    - ≥2 catégories distinctes avec dominance faible       → « melange »
    - 1 catégorie + confiance moyenne ≥ 0.75                → « propre »
    - 1 catégorie + confiance faible                        → « sale »
    """
    nb_categories = len(resume_quantite)
    if nb_categories == 0:
        return "inconnu"

    total = sum(resume_quantite.values())
    if total == 0:
        return "inconnu"

    dominant_ratio = max(resume_quantite.values()) / total

    if nb_categories >= 2:
        return "trie" if dominant_ratio >= 0.8 else "melange"

    # Une seule catégorie détectée — on juge sur la confiance moyenne
    confidences = [it.get("confidence", 0) for it in items]
    conf_moy = sum(confidences) / len(confidences) if confidences else 0
    return "propre" if conf_moy >= 0.75 else "sale"


def score_qualite(items: list[dict], resume_quantite: dict, etat: str) -> int:
    """
    Calcule un score de qualité 0-100.

    Décomposition :
    - 40 % : propreté moyenne des catégories (facteur_par_catégorie)
    - 25 % : confiance moyenne des détections
    - 20 % : état du tas (trie > propre > sale > melange)
    - 15 % : pénalité signaux dégradants Abidjan (person, lagune…)
    """
    if not items:
        return 0

    # 1. Propreté moyenne par catégorie (pondérée par compte)
    total = sum(resume_quantite.values()) or 1
    s1 = sum(
        facteur_proprete(cat) * compte
        for cat, compte in resume_quantite.items()
    ) / total

    # 2. Confiance moyenne
    confidences = [it.get("confidence", 0) for it in items]
    s2 = (sum(confidences) / len(confidences)) if confidences else 0

    # 3. Bonus / malus d'état
    bonus_etat = {
        "trie": 1.0,
        "propre": 0.85,
        "melange": 0.4,
        "sale": 0.3,
        "inconnu": 0.5,
    }.get(etat, 0.5)

    # 4. Pénalité signaux dégradants
    nb_signaux = sum(
        1 for it in items
        if (it.get("classe_brute") or "") in SIGNAUX_ABIDJAN_DEGRADANTS
    )
    s4 = max(0.0, 1.0 - 0.2 * nb_signaux)

    score = (0.40 * s1 + 0.25 * s2 + 0.20 * bonus_etat + 0.15 * s4) * 100
    return int(min(100, max(0, round(score))))


def evaluer_collectabilite(poids_par_cat: dict[str, float]) -> dict:
    """
    Détermine pour chaque catégorie si la collecte est rentable,
    puis la collectabilité globale du lot.
    """
    par_categorie: dict[str, Any] = {}
    rentable_global = False
    meilleure_cat = None
    meilleure_marge = 0.0

    for cat, poids in poids_par_cat.items():
        seuil = seuil_collectabilite(cat)
        rentable = poids >= seuil
        if rentable:
            rentable_global = True
            marge = poids - seuil
            if marge > meilleure_marge:
                meilleure_marge = marge
                meilleure_cat = cat
        par_categorie[cat] = {
            "poids_estime_kg": poids,
            "seuil_rentabilite_kg": seuil,
            "rentable": rentable,
        }

    return {
        "rentable": rentable_global,
        "meilleure_categorie": meilleure_cat,
        "raison": (
            f"Catégorie dominante rentable : {meilleure_cat} "
            f"({poids_par_cat.get(meilleure_cat, 0)} kg ≥ seuil)."
            if rentable_global else
            "Aucune catégorie n'atteint le seuil de rentabilité de collecte. "
            "Conseil : regrouper avec d'autres producteurs voisins (agrégation multi-lots)."
        ),
        "par_categorie": par_categorie,
    }


def recommander_actions(
    etat: str,
    resume_quantite: dict,
    poids_par_cat: dict,
    collectable: bool,
) -> list[str]:
    """
    Génère des actions simples, concrètes, compréhensibles par un producteur
    Abidjanais (restaurateur, hôtelier, directeur d'école…).
    """
    actions: list[str] = []

    if etat == "melange":
        actions.append(
            "Séparez les déchets par matière (plastique d'un côté, métal d'un autre) "
            "pour augmenter leur valeur marchande."
        )
    if etat == "sale":
        actions.append(
            "Rincez ou essuyez les emballages avant de les photographier à nouveau "
            "— un déchet propre vaut plus cher chez le recycleur."
        )
    if etat in ("trie", "propre"):
        actions.append(
            "Bien trié ! Conservez ce lot groupé, il est collectable "
            "par un collecteur de la zone."
        )

    if not collectable:
        actions.append(
            "Le volume seul est trop faible. Utilisez l'option « regroupement par zone » "
            "pour associer ce lot à ceux de producteurs voisins (même quartier)."
        )
    else:
        cat_dominante = max(resume_quantite, key=resume_quantite.get)
        actions.append(
            f"Publiez ce lot sur la marketplace — catégorie dominante "
            f"« {cat_dominante} », les collecteurs proches seront notifiés."
        )

    # Confort supplémentaire
    if any(cat == "organique" for cat in resume_quantite):
        actions.append(
            "Les déchets organiques ne sont pas recyclables seuls : envisagez un "
            "composteur ou un partenariat avec une ferme périurbaine d'Abidjan."
        )
    if any(cat == "dangereux" for cat in resume_quantite):
        actions.append(
            "Déchet dangereux détecté — ne pas mélanger. Mettez-le en sac séparé "
            "et déclarez-le pour collecte spécialisée."
        )

    return actions


def analyser(resultat_brut: dict) -> dict:
    """
    Point d'entrée unique : à partir du dict renvoyé par WasteClassifier.predict(),
    renvoie l'analyse complète attendue par la vision produit EcoLoop.

    Args :
        resultat_brut : { total_items, type_dominant, resume_quantite, items_trouves }

    Returns :
        dict complet :
            total_items, type_dominant, resume_quantite, items_trouves,
            etat, score_qualite, poids_estime (total + par catégorie),
            collectable (bool + raison + par_categorie),
            recommandations (list[str])
    """
    items = resultat_brut.get("items_trouves", [])
    resume = resultat_brut.get("resume_quantite", {})

    poids_par_cat = estimer_poids(items)
    poids_total = round(sum(poids_par_cat.values()), 2)
    etat = evaluer_etat(items, resume)
    qualite = score_qualite(items, resume, etat)
    coll = evaluer_collectabilite(poids_par_cat)
    recommandations = recommander_actions(etat, resume, poids_par_cat, coll["rentable"])

    return {
        "total_items": resultat_brut.get("total_items", 0),
        "type_dominant": resultat_brut.get("type_dominant", "inconnu"),
        "resume_quantite": resume,
        "items_trouves": items,
        "etat": etat,
        "score_qualite": qualite,
        "poids_estime_kg": poids_total,
        "poids_par_categorie_kg": poids_par_cat,
        "collectable": coll["rentable"],
        "raison_collectabilite": coll["raison"],
        "details_collectabilite": coll["par_categorie"],
        "recommandations": recommandations,
    }