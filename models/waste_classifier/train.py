"""
Script d'entraînement du classificateur de déchets - EcoLoop AI.

Ce module implémente le pipeline complet d'entraînement du modèle
de classification de déchets avec transfer learning en deux phases :
    - Phase 1 : Entraînement avec couches MobileNetV2 gelées
    - Phase 2 : Fine-tuning avec les 20 dernières couches dégelées

Il inclut également la conversion du modèle en format TFLite pour
le déploiement sur appareils mobiles.

Auteur : EcoLoop AI Team

Usage:
    python train.py --data_dir data/processed/waste_images --epochs 20 --batch_size 32
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# Ajouter le répertoire parent au path pour les imports locaux
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.waste_classifier.model import WasteClassifier

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Catégories de déchets attendues dans les sous-dossiers
CATEGORIES = ['plastique', 'metal', 'verre', 'papier', 'organique', 'dangereux']

# Répertoire de sauvegarde des modèles
SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'saved_models'
)


def train_waste_classifier(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 32
) -> dict:
    """
    Entraîne le classificateur de déchets avec transfer learning en 2 phases.

    Phase 1 - Transfer Learning :
        Toutes les couches de MobileNetV2 sont gelées. Seules les couches
        de classification personnalisées sont entraînées.

    Phase 2 - Fine-tuning :
        Les 20 dernières couches de MobileNetV2 sont dégelées et
        entraînées avec un learning rate réduit (1e-5).

    Le modèle est ensuite converti en format TFLite pour le déploiement mobile.

    Args:
        data_dir (str): Chemin vers le répertoire de données d'images.
            Doit contenir des sous-dossiers nommés selon les catégories
            de déchets (plastique, metal, verre, papier, organique, dangereux).
        epochs (int): Nombre d'époques pour la Phase 1. Par défaut : 20.
        batch_size (int): Taille des lots d'entraînement. Par défaut : 32.

    Returns:
        dict: Dictionnaire contenant :
            - 'history_phase1' : Historique d'entraînement de la Phase 1.
            - 'history_phase2' : Historique d'entraînement de la Phase 2.
            - 'model_path' : Chemin du modèle sauvegardé (.h5).
            - 'tflite_path' : Chemin du modèle TFLite.
            - 'final_accuracy' : Précision finale sur les données de validation.

    Raises:
        FileNotFoundError: Si le répertoire de données n'existe pas.
        ValueError: Si les données sont insuffisantes pour l'entraînement.
    """
    # Vérifier l'existence du répertoire de données
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Répertoire de données introuvable : {data_dir}"
        )

    # Créer le répertoire de sauvegarde si nécessaire
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT DU CLASSIFICATEUR DE DÉCHETS")
    logger.info("=" * 60)
    logger.info(f"Répertoire de données : {data_dir}")
    logger.info(f"Époques Phase 1 : {epochs}")
    logger.info(f"Taille des lots : {batch_size}")

    # =====================================================
    # CONFIGURATION DE L'AUGMENTATION DES DONNÉES
    # =====================================================
    logger.info("\n--- 1. Configuration de l'augmentation des données ---")
    # Configuration EXTRÊME pour simuler des déchets mélangés, cachés ou en tas
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=90,          # Rotation extrême pour simuler des objets jetés au hasard
        width_shift_range=0.3,      # Décalage pour objets partiellement visibles
        height_shift_range=0.3,     # Décalage pour objets partiellement visibles
        shear_range=0.3,            # Déformation (objets écrasés)
        zoom_range=[0.5, 1.5],      # Zoom avant/arrière puissant (simule des petits déchets dans un grand tas)
        horizontal_flip=True,       # Retournement horizontal
        vertical_flip=True,         # Retournement vertical (très commun dans une poubelle)
        brightness_range=[0.5, 1.5], # Simule différentes conditions d'éclairage (soleil, nuit)
        fill_mode='nearest',
        validation_split=0.2        # 20% des données pour la validation
    )

    # Générateur pour les données d'entraînement
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=WasteClassifier.IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Générateur pour les données de validation
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=WasteClassifier.IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    logger.info(f"Classes détectées : {train_generator.class_indices}")
    logger.info(f"Échantillons d'entraînement : {train_generator.samples}")
    logger.info(f"Échantillons de validation : {validation_generator.samples}")

    # =====================================================
    # CONFIGURATION DES CALLBACKS
    # =====================================================
    model_save_path = os.path.join(SAVED_MODELS_DIR, 'waste_classifier.h5')

    callbacks = [
        # Arrêt anticipé si la perte de validation ne s'améliore plus
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Sauvegarde du meilleur modèle
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Réduction du learning rate en cas de plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # =====================================================
    # PHASE 1 : TRANSFER LEARNING (COUCHES GELÉES)
    # =====================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 : Transfer Learning (couches MobileNetV2 gelées)")
    logger.info("=" * 60)

    # Construire le modèle (couches MobileNetV2 gelées par défaut)
    classifier = WasteClassifier()
    model = classifier.model

    logger.info(f"Nombre total de couches : {len(model.layers)}")
    logger.info(
        f"Couches entraînables : "
        f"{sum(1 for l in model.layers if l.trainable)}"
    )

    # Entraînement Phase 1
    history_phase1 = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    logger.info(
        f"Phase 1 terminée. "
        f"Meilleure val_accuracy : "
        f"{max(history_phase1.history['val_accuracy']):.4f}"
    )

    # =====================================================
    # PHASE 2 : FINE-TUNING (20 DERNIÈRES COUCHES DÉGELÉES)
    # =====================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 : Fine-tuning (20 dernières couches dégelées)")
    logger.info("=" * 60)

    # Identifier le modèle de base (MobileNetV2)
    base_model = model.layers[1] if hasattr(model.layers[1], 'layers') else None

    if base_model is not None:
        # Dégeler les 20 dernières couches de MobileNetV2
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        logger.info(
            f"Couches dégelées : les 20 dernières de MobileNetV2 "
            f"(sur {len(base_model.layers)} couches au total)"
        )
    else:
        # Dégeler les 20 dernières couches du modèle complet
        for layer in model.layers[:-20]:
            layer.trainable = False
        for layer in model.layers[-20:]:
            layer.trainable = True

        logger.info("20 dernières couches du modèle dégelées pour le fine-tuning.")

    # Recompiler avec un learning rate très faible
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(
        f"Couches entraînables après dégel : "
        f"{sum(1 for l in model.layers if l.trainable)}"
    )

    # Callbacks pour la Phase 2
    fine_tune_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),
    ]

    # Entraînement Phase 2 (10 époques)
    fine_tune_epochs = 10
    history_phase2 = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        callbacks=fine_tune_callbacks,
        verbose=1
    )

    logger.info(
        f"Phase 2 terminée. "
        f"Meilleure val_accuracy : "
        f"{max(history_phase2.history['val_accuracy']):.4f}"
    )

    # =====================================================
    # ÉVALUATION FINALE
    # =====================================================
    logger.info("\n--- Évaluation finale ---")

    eval_results = model.evaluate(validation_generator, verbose=1)
    final_accuracy = eval_results[1]

    logger.info(f"Perte finale (validation) : {eval_results[0]:.4f}")
    logger.info(f"Précision finale (validation) : {final_accuracy:.4f}")

    # =====================================================
    # CONVERSION EN TFLITE
    # =====================================================
    logger.info("\n--- Conversion en TFLite ---")

    tflite_path = _convert_to_tflite(model, SAVED_MODELS_DIR)

    # =====================================================
    # RÉSUMÉ
    # =====================================================
    logger.info("\n" + "=" * 60)
    logger.info("ENTRAÎNEMENT TERMINÉ")
    logger.info("=" * 60)
    logger.info(f"Modèle H5 sauvegardé : {model_save_path}")
    logger.info(f"Modèle TFLite sauvegardé : {tflite_path}")
    logger.info(f"Précision finale : {final_accuracy:.2%}")

    return {
        'history_phase1': history_phase1.history,
        'history_phase2': history_phase2.history,
        'model_path': model_save_path,
        'tflite_path': tflite_path,
        'final_accuracy': final_accuracy,
    }


def _convert_to_tflite(model, output_dir: str) -> str:
    """
    Convertit un modèle Keras en format TensorFlow Lite.

    Le modèle TFLite est optimisé pour le déploiement sur appareils
    mobiles et edge devices avec une taille réduite et une inférence
    plus rapide.

    Args:
        model (tf.keras.Model): Modèle Keras à convertir.
        output_dir (str): Répertoire de sortie pour le fichier .tflite.

    Returns:
        str: Chemin vers le fichier TFLite sauvegardé.
    """
    try:
        # Convertir le modèle
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Sauvegarder le modèle TFLite
        tflite_path = os.path.join(output_dir, 'waste_classifier.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Calculer la taille du fichier
        taille_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        logger.info(
            f"Modèle TFLite sauvegardé : {tflite_path} "
            f"({taille_mb:.2f} Mo)"
        )

        return tflite_path

    except Exception as e:
        logger.error(f"Erreur lors de la conversion TFLite : {str(e)}")
        raise


if __name__ == '__main__':
    """
    Point d'entrée principal pour l'entraînement du classificateur.

    Usage :
        python train.py --data_dir chemin/vers/images --epochs 20 --batch_size 32

    Si aucun argument n'est fourni, utilise les valeurs par défaut :
        - data_dir : data/processed/waste_images
        - epochs : 20
        - batch_size : 32
    """
    parser = argparse.ArgumentParser(
        description='Entraînement du classificateur de déchets EcoLoop AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
    python train.py
    python train.py --data_dir data/processed/waste_images
    python train.py --data_dir data/images --epochs 30 --batch_size 16
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/waste_images',
        help='Chemin vers le répertoire des images de déchets '
             '(par défaut : data/processed/waste_images)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Nombre d\'époques pour la Phase 1 (par défaut : 20)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Taille des lots d\'entraînement (par défaut : 32)'
    )

    args = parser.parse_args()

    logger.info("Démarrage de l'entraînement avec les paramètres :")
    logger.info(f"  - Répertoire de données : {args.data_dir}")
    logger.info(f"  - Époques : {args.epochs}")
    logger.info(f"  - Taille des lots : {args.batch_size}")

    # Lancer l'entraînement
    results = train_waste_classifier(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("\n✅ Entraînement terminé avec succès !")
    print(f"   Précision finale : {results['final_accuracy']:.2%}")
    print(f"   Modèle H5 : {results['model_path']}")
    print(f"   Modèle TFLite : {results['tflite_path']}")
