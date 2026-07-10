"""
Configuration centralisée de l'application EcoLoop AI.

Utilise pydantic-settings pour charger les paramètres depuis les variables
d'environnement et/ou un fichier .env.

Les valeurs par défaut sont adaptées au développement local.
En production, toutes les valeurs sensibles doivent être définies
via des variables d'environnement ou un fichier .env sécurisé.
"""

from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Paramètres de configuration de l'application EcoLoop AI.

    Attributs:
        API_HOST: Adresse d'écoute du serveur API
        API_PORT: Port d'écoute du serveur API
        ENV: Environnement d'exécution (development, staging, production)
        JWT_SECRET: Clé secrète pour la signature des tokens JWT
        JWT_ALGORITHM: Algorithme de signature JWT
        MODEL_PATH: Chemin vers les modèles sauvegardés
        DATABASE_URL: URL de connexion à la base de données
        ALLOWED_ORIGINS: Origines CORS autorisées (séparées par des virgules)
        MAX_IMAGE_SIZE: Taille maximale des images en octets (défaut: 10 Mo)
        MAX_REQUESTS_PER_MINUTE: Limite de requêtes par minute par client
    """

    # --- Serveur API ---
    API_HOST: str = '0.0.0.0'
    API_PORT: int = Field(default=8000, validation_alias=AliasChoices("PORT", "API_PORT"))

    # --- Environnement ---
    ENV: str = 'development'

    # --- Authentification JWT ---
    JWT_SECRET: str
    JWT_ALGORITHM: str = 'HS256'

    # --- Modèles IA ---
    MODEL_PATH: str = 'saved_models'

    # --- Base de données ---
    DATABASE_URL: Optional[str] = None

    # --- Sécurité & CORS ---
    ALLOWED_ORIGINS: str = '*'

    # --- Limites ---
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10 Mo
    MAX_REQUESTS_PER_MINUTE: int = 100

    class Config:
        """Configuration du chargement des paramètres."""
        env_file = '.env'


# Instance globale des paramètres (singleton)
settings = Settings()
