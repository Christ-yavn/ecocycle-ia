# Utiliser une image Python officielle légère
FROM python:3.12-slim

# Définir l'utilisateur non-root requis par Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user

# Définir les variables d'environnement
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Définir le répertoire de travail
WORKDIR $HOME/app

# Revenir en root temporairement pour installer les paquets système
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Repasser sur l'utilisateur non-root
USER user

# Copier le fichier des dépendances avec les bons droits
COPY --chown=user requirements.txt .

# Installer PyTorch en version CPU uniquement
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY --chown=user . .

# Exposer le port de l'API (Hugging Face utilise 7860 par défaut)
EXPOSE 7860

# Commande pour démarrer le serveur
CMD uvicorn api.ai_server:app --host 0.0.0.0 --port 7860
