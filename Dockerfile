# Utiliser une image Python officielle légère
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
# Dossiers de config inscriptibles (YOLO/Ultralytics et Matplotlib)
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib

# Installer les dépendances système nécessaires pour Prophet et OpenCV/Pillow
# (Debian 13 "trixie" : libgl1-mesa-glx -> libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer PyTorch en version CPU uniquement (évite les lourds wheels CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Commande pour démarrer le serveur
CMD uvicorn api.ai_server:app --host 0.0.0.0 --port ${PORT:-8000}
