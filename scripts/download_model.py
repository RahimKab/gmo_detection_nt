"""
Script pour télécharger et sauvegarder le modèle HuggingFace en local.
A exécuter une seule fois pour éviter de re-télécharger le modèle à chaque fois.

Utilise snapshot_download pour télécharger les fichiers bruts directement,
évitant les erreurs de chargement liées aux incohérences de configuration/poids.
"""
import os
import shutil
from huggingface_hub import snapshot_download
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[1]

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-V2-250m-multi-species"
LOCAL_PATH = f"{PROJECT_PATH}/pretrained_model"

if __name__ == "__main__":

    # Supprime les fichiers existants (potentiellement corrompus)
    if os.path.exists(LOCAL_PATH):
        print(f"Suppression du répertoire existant : {LOCAL_PATH}")
        shutil.rmtree(LOCAL_PATH)

    os.makedirs(LOCAL_PATH, exist_ok=True)

    print(f"Téléchargement du modèle : {MODEL_NAME}")
    print("Téléchargement de tous les fichiers du dépôt (cela peut prendre un moment)...")

    # Télécharge tous les fichiers bruts du dépôt HuggingFace sans instancier le modèle,
    # ce qui évite les erreurs de correspondance de taille entre config et poids.
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_PATH,
        local_dir_use_symlinks=False,
    )

    print(f"\n[OK] Modèle sauvegardé dans : {LOCAL_PATH}")
    print("Contenu :")
    for f in os.listdir(LOCAL_PATH):
        filepath = os.path.join(LOCAL_PATH, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size / 1024 / 1024:.1f} MB)")