import os
import shutil
import torch
from transformers import AutoTokenizer
from Tuning import NTForGMO
from peft import LoraConfig, get_peft_model
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[2]

MODEL_PATH = f"{PROJECT_PATH}/result/best_model.pt"
MERGED_PATH = f"{PROJECT_PATH}/result/merged_model"

LOCAL_MODEL_PATH = f"{PROJECT_PATH}/pretrained_model"
HF_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-V2-250m-multi-species"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def merge_and_save():

    base_model = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else HF_MODEL_NAME
    print(f"Chargement du modèle de base depuis: {base_model}")
    
    # Charge le checkpoint pour récupérer les poids
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    
    lora_dict = checkpoint.get("lora_config", {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["query", "value"],
        "lora_dropout": 0.1,
        "bias": "none",
    })

    model = NTForGMO(base_model)
    lora_config = LoraConfig(**lora_dict)
    model.encoder = get_peft_model(model.encoder, lora_config)
    print(f"LoRA utilise pour fusion: {lora_dict}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Poids entraînés chargés")
    
    # Fusionne les poids LoRA dans le modèle de base
    print("Fusion des poids LoRA...")
    model.encoder = model.encoder.merge_and_unload()
    
    os.makedirs(MERGED_PATH, exist_ok=True)
    
    print("Sauvegarde de l'encodeur...")
    model.encoder.save_pretrained(f"{MERGED_PATH}/encoder")

    # Copie les fichiers Python de trust_remote_code dans l'artefact fusionne.
    if os.path.isdir(base_model):
        for py_file in Path(base_model).glob("*.py"):
            shutil.copy2(py_file, Path(MERGED_PATH) / "encoder" / py_file.name)

    torch.save({
        "classifier_state_dict": model.classifier.state_dict(),
    }, f"{MERGED_PATH}/classifier.pt")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(f"{MERGED_PATH}/tokenizer")
    
    # Sauvegarde la config pour un chargement facile
    torch.save({
        "hidden_size": model.encoder.config.hidden_size,
        "model_name": "GMO_DETECTION_NT_V2_250M_MERGED",
    }, f"{MERGED_PATH}/config.pt")
    
    print(f"\nModèle fusionné sauvegardé dans: {MERGED_PATH}")
    print("Contenu:")
    for item in os.listdir(MERGED_PATH):
        print(f"  - {item}")

if __name__ == "__main__":
    merge_and_save()