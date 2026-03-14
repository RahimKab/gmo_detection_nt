
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
from transformers import AutoTokenizer
from utils.GmoDataset import GMODataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.Tuning import NTForGMO, save_checkpoint, load_checkpoint, save_history, load_history
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score, accuracy_score)
import torch.nn.functional as F
import utils.notification as notif
from peft import LoraConfig, get_peft_model

from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parents[1]

TRAIN_PATH = f"{PROJECT_PATH}/data/processed/splits/train.csv"
VAL_PATH = f"{PROJECT_PATH}/data/processed/splits/val.csv"
TEST_PATH = f"{PROJECT_PATH}/data/processed/splits/test.csv"
MODEL_SAVE_PATH = f"{PROJECT_PATH}/result/best_model.pt"
CHECKPOINT_PATH = f"{PROJECT_PATH}/result/checkpoint.pt"
LOG_PATH = f"{PROJECT_PATH}/result/logs.txt"
HISTORY_PATH = f"{PROJECT_PATH}/result/history.json"
MY_MODEL_NAME = "GMO_DETECTION_NT_V2_250M"

# Chemin local du modèle pré-entraîné
LOCAL_MODEL_PATH = f"{PROJECT_PATH}/pretrained_model"
# Nom du modèle Hugging Face à utiliser si le modèle local n'est pas trouvé
HF_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-V2-250m-multi-species"

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """Crée un DataLoader pour charger les données en batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def train(model, train_loader, val_loader, optimizer, epochs, accum_steps, device, checkpoint_interval=100):
    scaler = GradScaler()
    
    # Chargement de l'historique d'entraînement
    history = load_history(HISTORY_PATH)
    # Charge le checkpoint s'il existe
    start_epoch, start_step, best_pr_auc = load_checkpoint(model, optimizer, scaler, device, CHECKPOINT_PATH)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            # Saute des etapes deja executer lorsqu'il y a une checkpoint et reprend au dernier epoch
            if epoch == start_epoch and step < start_step:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device):
                outputs = model(**batch)
                loss = outputs["loss"] / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

            progress_bar.set_postfix({
                "loss": total_loss / (step+1)
            })

            # Enregistre les checkpoints periodiquement
            if (step + 1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scaler, epoch, step, best_pr_auc, CHECKPOINT_PATH)

        # Reprend start_step apres le premier epoch
        start_step = 0

        # Enregistre le checkpoint a la fin de chaque epoch
        save_checkpoint(model, optimizer, scaler, epoch + 1, 0, best_pr_auc)

        avg_loss, acc, roc_auc, pr_auc = evaluate(model, val_loader, device)
        
        # Sauvegarde des métriques pour le suivi de l'entraînement
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(avg_loss)
        history['val_acc'].append(acc)
        history['val_roc_auc'].append(roc_auc)
        history['val_pr_auc'].append(pr_auc)
        save_history(history, HISTORY_PATH)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "pr_auc": best_pr_auc,
                'model_name': MY_MODEL_NAME,
                'lora_config': {
                    'r': 8,
                    'lora_alpha': 16,
                    'target_modules': ["query", "value"],
                    'lora_dropout': 0.1,
                    'bias': "none"
                }
            }, MODEL_SAVE_PATH)
            save_checkpoint(model, optimizer, scaler, epoch + 1, 0, best_pr_auc, path=CHECKPOINT_PATH)
            with open(LOG_PATH,"a") as file :
                file.write(f"\nMeilleur modele sauvegarde! \n avg_loss: {avg_loss:.4f}\n accuracy: {acc:.4f}\n roc_auc: {roc_auc:.4f}\n pr_auc: {pr_auc:.4f} \n")
            print(" Meilleur modele sauvegarde! ")

        print(f"Epoch {epoch+1} finished | Avg loss: {total_loss/len(train_loader):.4f}")
    return history

def evaluate(model, val_loader, device):
    """Évalue le modèle sur un ensemble de données."""
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    progress_bar = tqdm(val_loader, desc="Evaluation en cours")

    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(batch["labels"].cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            progress_bar.set_postfix({
                "val_loss": total_loss / (step + 1)
            })
    avg_loss = total_loss / len(val_loader)

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    with open(LOG_PATH,"a") as file :
        file.write(f"\n {dt.datetime.now()}")
        file.write(f"\nResultat de la Validation")
        file.write(f"\nLoss: {avg_loss:.4f}")
        file.write(f"\nAccuracy: {acc:.4f}")
        file.write(f"\nROC-AUC: {roc_auc:.4f}")
        file.write(f"\nPR-AUC: {pr_auc:.4f}")
        file.write(f"\nRapport de la classification :\n {classification_report(y_true, y_pred, target_names=['Non-GMO', 'GMO'])}")
        
    return avg_loss, acc, roc_auc, pr_auc


def compute_class_weights(csv_path):
    """Calcule les poids des classes en inverse de leur fréquence."""
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index()
    total = len(df)
    # Pondération inverse de la fréquence
    weights = total / (len(class_counts) * class_counts.values)
    return torch.FloatTensor(weights)


def create_weighted_sampler(dataset, csv_path):
    """Crée un échantillonneur pondéré pour équilibrer les batches."""
    df = pd.read_csv(csv_path)
    class_counts = df['label'].value_counts().sort_index()
    
    # Poids pour chaque classe (inverse de sa fréquence)
    class_weights = 1.0 / class_counts.values
    
    # Attribue un poids à chaque échantillon en fonction de son label
    sample_weights = []
    for idx in range(len(dataset)):
        # Récupère le label pour ce chunk (remonte à la ligne originale)
        import bisect
        row_idx = bisect.bisect_right(dataset.cum, idx)
        label = int(df.iloc[row_idx]['label'])
        sample_weights.append(class_weights[label])
    
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def test():
    """Évalue le modèle sur l'ensemble de test."""
    if os.path.exists(TEST_PATH):
        # Charge le meilleur modèle
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_ds = GMODataset(TEST_PATH, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        
        # Évaluation finale sur les données de test
        test_loss, test_acc, test_roc, test_pr = evaluate(model, test_loader, DEVICE)

        print(f"\n--- RÉSULTATS DU TEST ---")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ROC-AUC: {test_roc:.4f}")
        print(f"Test PR-AUC: {test_pr:.4f}")
        
        with open(LOG_PATH, "a") as f:
            f.write(f"\n{'='*50}\nRÉSULTATS FINAUX TEST\n{'='*50}\n")
            f.write(f"Test Loss: {test_loss:.4f}\nTest Acc: {test_acc:.4f}\n")
            f.write(f"Test ROC-AUC: {test_roc:.4f}\nTest PR-AUC: {test_pr:.4f}\n")
    else:
        print(f"Fichier test introuvable {TEST_PATH}. test non effectuer.")

if __name__ == "__main__":
    try:
        import os
        # Charge le modèle ou utilise la version en ligne s'il n'existe pas localement
        MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else HF_MODEL_NAME
        print(f"Chargement du modele depuis: {MODEL_NAME}")
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MAX_LEN = 2048
        STRIDE = MAX_LEN // 2 + MAX_LEN // 4
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        train_ds = GMODataset(TRAIN_PATH, tokenizer)
        val_ds   = GMODataset(VAL_PATH, tokenizer)
        
        # Calcul des poids des classes pour les données déséquilibrées
        class_weights = compute_class_weights(TRAIN_PATH)
        print(f"Poids des classes: Non-GMO={class_weights[0]:.4f}, GMO={class_weights[1]:.4f}")
        
        # Crée un échantillonneur pondéré pour équilibrer les batches
        train_sampler = create_weighted_sampler(train_ds, TRAIN_PATH)
        
        # Utilise l'échantillonneur au lieu de shuffle pour équilibrer l'entraînement
        train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        # Passe les poids des classes au modèle
        model = NTForGMO(MODEL_NAME, class_weights=class_weights).to(DEVICE)

        # optimizer = AdamW(model.parameters(), lr=2e-5)
        EPOCHS = 5
        accum_steps = 4

        # Configuration de LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )

        model.encoder = get_peft_model(model.encoder, lora_config)
        model.encoder.print_trainable_parameters()

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

        train(model, train_loader, val_loader, optimizer, EPOCHS, accum_steps, DEVICE)
        test()
        notif.send_message("Entraînement du model terminé !")
    except Exception as e:
        print(f"Erreur survenue: {str(e)}")
        notif.send_message(f"❌ Entraînement crash: {str(e)}")


