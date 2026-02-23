import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.GmoDataset import GMODataset
from torch.utils.data import DataLoader
from utils.Tuning import NTForGMO
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score, accuracy_score)
import torch.nn.functional as F
import utils.notification as notif

PROJECT_PATH = "/home/strange/Documents/master_2/internship/model"
TRAIN_PATH = f"{PROJECT_PATH}/data/processed/splits/sample.csv"
VAL_PATH = f"{PROJECT_PATH}/data/processed/splits/sample_val.csv"
MODEL_SAVE_PATH = f"{PROJECT_PATH}/result/best_model.pt"
LOG_PATH = f"{PROJECT_PATH}/result/logs.txt"

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def train(model, train_loader, val_loader, optimizer, epochs, accum_steps, device): # Real batch_size = batch_size * accum_steps
    scaler = GradScaler()
    best_pr_auc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):

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

        avg_loss, acc, roc_auc, pr_auc = evaluate(model, val_loader, device)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            with open(LOG_PATH,"a") as file :
                file.write(f"\nMeilleur modele sauvegarde! \n avg_loss: {avg_loss:.4f}\n accuracy: {acc:.4f}\n roc_auc: {roc_auc:.4f}\n pr_auc: {pr_auc:.4f} ")
            print(" Meilleur modele sauvegarde! ")

        print(f"Epoch {epoch+1} finished | Avg loss: {total_loss/len(train_loader):.4f}")

def evaluate(model, val_loader, device):
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
        file.write(f"\nResultat de la Validation")
        file.wrtite(f"\nLoss: {avg_loss:.4f}")
        file.write(f"\nAccuracy: {acc:.4f}")
        file.write(f"\nROC-AUC: {roc_auc:.4f}")
        file.write(f"\nPR-AUC: {pr_auc:.4f}")
        file.write(f"\Rapport de la classification :\n {classification_report(y_true, y_pred, target_names=['Non-GMO', 'GMO'])}")
        
    return avg_loss, acc, roc_auc, pr_auc


if __name__ == "__main__":
    try:
        MODEL_NAME = "InstaDeepAI/nucleotide-transformer-V2-250m-multi-species"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        train_ds = GMODataset(TRAIN_PATH, tokenizer)
        val_ds   = GMODataset(VAL_PATH, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        model = NTForGMO(MODEL_NAME).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=2e-5)
        EPOCHS = 5
        accum_steps = 4

        train(model, train_loader, val_loader, optimizer, EPOCHS, accum_steps, DEVICE)

        notif.send_message("Entraînement du model terminé !")
    except Exception as e:
        notif.send_message(f"❌ Entraînement crashed: {str(e)}")


