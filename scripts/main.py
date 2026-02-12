import torch
import os
from transformers import AutoTokenizer
from utils.GmoDataset import GMODataset
from torch.utils.data import DataLoader
from utils.Tuning import NTForGMO
from transformers import AdamW
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report
import utils.notification as notif

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-V2-250m-multi-species"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def train():
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
                loss = outputs["loss"] / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print(f"Epoch {epoch+1} | Train loss: {total_loss/len(train_loader):.4f}")

def evaluation():
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs["logits"], dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch["labels"].cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=["Non-GMO", "GMO"]))


if __name__ == "__main__":
    train_ds = GMODataset("data/processed/splits/train.csv", tokenizer)
    val_ds   = GMODataset("data/processed/splits/val.csv", tokenizer)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    model = NTForGMO(MODEL_NAME).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    EPOCHS = 5
    scaler = GradScaler()
    accum_steps = 4

    train()
    evaluation()

    notif.send_message(" Entraînement du model terminé ! :tada: :tada: :tada:")


