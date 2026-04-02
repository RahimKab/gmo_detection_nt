
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from utils.GmoDataset import GMODataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.Tuning import NTForGMO, save_checkpoint, load_checkpoint, save_history, load_history
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import torch.nn.functional as F
import utils.notification as notif
from peft import LoraConfig, get_peft_model

from pathlib import Path
from huggingface_hub import snapshot_download
PROJECT_PATH = Path(__file__).resolve().parents[1]

TRAIN_PATH = f"{PROJECT_PATH}/data/processed/splits/train.csv"
VAL_PATH = f"{PROJECT_PATH}/data/processed/splits/val.csv"
TEST_PATH = f"{PROJECT_PATH}/data/processed/splits/test.csv"
MODEL_SAVE_PATH = f"{PROJECT_PATH}/result/best_model.pt"
CHECKPOINT_PATH = f"{PROJECT_PATH}/result/checkpoint.pt"
LOG_PATH = f"{PROJECT_PATH}/result/logs.txt"
HISTORY_PATH = f"{PROJECT_PATH}/result/history.json"
PLOTS_DIR = f"{PROJECT_PATH}/result/plots"
MY_MODEL_NAME = "GMO_DETECTION_NT_V2_250M"

# Chemin local du modèle pré-entraîné
LOCAL_MODEL_PATH = f"{PROJECT_PATH}/pretrained_model"
# Nom du modèle Hugging Face à utiliser si le modèle local n'est pas trouvé
HF_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-V2-250m-multi-species"
TARGET_GMO_RECALL = 0.90

# Garde-fous anti-overfitting
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 1
MIN_LR = 1e-7


def ensure_local_pretrained_model(local_path, hf_model_name):
    """Telecharge le snapshot HF en local si le dossier est absent/incomplet."""
    local_path = Path(local_path)
    has_config = (local_path / "config.json").exists()
    has_weights = any(local_path.glob("*.safetensors")) or any(local_path.glob("*.bin"))

    if has_config and has_weights:
        return str(local_path)

    print(f"Modele local absent/incomplet. Telechargement vers: {local_path}")
    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_model_name,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
    )
    return str(local_path)


def ensure_history_schema(history):
    """Garantit les cles necessaires, y compris en reprise d'anciens historiques."""
    for key in [
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_f1",
        "val_f1",
        "val_precision_gmo",
        "val_recall_gmo",
        "val_threshold",
        "val_roc_auc",
        "val_pr_auc",
    ]:
        history.setdefault(key, [])
    return history


def compute_binary_metrics(y_true, y_prob, threshold):
    """Calcule les metriques binaires a partir des probabilites et d'un seuil."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_gmo": precision_score(y_true, y_pred, zero_division=0),
        "recall_gmo": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "y_pred": y_pred.tolist(),
    }


def find_operating_threshold(y_true, y_prob, target_recall=0.90):
    """Trouve un seuil qui respecte le recall cible, puis maximise le F1."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    candidate_thresholds = np.linspace(0.40, 0.95, 111) # 111 valeur pour une pas de 0.005

    best_with_constraint = None
    best_global = None

    for thr in candidate_thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)

        entry = {
            "threshold": float(thr),
            "f1": float(f1),
            "recall_gmo": float(rec),
            "precision_gmo": float(prec),
        }

        if (best_global is None) or (entry["f1"] > best_global["f1"]):
            best_global = entry

        if rec >= target_recall:
            if (best_with_constraint is None) or (entry["f1"] > best_with_constraint["f1"]):
                best_with_constraint = entry

    if best_with_constraint is not None:
        best_with_constraint["meets_target_recall"] = True
        return best_with_constraint

    best_global["meets_target_recall"] = False
    return best_global


def save_confusion_matrix(y_true, y_pred, split_name, epoch):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Non-GMO", "GMO"]).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Matrice de confusion ({split_name})")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/confusion_matrix_{split_name.lower()}_epoch_{epoch + 1}.png", dpi=180)
    plt.close(fig)


def save_roc_curve(y_true, y_prob, split_name, epoch):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("Taux de False Positive")
    ax.set_ylabel("Taux de True Positive")
    ax.set_title(f"Courbe ROC ({split_name})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/roc_{split_name.lower()}_epoch_{epoch + 1}.png", dpi=180)
    plt.close(fig)


def save_training_curves(history):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss: Train vs Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Val Accuracy")
    axes[1].set_title("Precision: Train vs Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/accuracy_loss_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_f1"], marker="o", label="Train F1")
    ax.plot(epochs, history["val_f1"], marker="o", label="Val F1")
    ax.set_title("F1 Score: Train vs Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/f1_curve.png", dpi=180)
    plt.close(fig)

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """Crée un DataLoader pour charger les données en batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def train(model,train_loader,val_loader,optimizer,epochs,accum_steps,device,
    checkpoint_interval=100,scheduler=None,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    lora_config=None,
):
    scaler = GradScaler()
    
    # Chargement de l'historique d'entraînement
    history = ensure_history_schema(load_history(HISTORY_PATH))
    # Charge le checkpoint s'il existe
    start_epoch, start_step, best_pr_auc = load_checkpoint(model, optimizer, scaler, device, CHECKPOINT_PATH)
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        ema_loss = None
        train_true, train_pred, train_prob = [], [], []

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
                logits = outputs["logits"]

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                # Stabilise l'entrainement en precision mixte avant l'update.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            batch_loss = loss.item() * accum_steps
            ema_loss = batch_loss if ema_loss is None else (0.98 * ema_loss + 0.02 * batch_loss)

            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            train_true.extend(batch["labels"].detach().cpu().numpy())
            train_pred.extend(preds.detach().cpu().numpy())
            train_prob.extend(probs.detach().cpu().numpy())

            progress_bar.set_postfix({
                "batch_loss": f"{batch_loss:.4f}",
                "ema_loss": f"{ema_loss:.4f}",
                "avg_loss": f"{(total_loss / (step + 1)):.4f}",
            })

            # Enregistre les checkpoints periodiquement
            if (step + 1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scaler, epoch, step, best_pr_auc, CHECKPOINT_PATH)

        # Reprend start_step apres le premier epoch
        start_step = 0

        # Enregistre le checkpoint a la fin de chaque epoch
        save_checkpoint(model, optimizer, scaler, epoch + 1, 0, best_pr_auc)

        train_avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, zero_division=0)

        val_base = evaluate(model, val_loader, device, threshold=0.5, log_tag="Validation@0.50")
        
        threshold_choice = find_operating_threshold(
            val_base["y_true"],
            val_base["y_prob"],
            target_recall=TARGET_GMO_RECALL,
        )
        val_metrics = {
            **compute_binary_metrics(
                val_base["y_true"],
                val_base["y_prob"],
                threshold_choice["threshold"],
            ),
            "loss": val_base["loss"],
            "roc_auc": val_base["roc_auc"],
            "pr_auc": val_base["pr_auc"],
            "threshold": threshold_choice["threshold"],
            "y_true": val_base["y_true"],
            "y_prob": val_base["y_prob"],
        }
        
        # Sauvegarde des métriques pour le suivi de l'entraînement
        history['train_loss'].append(train_avg_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['acc'])
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision_gmo'].append(val_metrics['precision_gmo'])
        history['val_recall_gmo'].append(val_metrics['recall_gmo'])
        history['val_threshold'].append(val_metrics['threshold'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        save_history(history, HISTORY_PATH)

        print(
            f"Seuil choisi={val_metrics['threshold']:.3f}"
            f" | Recall GMO={val_metrics['recall_gmo']:.4f}"
            f" | Precision GMO={val_metrics['precision_gmo']:.4f}"
            f" | cible recall={TARGET_GMO_RECALL:.2f}"
        )

        save_confusion_matrix(train_true, train_pred, "Train", epoch)
        save_confusion_matrix(val_metrics['y_true'], val_metrics['y_pred'], "Validation", epoch)
        save_roc_curve(train_true, train_prob, "Train", epoch)
        save_roc_curve(val_metrics['y_true'], val_metrics['y_prob'], "Validation", epoch)
        save_training_curves(history)

        # Pilotage du LR sur la loss validation pour limiter la memorisation.
        old_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]["lr"]

        improved_val_loss = val_metrics['loss'] < (best_val_loss - early_stopping_min_delta)
        if improved_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"LR: {old_lr:.2e} -> {new_lr:.2e}"
            f" | best_val_loss: {best_val_loss:.4f}"
            f" | no_improve: {epochs_without_improvement}/{early_stopping_patience}"
        )

        if val_metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            torch.save({
                "model_state_dict": model.state_dict(),
                "pr_auc": best_pr_auc,
                "decision_threshold": float(val_metrics['threshold']),
                "target_gmo_recall": TARGET_GMO_RECALL,
                'model_name': MY_MODEL_NAME,
                'lora_config': {
                    'r': int(lora_config.r) if lora_config is not None else 8,
                    'lora_alpha': int(lora_config.lora_alpha) if lora_config is not None else 16,
                    'target_modules': list(lora_config.target_modules) if lora_config is not None else ["query", "value"],
                    'lora_dropout': float(lora_config.lora_dropout) if lora_config is not None else 0.1,
                    'bias': str(lora_config.bias) if lora_config is not None else "none"
                }
            }, MODEL_SAVE_PATH)
            save_checkpoint(model, optimizer, scaler, epoch + 1, 0, best_pr_auc, path=CHECKPOINT_PATH)
            with open(LOG_PATH,"a") as file :
                file.write(
                    f"\nMeilleur modele sauvegarde!"
                    f"\n avg_loss: {val_metrics['loss']:.4f}"
                    f"\n accuracy: {val_metrics['acc']:.4f}"
                    f"\n roc_auc: {val_metrics['roc_auc']:.4f}"
                    f"\n pr_auc: {val_metrics['pr_auc']:.4f} \n"
                )
            print(" Meilleur modele sauvegarde! ")

        print(
            f"Epoch {epoch+1} finished"
            f" | Train loss: {train_avg_loss:.4f}"
            f" | Train acc: {train_acc:.4f}"
            f" | Val loss: {val_metrics['loss']:.4f}"
            f" | Val acc: {val_metrics['acc']:.4f}"
            f" | Val f1: {val_metrics['f1']:.4f}"
            f" | Val rec GMO: {val_metrics['recall_gmo']:.4f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping active: val_loss n'ameliore plus depuis "
                f"{epochs_without_improvement} epoch(s)."
            )
            break
    return history

def evaluate(model, val_loader, device, threshold=0.5, log_tag="Validation"):
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

            y_true.extend(batch["labels"].cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            progress_bar.set_postfix({
                "val_loss": total_loss / (step + 1)
            })
    avg_loss = total_loss / len(val_loader)

    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    y_pred = metrics["y_pred"]
    acc = metrics["acc"]
    roc_auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]
    f1 = metrics["f1"]
    precision_gmo = metrics["precision_gmo"]
    recall_gmo = metrics["recall_gmo"]

    with open(LOG_PATH,"a") as file :
        file.write(f"\n {dt.datetime.now()}")
        file.write(f"\nResultat de {log_tag}")
        file.write(f"\nSeuil de decision GMO: {threshold:.3f}")
        file.write(f"\nLoss: {avg_loss:.4f}")
        file.write(f"\nAccuracy: {acc:.4f}")
        file.write(f"\nROC-AUC: {roc_auc:.4f}")
        file.write(f"\nPR-AUC: {pr_auc:.4f}")
        file.write(f"\nPrecision GMO: {precision_gmo:.4f}")
        file.write(f"\nRecall GMO: {recall_gmo:.4f}")
        file.write(f"\nF1-score: {f1:.4f}")
        file.write(f"\nRapport de la classification :\n {classification_report(y_true, y_pred, target_names=['Non-GMO', 'GMO'])}")

    return {
        "loss": avg_loss,
        "acc": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision_gmo": precision_gmo,
        "recall_gmo": recall_gmo,
        "threshold": threshold,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


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
        # Récupère le label pour ce morceau (remonte à la ligne originale)
        import bisect
        row_idx = bisect.bisect_right(dataset.cum, idx)
        label = int(df.iloc[row_idx]['label'])
        sample_weights.append(class_weights[label])
    
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def test(model, device, tokenizer):
    """Évalue le modèle sur l'ensemble de test."""
    if os.path.exists(TEST_PATH):
        # Charge le meilleur modèle si compatible, sinon evalue le modele courant.
        if os.path.exists(MODEL_SAVE_PATH):
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as exc:
                print(
                    "best_model.pt incompatible avec l'architecture courante. "
                    "Evaluation du modele en memoire."
                )
                print(f"Detail (resume): {str(exc).splitlines()[0]}")
                decision_threshold = 0.5
        else:
            print("Aucun best_model.pt trouve. Evaluation du modele en memoire.")
            decision_threshold = 0.5
        
        test_ds = GMODataset(TEST_PATH, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        
        # Évaluation finale sur les données de test
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            threshold=decision_threshold,
            log_tag="Test",
        )

        save_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], "Test", epoch=0)
        save_roc_curve(test_metrics["y_true"], test_metrics["y_prob"], "Test", epoch=0)

        print(f"\n--- RÉSULTATS DU TEST ---")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['acc']:.4f}")
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"Test Precision GMO: {test_metrics['precision_gmo']:.4f}")
        print(f"Test Recall GMO: {test_metrics['recall_gmo']:.4f}")
        print(f"Test F1-score: {test_metrics['f1']:.4f}")
        print(f"Seuil de decision utilise: {decision_threshold:.3f}")
        
        with open(LOG_PATH, "a") as f:
            f.write(f"\n{'='*50}\nRÉSULTATS FINAUX TEST\n{'='*50}\n")
            f.write(f"Test Loss: {test_metrics['loss']:.4f}\nTest Acc: {test_metrics['acc']:.4f}\n")
            f.write(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}\nTest PR-AUC: {test_metrics['pr_auc']:.4f}\n")
            f.write(f"Test Precision GMO: {test_metrics['precision_gmo']:.4f}\n")
            f.write(f"Test Recall GMO: {test_metrics['recall_gmo']:.4f}\n")
            f.write(f"Seuil de decision: {decision_threshold:.3f}\n")
            f.write(f"Test F1-score: {test_metrics['f1']:.4f}\n")
    else:
        print(f"Fichier test introuvable {TEST_PATH}. test non effectuer.")

if __name__ == "__main__":
    try:
        # Utilise un snapshot local propre pour eviter les incoherences cache/config/poids.
        MODEL_NAME = ensure_local_pretrained_model(LOCAL_MODEL_PATH, HF_MODEL_NAME)
        print(f"Chargement du modele depuis: {MODEL_NAME}")
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MAX_LEN = 2048
        STRIDE = MAX_LEN // 2 + MAX_LEN // 4
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
        train_ds = GMODataset(TRAIN_PATH, tokenizer)
        val_ds   = GMODataset(VAL_PATH, tokenizer)
        
        # Choisir une seule strategie d'equilibrage pour eviter l'instabilite de la loss.
        use_weighted_sampler = True
        use_class_weights = not use_weighted_sampler

        class_weights = compute_class_weights(TRAIN_PATH)
        print(f"Poids des classes: Non-GMO={class_weights[0]:.4f}, GMO={class_weights[1]:.4f}")
        
        # Cree un echantillonneur pondere pour equilibrer les batches.
        train_sampler = create_weighted_sampler(train_ds, TRAIN_PATH) if use_weighted_sampler else None

        # Utilise l'echantillonneur ou un shuffle classique.
        if train_sampler is not None:
            print("Strategie d'equilibrage: WeightedRandomSampler (sans class_weights).")
            train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler)
            class_weights_for_model = None
        else:
            print("Strategie d'equilibrage: class_weights (sans sampler).")
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
            class_weights_for_model = class_weights if use_class_weights else None

        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        # Passe les poids des classes au modèle
        model = NTForGMO(MODEL_NAME, class_weights=class_weights_for_model).to(DEVICE)

        # optimizer = AdamW(model.parameters(), lr=2e-5)
        EPOCHS = 12
        accum_steps = 4

        # Configuration de LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )

        model.encoder = get_peft_model(model.encoder, lora_config)
        model.encoder.print_trainable_parameters()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=5e-3,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )

        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            EPOCHS,
            accum_steps,
            DEVICE,
            scheduler=scheduler,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            lora_config=lora_config,
        )
        test(model=model, device=DEVICE, tokenizer=tokenizer)
        notif.send_message("Entraînement du model terminé !")
    except Exception as e:
        print(f"Erreur survenue: {str(e)}")
        notif.send_message(f"[X] Entraînement crash: {str(e)}")


