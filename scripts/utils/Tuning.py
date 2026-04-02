import os
import torch.nn as nn
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoConfig

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = f"{PROJECT_PATH}/result/checkpoint.pt"

class NTForGMO(nn.Module):
    """ Prends
        un nom de modèle pré-entraîné.
        Initialise un modèle de classification binaire pour les séquences d'ADN en utilisant 
        un encodeur de modèle pré-entraîné et une tête de classification.
        Retourne une instance du modèle de classification.
    """
    def __init__(self, model_name, class_weights=None):
        super().__init__()
        local_files_only = os.path.isdir(model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_attentions = True
        config.num_labels = 2
        config.problem_type = "single_label_classification"
        if not hasattr(config, 'is_decoder'):
            config.is_decoder = False
        if not hasattr(config, 'add_cross_attention'):
            config.add_cross_attention = False

        try:
            self.encoder = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        except RuntimeError as exc:
            if "size mismatch" in str(exc).lower():
                raise RuntimeError(
                    "Echec du chargement strict des poids pre-entraines (size mismatch). "
                    "Le cache/les fichiers locaux peuvent etre incoherents. "
                    "Re-telecharge le modele avec scripts/download_model.py puis relance l'entrainement."
                ) from exc
            raise

        # Sauvegarde les poids de classe pour une utilisation dans la fonction de perte,
        self.class_weights = class_weights

        # Garde un alias pour compatibilite avec scripts utilitaires existants.
        self.classifier = self.encoder.classifier

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Utilise les poids de classe si fournis, 
            # pour gérer les classes déséquilibrées
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight)(logits, labels)
            else:
                loss = outputs.loss

        return {
            "loss": loss,
            "logits": logits
        }
    


def save_checkpoint(model, optimizer, scaler, epoch, step, best_pr_auc, path=CHECKPOINT_PATH):
    """Enregistre le checkpoint de l'entrainement pour une reprise ulterieur ."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_pr_auc': best_pr_auc,
    }
    torch.save(checkpoint, path)
    print(f" Checkpoint enregistrer a epoch {epoch+1}, step {step+1}")

def load_checkpoint(model, optimizer, scaler, device, path=CHECKPOINT_PATH):
    """Charge le checkpoint pour reprendre l'entrainement."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except RuntimeError as exc:
            # Permet de continuer si l'architecture change (ex: AutoModel -> AutoModelForSequenceClassification).
            print(
                "Checkpoint incompatible avec l'architecture courante. "
                "Reprise ignoree et entrainement depuis zero."
            )
            print(f"Detail (resume): {str(exc).splitlines()[0]}")
            try:
                os.remove(path)
                print(f"Ancien checkpoint supprime: {path}")
            except OSError:
                pass
            return 0, 0, 0

        epoch = checkpoint['epoch']
        step = checkpoint['step']
        best_pr_auc = checkpoint['best_pr_auc']
        print(f" Reprise depuis l'epoch {epoch+1}, step {step+1}, best_pr_auc: {best_pr_auc:.4f}")
        return epoch, step, best_pr_auc
    return 0, 0, 0


def visualize_dna_attention(sequence, model, tokenizer, layer=-1):
    """
    Génère un graphique montrant où le modèle 'regarde' dans la séquence.
    """

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Shape: (batch, heads, seq_len, seq_len)
    attention = outputs.attentions[layer] 
    attention = attention.mean(dim=1).squeeze().cpu().numpy()
    
    importance_scores = attention.sum(axis=0)

    # Affichage
    plt.figure(figsize=(15, 3))
    plt.fill_between(range(len(importance_scores)), importance_scores, color="skyblue", alpha=0.4)
    plt.plot(range(len(importance_scores)), importance_scores, color="Slateblue", lw=1.5)
    
    plt.title(f"Carte d'Attention Génomique (Couche {layer})")
    plt.xlabel("Position dans la séquence (tokens)")
    plt.ylabel("Score d'Importance")
    
    # Overlay des bases lorsque la séquence est courte
    if len(sequence) < 100:
        plt.xticks(range(len(sequence)), list(sequence))
    
    plt.tight_layout()
    plt.show()


def save_history(history, path):
    """Sauvegarde l'historique d'entrainement dans un fichier json."""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Historique sauvegardee: {path}")


def load_history(path):
    """Charge l'historique lorsqu'il existe."""
    default_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'val_roc_auc': [],
        'val_pr_auc': []
    }

    if os.path.exists(path):
        with open(path, 'r') as f:
            history = json.load(f)

        # Complete les anciennes structures d'historique sans casser la reprise.
        for key, value in default_history.items():
            history.setdefault(key, value)
        return history

    return default_history