import os
import torch.nn as nn
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoConfig

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
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_attentions = True
        if not hasattr(config, 'is_decoder'):
            config.is_decoder = False
        if not hasattr(config, 'add_cross_attention'):
            config.add_cross_attention = False

        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )
        hidden_size = self.encoder.config.hidden_size
        
        # Sauvegarde les poids de classe pour une utilisation dans la fonction de perte,
        self.class_weights = class_weights

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token
        cls_emb = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_emb)

        loss = None
        if labels is not None:
            # Utilise les poids de classe si fournis, 
            # pour gérer les classes déséquilibrées
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
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
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_roc_auc': [],
        'val_pr_auc': []
    }