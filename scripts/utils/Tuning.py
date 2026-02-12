import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class NTForGMO(nn.Module):
    """ Prends
        un nom de modèle pré-entraîné.
        Initialise un modèle de classification binaire pour les séquences d'ADN en utilisant 
        un encodeur de modèle pré-entraîné et une tête de classification.
        Retourne une instance du modèle de classification.
    """
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )
        hidden_size = self.encoder.config.hidden_size

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
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }
