import bisect
import torch
import pandas as pd
from torch.utils.data import Dataset


class GMODataset(Dataset):
    def __init__(self, csv_path, tokenizer, chunk_size=512, stride=256, max_chunks_per_seq=None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.max_chunks = max_chunks_per_seq

        # Fais le traitement avec les le nombre de sequence maximal(512) valides (sans 'N')
        self.chunks_per_row = []
        for seq in self.df['sequence'].astype(str):
            s = seq.upper()
            n_chunks = max(0, (len(s) - chunk_size) // stride + 1) if len(s) >= chunk_size else 1 if len(s) > 0 else 0
            # Compte uniquement les morceaux de sequence sans 'N'
            count = 0
            for i in range(0, len(s) - chunk_size + 1, stride):
                chunk = s[i:i+chunk_size]
                if 'N' not in chunk:
                    count += 1
                    if self.max_chunks and count >= self.max_chunks:
                        break
            # Traite les séquences courtes (< chunk_size) comme un seul morceau de sequence si valide
            if len(s) < chunk_size and 'N' not in s:
                count = max(count, 1)
            self.chunks_per_row.append(count)

        # Fais le cumule des indice pour le mappage.
        self.cum = []
        total = 0
        for c in self.chunks_per_row:
            total += c
            self.cum.append(total)

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    def __getitem__(self, idx):
        # Trouve la ligne contenant l'indice
        row_idx = bisect.bisect_right(self.cum, idx)
        row_start_cum = self.cum[row_idx - 1] - self.chunks_per_row[row_idx] if row_idx > 0 else 0
        local_idx = idx - row_start_cum # 

        seq = str(self.df.iloc[row_idx]['sequence']).upper()
        label = int(self.df.iloc[row_idx]['label'])

        # Parcoure la sequence jusqu'a ce qu'on trouve le ie indice valide (pas de 'N')
        valid_seen = 0
        for i in range(0, len(seq) - self.chunk_size + 1, self.stride):
            chunk = seq[i:i + self.chunk_size]
            if 'N' in chunk:
                continue
            if valid_seen == local_idx:
                selected = chunk
                break
            valid_seen += 1
        else:
            # 
            selected = seq[:self.chunk_size].ljust(self.chunk_size, 'A')

        tokens = self.tokenizer(
            selected,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.chunk_size
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }