import os
import csv
import random
from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split
from common import clean_sequence, random_subseq, mutate


def load_fasta(path):
    """ Prends un chemin vers un fichier FASTA
        Retourne une liste de séquences nettoyées 
        (en majuscules, sans caractères invalides). """
    seqs = []
    for rec in SeqIO.parse(path, "fasta"):
        seqs.append(clean_sequence(str(rec.seq)))
    return seqs


def append_to_csv(samples, csv_path):
    """ Prends une liste de dicts représentant des échantillons et un chemin vers un fichier CSV.
        Ajoute les échantillons au fichier CSV, en créant le fichier et les dossiers nécessaires si ils n'existent pas,
        et en écrivant l'en-tête si le fichier est créé. 
        Retourne Rien(None)."""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "name", "sequence", "label", "source", "length"])

        for s in samples:
            writer.writerow([s["id"], s["name"], s["sequence"], s["label"], s["source"], s["length"]])

def fasta_to_csv(fasta_path, csv_writer, label, name, source="GenBank", min_len=200 ):
    """ Prends un chemin vers un fichier FASTA, un objet csv_writer, un label (0 ou 1), 
        un nom de séquence et une source.
        Lit les séquences du fichier FASTA, les nettoie, et écrit une ligne dans 
        le CSV pour chaque séquence valide 
        (longueur >= min_len) avec les champs id, name, sequence, label, source et length. 
        Retourne Rien(None)."""
    count = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = clean_sequence(str(record.seq))
        if len(seq) < min_len:
            continue

        seq_id = record.id

        csv_writer.writerow([
            seq_id,
            name,
            seq,
            label,
            source,
            len(seq)
        ])
        count += 1

    print(f"[✓] {count} séquences enregistrees ")


def build_dataset(fasta_path, output_csv, isGMO=False):
    """ Prends un chemin vers un fichier FASTA, un chemin vers un fichier CSV de sortie, 
        et un booléen indiquant si les séquences sont des OGM.
        Lit les séquences du fichier FASTA, les nettoie, et écrit une ligne dans le CSV 
        pour chaque séquence valide 
        (longueur >= 200) avec les champs id, name, sequence, label, source et length. 
        Le champ label est 1 si isGMO est True, sinon 0. 
        Le champ name est dérivé du nom du fichier FASTA. 
        Retourne Rien(None)."""
    file_exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    filename = os.path.basename(fasta_path)
    name = os.path.splitext(filename)[0]
    seqName = "_".join(name.split("_")[:2])

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "name", "sequence", "label", "source", "length"])

        fasta_to_csv(
            fasta_path=fasta_path,
            csv_writer=writer,
            label= 1 if isGMO else 0,
            name=seqName
        )

    print(f"[✓] Dataset Enregistree vers {output_csv}")

def build_synthetic_gmo(plant_genomes, gmo_name, promoter_seqs, gene_seqs, terminator_seqs,
        n_samples=1000, flank_min=300, flank_max=2000, mutation_rate=0.0005):
    """ Prends 
    - une liste de séquences génomiques de plantes,
    - un nom de GMO (ex: "Gossypium_hirsutum"),
    - une liste de séquences de promoteurs,
    - une liste de séquences de gènes,
    - une liste de séquences de terminateurs,
    - un nombre d'échantillons à générer,
    - une longueur minimale et maximale pour les séquences de flancs génomiques,
    - un taux de mutation pour introduire des mutations aléatoires dans les séquences synthétiques.
    Génère des séquences synthétiques d'OGM en combinant aléatoirement des flancs génomiques de plantes,
    des promoteurs, des gènes et des terminateurs, puis en introduisant des mutations aléatoires.
    Retourne une liste de dicts représentant les échantillons synthétiques, avec les champs id, name, sequence, label, source et length. 
    Le champ label est toujours 1"""
    samples = []

    for i in range(n_samples):
        plant = random.choice(plant_genomes)
        promoter = random.choice(promoter_seqs)
        gene = random.choice(gene_seqs)
        terminator = random.choice(terminator_seqs)

        left_flank  = random_subseq(plant, flank_min, flank_max)
        right_flank = random_subseq(plant, flank_min, flank_max)

        synthetic = (left_flank + promoter + gene + terminator + right_flank)

        synthetic = mutate(synthetic, mutation_rate)
        
        samples.append({
            "id": f"{gmo_name}'_'{i:06d}",
            "name" : "synthetic_GMO",
            "sequence": synthetic,
            "label": 1,
            "source": "me",
            "length": len(synthetic)
        })

    return samples


def split_dataset(input_csv, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    """ prends
    - un chemin vers un fichier CSV d'entrée,
    - des proportions pour les splits train, validation et test,
    - une graine pour la reproductibilité du split. 
    Lit le dataset à partir du fichier CSV, puis le divise en trois parties: train, validation et test,
    en respectant les proportions données et en stratifiant par label pour maintenir la même distribution de classes dans chaque partie. 
    Retourne les trois DataFrames correspondants aux splits train, validation et test."""

    df = pd.read_csv(input_csv)

    # Premier decoupage: train et temp (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), 
        stratify=df["label"], 
        random_state=random_state
    )

    # Second decoupage: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df["label"],
        random_state=random_state
    )
    return train_df, val_df, test_df


def save_split_datasets(train_df, val_df, test_df, output_dir):
    """ Prends
    - trois DataFrames correspondant aux splits train, validation et test,
    - un chemin vers un répertoire de sortie.
    Enregistre les trois DataFrames dans des fichiers CSV séparés (train.csv, val.csv, test.csv) 
    dans le répertoire de sortie donné"""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)