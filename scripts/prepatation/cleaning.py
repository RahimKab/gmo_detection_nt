from Bio import SeqIO
import random
import csv
import os
import re
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def chunk_sequence(seq, chunk_size=512, stride=256):
	seq = seq.upper()
	for i in range(0, len(seq) - chunk_size + 1, stride):
		yield seq[i : i + chunk_size]


def clean_sequence(seq):
	seq = seq.upper()
	return re.sub(r"[^ACGTN]", "N", seq)


def fasta_to_csv(fasta_path, csv_writer, label, name, source="GenBank", num_seq=10, 
				 min_len=200, max_len=512):
	count = 0
	for record in SeqIO.parse(str(fasta_path), "fasta"):
		seq = clean_sequence(str(record.seq))
		if len(seq) < min_len:
			continue

		seq_id = record.id
		if len(seq) > max_len:
			for chunk_seq in chunk_sequence(seq, max_len, math.ceil(max_len / 2)):
				csv_writer.writerow(
					[
						seq_id,
						name,
						chunk_seq,
						label,
						source,
						len(chunk_seq),
					]
				)
				count += 1
				if count >= num_seq:
					break
		else:
			csv_writer.writerow([seq_id, name, seq, label, source, len(seq)])
			count += 1

		if count >= num_seq:
			break

	print(f"[OK] {count} Sequence sauvegarder vers {fasta_path}")
	

def build_dataset(fasta_path, output_csv, is_gmo=False, num_seq=10, max_len=2048):
	output_csv = Path(output_csv)
	file_exists = output_csv.exists()
	output_csv.parent.mkdir(parents=True, exist_ok=True)

	filename = Path(fasta_path).name
	name = Path(filename).stem
	seq_name = "_".join(name.split("_")[:2])

	with output_csv.open("a", newline="") as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["id", "name", "sequence", "label", "source", "length"])

		fasta_to_csv(
			fasta_path=fasta_path,
			csv_writer=writer,
			label=1 if is_gmo else 0,
			name=seq_name,
			num_seq=num_seq,
			max_len=max_len,
		)

	print(f"[OK] Donnees mis a jour: {output_csv}")


def clean_strict(seq):
	seq = seq.upper()
	return re.sub(r"[^ACGT]", "N", seq)


def random_subseq(seq, min_len, max_len):
	seq_len = len(seq)
	size = random.randint(min_len, max_len)
	if seq_len <= size:
		return seq
	start = random.randint(0, seq_len - size)
	return seq[start : start + size]


def mutate(seq, rate=0.001):
	bases = ["A", "C", "G", "T", "N"]
	seq = list(seq)
	for i in range(len(seq)):
		if random.random() < rate:
			seq[i] = random.choice(bases)
	return "".join(seq)


def build_synthetic_gmo(plant_genomes, gmo_name, promoter_seqs, gene_seqs, terminator_seqs,
    n_samples=1000, min_flank=50, mutation_rate=0.0005 ):

    samples = []
    min_total = 1024
    max_total = 2048
    attempts = 0
    max_attempts = n_samples * 10

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        plant = random.choice(plant_genomes)
        promoter = random.choice(promoter_seqs)
        gene = random.choice(gene_seqs)
        terminator = random.choice(terminator_seqs)

        core_length = len(promoter) + len(gene) + len(terminator)

        if core_length + 2 * min_flank > max_total:
            continue

        min_flank_total = max(2 * min_flank, min_total - core_length)
        max_flank_total = max_total - core_length

        if min_flank_total > max_flank_total:
            continue

        total_flank_len = random.randint(min_flank_total, max_flank_total)
        left_flank_len = random.randint(min_flank, total_flank_len - min_flank)
        right_flank_len = total_flank_len - left_flank_len

        left_flank = random_subseq(plant, left_flank_len, left_flank_len)
        right_flank = random_subseq(plant, right_flank_len, right_flank_len)

        synthetic = left_flank + promoter + gene + terminator + right_flank
        synthetic = mutate(synthetic, mutation_rate)

        samples.append(
            {
                "id": f"{gmo_name}_{len(samples):06d}",
                "name": "synthetic_GMO",
                "sequence": synthetic,
                "label": 1,
                "source": "synthetic",
                "length": len(synthetic),
            }
        )

    return samples


def load_fasta(path):
	seqs = []
	for rec in SeqIO.parse(str(path), "fasta"):
		seqs.append(clean_strict(str(rec.seq)))
	return seqs


def append_to_csv(samples, csv_path):
	csv_path = Path(csv_path)
	file_exists = csv_path.exists()
	csv_path.parent.mkdir(parents=True, exist_ok=True)

	with csv_path.open("a", newline="") as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["id", "name", "sequence", "label", "source", "length"])

		for s in samples:
			writer.writerow(
				[s["id"], s["name"], s["sequence"], s["label"], s["source"], s["length"]]
			)


def shuffle_dataset(input_csv):
	df = pd.read_csv(input_csv)
	df = df.sample(frac=1, random_state=30).reset_index(drop=True)
	df.to_csv(input_csv, index=False)
	print(f"[OK] Melange du jeu donnees termine: {input_csv}")


def split_dataset(input_csv, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
	random_state=42 ):
	
	assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

	os.makedirs(output_dir, exist_ok=True)
	df = pd.read_csv(input_csv)
	df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

	train_df, temp_df = train_test_split(
		df,
		test_size=(1 - train_ratio),
		stratify=df["label"],
		random_state=random_state,
	)

	val_size = val_ratio / (val_ratio + test_ratio)
	val_df, test_df = train_test_split(
		temp_df,
		test_size=(1 - val_size),
		stratify=temp_df["label"],
		random_state=random_state,
	)

	train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
	val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
	test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

	print("[OK] Decoupage du jeu de donnees terminé")
	print(f"  Train: {len(train_df)}")
	print(f"  Val:   {len(val_df)}")
	print(f"  Test:  {len(test_df)}")


def split_dataset_train_test(input_csv, output_dir, train_ratio=0.8, test_ratio=0.2, random_state=42):
	assert abs(train_ratio + test_ratio - 1.0) < 1e-6

	os.makedirs(output_dir, exist_ok=True)
	df = pd.read_csv(input_csv)
	df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

	train_df, test_df = train_test_split(
		df,
		test_size=(1 - train_ratio),
		stratify=df["label"],
		random_state=random_state,
	)

	train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
	test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

	print("[OK] Decoupage du jeu de donnees terminer")
	print(f"  Train: {len(train_df)}")
	print(f"  Test:  {len(test_df)}")


def build_sequences(gen_files, output_csv, is_gmo=False, max_len=2048, num_seq=10):
	for gen_file in gen_files:
		build_dataset(
			fasta_path=RAW_DIR / gen_file,
			output_csv=output_csv,
			is_gmo=is_gmo,
			max_len=max_len,
			num_seq=num_seq,
		)


def build_synthetic_sequences(nb_seq=9976):
	gossypium_genomes = load_fasta(RAW_DIR / "Gossypium_hirsutum_v2.1_genomic.fna")
	vigna_genomes = load_fasta(RAW_DIR / "Vigna_unguiculata_v2_genomic.fna")
	promoters = load_fasta(RAW_DIR / "promoters.fasta")
	genes = load_fasta(RAW_DIR / "gmo_genes.fasta")
	terminators = load_fasta(RAW_DIR / "terminators.fasta")

	nb_seq_per = nb_seq // 2
	count = 0
	output_csv = PROCESSED_DIR / "dataset.csv"

	while count < nb_seq:
		gossypium_samples = build_synthetic_gmo(
			plant_genomes=gossypium_genomes,
			gmo_name="Gossypium_hirsutum",
			promoter_seqs=promoters,
			gene_seqs=genes,
			terminator_seqs=terminators,
			n_samples=nb_seq_per,
			mutation_rate=0.0003,
		)

		vigna_samples = build_synthetic_gmo(
			plant_genomes=vigna_genomes,
			gmo_name="Vigna_unguiculata",
			promoter_seqs=promoters,
			gene_seqs=genes,
			terminator_seqs=terminators,
			n_samples=nb_seq_per,
			mutation_rate=0.0003,
		)

		synthetic_samples = gossypium_samples + vigna_samples
		count += len(synthetic_samples)
		append_to_csv(synthetic_samples, csv_path=output_csv)

	print(f"[OK] {count} Sequences OMG synthetique generer")

def build_val_set():
	oreochromis_genomes = load_fasta(RAW_DIR / "Oreochromis_niloticus_genomic.fna")
	oreochromis_name = "Oreochromis_niloticus"
	promoters = load_fasta(RAW_DIR / "promoters.fasta")
	genes = load_fasta(RAW_DIR / "gmo_genes.fasta")
	terminators = load_fasta(RAW_DIR / "terminators.fasta")

	count = 0
	nb_seq = 2000
	while count < nb_seq:
		Oreochromis_samples = build_synthetic_gmo(plant_genomes=oreochromis_genomes, gmo_name=oreochromis_name, promoter_seqs=promoters, gene_seqs=genes,
			terminator_seqs=terminators, n_samples=nb_seq, mutation_rate=0.0003)
		
		count += len(Oreochromis_samples)
		append_to_csv(Oreochromis_samples, csv_path=PROCESSED_DIR / "splits" / "val.csv")

	print(f"[✓] Generated {count} synthetic GMO sequences")


if __name__ == "__main__":
	MAX_LEN = 2048
	NB_SEQ_TRAIN_TEST = 9976
	dataset_path = PROCESSED_DIR / "dataset.csv"
	split_dir = PROCESSED_DIR / "splits"
	
	gen_files = [
		"Gossypium_hirsutum_v2.1_genomic.fna",
		"Vigna_unguiculata_v2_genomic.fna",
	]

	gmo_files = [
		"bt_cry1Ac_sequences.fasta",
		"bt_cp4epsps_sequences.fasta",
		"cotton_gmo.fna",
	]

	val_files = [
		"Oreochromis_niloticus_genomic.fna"
	]
	
	build_sequences(gen_files, dataset_path, max_len=MAX_LEN)
	build_sequences(gmo_files, dataset_path, max_len=MAX_LEN)
	build_synthetic_sequences(nb_seq=NB_SEQ_TRAIN_TEST)

	shuffle_dataset(dataset_path)
	split_dataset(input_csv=dataset_path, output_dir=split_dir)
	split_dataset_train_test(input_csv=dataset_path, output_dir=split_dir)

	build_dataset(val_files, split_dir / "val.csv", max_len=MAX_LEN, num_seq=2500)
	build_val_set()
