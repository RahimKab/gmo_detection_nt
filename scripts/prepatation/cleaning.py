from __future__ import annotations

from pathlib import Path
import argparse
import logging
from typing import List, Dict, Tuple

import pandas as pd

import scripts.utils.common as common
import scripts.utils.utils as utils

LOG = logging.getLogger(__name__)


def build_all_datasets(fasta_info: List[Tuple[str, bool]], output_csv: str) -> None:
    """Construire des fichiers CSV de dataset à partir de plusieurs fichiers FASTA.

    fasta_info : liste de tuples (fasta_path, isGMO)
    """
    for fasta_path, is_gmo in fasta_info:
        LOG.info("Construction du jeu de données depuis %s (isGMO=%s)", fasta_path, is_gmo)
        utils.build_dataset(fasta_path=fasta_path, output_csv=output_csv, isGMO=is_gmo)


def generate_synthetic_samples(genome_fasta: str, gmo_name: str, promoters_fasta: str,
    genes_fasta: str, terminators_fasta: str, n_samples: int = 5000, flank_min: int = 500,
    flank_max: int = 3000, mutation_rate: float = 0.0003, ) -> List[Dict]:
    """Génère des échantillons GMO synthétiques pour un génome de plante donné.
    
    Retourne une liste de dictionnaires d'échantillons compatibles avec `utils.append_to_csv`.
    """
    LOG.info("Chargement du génome : %s", genome_fasta)
    plant_genomes = utils.load_fasta(genome_fasta)
    promoters = utils.load_fasta(promoters_fasta)
    genes = utils.load_fasta(genes_fasta)
    terminators = utils.load_fasta(terminators_fasta)

    LOG.info("Génération de %d échantillons synthétiques pour %s", n_samples, gmo_name)
    samples = utils.build_synthetic_gmo(
        plant_genomes=plant_genomes,
        gmo_name=gmo_name,
        promoter_seqs=promoters,
        gene_seqs=genes,
        terminator_seqs=terminators,
        n_samples=n_samples,
        flank_min=flank_min,
        flank_max=flank_max,
        mutation_rate=mutation_rate,
    )
    return samples


def split_and_save(csv_path: str, output_dir: str, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    LOG.info("Séparation du jeu de données %s -> %s", csv_path, output_dir)
    train_df, test_df, val_df = utils.split_dataset(
        input_csv=csv_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    utils.save_split_datasets(train_df, val_df, test_df, output_dir)
    return train_df, val_df, test_df


def validate_splits(output_dir: str) -> None:
    """Vérifie qu'il n'y a pas de chevauchement d'IDs entre les splits train/val/test."""
    
    train_ids = set(pd.read_csv(Path(output_dir) / "train.csv")["id"])
    val_ids = set(pd.read_csv(Path(output_dir) / "val.csv")["id"])
    test_ids = set(pd.read_csv(Path(output_dir) / "test.csv")["id"])
    assert train_ids.isdisjoint(val_ids), "Chevauchement train/val détecté"
    assert train_ids.isdisjoint(test_ids), "Chevauchement train/test détecté"
    assert val_ids.isdisjoint(test_ids), "Chevauchement val/test détecté"
    LOG.info("Aucun chevauchement d'ID détecté entre les splits")


def main(args: argparse.Namespace) -> None:
    output_csv = args.output_csv

    fasta_info = [
        (args.hirsutum_fasta, False),
        (args.unguiculata_fasta, False),
        (args.cry1ac_fasta, True),
        (args.cp4epsps_fasta, True),
        (args.cotton_gmo_fasta, True),
    ]

    build_all_datasets(fasta_info=fasta_info, output_csv=output_csv)

    samples_all: List[Dict] = []
    samples_all.extend(
        generate_synthetic_samples(
            genome_fasta=args.hirsutum_fasta,
            gmo_name="Gossypium_hirsutum",
            promoters_fasta=args.promoters_fasta,
            genes_fasta=args.genes_fasta,
            terminators_fasta=args.terminators_fasta,
            n_samples=args.n_samples,
        )
    )
    
    samples_all.extend(
        generate_synthetic_samples(
            genome_fasta=args.unguiculata_fasta,
            gmo_name="Vigna_unguiculata",
            promoters_fasta=args.promoters_fasta,
            genes_fasta=args.genes_fasta,
            terminators_fasta=args.terminators_fasta,
            n_samples=args.n_samples,
        )
    )

    utils.append_to_csv(samples_all, csv_path=output_csv)
    LOG.info("Génération et ajout de %d échantillons synthétiques", len(samples_all))

    train_df, val_df, test_df = split_and_save(csv_path=output_csv, output_dir=args.output_splits)

    LOG.info("Séparation du jeu de données terminée : Train=%d Val=%d Test=%d", len(train_df), len(val_df), len(test_df))

    LOG.info("Distribution des labels - Train :\n%s", train_df["label"].value_counts(normalize=True))
    LOG.info("Distribution des labels - Val :\n%s", val_df["label"].value_counts(normalize=True))
    LOG.info("Distribution des labels - Test :\n%s", test_df["label"].value_counts(normalize=True))

    validate_splits(args.output_splits)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit et nettoie le jeu de données à partir de fichiers FASTA")
    parser.add_argument("--hirsutum-fasta", default="data/raw/Gossypium_hirsutum_v2.1_genomic.fna")
    parser.add_argument("--unguiculata-fasta", default="data/raw/Vigna_unguiculata_v2_genomic.fna")
    parser.add_argument("--cry1ac-fasta", default="data/raw/bt_cry1Ac_sequences.fasta")
    parser.add_argument("--cp4epsps-fasta", default="data/raw/bt_cp4epsps_sequences.fasta")
    parser.add_argument("--cotton-gmo-fasta", default="data/raw/cotton_gmo.fna")
    parser.add_argument("--promoters-fasta", default="data/raw/promoters.fasta")
    parser.add_argument("--genes-fasta", default="data/raw/gmo_genes.fasta")
    parser.add_argument("--terminators-fasta", default="data/raw/terminators.fasta")
    parser.add_argument("--output-csv", default="data/processed/datas.csv")
    parser.add_argument("--output-splits", default="data/processed/splits")
    parser.add_argument("--n-samples", type=int, default=5000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    main(args)