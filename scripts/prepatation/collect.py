from Bio import Entrez, SeqIO
Entrez.email = "your_email@example.com"

from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    file_Path = f"{PROJECT_PATH}/data/raw"

    # Liste des nom des sequences à récupérer depuis NCBI
    accession_names = {
        "hirsutum":"Gossypium hirsutum[Organism]",
        "cry1Ac":"cry1Ac",
        "cry2Ab":"cry2Ab",
        "cp4epsps":"cp4epsps",
        "nptII":"nptII",
        "promoter_camv_35S":"CaMV 35S promoter [All Fields]",
        "promoter_FMV" : "FMV promoter [All Fields]",
        "promoter_zmubi" : "ZmUbi1 promoter [All Fields]",
        "promoter_nos" : "NOS promoter [All Fields]",
        "terminator_nos" : "NOS terminator [All Fields]",
        "terminator_ocs" : "OCS terminator [All Fields]",
        "terminator_35s" : "35S terminator [All Fields]"

    }

    for acc_name, acc_value in accession_names.items():
        print(f"Récupération de la séquence pour {acc_name}...")
        handle = Entrez.esearch(db="nucleotide", term=acc_value)
        record = Entrez.read(handle)
        ids = record["IdList"]
        
        sequences = []
        for seq_id in ids:
            handle = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            sequences.append(record)
        filename = f"{acc_name}.fasta"
        SeqIO.write(record, f"{file_Path}/{filename}", "fasta")
        print(f"Séquence sauvegardée dans {filename}")