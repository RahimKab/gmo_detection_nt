from Bio import Entrez, SeqIO
import urllib.request
from pathlib import Path

Entrez.email = "your_email@example.com"
PROJECT_PATH = Path(__file__).resolve().parents[2]

def download_reference_genome(organism_name, out_dir, assembly_name=None):
    """
    Télécharge le génome de référence pour un organisme donné depuis NCBI Assembly.
    Si assembly_name est fourni, il sera utilisé pour une recherche plus précise.
    """
    if assembly_name:
        search_term = f"{organism_name}[Organism] AND {assembly_name}[Assembly Name]"
    else:
        search_term = f"{organism_name}[Organism] AND reference genome[All Fields]"

    print(f"Recherche de : {search_term}")
    handle = Entrez.esearch(db="assembly", term=search_term)
    record = Entrez.read(handle)
    if not record["IdList"]:
        print(f"Aucun assemblage trouvé pour {organism_name}.")
        return
    
    uid = record["IdList"][0]
    summary_handle = Entrez.esummary(db="assembly", id=uid, report="full")
    summary = Entrez.read(summary_handle, validate=False)
    ftp_path = summary['DocumentSummarySet']['DocumentSummary'][0]['FtpPath_GenBank']

    if not ftp_path:
        print(f"Aucun chemin FTP trouvé pour {organism_name}.")
        return
    fasta_url = f"{ftp_path}/{ftp_path.split('/')[-1]}_genomic.fna.gz"
    print(f"Téléchargement : {fasta_url}")

    out_path = Path(out_dir) / f"{organism_name.replace(' ', '_')}_genomic.fna.gz"
    try:
        urllib.request.urlretrieve(fasta_url, out_path)
        print(f"Téléchargement terminé : {out_path}")
    except Exception as e:
        print(f"Échec du téléchargement de {fasta_url} : {e}")

def download_nucleotide_seq(accession_names, out_dir):
    """
    Télécharge les séquences nucléotidiques pour chaque nom d'accession fourni.
    Les séquences sont sauvegardées dans le dossier de sortie.
    """
    for acc_name, acc_value in accession_names.items():
        print(f"Récupération de la séquence pour {acc_name}...")
        handle = Entrez.esearch(db="nucleotide", term=acc_value)
        record = Entrez.read(handle)
        ids = record["IdList"]

        sequences = []
        for seq_id in ids:
            handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            sequences.append(record)
        filename = f"{acc_name}.fasta"
        SeqIO.write(record, f"{out_dir}/{filename}", "fasta")
        print(f"Séquence sauvegardée dans {filename}")


if __name__ == "__main__":
    file_Path = f"{PROJECT_PATH}/data/raw"

    # Référence pour Gossypium hirsutum v2.1 (Cotton)
    download_reference_genome(
        organism_name="Gossypium hirsutum",
        out_dir=file_Path,
        assembly_name="Gossypium_hirsutum_v2.1"
    )

    # Référence pour Vigna unguiculata (Niébé)
    download_reference_genome(
        organism_name="Vigna unguiculata",
        out_dir=file_Path,
        assembly_name="ASM411807v2"
    )
    # Référence pour Oreochromis niloticus (Tilapia du Nile)
    download_reference_genome(
        organism_name="Oreochromis niloticus",
        out_dir=file_Path,
        assembly_name="O_niloticus_UMD_NMBU"
    )
    
    accession_names = {
        "cry1Ac":"cry1Ac", # protéine de résistance aux insectes
        "cry2Ab":"cry2Ab", # protéine de résistance aux insectes
        "cp4epsps":"cp4epsps", # protéine de résistance au glyphosate (herbicide)
        "nptII":"nptII",
        "promoter_camv_35S":"CaMV 35S promoter [All Fields]",
        "promoter_FMV" : "FMV promoter [All Fields]",
        "promoter_zmubi" : "ZmUbi1 promoter [All Fields]",
        "promoter_nos" : "NOS promoter [All Fields]",
        "terminator_nos" : "NOS terminator [All Fields]", # Le gène terminator enlève le pouvoir de reproduction de la semence
        "terminator_ocs" : "OCS terminator [All Fields]",
        "terminator_35s" : "35S terminator [All Fields]"
    }
    download_nucleotide_seq(accession_names, file_Path)