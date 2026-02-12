import re
import random



def chunk_sequence(seq, chunk_size=512, stride=256):
    """ Prends
        une séquence d'ADN, une taille de chunk et un stride.
        Découpe la séquence en morceaux de taille chunk_size avec un chevauchement de stride.
         Retourne un des morceaux de sequences.
    """
    seq = seq.upper()
    for i in range(0, len(seq) - chunk_size + 1, stride):
        yield seq[i:i + chunk_size]

def clean_sequence(seq):
    """ Prends
        une séquence d'ADN brute.
        Nettoie la séquence en supprimant les caractères non valides et en la mettant en majuscules.
        Retourne la séquence nettoyée.
    """
    seq = seq.upper()
    seq = re.sub(r"[^ACGTN]", "", seq)
    return seq

def random_subseq(seq, min_len, max_len):
    """ PRends
        une séquence d'ADN, une longueur minimale et une longueur maximale.
        Génère une sous-séquence aléatoire de la séquence donnée avec une longueur comprise entre min_len et max_len.
        Retourne la sous-séquence générée."""
    L = len(seq)
    size = random.randint(min_len, max_len)
    if L <= size:
        return seq
    start = random.randint(0, L - size)
    return seq[start:start+size]


def mutate(seq, rate=0.001):
    """ Prends
        une séquence d'ADN et un taux de mutation.
        Crée une mutation sur la séquence en changeant aléatoirement des bases avec un taux de mutation donné.
        Retourne la séquence mutée."""
    bases = ["A","C","G","T"]
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < rate:
            seq[i] = random.choice(bases)
    return "".join(seq)
