# Détection des OGM basée sur l'IA 

Ce projet propose une solution pour la détection de séquences OGM dans l'ADN 
en utilisant des modèles de langage génomiques. Dans notre cas, Nucleotide Transformers.

cette approche réside dans sa capacité à être entraînée sur un PC aux caractéristiques modestes 
grâce aux techniques de Parameter-Efficient Fine-Tuning (PEFT) comme LoRA et la quantification 4-bit.

###### Points Forts

✓  Modèle Génomique : Utilisation de Nucleotide-Transformer-v2 (InstaDeep).
✓ Optimisation Low-Resource : Entraînement possible sur GPU grand public .
✓ Génération In-Silico : Pipeline d'assemblage de séquences synthétiques (Promoteurs, Gènes BT, Terminateurs).
✓ Métriques de Précision : Focus sur le F1-Score et le Rappel pour minimiser les faux négatifs.


###### Utilisation de snakemake
Créer un environement different et installer snakemake
Il y'a des conflits de versions entre les bibliotheque de snakemake et ceux de transformers.

• python3 -m venv .snakemake_env