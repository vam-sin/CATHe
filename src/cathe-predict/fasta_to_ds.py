import pandas as pd 
from Bio import SeqIO

seq = []
desc = []

with open("proteins.fa") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc.append(record.description)

df = pd.DataFrame(list(zip(desc, seq)),
               columns =['Record', 'Sequence'])

print(df)
df.to_csv('Dataset.csv')