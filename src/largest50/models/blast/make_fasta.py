import pandas as pd
import pickle

# sf import
infile = open('top50.pickle','rb')
top50 = pickle.load(infile)
infile.close()

# make fasta
df = pd.read_csv('Test.csv')

seq_full = list(df["Sequence"])
desc_full = list(df["Description"])

train_index = df.index[df['SF'].isin(top50)].tolist()

seq = [seq_full[k] for k in train_index]
desc = [desc_full[k] for k in train_index]

f = open("test.fasta", "w")

for i in range(len(seq)):
	print(i, len(seq))
	f.write(">" + str(desc[i]) + "\n")
	f.write(seq[i])
	f.write("\n")

f.close()