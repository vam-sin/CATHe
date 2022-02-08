# libraries
import numpy as np
from bio_embeddings.embed import ProtTransT5BFDEmbedder
import pandas as pd 

embedder = ProtTransT5BFDEmbedder()

ds = pd.read_csv('Sequences_Predict.csv')

sequences_Example = list(ds["Sequence"])
num_seq = len(sequences_Example)

i = 0
length = 1000
while i < num_seq:
	print("Doing", i, num_seq)
	start = i 
	end = i + length

	sequences = sequences_Example[start:end]

	embeddings = []
	for seq in sequences:
		embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

	s_no = start / length
	filename = 'Embeddings/' + 'T5_' + str(s_no) + '.npz'
	embeddings = np.asarray(embeddings)
	# print(embeddings.shape)
	np.savez_compressed(filename, embeddings)
	i += length