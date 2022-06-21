import numpy as np

filename = 'Embeddings/' + 'T5_' + '0.0' + '.npz'
pb_arr = np.load(filename)['arr_0']

for i in range(1, 1000000):
	print(i, pb_arr.shape)
	try:
		filename = 'Embeddings/' + 'T5_' + str(i) + '.0' + '.npz'
		arr = np.load(filename)['arr_0']
		pb_arr = np.append(pb_arr, arr, axis = 0)
	except:
		pass

np.savez_compressed('Embeddings_T5.npz', pb_arr)