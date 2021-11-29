import numpy as np 
import pandas as pd 
import h5py 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

f = h5py.File('results/test.h5', 'r')
keys = list(f.keys())
print(keys)
ds = pd.read_csv('results/Y_Test_SF.csv')
print(ds["SF"].value_counts())
y = list(ds["SF"])

# print(y)

filename = 'results/SF_Test_ProtT5.npz'
X_test = np.load(filename)['arr_0']
# X_test = np.expand_dims(X_test, axis = 1)

print(X_test.shape)

# X_embedded = TSNE(n_components=2).fit_transform(X_test)

# print(X_embedded.shape)

filename = 'results/y_pred.pickle'
infile = open(filename,'rb')
y_pred = pickle.load(infile)
infile.close()

filename = 'X_embedded.pickle'
# outfile = open(filename,'wb')
# pickle.dump(X_embedded,outfile)
# outfile.close()

infile = open(filename,'rb')
X_embedded = pickle.load(infile)
infile.close()

le = LabelEncoder()
le.fit(y)
y_pred_prob = y_pred
y_pred = y_pred.argmax(axis=1)
y_pred_inv = le.inverse_transform(y_pred)
# print(y_pred_prob[116])
# print(y_pred_inv, y)
count = 0
for i in range(len(y_pred_inv)):
	if y_pred_inv[i] != y[i]:
		# print(i)
		print(count, keys[count], y[i], y_pred_inv[i], max(y_pred_prob[i]), np.argmax(y_pred_prob[i]))
	count += 1 

print(count)

# # y = np.asarray(le.transform(y))

# y_arch = []

# for i in y_pred_inv:
# 	sp = i.split('.')
# 	a = sp[0] + '.' + sp[1]
# 	y_arch.append(a)

# print(y)

# X_non_340 = []
# y_non_340 = []

# for i in range(len(y_arch)):
# 	# if (y_arch[i] in ['3.40']):
# 	y_non_340.append(y_arch[i])
# 	X_non_340.append(X_embedded[i])

# X_non_340 = np.asarray(X_non_340)
# print(X_non_340.shape)
# df = pd.DataFrame()
# df['T-SNE Feature 1'] = X_non_340[:,0]
# df['T-SNE Feature 2'] = X_non_340[:,1] 
# df['Label'] = y_non_340

# print(len(list(set(y_arch))))
# colors = ["#1abc9c", "#39ff13", "#2980b9", "#9b59b6", "#34495e", "#f368e0", "#fffa65", "#ffa502", "#ff5252", "#95a5a6", "#ffb8b8", "#7efff5", "#6D214F", "#c39b77"]
# #colors = ["#1abc9c", "#39ff13", "#2980b9", "#9b59b6", "#34495e", "#f368e0", "#fffa65", "#ffa502", "#ff5252", "#7efff5", "#6D214F", "#c39b77"]

# sns.set(style="whitegrid", font_scale=2, font = 'Arial')
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="T-SNE Feature 1", y="T-SNE Feature 2",
#     hue="Label",
#     palette=sns.color_palette(colors, n_colors = len(list(set(y_non_340)))),
#     data=df,
#     legend="full",
#     alpha=1.0, s=50
# )
# plt.legend(ncol=1, markerscale=2.5, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.show()

# '''
# 3.40: 13 superfamilies
# 3.30: 9 superfamilies
# '''

'''
116 1xiwB00 testsequence_1104_2_60_40_10 2.60.40.10 2.40.50.140 0.18229958 17
195 1ev7B02 testsequence_1176_1_10_10_10 1.10.10.10 2.40.50.140 0.9929582 17
241 2hxiA01 testsequence_1217_1_10_10_60 1.10.10.60 1.10.357.10 0.6021161 4
281 2nn6G01 testsequence_1253_2_40_50_100 2.40.50.100 2.40.50.140 0.23045656 17
726 2ea9A01 testsequence_1654_3_30_450_20 3.30.450.20 3.40.50.300 0.08661755 39
844 1cidA02 testsequence_1760_2_60_40_10 2.60.40.10 2.40.50.100 0.9931625 16
1020 1t0fA02 testsequence_191_1_10_10_10 1.10.10.10 3.40.190.10 0.10683096 32
1026 4pz8A01 testsequence_1925_2_40_50_140 2.40.50.140 3.40.50.300 0.14096336 39
1196 1c1kA01 testsequence_282_1_10_8_60 1.10.8.60 3.40.50.300 0.10751628 39
1384 4ccdA02 testsequence_451_3_20_20_70 3.20.20.70 3.20.20.80 0.99983096 22
1459 1hkgA02 testsequence_519_3_30_420_40 3.30.420.40 3.40.190.10 0.28244227 32
1525 1mu2B04 testsequence_579_3_30_70_270 3.30.70.270 2.40.50.140 0.8653514 17
1572 3d31A02 testsequence_620_2_40_50_140 2.40.50.140 2.40.50.100 0.9995307 16
1686 2lvsA01 testsequence_725_1_10_10_60 1.10.10.60 1.10.10.10 0.999956 0
1724 1pryA01 testsequence_75_3_30_200_20 3.30.200.20 2.40.50.140 0.8664682 17
1796 3u3wA01 testsequence_825_1_10_260_40 1.10.260.40 1.25.40.10 0.9419004 10
1963 2q7nA03 testsequence_977_2_60_40_10 2.60.40.10 3.40.50.300 0.93441784 39
'''