import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

ds = pd.read_csv('results/Y_Test_SF.csv')

y = list(ds["SF"])

print(y)

filename = 'results/SF_Test_ProtT5.npz'
X_test = np.load(filename)['arr_0']
# X_test = np.expand_dims(X_test, axis = 1)

print(X_test.shape)

# X_embedded = TSNE(n_components=2).fit_transform(X_test)

# print(X_embedded.shape)

filename = 'X_embedded.pickle'
# outfile = open(filename,'wb')
# pickle.dump(X_embedded,outfile)
# outfile.close()

infile = open(filename,'rb')
X_embedded = pickle.load(infile)
infile.close()

# le = LabelEncoder()
# le.fit(y)

# y = np.asarray(le.transform(y))

y_arch = []

for i in y:
	sp = i.split('.')
	a = sp[0] + '.' + sp[1]
	y_arch.append(a)

print(y)

X_non_340 = []
y_non_340 = []

for i in range(len(y_arch)):
	# if (y_arch[i] in ['3.40']):
	y_non_340.append(y_arch[i])
	X_non_340.append(X_embedded[i])

X_non_340 = np.asarray(X_non_340)
print(X_non_340.shape)
df = pd.DataFrame()
df['X'] = X_non_340[:,0]
df['Y'] = X_non_340[:,1] 
df['Label'] = y_non_340

print(len(list(set(y_arch))))
colors = ["#1abc9c", "#39ff13", "#2980b9", "#9b59b6", "#34495e", "#f368e0", "#fffa65", "#ffa502", "#ff5252", "#95a5a6", "#ffb8b8", "#7efff5", "#6D214F", "#c39b77"]
#colors = ["#1abc9c", "#39ff13", "#2980b9", "#9b59b6", "#34495e", "#f368e0", "#fffa65", "#ffa502", "#ff5252", "#7efff5", "#6D214F", "#c39b77"]

sns.set(style="whitegrid", font_scale=2, font = 'Arial')
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="X", y="Y",
    hue="Label",
    palette=sns.color_palette(colors, n_colors = len(list(set(y_non_340)))),
    data=df,
    legend="full",
    alpha=1.0, s=50
)
plt.legend(ncol=1, markerscale=2.5, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.xticks(color='w')
plt.yticks(color='w')
plt.xlabel('X axis', fontsize=0)
plt.ylabel('Y axis', fontsize=0)
plt.show()

'''
3.40: 13 superfamilies
3.30: 9 superfamilies
'''