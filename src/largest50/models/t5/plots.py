import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('results/res.csv')
df = df.drop(["Struc-Clus9", "Seq-Clus"], axis=1)
print(df)
df.columns = ["one", "two", "SF", "Number of Sequences in the Training Set", "F1-Score", "t", "Number of SSG5 Classes", "Sequence Clusters to Structural Clusters Ratio"]
df = df.drop(["one", "two", "t", "Sequence Clusters to Structural Clusters Ratio"], axis=1)
print(df)

sf = list(df["SF"])

arch = []

for i in sf:
    sp = i.split('.')
    a = sp[0] + '.' + sp[1]
    arch.append(a)

# df['T-SNE Feature 1'] = X_non_340[:,0]
# df['T-SNE Feature 2'] = X_non_340[:,1] 
df['Label'] = arch

# print(len(list(set(y_arch))))
sns.set(style="whitegrid", font_scale=2.5, font = 'Arial')
colors = ["#1abc9c", "#39ff13", "#2980b9", "#9b59b6", "#34495e", "#f368e0", "#fffa65", "#ffa502", "#ff5252", "#95a5a6", "#ffb8b8", "#7efff5", "#6D214F", "#c39b77"]
#colors = ["#1abc9c", "#39ff13", "#2980b9", "#9b59b6", "#34495e", "#f368e0", "#fffa65", "#ffa502", "#ff5252", "#95a5a6", "#7efff5", "#6D214F", "#c39b77"]
cmap = sns.color_palette() 
sns.set_palette(cmap, n_colors=14)
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="F1-Score", y="Number of Sequences in the Training Set",
    hue="Label",
    palette=sns.color_palette(colors),
    data=df,
    legend="full",
    alpha=1, s=500
)
# sns.regplot(x="F1-Score", y="Number of Sequences in the Training Set",
#     data=df)
# sns.scatterplot(sns.color_palette(n_colors=14))
plt.legend(ncol=1, markerscale=2.5, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()