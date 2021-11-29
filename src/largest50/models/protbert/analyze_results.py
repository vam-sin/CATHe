import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import numpy as np

top50 = pd.read_csv('results/top50.tsv', sep='\t')

# top50_size = list(top50["Sequences"])
# top50 = list(top50["SFAM"])

actual_train = pd.read_csv('../../../data/new_pdb_data/final/Y_Train_SF.csv')
actual_train = actual_train["SF"]
size = dict(actual_train.value_counts())
# print(top50)
top50 = list(size.keys())
print(top50)
top50_size = []

for i in top50:
	top50_size.append(size[i])

print(np.sum(top50_size)/2)
print(np.sum(top50_size[:10]))
print(len(top50_size[10:]))
print(len(top50_size[:10]))
le = LabelEncoder()
le.fit(top50)

classes = le.classes_
# print(classes)
res = pd.read_csv('results/CR_ANN_ProtBert.csv')

# print(res)

f1_score = list(res["f1-score"])
f1_score = f1_score[:50]

class_f1_score_map = {}

for i in range(len(classes)):
	class_f1_score_map[classes[i]] = f1_score[i]

# print(class_f1_score_map)

top50_score = []

for i in top50:
	top50_score.append(class_f1_score_map[i])

# print(top50_score)

df = pd.DataFrame(list(zip(top50, top50_size, top50_score)), columns =['SF', 'Size', 'F1-Score'])

print(df)

df.to_csv('results/sizevperf.csv')

# print(np.sum())

print(np.sum(top50_score[:10])/10)
print(np.sum(top50_score[10:])/40)
print(np.sum(top50_score)/50)