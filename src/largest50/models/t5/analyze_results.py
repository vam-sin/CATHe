import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import numpy as np
import pickle

# dataset import
infile = open('tsne/top50.pickle','rb')
top50 = pickle.load(infile)
infile.close()

# train 
ds_train = pd.read_csv('/home/vamsi/UCL/projects/CATHe/src/all/data/final/CSV/Train.csv')
y_train_full = list(ds_train["SF"])

train_index = ds_train.index[ds_train['SF'].isin(top50)].tolist()

y_train = [y_train_full[k] for k in train_index]
y_train = pd.DataFrame(y_train)
y_train.columns = ["SF"]
print(y_train)
size = dict(y_train.value_counts())
print(size)
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
res = pd.read_csv('new_results/CATHe_L50.csv')

# print(res)

f1_score = list(res["f1-score"])
f1_score = f1_score[:50]

class_f1_score_map = {}

for i in range(len(classes)):
	class_f1_score_map[classes[i]] = f1_score[i]

# print(class_f1_score_map)

top50_score = []
print(class_f1_score_map)
for i in top50:
	k = str(str(i).replace('(', '').replace(')', '').replace(',', '')).replace("'",'')
	top50_score.append(class_f1_score_map[k])

# print(top50_score)
top50_p = []
for i in top50:
	top50_p.append(str(str(i).replace('(', '').replace(')', '').replace(',', '')).replace("'",''))
df = pd.DataFrame(list(zip(top50_p, top50_size, top50_score)), columns =['SF', 'Size', 'F1-Score'])

print(df)

df.to_csv('new_results/sizevperf.csv')

# print(np.sum())
print(np.sum(top50_size))
print(np.sum(top50_score[:10])/10)
print(np.sum(top50_score[10:])/40)
print(np.sum(top50_score)/50)

print(np.sum(top50_score[:1])/1)
print(np.sum(top50_score[33:])/17)

'''
Top 50%: 0.9944572650745659
Bottom 50%: 0.9916546312493963


Top 10%: 0.98989898989899
Bottom 10%: 0.9943768919353824
'''