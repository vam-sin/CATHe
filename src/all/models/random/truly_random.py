import pandas as pd 
import random
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score

test = pd.read_csv('../../data/final/CSV/Test.csv')
y_test = list(test["SF"])
for i in range(150):
	y_test.append('other')

train = pd.read_csv('../../data/final/CSV/Train.csv')
train_sf = list(train["SF"])
for i in range(3456):
	train_sf.append('other')

acc = []
f1 = []
mcc = []
ba = []

num_iter = 1000

for j in range(num_iter):
	print(j)
	y_pred = []
	y_test_resample = resample(y_test, n_samples = len(y_test), random_state = 42)

	for i in range(len(y_test_resample)):
		y_pred.append(random.choice(train_sf))

	acc.append(accuracy_score(y_test_resample, y_pred))
	f1.append(f1_score(y_test_resample, y_pred, average='macro'))
	mcc.append(matthews_corrcoef(y_test_resample, y_pred))
	ba.append(balanced_accuracy_score(y_test_resample, y_pred))

print(np.mean(acc), np.std(acc))
print(np.mean(f1), np.std(f1))
print(np.mean(mcc), np.std(mcc))
print(np.mean(ba), np.std(ba))

# cr = classification_report(y_test, y_pred, digits=4)
# print(cr)

'''
0.004453803555814631 0.0008246335652253825
0.00046690909257299316 0.00020300077857971955
-3.398730206220937e-05 0.0008245573165193321
0.0007073635938447099 0.00030933828068284745
'''
