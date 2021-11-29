import pandas as pd 
import random
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score

test = pd.read_csv('../../data/final/Test.csv')
y_test = list(test["SF"])
train = pd.read_csv('../../data/final/Train.csv')
train_sf = list(train["SF"])

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
Accuracy: 0.0045119189511323 0.0008166512714633589
F1-Score: 0.0004619216238013885 0.00020521450597855932
MCC: -1.216621567399447e-05 0.0008190699742340104
Balanced Accuracy: 0.0007005831158197104 0.00031416093242522174
'''
