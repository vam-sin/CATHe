import pandas as pd 
import random
import pickle
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score

# dataset import
infile = open('../../data/final/top50.pickle','rb')
top50 = pickle.load(infile)
infile.close()

# train 
ds_train = pd.read_csv('../../../all/data/final/Train.csv')
y_train_full = list(ds_train["SF"])
print(len(y_train_full))

train_index = ds_train.index[ds_train['SF'].isin(top50)].tolist()

train_sf = [y_train_full[k] for k in train_index]

# test
ds_test = pd.read_csv('../../../all/data/final/Test.csv')
y_test_full = list(ds_test["SF"])

test_index = ds_test.index[ds_test['SF'].isin(top50)].tolist()
y_test = [y_test_full[k] for k in test_index]

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
0.02684994861253854 0.0036400137026782057
0.01607407095107804 0.002817064088301045
-9.889507646715503e-05 0.0037323095376247998
0.020449992270371805 0.005722759969054513
'''
