# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import biovec
import math
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample 

# dataset import
infile = open('top50.pickle','rb')
top50 = pickle.load(infile)
infile.close()

# train 
ds_train = pd.read_csv('Train.csv')
y_train_full = list(ds_train["SF"])
filename = 'SF_Train_ProtBert.npz'
X_train_full = np.load(filename)['arr_0']

train_index = ds_train.index[ds_train['SF'].isin(top50)].tolist()

X_train = [X_train_full[k] for k in train_index]
y_train = [y_train_full[k] for k in train_index]

# val
ds_val = pd.read_csv('Val.csv')
y_val_full = list(ds_val["SF"])
filename = 'SF_Val_ProtBert.npz'
X_val_full = np.load(filename)['arr_0']

val_index = ds_val.index[ds_val['SF'].isin(top50)].tolist()

X_val = [X_val_full[k] for k in val_index]
y_val = [y_val_full[k] for k in val_index]

# test
ds_test = pd.read_csv('Test.csv')
y_test_full = list(ds_test["SF"])
filename = 'SF_Test_ProtBert.npz'
X_test_full = np.load(filename)['arr_0']

test_index = ds_test.index[ds_test['SF'].isin(top50)].tolist()

X_test = [X_test_full[k] for k in test_index]
y_test = [y_test_full[k] for k in test_index]

# y process
y_tot = []

for i in range(len(y_train)):
    y_tot.append(y_train[i])

for i in range(len(y_val)):
    y_tot.append(y_val[i])

for i in range(len(y_test)):
    y_tot.append(y_test[i])

le = preprocessing.LabelEncoder()
le.fit(y_tot)

y_train = np.asarray(le.transform(y_train))
y_val = np.asarray(le.transform(y_val))
y_test = np.asarray(le.transform(y_test))

num_classes = len(np.unique(y_tot))
print(num_classes)
print("Loaded X and y")

X_train, y_train = shuffle(X_train, y_train, random_state=42)

# logreg model
clf = LogisticRegression(solver='lbfgs', random_state=0, max_iter=5000, verbose=1).fit(X_train, y_train)
# val_score = clf.score(X_val, y_val)
# test_score = clf.score(X_test, y_test)

# print("Val Score: ", val_score)
# print("Test Score: ", test_score)

y_pred_test = clf.predict(X_test)
f1_score_test = f1_score(y_test, y_pred_test, average = 'weighted')
acc_score_test = accuracy_score(y_test, y_pred_test)
mcc_score = matthews_corrcoef(y_test, y_pred_test)
bal_acc = balanced_accuracy_score(y_test, y_pred_test)
print("Test Results")
print("F1 Score: ", f1_score_test)
print("Acc Score: ", acc_score_test)
print("MCC: ", mcc_score)
print("Bal Acc: ", bal_acc)

print("Bootstrapping Results")
num_iter = 1000
f1_arr = []
acc_arr = []
mcc_arr = []
bal_arr = []
for it in range(num_iter):
    print("Iteration: ", it)
    X_test_re, y_test_re = resample(X_test, y_test, n_samples = len(y_test), random_state=it)
    y_pred_test_re = clf.predict(X_test_re)
    # print(y_test_re)
    f1_arr.append(f1_score(y_test_re, y_pred_test_re, average = 'macro'))
    acc_arr.append(accuracy_score(y_test_re, y_pred_test_re))
    mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re))
    bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re))


print("Accuracy: ", np.mean(acc_arr), np.std(acc_arr))
print("F1-Score: ", np.mean(f1_arr), np.std(f1_arr))
print("MCC: ", np.mean(mcc_arr), np.std(mcc_arr))
print("Bal Acc: ", np.mean(bal_arr), np.std(bal_arr))

'''(CS) 
'''


