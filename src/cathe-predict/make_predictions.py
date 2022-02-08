# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, LeakyReLU, Add
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
from tensorflow.keras import backend as K
from tensorflow import keras

# load data
filename = 'Pfam_Human_CRH_ProtT5.npz'
embeds = np.load(filename)['arr_0']

# # annotations
# ds_anno = pd.read_csv('Annotations.csv')
# y_test = list(ds_anno["SF"])

# test
ds_test = pd.read_csv('Y_Test_SF.csv')

y_test = list(ds_test["SF"])

le = preprocessing.LabelEncoder()
le.fit(y_test)

classes = le.classes_

# sensitivity metric
def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	return true_positives / (possible_positives + K.epsilon())

model = load_model('cath-t5.h5', custom_objects={'sensitivity':sensitivity})

y_pred = model.predict(embeds)

print(y_pred.shape)

count = 0
sfam_thresh = []
sequence_thresh = []
record_thresh = []

ds = pd.read_csv('pfam-human-crh.csv')

sequences = list(ds["Sequence"])
record = list(ds["Record"])

for i in range(len(y_pred)):
	if max(y_pred[i]) >= 0.95:
		count += 1
		sfam_thresh.append(classes[np.argmax(y_pred[i])])
		sequence_thresh.append(sequences[i])
		record_thresh.append(record[i])
		print(classes[np.argmax(y_pred[i])])

print(count, y_pred.shape)

df = pd.DataFrame(list(zip(record_thresh, sequence_thresh, sfam_thresh)), columns =['Record', 'Sequence', 'CATHe_Predicted_SFAM'])
print(df)
df.to_csv('pfam-human-crh-1percent-cathe.csv')

print(len(list(set(sfam_thresh))))

# f1_score_test = f1_score(y_test, y_pred.argmax(axis=1), average = 'weighted')
# acc_score_test = accuracy_score(y_test, y_pred.argmax(axis=1))
# mcc_score = matthews_corrcoef(y_test, y_pred.argmax(axis=1))
# bal_acc = balanced_accuracy_score(y_test, y_pred.argmax(axis=1))

# print("F1 Score: ", f1_score_test)
# print("Acc Score: ", acc_score_test)
# print("MCC: ", mcc_score)
# print("Bal Acc: ", bal_acc)

''' 19368 sequences in total
0.99: 11269 (0.5% error rate)
0.95: 11939 (1% error rate, 588 unique superfamilies)
0.9: 12399
0.8: 12983
0.7: 13565
0.6: 14134
0.5: 14754
0.4: 15476
0.3: 16187
0.2: 17099 (5% error rate, 816 unique superfamilies)
0.0: 19368 (900 unique superfamilies)
'''

'''
check how many superfamilies are hit.
'''