'''
Reads the all-vs-all.tsv file from the BLAST scan output to 
find the best hits for all the sequences in the query set.

Output: Performance of the BLAST model. (Bootstrapped) 
'''

# libraries
import pandas as pd 
import numpy as np
import pickle
from Bio import SeqIO
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
from sklearn.utils import resample 

# load data and check
ds = pd.read_csv('all-vs-all.tsv', sep='\t', header=None)
ds.columns = ["q" , "t", "pid", "length", "slen", "qlen", "eval"]
print(ds)

q_un = []

# y_pred generation
for record in SeqIO.parse("test.fasta", "fasta"):
    q_un.append(record.description)

test_df = pd.read_csv('Test.csv')

infile = open('top50.pickle','rb')
top50 = pickle.load(infile)
infile.close()

test_index = test_df.index[test_df['SF'].isin(top50)].tolist()

y_test_full = list(test_df["SF"])

y_test = [y_test_full[k] for k in test_index]

y_pred = []
count = 0
for i in range(len(q_un)):
	print(i, q_un[i])
	try:
		# hits for query sequence
		small_ds = ds[ds["q"] == q_un[i]]
		# print(small_ds)

		# hit with least evalue
		min_eval = np.min(np.asarray(small_ds["eval"]))
		small_ds = small_ds[small_ds["eval"] == min_eval]
		
		# hit with max pid
		max_val = np.max(np.asarray(small_ds["pid"]))
		small_ds = small_ds[small_ds["pid"] == max_val]
		
		# print best hit
		print(small_ds)

		target = list(small_ds["t"])
	except:
		target = []

	if len(target) == 0:
		# if no hit
		for query in q_un:
			if query.split('_')[3] != y_test[i]:
				sf = query.split('_')[3]
				break
	else:
		# found a good hit
		count += 1
		sf = target[0].split('_')[3]
	print(sf, y_test[i])
	y_pred.append(sf)

print("Accuracy: ", accuracy_score(y_test, y_pred))

tp = 0 
fp = 0 
for i in range(len(y_test)):
	if y_test[i] == y_pred[i]:
		tp += 1
	else:
		fp += 1

print(tp, fp)
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("MCC Score: ", matthews_corrcoef(y_test, y_pred))
print("Bal Acc: ", balanced_accuracy_score(y_test, y_pred))

print("Number of Hits: ", count)

print("Bootstrapping Results")
num_iter = 1000
f1_arr = []
acc_arr = []
mcc_arr = []
bal_arr = []
for it in range(num_iter):
    print("Iteration: ", it)
    y_pred_test_re, y_test_re = resample(y_pred, y_test, n_samples = len(y_test), random_state=it)
    # y_pred_test_re = clf.predict(X_test_re)
    # print(y_test_re)
    f1_arr.append(f1_score(y_test_re, y_pred_test_re, average = 'macro'))
    acc_arr.append(accuracy_score(y_test_re, y_pred_test_re))
    mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re))
    bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re))


print("Accuracy: ", np.mean(acc_arr), np.std(acc_arr))
print("F1-Score: ", np.mean(f1_arr), np.std(f1_arr))
print("MCC: ", np.mean(mcc_arr), np.std(mcc_arr))
print("Bal Acc: ", np.mean(bal_arr), np.std(bal_arr))

'''
Accuracy:  0.5390662898252826 0.01124982295333954
F1-Score:  0.4977628122858198 0.014228630994095149
MCC:  0.5266313402076651 0.011455595731603405
Bal Acc:  0.578437145222221 0.015448843258183197
'''