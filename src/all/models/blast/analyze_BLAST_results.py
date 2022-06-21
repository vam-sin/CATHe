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
ds.columns = ["q" , "t", "pid", "four","five","six","seven","eight","nine","ten","eval","twelve"]
# ds.columns = ["q" , "t", "pid", "length", "slen", "qlen", "eval", "bitscore"]
print(ds)

q_un = []
y_test = []

# y_pred generation
for record in SeqIO.parse("test_all.fasta", "fasta"):
    q_un.append(record.description)
    spl = record.description.split('_')
    y_test.append(spl[len(spl) - 1])

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
			spl = query.split('_')
			if spl[len(spl) - 1] != y_test[i]:
				sf = spl[len(spl) - 1]
				break
	else:
		# found a good hit
		count += 1
		spl = target[0].split('_')
		sf = spl[len(spl) - 1]
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
Accuracy:  0.31245161760419704 0.005658719323554423
F1-Score:  0.21536128597311935 0.006040697628749656
MCC:  0.310088764795035 0.005670890714957613
Bal Acc:  0.2531639426479432 0.006609206424256284
'''