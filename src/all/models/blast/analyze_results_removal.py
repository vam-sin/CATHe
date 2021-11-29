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

# make overlap feature
# overlap = []
# aln_length = list(ds["length"])
# qlen = list(ds["qlen"])
# slen = list(ds["slen"])

# for i in range(len(aln_length)):
# 	overlap.append(aln_length[i] / max(qlen[i], slen[i]))

# ds["overlap"] = overlap

# remove certain train sequences
# infile = open('remove_seq_mmseqs_condn_10.pickle','rb')
# remove_seq = pickle.load(infile)
# infile.close()
# remove_seq = set(remove_seq)

# ds = ds[~ds["t"].isin(remove_seq)]
# print(ds)

# evalue cutoff
# ds = ds[ds["eval"] <= 10]

# overlap cutoff 
# ds = ds[ds["overlap"] >= 0.3]

q_un = []

# y_pred generation
for record in SeqIO.parse("test.fasta", "fasta"):
    q_un.append(record.description)

y_test = pd.read_csv('Test.csv')
y_test = list(y_test["SF"])

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
Accuracy:  0.3195542312276519 0.005711106662232276
F1-Score:  0.2167746979988557 0.006063206363987007
MCC:  0.31721905583723653 0.005722178290308073
Bal Acc:  0.2529937488587626 0.006604891604146327
'''