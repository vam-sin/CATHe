import pandas as pd

# make fasta
df = pd.read_csv('Train_Overlap.csv')

seq1 = list(df["Sequence"])
dom1 = list(df["Description"])
print(len(seq1))


# # make fasta
# df3 = pd.read_csv('X_Test_SF.csv')
# df4 = pd.read_csv('Y_Test_SF.csv')

# seq2 = list(df3["Sequence"])
# dom2 = list(df3["Domain"])
# print(len(seq2))
# sf2 = list(df4["SF"])
# print(len(sf2))

f = open("train.fasta", "w")

for i in range(len(seq1)):
	print(i, len(seq1))
	f.write(">" + str(dom1[i]) + "\n")
	f.write(seq1[i])
	f.write("\n")

# for i in range(len(seq2)):
# 	print(i, len(seq2))
# 	f.write(">test_valsequence_" + str(i) + '_' + str(dom2[i]) + '_' + str(sf2[i]) + "\n")
# 	f.write(seq2[i])
# 	f.write("\n")

f.close()