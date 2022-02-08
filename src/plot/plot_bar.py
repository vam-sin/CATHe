import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style 

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot') 

ds = pd.read_csv('results/L50.csv')

model = list(ds["Model"])

X = ['Accuracy', 'F1-Score', 'MCC', 'Bal Acc']

models =  ['ANN + ProtBert', 'LR + ProtBert', 'CATHe (ANN + ProtT5)', 'LR + ProtT5', 'ANN + Length', 'BLAST', 'Random']

ann_pb_full = list(ds.loc[0])[3:7]
ann_pb_val = [float(x.split('+-')[0]) for x in ann_pb_full]
ann_pb_err = [float(x.split('+-')[1])*1.96 for x in ann_pb_full]

lr_pb_full = list(ds.loc[1])[3:7]
lr_pb_val = [float(x.split('+-')[0]) for x in lr_pb_full]
lr_pb_err = [float(x.split('+-')[1])*1.96 for x in lr_pb_full]

cathe_full = list(ds.loc[2])[3:7]
cathe_val = [float(x.split('+-')[0]) for x in cathe_full]
cathe_err = [float(x.split('+-')[1])*1.96 for x in cathe_full]

lr_t5_full = list(ds.loc[3])[3:7]
lr_t5_val = [float(x.split('+-')[0]) for x in lr_t5_full]
lr_t5_err = [float(x.split('+-')[1])*1.96 for x in lr_t5_full]

ann_l_full = list(ds.loc[4])[3:7]
ann_l_val = [float(x.split('+-')[0]) for x in ann_l_full]
ann_l_err = [float(x.split('+-')[1])*1.96 for x in ann_l_full]

blast_full = list(ds.loc[5])[3:7]
blast_val = [float(x.split('+-')[0]) for x in blast_full]
blast_err = [float(x.split('+-')[1])*1.96 for x in blast_full]

random_full = list(ds.loc[6])[3:7]
random_val = [float(x.split('+-')[0]) for x in random_full]
random_err = [float(x.split('+-')[1])*1.96 for x in random_full]

df_val = pd.DataFrame(list(zip(ann_pb_val, lr_pb_val, cathe_val, lr_t5_val, ann_l_val, blast_val, random_val)), columns =models)
df_err = pd.DataFrame(list(zip(ann_pb_err, lr_pb_err, cathe_err, lr_t5_err, ann_l_err, blast_err, random_err)), columns =models)

df_val["Metrics"] = X
df_val.index = X

df_val.plot(y=models, yerr= df_err[models].values.T, kind="bar",figsize=(8,8), fontsize=16, color=['#f1c40f', '#e67e22', '#27ae60', '#2c3e50', '#e74c3c', '#3498db', '#e056fd'])
plt.xticks(rotation=360, ha='center')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Top 50 Superfamilies.png', bbox_inches='tight')
plt.show()

  