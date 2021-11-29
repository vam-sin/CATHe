import h5py
import numpy as np
import pandas as pd

h5_f = h5py.File("train.h5",'r')
dataset = { seq_id : np.array(embd) for seq_id, embd in h5_f.items() }

y = []

for i in dataset.keys():
	sp = i.split('_')
	y.append(sp[2] + '.' + sp[3] + '.' + sp[4] + '.' + sp[5])

dict = {'SF': y}  
       
df = pd.DataFrame(dict) 
    
# saving the dataframe 
df
df.to_csv('Y_Train_SF.csv') 