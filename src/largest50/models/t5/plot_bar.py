import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style 

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot') 

ds = pd.read_csv('new_results/ann_breakdown_all.csv')

model = list(ds["Model Serial Number"])

X = ['F1-Score']

models =  ['1', '2', '3', '4', '5 (CATHe)', '6', '7', '8', '9', '10', '11']

m1_full = list(ds.loc[0])[2:3]
m1_val = [float(x.split('+-')[0]) for x in m1_full]
m1_err = [float(x.split('+-')[1]) for x in m1_full]

m2_full = list(ds.loc[1])[2:3]
m2_val = [float(x.split('+-')[0]) for x in m2_full]
m2_err = [float(x.split('+-')[1]) for x in m2_full]

m3_full = list(ds.loc[2])[2:3]
m3_val = [float(x.split('+-')[0]) for x in m3_full]
m3_err = [float(x.split('+-')[1]) for x in m3_full]

m4_full = list(ds.loc[3])[2:3]
m4_val = [float(x.split('+-')[0]) for x in m4_full]
m4_err = [float(x.split('+-')[1]) for x in m4_full]

m5_full = list(ds.loc[4])[2:3]
m5_val = [float(x.split('+-')[0]) for x in m5_full]
m5_err = [float(x.split('+-')[1]) for x in m5_full]

m6_full = list(ds.loc[5])[2:3]
m6_val = [float(x.split('+-')[0]) for x in m6_full]
m6_err = [float(x.split('+-')[1]) for x in m6_full]

m7_full = list(ds.loc[6])[2:3]
m7_val = [float(x.split('+-')[0]) for x in m7_full]
m7_err = [float(x.split('+-')[1]) for x in m7_full]

m8_full = list(ds.loc[7])[2:3]
m8_val = [float(x.split('+-')[0]) for x in m8_full]
m8_err = [float(x.split('+-')[1]) for x in m8_full]

m9_full = list(ds.loc[8])[2:3]
m9_val = [float(x.split('+-')[0]) for x in m9_full]
m9_err = [float(x.split('+-')[1]) for x in m9_full]

m10_full = list(ds.loc[9])[2:3]
m10_val = [float(x.split('+-')[0]) for x in m10_full]
m10_err = [float(x.split('+-')[1]) for x in m10_full]

m11_full = list(ds.loc[10])[2:3]
m11_val = [float(x.split('+-')[0]) for x in m11_full]
m11_err = [float(x.split('+-')[1]) for x in m11_full]


df_val = pd.DataFrame(list(zip(m1_val, m2_val, m3_val, m4_val, m5_val, m6_val, m7_val, m8_val, m9_val, m10_val, m11_val)), columns =models)
df_err = pd.DataFrame(list(zip(m1_err, m2_err, m3_err, m4_err, m5_err, m6_err, m7_err, m8_err, m9_err, m10_err, m11_err)), columns =models)

df_val["Metrics"] = X
df_val.index = X

df_val.plot(x = "Metrics", y=models, yerr= df_err[models].values.T, kind="bar",figsize=(9,8), color=['#f1c40f', '#e67e22', '#27ae60', '#2c3e50', '#e74c3c', '#3498db', '#e056fd'])
plt.xticks(rotation=360, ha='right')
plt.show()
  