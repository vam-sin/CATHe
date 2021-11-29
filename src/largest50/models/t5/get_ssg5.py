import pandas as pd 

ds = pd.read_csv('new_results/sizevperf.csv')
sfam = list(ds["SF"])
ssg5 = []

ssg_csv = pd.read_csv('cath.ssg5_clusters.v4_3_0.csv', sep='\t')
vc = dict(ssg_csv["SUPERFAMILY_ID"].value_counts())

print(vc)

# print(col1, col2)
for i in range(len(sfam)):
	print(sfam[i], i)
	try:
		ssg5.append(vc[sfam[i]])
	except:
		ssg5.append(1)

ds["SSG5"] = ssg5

print(ds)

ds.to_csv('new_results/sizevperf_ssg.csv')
	