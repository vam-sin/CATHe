import os 

cmd = 'mkdir Embeddings'
os.system(cmd)

cmd = 'python3 fasta_to_ds.py'
os.system(cmd)

cmd = 'python3 predict_embed.py'
os.system(cmd)

cmd = 'python3 append_embed.py'
os.system(cmd)

cmd = 'python3 make_predictions.py'
os.system(cmd)

