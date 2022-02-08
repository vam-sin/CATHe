#$ -l tmem=20G
#$ -l gpu=true
#$ -l h_rt=24:00:00

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N cath-t5-predict_sfam
#$ -cwd 

source ml-actual/bin/activate
cd cath-t5-predict/
mkdir Embeddings
source /share/apps/source_files/cuda/cuda-10.1.source
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 process_fasta.py
python3 predict_embed.py
python3 append_embed.py
python3 make_predictions.py 	