#$ -l tmem=20G
#$ -l m_core=1
#$ -l h_rt=12:00:00

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N cathe
#$ -cwd 

source ml-actual/bin/activate
cd /SAN/cath/cath_v4_0/vnallapareddy/CATHe/src/cathe-predict
mkdir Embeddings
source /share/apps/source_files/cuda/cuda-10.1.source
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 cathe_predictions.py
