#$ -l tmem=16G
#$ -l h_rt=12:00:00
#$ -l m_core=1

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N logreg_top50_prott5
#$ -cwd 

source ml-actual/bin/activate
cd SF/dl_models/prott5/
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 logreg_prott5_top50.py 	