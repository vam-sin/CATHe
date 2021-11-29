#$ -l tmem=24G
#$ -l h_rt=24:00:00
#$ -l m_core=1

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N logreg_all_prott5
#$ -cwd 

source ml-actual/bin/activate
cd SF-all/models/prott5/
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 logreg_prott5_all.py 	