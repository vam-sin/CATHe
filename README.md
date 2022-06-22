[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6327572.svg)](https://doi.org/10.5281/zenodo.6327572)



# CATHe

CATHe (short for CATH embeddings) is a deep learning tool designed to detect remote homologues (up to 20% sequence similarity) for superfamilies in the CATH database. CATHe consists of an artificial neural network model which was trained on sequence embeddings from the ProtT5 protein Language Model (pLM). It was able to achieve an accuracy of 85.6% +- 0.4%, and outperform the other baseline models derived from both, simple machine learning algorithms such as Logistic Regression, and homology-based inference using BLAST. 

# Requirements

```python3
pip3 install -r requirements.txt
```

# Data

The dataset used for training, optimizing, and testing CATHe was derived from the CATH database. The datasets, along with the weights for the CATHe artificial neural network can be downloaded from Zenodo from this link: [Dataset](https://doi.org/10.5281/zenodo.6327572).

# CATHe Predictions

Folder /src/cathe-predict

Before running the scripts, download the following files from Zenodod Dataset (mentioned above) and place them in the "/src/cathe-predict" folder:

a) CATHe.h5 

This is the CATHe neural network model.

b) Y_Train_SF.csv, Y_Test_SF.csv, Y_Val_SF.csv

These are the superfamily label files from the dataset. 

Additionally, set the location of the protein fasta file in fasta_to_ds.py script. 

Run the following command to make predictions using CATHe. 

```python3
python3 cathe_predictions.py
```

The CATHe predictions would be stored in a file named "results.csv" in the same folder. The results.csv has 4 columns: ['Record', 'Sequence', 'CATHe_Predicted_SFAM', 'CATHe_Prediction_Probability']. The "Record" column is the identifier for the protein sequence. The "Sequence" column stores the primary sequence of the protein. The "CATHe_Predicted_SFAM" is the CATH superfamily prediction made by CATHe, and the probability of this prediction is mentioned in the "CATHe_Prediction_Probability" column. 
