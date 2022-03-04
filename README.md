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

Set the following values before running the predictions script: 
  a) Location of the protein fasta file in fasta_to_ds.py
  b) Prediction Probability Threshold in make_predictions.py

```python3
python3 cathe_predictions.py
```

In a file named "results.csv", the results from the batch prediction will be stored. 
