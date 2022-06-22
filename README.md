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

# Pre-Print

If you found this work useful, please consider citing the following article:

@article {CATHe2022,
	author = {Nallapareddy, Vamsi and Bordin, Nicola and Sillitoe, Ian and Heinzinger, Michael and Littmann, Maria and Waman, Vaishali and Sen, Neeladri and Rost, Burkhard and Orengo, Christine},
	title = {CATHe: Detection of remote homologues for CATH superfamilies using embeddings from protein language models},
	elocation-id = {2022.03.10.483805},
	year = {2022},
	doi = {10.1101/2022.03.10.483805},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {CATH is a protein domain classification resource that combines an automated workflow of structure and sequence comparison alongside expert manual curation to construct a hierarchical classification of evolutionary and structural relationships. The aim of this study was to develop algorithms for detecting remote homologues that might be missed by state-of-the-art HMM-based approaches. The proposed algorithm for this task (CATHe) combines a neural network with sequence representations obtained from protein language models. The employed dataset consisted of remote homologues that had less than 20\% sequence identity. The CATHe models trained on 1773 largest, and 50 largest CATH superfamilies had an accuracy of 85.6+-0.4, and 98.15+-0.30 respectively. To examine whether CATHe was able to detect more remote homologues than HMM-based approaches, we employed a dataset consisting of protein regions that had annotations in Pfam, but not in CATH. For this experiment, we used highly reliable CATHe predictions (expected error rate \&lt;0.5\%), which provided CATH annotations for 4.62 million Pfam domains. For a subset of these domains from homo sapiens, we structurally validated 90.86\% of the predictions by comparing their corresponding AlphaFold structures with experimental structures from the CATHe predicted superfamilies.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/03/13/2022.03.10.483805},
	eprint = {https://www.biorxiv.org/content/early/2022/03/13/2022.03.10.483805.full.pdf},
	journal = {bioRxiv}
}
