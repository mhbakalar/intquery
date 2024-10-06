## Overview
Intquery: Deep learning prediction of LSI off-targets

IntQuery leverages deep learning to predict LSI activity at candidate off-target sites across the genome, offering an **in silico** method for LSI off-target prediction.

# Abstract
Large Serine Integrases (LSIs) hold significant therapeutic promise due to their ability to efficiently incorporate gene-sized DNA into the human genome, offering a method to integrate healthy genes in patients with monogenic disorders or to insert gene circuits for the development of advanced cell therapies. To advance the application of LSIs for human therapeutic applications, new technologies and analytical methods for predicting and characterizing off-target recombination by LSIs are required. It is not experimentally tractable to validate off-target editing at all potential off-target sites in therapeutically relevant cell types because of sample limitations and genetic variation in the human population. To address this gap, we constructed a deep learning model named IntQuery that can predict LSI activity at candidate off-target sites genome-wide. For Bxb1 integrase, IntQuery was trained on quantitative off-target data from 410,776 sequences discovered by Cryptic-Seq, an unbiased in vitro discovery technology for LSI off-target recombination. We show that IntQuery can accurately predict in vitro LSI activity, providing a tool for in silico off-target prediction of large serine integrases to advance therapeutic applications.

## Scripts and Notebooks

### `train.py`
A script for training a multi-layer perceptron (MLP) using the full Cryptic-Seq dataset for Bxb1 integrase. The training data is provided in `data/train.csv`. This script handles loading the dataset, training the model, and saving the trained model's weights.

### `predict.ipynb`
A Jupyter notebook for predicting cryptic site activity from input sequences. Sequences are loaded from a CSV file, and predictions are made using a pre-trained MLP model. The notebook uses weights stored in `weights/200_200_mlp.pt`, which were obtained by training a model with hidden dimensions `[200, 200]` on the full training dataset (`train.csv`).

### `train_fold.ipynb`
This notebook performs **5-fold cross-validation** on the training set to evaluate model performance. It includes code for splitting the dataset into folds, training on each fold, and generating relevant plots (e.g., performance metrics, validation losses).

## Data
The training data used in this repository comes from **Cryptic-Seq**, an in vitro discovery technology for unbiased detection of LSI off-target recombination. The data includes 410,776 sequences tested for off-target activity of the Bxb1 integrase.

## Usage
1. **Training the Model**: 
    To train the model from scratch, run:
    ```bash
    python train.py
    ```
    This script will train the MLP model using the training data in `data/train.csv` and save the model weights to `weights/200_200_mlp.pt`.

2. **Predicting Off-Target Activity**:
    To predict activity on new cryptic sites, use the `predict.ipynb` notebook. Simply load the sequences from a CSV file and apply the pre-trained model to generate predictions.

3. **Cross-Validation**:
    For cross-validation and performance evaluation, open and run the `train_fold.ipynb` notebook. This will perform 5-fold cross-validation on the training data and generate plots for analysis.

## BioRxiv Preprint
For a detailed explanation of the methodology and results, please refer to our [BioRxiv preprint](#).

---


