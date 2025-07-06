# Master-Thesis

This repository contains all code used in my Master's thesis project:
"Optimization of an Adaptive Closed-Loop Neurorehabilitation System for Gait Recovery in Mice After Spinal Cord Injury"
MSc in Biomedical Engineering, University of Copenhagen and the Technical University of Denmark


## Overview
The project uses a Conditional Variational Autoencoder (CVAE) model to structure high-dimensional kinematic locomoter data from mice with spinal cord injuries (SCI). The learned latent space is used to identify injury severity levels, which are then used to train supervised classifiers.

## Repository Structure

| File | Description |
|------|-------------|
| `File_preparation.py` | Prepares the KineMatrix-extracted data for use in the CVAE model. This includes data formatting, standardization, and preprocessing. |
| `CVAE_with_covariets.py` | The main CVAE model used in the thesis. Includes covariates (e.g., asymmetry) to improve latent space organization by injury severity. |
| `CVAE_without_covariets.py` | A baseline CVAE model without covariates. Used for comparison to assess the added value of conditioning. |
| `classification.py` | Contains data preparation, model training, and evaluation for supervised classification. Uses CVAE-derived cluster labels as targets. Includes Random Forest, SVM, MLP, and Logistic Regression classifiers. |
