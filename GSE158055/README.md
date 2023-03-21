# COVID-19 Classification Using the GSE158055 Dataset

This repository contains scripts and resources to analyze the GSE158055 dataset and develop classification models for all proposed scenario.

## Repository Schema

Below is an outline of the repository structure:

```bash
GSE158055/
│
├── data/ # Save all .h5ad data.
│   └── GSE_158055_COVID19_ALL.h5ad # This dataset can be downloaded from here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158055
│
├── experiments/
│   └── scenario_1/ # Distribution similarity-based classification
│       ├── PCA_KL-D_sample_classification_GSE158055.ipynb # Notebook in PCA rep as an input
│       ├── UMAP_KL-D_sample_classification_GSE158055.ipynb # Notebook in UMAP rep as an input
│       ├── helper_func.py # Helper function for the below k-fold cross validation
│       └── KL-D_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 1
│   └── scenario_2/ # Centroid-based classification
│       ├── PCA_KL-PCA_centroid_sample_classification_GSE158055.ipynb # Notebook in PCA rep as an input
│       ├── UMAP_KL-2_1_centroid_sample_classification_GSE158055.ipynb # Notebook in UMAP rep as an input
│       ├── helper_func.py # Helper function for the below k-fold cross validation
│       ├── 2_1_centroid_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 2.1
│       ├── 2_2_centroid_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 2.2
│       ├── 2_3_centroid_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 2.3
│       └── 2_4_centroid_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 2.4
│   └── scenario_3/ # Centroid-based classification with using SVM
│       ├── PCA_KL-PCA_centroid_sample_classification_2_GSE159812.ipynb # Notebook in PCA rep as an input
│       ├── UMAP_KL-UMAP_centroid_sample_classification_2_GSE158055.ipynb.ipynb # Notebook in UMAP rep as an input
│       ├── helper_func.py # Helper function for the below k-fold cross validation
│       ├── 3_1_centroid_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 3.1
│       └── 3_2_centroid_sample_classification_GSE158055.py # Scripts to run k-fold cross validation in scenario 3.2
│
├── calculate_k_fold_performance_.ipynb # Stores results from all scenarios and subsequently derives the final outputs for all model performances.
│
├── generate_figures.ipynb.ipynb
│
├── generate_k_fold_adata.ipynb.ipynb # Split the dataset into training and test datasets for k-fold cross validation.
│
├── ingest_GSE_158055.ipynb
│
├── generate_k_fold_adata.py # Split the dataset into training and test datasets for k-fold cross validation.
│
├── figures/                  # Generated figures for the report
│
└── README.md                 # This README file
```

## Dataset

GSE158055 is used for this experiment. We can download it from [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158055). Once you have done download it, place it under `./data` folder. 