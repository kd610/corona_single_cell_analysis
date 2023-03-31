"""
Due to the memory limitation, we may not want to use this script to generate the adata for k-fold cross validation.
Recommended to use the script: generate_k_fold_adata.ipynb to generate the adata for k-fold cross validation.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import warnings
from tqdm import tqdm
import time
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
import sys
from scipy.stats import laplace
from scipy.sparse import csr_matrix, vstack
import psutil
import gc
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

np.random.seed(41)
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()
RANDOM_SEED = 110011
seed_list = [110011, 1234, 1235, 2222, 123445]

def get_size_gb(matrix):
    if isinstance(matrix, np.ndarray):
        # If it's a NumPy array (dense matrix)
        size_bytes = matrix.nbytes
    elif isinstance(matrix, csr_matrix):
        # If it's a CSR sparse matrix
        size_bytes = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    else:
        raise ValueError("Unsupported matrix type.")

    size_gb = size_bytes / (1024 * 1024 * 1024)
    return size_gb

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")


### Main function
def main(k=0):
    # Read adata
    print("Reading a noised h5ad data file...")
    # This adata is already clipped with covid_saples: 120 and control samples: 28 and added noise.
    
            
    # Run experimetns for the representation of UMAP and PCA.
    for rep in ['X_pca', 'X_umap']:
        print("----------------Using an input representation: ", rep, " ------------------")
        if rep == 'X_umap':
            rep_name = 'UMAP'
        else:
            rep_name = 'PCA'
        
        adata = sc.read_h5ad('./data/GSE_158055_COVID19_clipped_add_noise.h5ad')
        
         # Create adata for train and test.
        print(f"k_fold cross validation sets (split randomly for train and test) with k={k}")
        list_samples = list(adata.obs['sampleID_label'].unique())
        _, y_test = train_test_split(list_samples, test_size=0.20, random_state=seed_list[k])
        
        # Make adata for test data.
        # Make a column that contains bool values based on whether the sample_name holds one of the samples in the y_train..
        adata.obs['contain_y_test'] = adata.obs['sampleID_label'].isin(y_test)
        adata_test = adata[adata.obs['contain_y_test'] == True,:].copy()
        del adata
        gc.collect()
        
        adata_train = sc.read_h5ad(f"./data/k{k}_{rep_name}_adata_train_GSE_158055_COVID19_DP.h5ad")
        print_memory_usage()
        
        print("Run PCA on the adata_train.")

        if rep == 'X_umap':
            print("Run sc.tl.ingest on the adata_test by the fit reducer (UMAP).")
            sc.tl.ingest(adata_test, adata_train, embedding_method='umap')
        else:
            print("Run sc.tl.ingest on the adata_test by the fit reducer (PCA).")
            sc.tl.ingest(adata_test, adata_train, embedding_method='pca')
       
        gc.collect()
        # Concatenate adata_train and adata_test into adata
        adata_concat = adata_train.concatenate(adata_test, batch_categories=['ref', 'new'])
        del adata_train
        del adata_test
        del adata_concat.X
        print_memory_usage()
        gc.collect()
        
        # Change the type of contain_y_train from bool to string.
        adata_concat.obs['contain_y_train'] = adata_concat.obs['contain_y_train'].astype('str')
        adata_concat.obs['contain_y_test'] = adata_concat.obs['contain_y_test'].astype('str')
        
        print(f"Saving {k}th adata_concat in {rep_name} rep to a noised h5ad file...")
        adata_concat.write(f"./data/k{k}_{rep_name}_adata_GSE_158055_COVID19_DP.h5ad") # type: ignore
        
        del adata_concat
        gc.collect()
                
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    #parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset',)
    parser.add_argument('--k', type=int, help='Number of k fold cross validation', default=5)
    # Parse the arguments
    args = parser.parse_args()
    
    main(k=args.k)
                