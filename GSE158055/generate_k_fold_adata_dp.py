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


warnings.filterwarnings("ignore")

np.random.seed(41)
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()
RANDOM_SEED = 110011

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

# def apply_DP(adata, sensitivity, epsilon):
#     '''
#     Apply differential privacy to the data
#     X: numpy array
#     sensitivity: float
#     epsilon: float
#     return: numpy array
#     '''
#     print("Start applying DP...")
#     print("Sensitivity: ", sensitivity)
#     print("Epsilon: ", epsilon)
    
#     # Convert the dense matrix to a pandas DataFrame
#     X_df = pd.DataFrame(adata.X.toarray())

#     # Generate Laplace noise
    
#     print("Start adding noise...")
#     f = lambda x: x + np.random.laplace(loc=0, scale=sensitivity/epsilon)
#     # Apply the Laplace noise to the DataFrame
#     X_noisy_df = X_df.applymap(f)
#     del X_df
    
#     # Convert the noisy DataFrame back to a CSR sparse matrix
#     X_noisy_sparse = csr_matrix(X_noisy_df.values)
#     del X_noisy_df
    
#     # Assign the noisy sparse matrix back to adata.X
#     adata.X = X_noisy_sparse
    
#     # for i in tqdm(range(adata.X.shape[0])):  
#     #     laplace_noise = np.random.laplace(scale=scale_cells, size=adata.X.shape[1])
        
#     #     lambda x: x + np.random.laplace(loc=0, scale=1/epsilon)
#     #     # Apply input perturbation
#     #     adata.X[i] = adata.X[i] + laplace_noise
        
#         # Addding LAPLACE noise to the adata.X from 0 to 10
#         #laplace_noise = np.random.laplace(scale=scale_cells, size=adata.X.shape[1])
    
#     print("Finish adding noise...")
    
#     return adata

def apply_DP(adata, sensitivity, epsilon, chunk_size=1000):
    '''
    Apply differential privacy to the data
    adata: AnnData object
    sensitivity: float
    epsilon: float
    chunk_size: int, number of rows to process at a time
    '''
    print("Start applying DP...")
    print("Sensitivity: ", sensitivity)
    print("Epsilon: ", epsilon)
    
    # Get the sparse matrix from adata.X
    X_sparse = adata.X.copy()

    # Calculate the number of chunks to process
    num_chunks = (X_sparse.shape[0] + chunk_size - 1) // chunk_size

    # Generate and add Laplace noise in chunks
    print("Start adding noise...")
    print("Number of chunks: ", num_chunks)
    for i in tqdm(range(num_chunks)):
        # Get the current chunk's start and end row indices
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, X_sparse.shape[0])
        print("Start row: ", start_row)
        print("End row: ", end_row)

        # Get the current chunk as a dense matrix
        chunk_dense = X_sparse[start_row:end_row].toarray()

        # Generate Laplace noise for the current chunk
        laplace_noise = np.random.laplace(loc=0, scale=sensitivity/epsilon, size=chunk_dense.shape)

        # Add Laplace noise to the current chunk
        chunk_noisy = chunk_dense + laplace_noise
        del chunk_dense

        # Convert the noisy chunk back to a sparse matrix
        chunk_noisy_sparse = csr_matrix(chunk_noisy)
        del chunk_noisy

        # Update the corresponding rows in adata.X directly
        adata.X[start_row:end_row] = chunk_noisy_sparse
        del chunk_noisy_sparse


    print("Finish adding noise...")

def n_random_sampling(adata, n=1, max_samples_covid=128, max_samples_control=28, RANDOM_SEED=110011):
    # Set the maximum number of samples for each severity group
    max_samples_covid = max_samples_covid
    max_samples_control = max_samples_control
    
    # Get the samples for each severity group
    severe_critical_samples = adata.obs[adata.obs['CoVID-19 severity'] == 'severe/critical']['sampleID_label'].unique()
    mild_moderate_samples = adata.obs[adata.obs['CoVID-19 severity'] == 'mild/moderate']['sampleID_label'].unique()
    control_samples = adata.obs[adata.obs['CoVID-19 severity'] == 'control']['sampleID_label'].unique()

    # Get lists of all the samples for each fold
    # key: fold number, value: list of samples
    k_sets_samples_severe_critical = dict() 
    k_set_samples_mild_moderate = dict()
    k_sets_samples_control = dict()
    
    for i in range(n):
        rng = default_rng(seed=RANDOM_SEED)
        # Randomly select max_samples samples for each severity group
        clip_samples_severe_critical = rng.choice(a=severe_critical_samples, size=max_samples_covid, replace=False, shuffle=True)
        clip_samples_mild_moderate = rng.choice(a=mild_moderate_samples, size=max_samples_covid, replace=False, shuffle=True)
        clip_samples_control = rng.choice(a=control_samples, size=max_samples_control, replace=False, shuffle=True) # There are only 28 control samples.
        
        k_sets_samples_severe_critical[i] = clip_samples_severe_critical
        k_set_samples_mild_moderate[i] = clip_samples_mild_moderate
        k_sets_samples_control[i] = clip_samples_control
    
    # Make assert if each value of k_sets_samples_severe_critical are the same.
    for i in range(n):
        for j in range(i+1, n):
            if set(k_sets_samples_severe_critical[i]) == set(k_sets_samples_severe_critical[j]):
                raise AssertionError("Not all values in k_sets_samples_severe_critical are the same")
                
    # Make assert if each value of k_set_samples_mild_moderate are the same.
    for i in range(n):
        for j in range(i+1, n):
            if set(k_set_samples_mild_moderate[i]) == set(k_set_samples_mild_moderate[j]):
                raise AssertionError("Not all values in k_set_samples_mild_moderate are the same")

    return k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control

def main(num_kfold=5, save_add_noise_adata=False):
    # Read adata
    print("Reading a h5ad data file...")
    adata = sc.read_h5ad('./data/GSE_158055_COVID19_ALL.h5ad')
    #print("adada shape: ", adata.X.shape)
    #print(adata_all.X)
    
    # Delete objects to prepare for splitting the adata to train and test.
    del adata.uns['neighbors']
    del adata.uns['pca']
    del adata.obsm['X_pca']
    del adata.obsm['X_tsne']
    del adata.obsp

    # Change the type of the sampleID and CoVID-19 severity to string.
    adata.obs['sampleID'] = adata.obs['sampleID'].astype('str')
    adata.obs['CoVID-19 severity'] = adata.obs['CoVID-19 severity'].astype('str')

    # Make a new column that has the combined information of the sampleID and CoVID-19 severity.
    adata.obs['sampleID_label'] = adata.obs['sampleID'] + '_' + adata.obs['CoVID-19 severity']

    # Dataset class distribution: “severe/critical” : 134, “mild/moderate”: 122, and “control”: 28.
    num_dataset_sampling = 1 # The number of datasets to be sampled.
    max_samples_covid = 60
    max_samples_control = 28 # There is only 28 control samples.
    k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control = \
                                                                    n_random_sampling(adata, num_dataset_sampling, max_samples_covid, max_samples_control)
                                                                    
    # print("Sets of samples for severe/critical: ", k_sets_samples_severe_critical)
    # print("Sets of samples for mild/moderate: ", k_set_samples_mild_moderate)
    # print("Sets of samples for control: ", k_sets_samples_control)

    # n-random sampling from the adata. (Default is 1 with the maximum number of samples for covid and control that can be handled by the memory.)
    for i in range(num_dataset_sampling):       
        print(f'-------------{i}st data sampling------------')                                            
        # Filter out adata by the k selected samples' group.
        adata = adata[adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
                                                    | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
                                                    | adata.obs['sampleID_label'].isin(k_sets_samples_control[i])]
        # Run assert test for concatenated adata.
        assert all(adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
                                                    | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
                                                    | adata.obs['sampleID_label'].isin(k_sets_samples_control[i]))
        #assert adata.obs['sampleID_label'].unique().size == (len(k_sets_samples_severe_critical[i])+len(k_set_samples_mild_moderate[i])+len(k_sets_samples_control[i]))
        result_k_dataset_clip = dict()
        
        adata_X_size_gb = get_size_gb(adata.X)
        print(f"Size of adata.X: {adata_X_size_gb} GB")
        print("the size of adata.X: ", adata.X.shape)
        
        # Apply DP
        adata = apply_DP(adata, sensitivity=1, epsilon=1.0, chunk_size=500)
        
        if save_add_noise_adata:
            print("Saving the adata with added noise...")
            adata.write_h5ad('./data/GSE_158055_COVID19_clipped_ALL_DP.h5ad')
            
        # Run experimetns for the representation of UMAP and PCA.
        result_rep = dict()
        for rep in ['X_pca', 'X_umap']:
            print("----------------Using an input representation: ", rep, " ------------------")
            if rep == 'X_umap':
                rep_name = 'UMAP'
            else:
                rep_name = 'PCA'

            # Create adata for train and test.
            for k in tqdm(range(num_kfold)):
                start_time = time.time()
                print("k_fold cross validation sets (split randomly for train and test)", k, " on the ", i, "st clipped dataset","--------")
                list_samples = list(adata.obs['sampleID_label'].unique())
                y_train, y_test = train_test_split(list_samples, test_size=0.20)
                
                # Make adata for train/existing data.
                # Make a column that contains bool values based on whether the sample_name holds one of the samples in the y_train..
                adata.obs['contain_y_train'] = adata.obs['sampleID_label'].isin(y_train)
                adata_train = adata[adata.obs['contain_y_train'] == True,:].copy()
                # Make adata for train/existing data.
                # Make a column that contains bool values based on whether the sample_name holds one of the samples in the y_train..
                adata.obs['contain_y_test'] = adata.obs['sampleID_label'].isin(y_test)
                adata_test = adata[adata.obs['contain_y_test'] == True,:].copy()
                
                # Delete adata to free up memory.
                del adata
                
                # adata_X_size_gb = get_size_gb(adata.X)
                # print(f"Size of adata.X: {adata_X_size_gb} GB")
                print("the size of adata.X: ", adata_train.X.shape)
                
                # # Apply DP
                # print("Apply Dp to the adata_train.")
                # adata_train = apply_DP(adata_train, sensitivity=1, epsilon=1.0, chunk_size=500) 
                
                
                # # if save_add_noise_adata:
                # #     print("Saving the adata with added noise...")
                # #     adata.write_h5ad('./data/GSE_158055_COVID19_clipped_ALL_DP.h5ad')
                
                # print("Apply Dp to the adata_test.")
                # adata_test = apply_DP(adata_test, sensitivity=1, epsilon=1.0, chunk_size=1000) 
                
                print("Run PCA and UMAP on the adata_train.")
                # Run PCA and neighbour graph on the adata_train. 
                sc.pp.neighbors(adata_train, n_pcs = 30, n_neighbors = 20) 
                sc.tl.pca(adata_train)
                if rep == 'X_umap':
                    print("Run sc.tl.ingest on the adata_test by the fit reducer (UMAP).")
                    sc.tl.umap(adata_train)
                    sc.tl.ingest(adata_test, adata_train, embedding_method='umap')
                else:
                    print("Run sc.tl.ingest on the adata_test by the fit reducer (PCA).")
                    sc.tl.ingest(adata_test, adata_train, embedding_method='pca')
                
                # Concatenate adata_train and adata_test into adata
                adata_concat = adata_train.concatenate(adata_test, batch_categories=['ref', 'new'])
                # Change the type of contain_y_train from bool to string.
                adata_concat.obs['contain_y_train'] = adata_concat.obs['contain_y_train'].astype('str')
                adata_concat.obs['contain_y_test'] = adata_concat.obs['contain_y_test'].astype('str')
                
                print(f"Saving {k}th adata in {rep_name} rep to a h5ad file...")
                adata_concat.write(f"./data/k{k}_{rep_name}_adata_GSE_158055_COVID19_DP.h5ad") # type: ignore
        
                
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    #parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset',)
    parser.add_argument('--num_kfold', type=int, help='Number of k fold cross validation', default=5)
    parser.add_argument('--save_add_noise_adata', type=str, help='Save the adata with added noise to a h5ad file', default='False')
    # Parse the arguments
    args = parser.parse_args()

    # Conver the string to bool.
    if args.save_add_noise_adata == 'True':
        save_add_noise_adata = True 
    else:
        save_add_noise_adata = False
        
    main(num_kfold=args.num_kfold, save_add_noise_adata=save_add_noise_adata)
                