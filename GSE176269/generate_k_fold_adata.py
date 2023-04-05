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
import gc

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

np.random.seed(41)
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()
RANDOM_SEED = 110011

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

def main(num_dataset_sampling=1, num_kfold=5):
    # Read adata
    print("Reading a h5ad data file...")
    adata = sc.read_h5ad('./data/gse176269_raw_covid_normalized_preprocessed.h5ad')

    # Delete objects to prepare for splitting the adata to train and test.
    del adata.uns['neighbors']
    del adata.uns['pca']
    del adata.obsm['X_pca']
    del adata.obsm['X_umap']
    del adata.obsp

    # Change the type of the sampleID and CoVID-19 severity to string.
    adata.obs['sampleID'] = adata.obs['sampleID'].astype('str')
    #adata.obs['CoVID-19 severity'] = adata.obs['CoVID-19 severity'].astype('str')

    # Make a new column that has the combined information of the sampleID and CoVID-19 severity.
    #adata.obs['sampleID_label'] = adata.obs['sampleID'] + '_' + adata.obs['CoVID-19 severity']

    # Dataset class distribution: {'covid': 32, 'flu': 56, 'control': 28}
    # {'covid': 32, 'non-covid': 84}
    max_samples_covid = 32
    max_samples_control = 84 
    # k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control = \
    #                                                                 n_random_sampling(adata, num_dataset_sampling, max_samples_covid, max_samples_control)
                                                                    
    # print("Sets of samples for severe/critical: ", k_sets_samples_severe_critical)
    # print("Sets of samples for mild/moderate: ", k_set_samples_mild_moderate)
    # print("Sets of samples for control: ", k_sets_samples_control)

    seed_list = [1777, 111000, 12245, 2000, 77777]
    # n-random sampling from the adata. (Default is 1 with the maximum number of samples for covid and control that can be handled by the memory.)
    for i in range(num_dataset_sampling):       
        print(f'-------------{i}st data sampling------------')                                            
        # Filter out adata by the k selected samples' group.
        # adata = adata[adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
        #                                             | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
        #                                             | adata.obs['sampleID_label'].isin(k_sets_samples_control[i])]
        # # Run assert test for concatenated adata.
        # assert all(adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
        #                                             | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
        #                                             | adata.obs['sampleID_label'].isin(k_sets_samples_control[i]))
        #assert adata.obs['sampleID_label'].unique().size == (len(k_sets_samples_severe_critical[i])+len(k_set_samples_mild_moderate[i])+len(k_sets_samples_control[i]))
        result_k_dataset_clip = dict()
        
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
                list_samples = list(adata.obs['sampleID'].unique())
                y_train, y_test = train_test_split(list_samples, test_size=0.20, random_state=seed_list[k])
                
                print("The number of samples in the train set: ", len(y_train))
                print("The number of samples in the test set: ", len(y_test))
                
                # Make adata for train/existing data.
                # Make a column that contains bool values based on whether the sample_name holds one of the samples in the y_train..
                adata.obs['contain_y_train'] = adata.obs['sampleID'].isin(y_train)
                adata_train = adata[adata.obs['contain_y_train'] == True,:].copy()
                # Make adata for train/existing data.
                # Make a column that contains bool values based on whether the sample_name holds one of the samples in the y_train..
                adata.obs['contain_y_test'] = adata.obs['sampleID'].isin(y_test)
                adata_test = adata[adata.obs['contain_y_test'] == True,:].copy()
                
                # Delete adata to free up memory.
                #del adata
                
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
                
                del adata_train
                del adata_test
                gc.collect()
                
                print(f"Saving {k}th adata in {rep_name} rep to a h5ad file...")
                adata_concat.write(f"./data/k{k}_{rep_name}_adata_GSE_176269_COVID19.h5ad") # type: ignore

                del adata_concat
                gc.collect()
                
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset', default=1)
    parser.add_argument('--num_kfold', type=int, help='Number of k fold cross validation', default=5)
    # Parse the arguments
    args = parser.parse_args()

    main(num_dataset_sampling=args.num_dataset_sampling, num_kfold=args.num_kfold)
                