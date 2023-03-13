import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import warnings
from tqdm import tqdm
import time
from helper_func import n_random_sampling
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

np.random.seed(41)
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()
RANDOM_SEED = 110011

def main(num_dataset_sampling=5, num_kfold=5):
    # Read adata
    print("Reading a h5ad data file...")
    adata = sc.read_h5ad('../../data/GSE_158055_COVID19_ALL.h5ad')

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
    max_samples_covid = 60
    max_samples_control = 28 # There is only 28 control samples.
    k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control = \
                                                                    n_random_sampling(adata, num_dataset_sampling, max_samples_covid, max_samples_control)
                                                                    
    print("Sets of samples for severe/critical: ", k_sets_samples_severe_critical)
    print("Sets of samples for mild/moderate: ", k_set_samples_mild_moderate)
    print("Sets of samples for control: ", k_sets_samples_control)

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
                
                print(f"Saving {k}th adata in {rep_name} rep to a h5ad file...")
                adata_concat.write(f'../../data/k{k}_{rep_name}_adata_GSE_158055_COVID19.h5ad')
        
                
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset')
    parser.add_argument('--num_kfold', type=int, help='Number of k fold cross validation')
    # Parse the arguments
    args = parser.parse_args()

    main(num_dataset_sampling=args.num_dataset_sampling, num_kfold=args.num_kfold)
                