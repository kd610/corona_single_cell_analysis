
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import warnings
from tqdm import tqdm
import time
from numpy.random import default_rng
import warnings
from sklearn.model_selection import train_test_split

#np.random.seed(41)
sc.settings.verbosity = 0
RANDOM_SEED = 110011
warnings.filterwarnings("ignore")


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

def sampling_adata(adata):
    # Read adata
    num_dataset_sampling=1
    #num_kfold=1

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
    print("Clipping the adata due to the memory limitation...")
    max_samples_covid = 60
    max_samples_control = 28 # There is only 28 control samples.
    k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control = \
                                                                    n_random_sampling(adata, num_dataset_sampling, max_samples_covid, max_samples_control, RANDOM_SEED)
                                                                    
    # print("Sets of samples for severe/critical: ", k_sets_samples_severe_critical)
    # print("Sets of samples for mild/moderate: ", k_set_samples_mild_moderate)
    # print("Sets of samples for control: ", k_sets_samples_control)

    i=0
    #print(f'-------------{i}st data sampling------------')                                            
    # Filter out adata by the k selected samples' group.
    adata = adata[adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
                                                | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
                                                | adata.obs['sampleID_label'].isin(k_sets_samples_control[i])]
    # Run assert test for concatenated adata.
    assert all(adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
                                                | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
                                                | adata.obs['sampleID_label'].isin(k_sets_samples_control[i]))
    
    return adata

# Function to create a neighboring dataset
def create_neighboring_dataset(adata):
    adata_neighbor = adata.copy()
    # Change one row values of the adata_neighbor to random values.
    adata_neighbor.X[np.random.randint(adata.X.shape[0])] = np.random.rand(adata.X.shape[1])
    return adata_neighbor

# Function to calculate the difference between two PCA or UMAP results
def rep_difference(comp1, comp2):
    return np.max(np.abs(comp1 - comp2)) #L1



def main(num_neighbors=100):
    # Number of neighboring datasets to generate
    
    print("Reading a h5ad data file...")
    #adata = sc.read_h5ad('./data/GSE_158055_COVID19_ALL.h5ad')
    num_kfold = 5
    # Perform PCA/UMAP on neighboring datasets and calculate the differences
    #for rep in ['X_pca', 'X_umap']:
    for rep in ['X_umap']:
        print("----------------Using an input representation: ", rep, " ------------------")
        if rep == 'X_umap':
            rep_name = 'UMAP'
        else:
            rep_name = 'PCA'
            
        for k in tqdm(range(num_kfold)):
            start_time = time.time()
            print('Started at: ', start_time)
            # adata = sampling_adata(adata)

            # # Split the samples into train and test. Train is only related to estimate sensitivity.
            # list_samples = list(adata.obs['sampleID_label'].unique())
            # y_train, _ = train_test_split(list_samples, test_size=0.20)
            # # Make adata for train/existing data.
            # # Make a column that contains bool values based on whether the sample_name holds one of the samples in the y_train..
            # adata.obs['contain_y_train'] = adata.obs['sampleID_label'].isin(y_train)
            # adata_train = adata[adata.obs['contain_y_train'] == True,:].copy()
            
            # # Perform PCA or UMAP on the original adata_train.
            # print(f"Run {rep_name} on the adata_train.")
            # sc.pp.neighbors(adata_train, n_pcs = 30, n_neighbors = 20) 
            # sc.tl.pca(adata_train)
            # if rep == 'X_umap':
            #     print("Running UMAP on the adata_train.")
            #     sc.tl.umap(adata_train)
            
            # Read pre-generated adata file for the k-th fold. 
            k_file = f'./data/k{k}_{rep_name}_adata_GSE_158055_COVID19.h5ad'
            print("Reading a h5ad data file:", k_file)
            adata = sc.read_h5ad(k_file)
            print("Finished reading the h5ad data file...")
            # Get adata with adata.obs['batch'] = 'ref'.
            adata_train = adata[adata.obs['batch'] == 'ref'].copy()
            del adata
                
            # Run estimation
            max_diff = 0
            for i in tqdm(range(num_neighbors)):
                adata_train_neighbor = create_neighboring_dataset(adata_train)
                # Run PCA and neighbour graph on the adata_train. 
                print("Running PCA and neighbour graph on the adata_train_neighbor.")
                sc.pp.neighbors(adata_train_neighbor, n_pcs = 30, n_neighbors = 20) 
                sc.tl.pca(adata_train_neighbor)
                if rep == 'X_umap':
                    print("Running UMAP on the adata_train_neighbor.")
                    sc.tl.umap(adata_train_neighbor)
                # Calculate the difference between the original adata_train and the neighboring adata_train.
                diff = rep_difference(adata_train.obsm[rep], adata_train_neighbor.obsm[rep])
                print(f"Diff in iteration {i}:", diff)
                max_diff = max(max_diff, diff)
                
                del adata_train_neighbor
                
            # Estimated sensitivity
            sensitivity_pca = max_diff
            print(f"Estimated sensitivity for {rep_name} output:", sensitivity_pca)
    
        

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    parser.add_argument('--num_neighbors', type=int, help='Number of neighboring datasets to generate (iteration of the estimation process)', default=100)
    # Parse the arguments
    args = parser.parse_args()

    main(num_neighbors=args.num_neighbors)