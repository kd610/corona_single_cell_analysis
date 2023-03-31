
import numpy as np
import scanpy as sc
import argparse
from tqdm import tqdm
import gc
import psutil
from numpy.random import default_rng
from scipy import sparse

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
            
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")


### Main function
def main(num_split=10, num_samples_covid=60, num_samples_control=28):
    print("Reading adata...")
    adata = sc.read_h5ad("./data/GSE_158055_COVID19_ALL.h5ad")
    print("Data type of adata.X:", adata.X.dtype)
    print("Type of adata.X:", type(adata.X))
    
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
    max_samples_covid = num_samples_covid # 30 was the highest number.
    max_samples_control = num_samples_control # There is only 28 control samples.
    k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control = \
                                                                    n_random_sampling(adata, num_dataset_sampling, max_samples_covid, max_samples_control)
    
    i=0
    # Filter out adata by the k selected samples' group.
    adata = adata[adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
                                                | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
                                                | adata.obs['sampleID_label'].isin(k_sets_samples_control[i])]
    # Run assert test for concatenated adata.
    assert all(adata.obs['sampleID_label'].isin(k_sets_samples_severe_critical[i]) \
                                                | adata.obs['sampleID_label'].isin(k_set_samples_mild_moderate[i]) \
                                                | adata.obs['sampleID_label'].isin(k_sets_samples_control[i]))
    

    num_splits = num_split
    split_size = int(np.ceil(adata.shape[0] / num_splits))
    
    print(f"Start splitting the adata into {num_split} smaller AnnData objects...")
    for i in tqdm(range(num_splits)):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, adata.shape[0])
        small_adata = adata[start_idx:end_idx, :].copy()
        gc.collect()
        print("Copied adata to small_adata...")
        print_memory_usage()
        
        # Convert the count matrix from a sparse matrix to a dense matrix.
        print("Converting the count matrix from float32 to float16...")
        small_adata.X = small_adata.X.toarray()
        
        print("Done converting the count matrix from float32 to float16...")
        print_memory_usage()
        print("Type of adata.X:", type(small_adata.X))
        print("Data type of small_adata.X:", small_adata.X.dtype)
        # Get the size of small_adata.X in GB
        print("Size of small_adata.X:", small_adata.X.nbytes / (1024 ** 3), "GB")
        
        # Convert the count matrix from float32 to float16.
        small_adata.X = small_adata.X.astype("float16")
        
        print("Type of adata.X:", type(small_adata.X))
        print("Data type of small_adata.X (after conversion):", small_adata.X.dtype)
        print("Size of small_adata.X (after conversion):", small_adata.X.nbytes / (1024 ** 3), "GB")

        # Save the small_adata to a h5ad file.
        small_adata.write_h5ad(f"./data/GSE_158055_COVID19_part{i}.h5ad")
        gc.collect()
        

if  __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    #parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset',)
    parser.add_argument('--num_split', type=int, help='To split the adata into k smaller AnnData objects', default=10)
    parser.add_argument('--num_samples_covid', type=int, default=50)
    parser.add_argument('--num_samples_control', type=int, default=28)
    # Parse the arguments
    args = parser.parse_args()
    
    main(args.num_split, args.num_samples_covid, args.num_samples_control)
