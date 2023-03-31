
import numpy as np
import scanpy as sc

from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy import sparse
import gc
import psutil
import argparse

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")

def main(num_split=20):
    
    final_adata = None
    
    for i in tqdm(range(num_split)):
        small_adata = sc.read_h5ad(f"./data/GSE_158055_COVID19_part{i}.h5ad")
        print("Size of small_adata.X: ", small_adata.X.shape)
        # small_adata.X = small_adata.X.toarray()
        # small_adata.X = small_adata.X.astype("float16")
        
        print("Adding noise to small_adata.X...")
        epsilon = 1
        # Create noise matrix intertively as it is always float64 initially which wont fit in memory
        noise = np.random.laplace(
            loc=0, scale=1 / epsilon, size=(small_adata.X.shape[0], small_adata.X.shape[1])
        ).astype("float16")
    
        # Add noise to data
        small_adata.X = small_adata.X + noise
        
        # Convert the count matrix from a dense matrix to a sparse matrix.
        print("Converting the count matrix from float16 to float32...")
        #small_adata.X = sparse.csr_matrix(small_adata.X)
        
        # Save adata
        save_file_dp = f"./data/GSE_158055_COVID19_part{i}_add_noise.h5ad"
        small_adata.write_h5ad(save_file_dp)
        
        # Concatenate small_adata to final_adata
        if final_adata is None:
            final_adata = small_adata
        else:
            final_adata = final_adata.concatenate(small_adata)
        
        # Print memory usage
        print_memory_usage()
            
        # Garbage collection
        del small_adata
        del noise
        gc.collect()
    
    # Save adata
    save_file_dp_final = f"./data/GSE_158055_COVID19_clipped_add_noise.h5ad"
    final_adata.write_h5ad(save_file_dp_final)
    

if  __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    #parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset',)
    parser.add_argument('--num_split', type=int, help='To split the adata into k smaller AnnData objects', default=10)
    # Parse the arguments
    args = parser.parse_args()
    
    main(args.num_split)
