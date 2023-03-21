"""
Sample Classification: Scenario 2.3

Method
Get centroids of each sample's cluster (covid and non-covid samples). 
When we get a new sample, we compute a centroid of each new sample. 
And compare them with those existing samples' centroids, get the nearest centroids, 
and determine whether the new sample is covid or non_covid based on 
the nearest sample's centroid.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import scipy.stats as st
from scipy import stats
from sklearn.model_selection import train_test_split
from helper_func import n_random_sampling, count_nearest_samples, performance_eval, calc_dist_centroids, calc_nearest_centroids
from scipy.stats import norm  
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
import warnings
from sklearn.neighbors import NearestCentroid
from collections import Counter
import time
warnings.filterwarnings("ignore")


#np.random.seed(41)
sc.settings.verbosity = 1 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()
RANDOM_SEED = 110011

def main(num_kfold=5):
    # Read adata
    result_rep = dict()
    for rep in ['X_umap']:
        print("----------------Using an input representation: ", rep, " ------------------")
        if rep == 'X_umap':
            rep_name = 'UMAP'
        else:
            rep_name = 'PCA'
        
        # K-fold cross validation for train and test.
        result_k_fold = dict()
        # Those are for the result of the k-fold cross validation in avg.
        acc_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        for k in range(3, num_kfold):
            start_time = time.time()
            print(f"k_fold cross validation (split randomly for train and test): k={k}")
            print("Started at: ", start_time)

            # Read pre-generated adata file for the k-th fold.
            k_file = f"../../data/k{k}_{rep_name}_adata_GSE_158055_COVID19.h5ad"
            print("Reading a h5ad data file:", k_file)
            adata = sc.read_h5ad(k_file)
            print("Finished reading the h5ad data file...")

            # Output the number of unique samples from the dataframe of adata.obs.
            print("Number of unique samples: ", len(adata.obs["sampleID_label"].unique()))
            new_value = None

            # Add a new columns: covid_non_covid to adata.obs
            for index, row in adata.obs.iterrows():
                if 'severe/critical' in row['CoVID-19 severity'] or 'mild/moderate' in row['CoVID-19 severity']:
                    new_value = 'covid'
                elif 'control' in row['CoVID-19 severity']:
                    new_value = 'non_covid'

                adata.obs.at[index, 'covid_non_covid'] = new_value
                
            # Create a dataframe that contains the UMAP values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.
            basis_values = adata.obsm[rep] # X_umap or X_pca
            sample_vector = adata.obs['sampleID_label'].values
            #covid_non_covid_vector = adata.obs['covid_non_covid'].values
            batch_vector = adata.obs['batch'].values # Define whether the cell is from the train or test set.
            x_basis_value = []
            y_basis_value = []

            ### I'll use the slice later...
            for b_v in basis_values:
                x_basis_value.append(b_v[0])
                y_basis_value.append(b_v[1])
            ###
            df = pd.DataFrame(list(zip(basis_values, sample_vector, batch_vector, x_basis_value, y_basis_value)),
                        columns =['basis_value', 'sample', 'batch', 'x_basis_value', 'y_basis_value'])
            print(f"Created a dataframe that contains the {rep_name} values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.")
            
            # Delete the below variables to free up memory.
            del basis_values
            del sample_vector
            del batch_vector
            del x_basis_value
            del y_basis_value
            
            # Get train and test samples
            y_train = np.unique(df.query("batch == 'ref'")['sample'].values)
            y_test = np.unique(df.query("batch == 'new'")['sample'].values)
            print("Num of train samples:", len(y_train))
            print("Num of test samples:", len(y_test))
            
            # Create dataframe for the train samples.
            y_train = list(y_train)
            df_exist = df.query("sample == @y_train")
            
            # Get centroids of each existing samples.
            X = np.stack(df_exist.basis_value.values.tolist()[:])
            y = df_exist['sample'].tolist()
            clf = NearestCentroid()
            clf.fit(X, y)
            
            # Create dataframe for the test samples.
            y_test = list(y_test)
            df_new = df.query("sample == @y_test")
            # Create input features and labels for the test samples.
            X_new = np.stack(df_new.basis_value.values.tolist()[:])
            y_new = df_new['sample'].tolist()
            # Get a centroid of each sample
            clf_new = NearestCentroid()
            clf_new.fit(X_new, y_new)
            
            exist_sample_centroids = clf.centroids_
            new_sample_centroids = clf_new.centroids_
            list_exist_sample = df_exist['sample'].unique()
            list_new_sample = df_new['sample'].unique()
            # Compute the nearest centroids for the test samples.
            y_true_nearest, y_pred_nearest = calc_nearest_centroids(exist_sample_centroids, new_sample_centroids, 
                                            list_exist_sample, list_new_sample) # type: ignore
            
            acc, precision, recall, f1 = performance_eval(y_true_nearest, y_pred_nearest)
            
            print("Result of k_fold cross validation k =", k)
            print("Accuracy =", acc)
            print("Precision = ", precision)
            print("Recall = ", recall)
            print("f1 score = ", f1)
            # Append results for getting avg.
            result_k_fold[k] = [acc, precision, recall, f1]
            acc_list.append(acc)
            precision_score_list.append(precision)
            recall_score_list.append(recall)
            f1_score_list.append(f1)
        print("----------------------------------------------------------------------")
        print(f"Reuslt of k_fold cross validation ({rep_name}): k={k}, acc={np.mean(acc_list)}, precision={np.mean(precision_score_list)}, recall={np.mean(recall_score_list)}, f1={np.mean(f1_score_list)}")
        print("----------------------------------------------------------------------")
        result_rep[rep_name] = result_k_fold
        del df_exist
        del df_new
        del X_new
        del X
        del y
        del y_new
        del clf
        del df
        del adata
        del exist_sample_centroids
        del new_sample_centroids
        
    print("result_rep:", result_rep)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset')
    parser.add_argument('--num_kfold', type=int, help='Number of k fold cross validation')
    # Parse the arguments
    args = parser.parse_args()

    print("Running the experiment of scenario 2.3...")
    main(num_kfold=args.num_kfold)
    print("Done experiment of scenario 2.3!")
                