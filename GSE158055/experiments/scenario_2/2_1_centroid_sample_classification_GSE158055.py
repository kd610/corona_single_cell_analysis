"""
Sample Classification: Scenario 2.1

Method
Get centroids of each sample's cluster (covid and non-covid samples).
When we get a new sample, count the nearest centroids of each cell in a new sample. 
And make a result based on the count (either `covid` or `non_covid`)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import scipy.stats as st
from scipy import stats
from sklearn.model_selection import train_test_split
from helper_func import n_random_sampling, count_nearest_samples, performance_eval
from scipy.stats import norm  
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
import warnings
from sklearn.neighbors import NearestCentroid
from collections import Counter
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")


#np.random.seed(41)
sc.settings.verbosity = 1 # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()
RANDOM_SEED = 110011

def main(num_dataset_sampling=5, num_kfold=5):
    # Read adata
    print("Reading preprocessed adata...")
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
    max_samples_covid = 80
    max_samples_control = 28 # There is only 28 control samples.
    k_sets_samples_severe_critical, k_set_samples_mild_moderate, k_sets_samples_control = \
                                                                    n_random_sampling(adata, num_dataset_sampling, max_samples_covid, max_samples_control)

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
            
            # K-fold cross validation for train and test.
            result_k_fold = dict()
            # Those are for the result of the k-fold cross validation in avg.
            acc_list = []
            precision_score_list = []
            recall_score_list = []
            f1_score_list = []
            
            # Measure time for loop execution
            
            for k in tqdm(range(num_kfold)):
                # Get preocessing time for each k-fold
                start_time = time.time()
                print("k_fold cross validation (split randomly for train and test)", k, " on the ", i, "st clipped dataset","--------")
                print("Started at: ", start_time)
                
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
                adata = adata_concat.copy()
                
                # Delete adata_concat to free up memory.
                del adata_concat

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
                # Get the nearest centroids with new samples.
                y_pred = []
                for x_new in X_new:
                    y_pred.append(clf.predict([x_new]))

                y_pred = np.squeeze(np.stack(y_pred[:]))
                
                # Count how many the nearest centroids of covid sample/non-covid samples 
                # has the new sample obtained.
                y_true_agg, y_pred_agg = count_nearest_samples(df_new, y_pred)
                # Compute the performance metrics.
                acc, precision, recall, f1 = performance_eval(y_true_agg, y_pred_agg)
                
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
                
                # Delete variables to save memory.
                del ext_covid_vector
                del ext_non_covid_vector
                del x_covid
                del x_non_covid
                del df
                del y_covid
                del y_non_covid
                del kde_x_covid
                del kde_x_n_covid
                del kde_y_covid
                del kde_y_n_covid
                del y_test
                del dict_sample_vec
                
            print("----------------------------------------------------------------------")
            print(f"Reuslt of k_fold cross validation ({rep_name}): k={k}, acc={np.mean(acc_list)}, precision={np.mean(precision_score_list)}, recall={np.mean(recall_score_list)}, f1={np.mean(f1_score_list)}")
            print("----------------------------------------------------------------------")
            result_rep[rep_name] = result_k_fold
            
        result_k_dataset_clip[i] = result_rep
        del df_exist
        del df_new
        del X_new
        del X
        del clf
        
    print("result_k_dataset_clip:", result_k_dataset_clip)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset')
    parser.add_argument('--num_kfold', type=int, help='Number of k fold cross validation')
    # Parse the arguments
    args = parser.parse_args()

    print("Running the experiment of scenario 2.1...")
    main(num_dataset_sampling=args.num_dataset_sampling, num_kfold=args.num_kfold)
    print("Done experiment of scenario 2.1!")
                