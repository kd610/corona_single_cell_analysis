"""
Sample Classification: Scenario 1

Method
Get KDEs of covid and non-covid from UMAP or PCA representation (UMAP/PCA~2), and compare them with a new sample cluster's KDE by KL-divergence.
* Sample for making KDE of covid and non-covid: Sample obtained from `train_test_split` (80% of the total samples)
* Sample to measure how similar to two KDEs: Sample obtained from `train_test_split` (20% of the total samples)

"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import scipy.stats as st
from scipy import stats
from sklearn.model_selection import train_test_split
from helper_func import n_random_sampling, KL_div, plot_dist_train_test
from scipy.stats import norm  
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings("ignore")


#np.random.seed(41)
sc.settings.verbosity = 1 # verbosity: errors (0), warnings (1), info (2), hints (3)
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
    max_samples_covid = 70
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
            
            # K-fold cross validation for train and test.
            result_k_fold = dict()
            # Those are for the result of the k-fold cross validation in avg.
            acc_list = []
            precision_score_list = []
            recall_score_list = []
            f1_score_list = []
            for k in tqdm(range(num_kfold)):
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
                covid_non_covid_vector = adata.obs['covid_non_covid'].values
                batch_vector = adata.obs['batch'].values # Define whether the cell is from the train or test set.
                df_pds = pd.DataFrame(list(zip(basis_values, sample_vector, batch_vector, covid_non_covid_vector)),
                            columns =['basis_value', 'sample', 'batch', 'covid_non_covid'])
                print(f"Created a dataframe that contains the {rep_name} values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.")
                
                # Delete the below variables to free up memory.
                del basis_values
                del sample_vector
                del covid_non_covid_vector
                del batch_vector
                
                # Get covid vectors and non-covid vectors from UMAP representation on the train (existing sample) data.
                
                ext_covid_vector = df_pds.query("batch == 'ref' and not sample.str.contains('control')")['basis_value'].values
                ext_non_covid_vector = df_pds.query("batch == 'ref' and sample.str.contains('control')")['basis_value'].values

                new_covid_vector = []

                for v in ext_covid_vector:
                    new_covid_vector.append(list(v))

                new_non_covid_vector = []

                for v in ext_non_covid_vector:
                    new_non_covid_vector.append(list(v))

                ext_covid_vector = np.array(new_covid_vector)
                ext_non_covid_vector = np.array(new_non_covid_vector)
                
                del new_covid_vector
                del new_non_covid_vector
                # print(f"Number of cells from selected covid samples: {len(ext_covid_vector)}")
                # print(f"Number of cells from selected non-covid samples: {len(ext_non_covid_vector)}")
                
                # Compute 
                x_covid = ext_covid_vector[:, 0]
                y_covid = ext_covid_vector[:, 1]
                x_n_covid = ext_non_covid_vector[:, 0]
                y_n_covid = ext_non_covid_vector[:, 1]

                # now determine nice limits by hand.
                binwidth = 1
                xymax = np.max([np.max(np.fabs(x_covid)), np.max(np.fabs(y_covid))])
                lim = (int(xymax/binwidth) + 1) * binwidth
                print("Getting the KDEs...")
                kernel_x_covid = stats.gaussian_kde(x_covid)
                kernel_y_covid = stats.gaussian_kde(y_covid)
                kernel_x_n_covid = stats.gaussian_kde(x_n_covid)
                kernel_y_n_covid = stats.gaussian_kde(y_n_covid)
                n_sample_covid = np.linspace(-lim+5, lim+5, len(ext_covid_vector)) #(start, stop, num=50 -> Number of samples to generate. Default is 50. Must be non-negative)

                # KDE for UMAP1 (x-axis)
                kde_x_covid = kernel_x_covid(n_sample_covid)
                kde_x_n_covid = kernel_x_n_covid(n_sample_covid)

                # KDE for UMAP2 (y-axis)
                kde_y_covid = kernel_y_covid(n_sample_covid)
                kde_y_n_covid = kernel_y_n_covid(n_sample_covid)
                
                # Getting the vector of test samples from the UMAP representation.
                y_test = df_pds.query("batch == 'new'")['sample'].unique()
                dict_sample_vec = {} #Key: sample_name, value: basis value as [UMAP1, UMAP2]
                for s in y_test:
                    dict_sample_vec[s] = np.stack(list(df_pds.query(f"sample == '{s}'")['basis_value'].values[:]))
                    #print(f"Sample: {s}, size of vecor {len(dict_sample_vec[s])}")
            
                result_pred = dict() # Key: sample_name, value: #[Avg of the exsiting covid sample's KL-divergence score between UMAP1 and UMAP2, Avg of the exsiting non-covid sample's KL-divergence score between UMAP1 and UMAP2]
                
                print("Computing KL-divergence with new samples")
                for s, vec in dict_sample_vec.items():
                    x_vec = vec[:, 0]
                    y_vec = vec[:, 1]

                    #Compute kernels of KDE for UMAP1 or PCA1: x-axis and UMAP2 or PCA2: y-axis
                    kernel_x = stats.gaussian_kde(x_vec)
                    kernel_y = stats.gaussian_kde(y_vec)

                    # KDE for UMAP1 or PCA1 (x-axis)
                    kde_x_test = kernel_x(n_sample_covid)

                    # KDE for UMAP2 or PCA2 (y-axis)
                    kde_y_test = kernel_y(n_sample_covid)

                    # Get KL-divergence in UMAP1 or PCA1: x-axis between the existing samples and test sample
                    x_KL_d_exis_covid = KL_div(kde_x_covid, kde_x_test)
                    x_KL_d_exis_non_covid = KL_div(kde_x_n_covid, kde_x_test)
                    # Get KL-divergence in UMAP2 or PCA2: y-axis between the existing samples and test sample
                    y_KL_d_exis_covid = KL_div(kde_y_covid, kde_y_test)
                    y_KL_d_exis_non_covid = KL_div(kde_y_n_covid, kde_y_test)
                    
                    # plot_dist_train_test(s, rep_name, x_KL_d_exis_covid, n_sample_covid, kde_x_covid, kde_x_test, \
                    #                 x_KL_d_exis_non_covid, kde_x_n_covid, y_KL_d_exis_covid, kde_y_covid, kde_y_test, y_KL_d_exis_non_covid, kde_y_n_covid):

                    # To produce a final prediction score, we sum up the KL-divergence score 
                    # between UMAP/PCA1 and UMAP/PCA2, and then divide by 2 to get an average of UMAP/PCA1 and UMAP/PCA2.
                    result_pred[s] = [(x_KL_d_exis_covid+y_KL_d_exis_covid)/2, (x_KL_d_exis_non_covid+y_KL_d_exis_non_covid)/2]

                print("Done computing KL-divergence with new samples")
                #print("result_pred:", result_pred)                        

                y_true = [] # Covid = 0, Non-Covid = 1
                y_pred = []

                for key, value in result_pred.items():
                    if "mild/moderate" in key or "severe/critical" in key:
                        y_true.append(0)
                    else:
                        y_true.append(1)

                    #print(value)
                    y_pred.append(np.argmin(value))
                    
                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro')
                recall = recall_score(y_true, y_pred, average='macro')
                f1 = f1_score(y_true, y_pred, average='macro')
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
                del df_pds
                del y_covid
                del x_n_covid
                del y_n_covid
                del kde_x_covid
                del kde_x_n_covid
                del kde_y_covid
                del kde_y_n_covid
                del y_test
                del dict_sample_vec
                del adata_train
                del adata_test
                print("Ended at: ", time.time())
                
            print("----------------------------------------------------------------------")
            print(f"Reuslt of k_fold cross validation ({rep_name}): k={k}, acc={np.mean(acc_list)}, precision={np.mean(precision_score_list)}, recall={np.mean(recall_score_list)}, f1={np.mean(f1_score_list)}")
            print("----------------------------------------------------------------------")
            result_rep[rep_name] = result_k_fold
            
        result_k_dataset_clip[i] = result_rep
        
    print("result_k_dataset_clip:", result_k_dataset_clip)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script')

    # Define the arguments
    parser.add_argument('--num_dataset_sampling', type=int, help='Number of k times splitting/clipping the dataset')
    parser.add_argument('--num_kfold', type=int, help='Number of k fold cross validation')
    # Parse the arguments
    args = parser.parse_args()

    main(num_dataset_sampling=args.num_dataset_sampling, num_kfold=args.num_kfold)
                