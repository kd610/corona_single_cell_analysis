"""
Sample Classification: Scenario 3.3

Method
Get clusters as well as a centroid of covid and non_covid in UMAP representation. Encode two sample clusters into further by Euclidian distance between those centroids. Apply some linear classifier to train. Count the label prediction of a new sample because it’s encoded into distance expression, so that it derives how many cells are close with each exciting cluster/samples (covid or non-covid). 

This implementation is the same as this paper: 
A study on using data clustering for feature extraction to improve the quality of classification
"""
import numpy as np
import pandas as pd
import scanpy as sc
from helper_func import count_nearest_samples, performance_eval
import argparse
import warnings
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import time
from sklearn.svm import SVC, LinearSVC
import gc

warnings.filterwarnings("ignore")


# np.random.seed(41)
sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
# sc.logging.print_versions()
RANDOM_SEED = 110011

def main(num_kfold=5):
    # Read adata
    result_k_dataset_clip = dict()

    # Run experimetns for the representation of UMAP and PCA.
    result_rep = dict()
    for rep in ["X_pca", "X_umap"]:
    #for rep in ["X_umap"]:
        print("----------------Using an input representation: ", rep, " ------------------")
        if rep == "X_umap":
            rep_name = "UMAP"
            rep_name_small = 'umap'
        else:
            rep_name = "PCA"
            rep_name_small='pca'

        # K-fold cross validation for train and test.
        result_k_fold = dict()
        # Those are for the result of the k-fold cross validation in avg.
        acc_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        # Measure time for loop execution

        for k in tqdm(range(num_kfold)):
            start_time = time.time()
            print(f"k_fold cross validation (split randomly for train and test): k={k}")
            print("Started at: ", start_time)

            # Read pre-generated adata file for the k-th fold.
            k_file = f"../../data/k{k}_{rep_name}_adata_GSE_176269_COVID19.h5ad"
            print("Reading a h5ad data file:", k_file)
            adata = sc.read_h5ad(k_file)
            print("Finished reading the h5ad data file...")

            # Output the number of unique samples from the dataframe of adata.obs.
            print("Number of unique samples: ", len(adata.obs["sampleID"].unique()))
            new_value = None
            # Add a new columns: covid_non_covid to adata.obs
            for index, row in adata.obs.iterrows():
                if "Cov" in row["sampleID"]:
                    new_value = "covid"
                else:
                    new_value = "non_covid"

                adata.obs.at[index, "covid_non_covid"] = new_value

            # Create a dataframe that contains the UMAP values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.
            if rep=='X_umap':
                basis_values = adata.obsm[rep]  # X_umap
            else:
                basis_values = adata.obsm['X_pca'][:, :2]  # X_pca
            sample_vector = adata.obs["sampleID"].values
            # covid_non_covid_vector = adata.obs['covid_non_covid'].values
            batch_vector = adata.obs["batch"].values  # Define whether the cell is from the train or test set.
            x_basis_value = []
            y_basis_value = []

            ### I'll use the slice later...
            for b_v in basis_values:
                x_basis_value.append(b_v[0])
                y_basis_value.append(b_v[1])
            ###
            df = pd.DataFrame(
                list(zip(basis_values, sample_vector, batch_vector, x_basis_value, y_basis_value)),
                columns=["basis_value", "sample", "batch", f"{rep_name_small}1", f"{rep_name_small}2"],
            )
            print(f"Created a dataframe that contains the {rep_name} values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.")
            
            list_covid_non_covid = []
            for index, row in df.iterrows():
                if "Cov" in row['sample']:
                    list_covid_non_covid.append('covid')
                else:
                    list_covid_non_covid.append('non_covid')

            df['covid_non_covid'] = list_covid_non_covid
            df

            # Delete the below variables to free up memory.
            del basis_values
            del sample_vector
            del batch_vector
            del x_basis_value
            del y_basis_value
            gc.collect()

            # Get train and test samples
            y_train = np.unique(df.query("batch == 'ref'")["sample"].values)
            y_test = np.unique(df.query("batch == 'new'")["sample"].values)
            print("Num of train samples:", len(y_train))
            print("Num of test samples:", len(y_test))

            # Create dataframe for the train samples.
            # Get a centroid of each sample: exist (train)
            y_train = list(y_train)
            df_exist = df.query("sample == @y_train")
            X = np.stack(df_exist.basis_value.values.tolist()[:])
            y = df_exist['sample'].tolist()
            clf = NearestCentroid()
            clf.fit(X, y)

            y_test = list(y_test)
            df_new = df.query("sample == @y_test")
            #df_new

            X_new = np.stack(df_new.basis_value.values.tolist()[:])
            y_new = df_new['sample'].tolist()

            y_pred = []
            for x_new in X_new:
                y_pred.append(clf.predict([x_new]))

            y_pred = np.squeeze(np.stack(y_pred[:]))

            # Get a centroid of each sample: new (test)
            X_new = np.stack(df_new.basis_value.values.tolist()[:])
            y_new = df_new['sample'].tolist()

            clf_new = NearestCentroid()
            clf_new.fit(X_new, y_new)

            exist_sample_centroids = clf.centroids_
            #print("exist_sample_centroids:", exist_sample_centroids)
            new_sample_centroids = clf_new.centroids_

            # result_dict = dict(zip(np.unique(np.array(y_new)),clf_new.centroids_))
            # print(result_dict)

            # Make covid_non_covid column
            list_covid_non_covid = []
            for index, row in df_exist.iterrows():
                if "Cov" in row['sample']:
                    list_covid_non_covid.append('covid')
                else:
                    list_covid_non_covid.append('non_covid')

            df_exist['covid_non_covid'] = list_covid_non_covid

            list_covid_non_covid = []
            for index, row in df_new.iterrows():
                if "Cov" in row['sample']:
                    list_covid_non_covid.append('covid')
                else:
                    list_covid_non_covid.append('non_covid')

            df_new['covid_non_covid'] = list_covid_non_covid
            
            ### Compute a distance
            print("Compute a distance...")
            for i in tqdm(range(len(exist_sample_centroids))):
                cur_centroid = np.array(exist_sample_centroids[i])
                # Traverse each row to calculate the distance between the centroid 
                # and the coordinates of each sample.
                temp_dist_sample_centroids = []
                #print("cur_centroid:", cur_centroid)
                for index, row in df_exist.iterrows():
                    coord = np.array([row[f'{rep_name_small}1'], row[f'{rep_name_small}2']])
                    #print("coord:", coord)
                    dist = np.linalg.norm(cur_centroid - coord)

                    temp_dist_sample_centroids.append(dist)
                
                # Add dinstances between each row and selected centroid to existing df.
                df_exist[f'dist_centorid_{i}'] = temp_dist_sample_centroids

            # Prepare training data for building the model
            X_train = df_exist.drop(['basis_value', 'sample', f'{rep_name_small}1', f'{rep_name_small}2', 'batch', 'covid_non_covid'], axis=1)
            y_train = df_exist['covid_non_covid']

            print("Train the SVM model...")
            # Instantiate the model
            cls = LinearSVC()
            # Train/Fit the model 
            cls.fit(X_train, y_train)
            
            # Get distance between a new sample and existing centroids
            print("Get distance between a new sample and existing centroids...")
            for i in range(len(exist_sample_centroids)):
                cur_centroid = np.array(exist_sample_centroids[i])
                # Traverse each row to calculate the distance between the centroid 
                # and the coordinates of each sample.
                temp_dist_sample_centroids = []
                for index, row in df_new.iterrows():
                    coord = np.array([row[f'{rep_name_small}1'], row[f'{rep_name_small}2']])
                    dist = np.linalg.norm(cur_centroid - coord)

                    temp_dist_sample_centroids.append(dist)
                
                # Add dinstances between each row and selected centroid to existing df.
                df_new[f'dist_centorid_{i}'] = temp_dist_sample_centroids

            # Run prediction
            print("Run prediction...")
            X_test = df_new.drop(['basis_value', 'sample', f'{rep_name_small}1', f'{rep_name_small}2', 'batch', 'covid_non_covid'], axis=1)
            y_test = df_new['covid_non_covid']
            y_pred = cls.predict(X_test)

            df_new['y_pred'] = y_pred
            
            pred_label_count = {key: [0, 0,]for key in df_new['sample'].unique()} # [count_covid, count_non_covid]
            for index, row in df_new.iterrows():
                if 'covid' == row['y_pred']:
                    pred_label_count[row['sample']][0] += 1
                else:
                    pred_label_count[row['sample']][1] += 1

            #print("pred_label_count:", pred_label_count)

            y_pred = []
            y_true = []

            # Finalize the output, get the label with the highest counts
            for sample, counts in pred_label_count.items():
                if counts[0] > counts[1]: #counts of covid > counts of non_covid
                    y_pred.append('covid')
                else:
                    y_pred.append('non_covid')
                
                if "Cov" in sample:
                    y_true.append("covid")
                else:
                    y_true.append("non_covid")
            
            # Compute the performance metrics.
            acc, precision, recall, f1 = performance_eval(y_true, y_pred)

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
            del df_exist
            del df_new
            del X_new
            del X
            del y
            del y_new
            del clf
            del df
            del X_train
            del y_train
            del X_test
            del y_test
            del y_pred
            del pred_label_count
            del list_covid_non_covid
            del clf_new
            del adata
            gc.collect()

        print("----------------------------------------------------------------------")
        print(
            f"Reuslt of k_fold cross validation ({rep_name}): k={k}, acc={round(np.mean(acc_list), 4)}, precision={round(np.mean(precision_score_list),4)}, recall={round(np.mean(recall_score_list), 4)}, f1={round(np.mean(f1_score_list), 4)}"
        )
        print("----------------------------------------------------------------------")
        print(f"Standard deviation for each metric: acc={round(np.std(acc_list), 4)}, precision={round(np.std(precision_score_list), 4)}, recall={round(np.std(recall_score_list), 4)}, f1={round(np.std(f1_score_list), 4)}")
        print("----------------------------------------------------------------------")
        result_rep[rep_name] = result_k_fold

    print("result_rep:", result_rep)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script")
    # Define the arguments
    parser.add_argument("--num_kfold", type=int, help="Number of k fold cross validation")
    # Parse the arguments
    args = parser.parse_args()

    print("Running the experiment of scenario 3.2...")
    main(num_kfold=args.num_kfold)
    print("Done experiment of scenario 3.2!")
