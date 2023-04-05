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
from helper_func import count_nearest_samples, performance_eval
import argparse
import warnings
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
import time
import gc

warnings.filterwarnings("ignore")


# np.random.seed(41)
sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)
# sc.logging.print_versions()
RANDOM_SEED = 110011

def main(num_kfold=5):
    
    print("Start scenario 2.1 with input DP: GSE176269")
    
    # Read adata
    result_k_dataset_clip = dict()

    # Run experimetns for the representation of UMAP and PCA.
    result_rep = dict()
    for rep in ["X_pca", "X_umap"]:
        print("----------------Using an input representation: ", rep, " ------------------")
        if rep == "X_umap":
            rep_name = "UMAP"
        else:
            rep_name = "PCA"

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
            k_file = f"../../data/k{k}_{rep_name}_adata_inputDP_GSE_176269_COVID19.h5ad"
            print("Reading a h5ad data file:", k_file)
            adata = sc.read_h5ad(k_file)
            print("Finished reading the h5ad data file...")

            # Output the number of unique samples from the dataframe of adata.obs.
            print("Number of unique samples: ", len(adata.obs["sampleID"].unique()))
            new_value = None
            # Add a new columns: covid_non_covid to adata.obs
            for index, row in adata.obs.iterrows():
                if 'Cov' in row['sampleID']:
                    new_value = 'covid'
                else:
                    new_value = 'non_covid'

                adata.obs.at[index, "covid_non_covid"] = new_value

            # Create a dataframe that contains the UMAP values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.
            basis_values = adata.obsm[rep]  # X_umap or X_pca
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
                columns=["basis_value", "sample", "batch", "x_basis_value", "y_basis_value"],
            )
            print(f"Created a dataframe that contains the {rep_name} values (1 & 2), sample_vector, batch_vector, covid_non_covid_vector.")

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
            y_train = list(y_train)
            print("Number of samples in the train set: ", len(y_train))
            df_exist = df.query("sample == @y_train")

            # Get centroids of each existing samples.
            X = np.stack(df_exist.basis_value.values.tolist()[:])
            y = df_exist["sample"].tolist()
            clf = NearestCentroid()
            clf.fit(X, y)

            # Create dataframe for the test samples.
            y_test = list(y_test)
            df_new = df.query("sample == @y_test")
            # Create input features and labels for the test samples.
            X_new = np.stack(df_new.basis_value.values.tolist()[:])
            y_new = df_new["sample"].tolist()
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
            del df_exist
            del df_new
            del X_new
            del X
            del y
            del y_new
            del y_pred
            del clf
            del df
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

    print("Running the experiment of scenario 2.1...")
    main(num_kfold=args.num_kfold)
    print("Done experiment of scenario 2.1!")
