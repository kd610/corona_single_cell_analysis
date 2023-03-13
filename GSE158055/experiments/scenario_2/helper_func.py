from numpy.random import default_rng
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import metrics

# Compute KL Divergence
"""KL Divergence(P|Q)"""
def KL_div(p_probs, q_probs):    
    KL_div = p_probs * np.log(p_probs / q_probs)
    
    return np.sum(KL_div)

def n_random_sampling(adata, n=1, max_samples_covid=128, max_samples_control=28):
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
        rng = default_rng()
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

def count_nearest_samples(df_new, y_pred):
    beginning = 0
    samples_range = dict()

    y_pred_agg = []
    y_true_agg = []

    # Get the range of each new samples.
    # Because each sample contains multiple cells, we need to get the range of each sample.
    for sample in df_new['sample'].unique():
        # Assing a new label (either covid or non_covid sample))
        if "mild/moderate" in sample or "severe/critical" in sample:
            y_true_agg.append('covid')
        else:
            y_true_agg.append('non_covid')
        samples_range[sample] = [beginning, beginning+len(df_new[df_new['sample'] == sample].index.to_list())]
        beginning = len(df_new[df_new['sample'] == sample].index.to_list())+beginning

    # Count the number of nearest train samples for each new sample.
    for s, r in samples_range.items():
        #print(f"Testing {s}")
        tmp_y_pred = y_pred[r[0]:r[1]]
        # count_detected_samples_key = list(Counter(tmp_y_pred).keys())
        # count_detected_samples = list(Counter(tmp_y_pred).values())
        count_pred_samples = Counter(tmp_y_pred)

        covid_sample_counter = 0
        non_covid_sample_counter = 0
        for s, count in count_pred_samples.items():
            if "mild/moderate" in s or "severe/critical" in s:
                covid_sample_counter+=count
            else:
                non_covid_sample_counter+=count
        
        if covid_sample_counter>non_covid_sample_counter:
            y_pred_agg.append('covid')
        else:
            y_pred_agg.append('non_covid')
    
    return y_true_agg, y_pred_agg

def calc_dist_centroids(exist_sample_centroids, new_sample_centroids, list_exist_sample, list_new_sample):
    """
    Calculate the distance between the centroids of the existing samples and the new samples.
    Next, get a label from checking the total distance of summing up the distance of existing/train covid sample 
    and non covid sample between each new/test sample.
    
    And if the total distance to centroid with corona patient < the total distance to centroid with non corona patient,
    -> The new sample is Corona (because the centroid of the new sample should be closer with the existing covid samples).
    Otherwise, The new sample is Non-covid.
    """
    dict_dist_results = {s: [] for s in list_new_sample}
    for i in range(len(exist_sample_centroids)):
        # print("---------------------")
        # print("Exist sample:", list_exist_sample[i])
        #sample_key = list_exist_sample[i]
        for j in range (len(new_sample_centroids)):
            sample = list_new_sample[j]
            #print('sample:', list_new_sample[j])
            dist = np.linalg.norm(np.array(exist_sample_centroids[i]) - np.array(new_sample_centroids[j]))
            #print(dist)
            dict_dist_results[sample].append(dist)
            

    new_dist_total = {s: {'covid': 0, 'non_covid': 0} for s in list_new_sample}

    # Summing up the distance of exist covid sample and non covid sample between each ewn sample.
    for i in range(len(list_exist_sample)):
        exist_sample = list_exist_sample[i]
        print(exist_sample)
        for new_s in dict_dist_results.keys():
            dist = dict_dist_results[new_s][i]
            if "mild/moderate" in exist_sample or "severe/critical" in exist_sample:
                new_dist_total[new_s]['covid'] += dist
            elif "control" in exist_sample:
                new_dist_total[new_s]['non_covid'] += dist

    y_true_total_dist = []
    y_pred_total_dist = []

    for new_s, dist_total in new_dist_total.items():
        if "mild/moderate" in new_s or "severe/critical" in new_s:
            y_true_total_dist.append('covid')
        else:
            y_true_total_dist.append('non_covid')

        # Compare the total distance to centroid with corona & non corona samples    
        if dist_total['covid'] < dist_total['non_covid']:
            y_pred_total_dist.append('covid')
        else:
            y_pred_total_dist.append('non_covid')
            
    
    return y_true_total_dist, y_pred_total_dist

def calc_nearest_centroids(exist_sample_centroids, new_sample_centroids, list_exist_sample, list_new_sample):
    """
    Calculate the nearest centroid of the existing samples and the new samples.
    """
    dict_dist_results = {s: [] for s in list_new_sample}
    for i in range(len(exist_sample_centroids)):
        # print("---------------------")
        # print("Exist sample:", list_exist_sample[i])
        #sample_key = list_exist_sample[i]
        for j in range (len(new_sample_centroids)):
            sample = list_new_sample[j]
            #print('sample:', list_new_sample[j])
            dist = np.linalg.norm(np.array(exist_sample_centroids[i]) - np.array(new_sample_centroids[j]))
            #print(dist)
            dict_dist_results[sample].append(dist)
            
    y_true_nearest = []
    y_pred_nearest = []

    for new_s, coords_centroid in dict_dist_results.items():
        nearest_sample_index = np.argmin(np.array(coords_centroid))
        nearest_sample = list_exist_sample[nearest_sample_index]

        # Assign a label to y_true
        if "mild/moderate" in new_s or "severe/critical" in new_s:
            y_true_nearest.append('covid')
        else:
            y_true_nearest.append('non_covid')
        
        # Assign a label to y_pred based on the nearest sample by distance
        if "mild/moderate" in nearest_sample or "severe/critical" in nearest_sample:
            y_pred_nearest.append('covid')
        else:
            y_pred_nearest.append('non_covid')
                
        
        return y_true_nearest, y_pred_nearest

def performance_eval(y_true, y_pred):
    # Model Accuracy: how often is the classifier correct?
    acc = metrics.accuracy_score(y_true, y_pred)

    # Model Precision: what percentage of positive tuples are labeled as such?
    precision = metrics.precision_score(y_true, y_pred,  average='micro')

    # Model Recall: what percentage of positive tuples are labelled as such?
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    
    # Model F1: what percentage of positive tuples are labelled as such?
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    
    return acc, precision, recall, f1
    
    
