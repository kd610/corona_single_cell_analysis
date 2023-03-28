from numpy.random import default_rng
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(41)

# Compute KL Divergence
"""KL Divergence(P|Q)"""
def KL_div(p_probs, q_probs):    
    KL_div = p_probs * np.log(p_probs / q_probs)
    
    return np.sum(KL_div)

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

def plot_dist_train_test(s, rep_name, x_KL_d_exis_covid, n_sample_covid, kde_x_covid, kde_x_test, x_KL_d_exis_non_covid, kde_x_n_covid, y_KL_d_exis_covid, kde_y_covid, kde_y_test, y_KL_d_exis_non_covid, kde_y_n_covid):
    """
    Plot distributions of the existing and test.
    """
    # Initialise the subplot function using number of rows and columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout(pad=5.0)

    print(f"Existing sample: covid, Test sample: {s} in {rep_name}1, KL-divergence: {x_KL_d_exis_covid}")
    axes[0,0].set_title('UMAP1, KL(P||Q) = %1.3f' % x_KL_d_exis_covid)  
    axes[0,0].plot(n_sample_covid, kde_x_covid, label='Covid (exist)')  
    axes[0,0].plot(n_sample_covid, kde_x_test, label=f'{s}', color='red')
    axes[0,0].legend(prop={'size': 8})
    print(f"Existing sample: non_covid, Test sample: {s} in UMAP1, KL-divergence: {x_KL_d_exis_non_covid}")
    axes[0,1].set_title('UMAP1, KL(P||Q) = %1.3f' % x_KL_d_exis_non_covid)  
    axes[0,1].plot(n_sample_covid, kde_x_n_covid, label='Non-Covid (exist)')  
    axes[0,1].plot(n_sample_covid, kde_x_test, label=f'{s}', color='red')
    axes[0,1].legend(prop={'size': 8})

    #result_UMAP1_pred[s] = [x_KL_d_exis_covid, x_KL_d_exis_non_covid]

    print(f"Existing sample: covid, Test sample: {s} in UMAP2, KL-divergence: {y_KL_d_exis_covid}")
    axes[1,0].set_title('UMAP2, KL(P||Q) = %1.3f' % y_KL_d_exis_covid)  
    axes[1,0].plot(n_sample_covid, kde_y_covid, label='Covid (exist)')  
    axes[1,0].plot(n_sample_covid, kde_y_test, label=f'{s}', color='red')
    axes[1,0].legend(prop={'size': 8})
    print(f"Existing sample: non_covid, Test sample: {s} in UMAP2, KL-divergence: {y_KL_d_exis_non_covid}")
    axes[1,1].set_title('UMAP2, KL(P||Q) = %1.3f' % y_KL_d_exis_non_covid)  
    axes[1,1].plot(n_sample_covid, kde_y_n_covid, label='Non-Covid (exist)')  
    axes[1,1].plot(n_sample_covid, kde_y_test, label=f'{s}', color='red')
    axes[1,1].legend(prop={'size': 8})
    
    #fig.savefig(f"../output_pdfs/dist_compare_{s}_GSE159812.pdf", bbox_inches='tight')
