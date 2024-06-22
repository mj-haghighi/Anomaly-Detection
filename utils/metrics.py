import numpy as np
import pandas as pd

def calc_entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities + 0.001))

def std_agg(values):
    # 6 time faster than values.apply(lambda x: x**2)
    # 7 time faster than np.power and 
    z = []
    for value in values:
        values_powered_by_2 = value ** 2
        z.append(values_powered_by_2)
    return np.sqrt(np.mean(z))

def diff_two_max(prob_array):
    sorted_probs = sorted(prob_array, reverse=True)
    return sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else np.nan

def aum(prob_array, assigned_label_index):
    sorted_probs = sorted(prob_array, reverse=True)
    return sorted_probs[0] - prob_array[assigned_label_index]

def find_max_proba(prob_array):
    return np.max(prob_array)

def argmax_list(lst):
    return max(range(len(lst)), key=lst.__getitem__)

def load_examins_auc(path, index_col):
    experiments_with_corretness_auc = None
    try:
        experiments_with_corretness_auc = pd.read_csv(path, index_col=index_col)
    except Exception as e:
        print(e)
    return experiments_with_corretness_auc

def load_experiments(path, index_col):
    experiments = pd.read_csv(path, index_col=index_col)
    try:
        experiments.drop(columns=['test_acc'], inplace=True)
    except Exception as e:
        print(e)
    return experiments

def filter_out_auc_calculated_experiments(experiments, experiments_with_corretness_auc):
    experiments['has_auc'] = False
    for idx in experiments_with_corretness_auc.index:
        if idx in experiments.index:
            experiments.loc[idx, 'has_auc'] = True

    return experiments[(experiments['has_auc'] == False)]
