import numpy as np

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

def find_max_proba(prob_array):
    return np.max(prob_array)

def argmax_list(lst):
    return max(range(len(lst)), key=lst.__getitem__)    