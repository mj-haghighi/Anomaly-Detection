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