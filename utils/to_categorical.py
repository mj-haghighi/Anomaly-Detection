from cProfile import label
from typing import List
import numpy as np

def batch(labels:np.ndarray, classes):
    one_hot = np.eye(len(classes))[labels]
    return one_hot

def sample(label:str, classes:List[str]):
    clabel = np.zeros(shape=(len(classes),))
    clabel[classes.index(label)] = 1
    return clabel
