import numpy as np
import pandas as pd

import os

# CLR implementation
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for clr normalization.")
    if (data <= 0).any().any():
        data = data.replace(0, eps)
    gm = np.exp(data.apply(np.log).mean(axis=1))
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)
    return clr_data

input_dir_matrices = '../../00_matrices'
input_dir_r2 = '../../../out_results/out_regression/'

