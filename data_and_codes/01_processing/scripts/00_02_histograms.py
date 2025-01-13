import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from statsmodels.stats.diagnostic import kstest_normal
from statsmodels.stats.diagnostic import lilliefors

# Define input and output directories
input_dir = '../../00_matrices/'
output_dir = '../../../out_results/out_histograms/'

os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

# CLR implementation

def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.

    Parameters:
    data (pandas.DataFrame): A DataFrame with samples as rows and components as columns.

    Returns:
    pandas.DataFrame: A clr-normalized DataFrame.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for clr normalization.")

    # Add small amount to cells with a value of 0
    if (data <= 0).any().any():
        data = data.replace(0, eps)

    # Calculate the geometric mean of each row
    gm = np.exp(data.apply(np.log).mean(axis=1))

    # Perform clr transformation
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)

    return clr_data

# Read and filter the files
matrix_files = [f for f in os.listdir(input_dir) if f.startswith('Matrix_') and not f.startswith('Matrix_GEN_') and f != 'Matrix_outsource.tsv' and f != 'Matrix_outsource_original.tsv' and f.endswith('.tsv')]

clr_dfs = {}
for file in matrix_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # Apply clr transformation
    clr_df = clr_(df)
    
    # Store the clr-transformed DataFrame with the file name as the key
    clr_dfs[file] = clr_df

# Generate histograms and perform Lilliefors test
for name, clr_df in clr_dfs.items():
    base_name = name.replace('Matrix_', '').replace('.tsv', '') # Store matrix type and subsample in the name
    # Calculate the layout dynamically based on the number of features
    n_features = clr_df.shape[1]
    n_cols = 6  # Adjust the number of columns as needed
    n_rows = n_features // n_cols + (n_features % n_cols > 0)  # Ensure all plots fit in the figure

    plt.figure(figsize=(n_cols * 4, n_rows * 3))  # Adjust figure size as needed

    for i, feature in enumerate(clr_df.columns, start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(clr_df[feature], kde=True)

        # Perform the Lilliefors test
        data = clr_df[feature].dropna()  # Remove missing values if any
        statistic, p_value = lilliefors(data, dist='norm', pvalmethod='table')

        # Display the test results in the plot
        plt.text(0.4, 0.8, f'Stat: {statistic:.3f}\nP-value: {p_value:.3f}', transform=plt.gca().transAxes, ha='right', fontsize=11)
        plt.title(f'Distribution of {feature}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_of_tfs_{base_name}.pdf'), bbox_inches='tight')
    plt.close()