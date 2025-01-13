#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as patches

from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress


# # CLR implementation

# In[2]:


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


# # Define output directory

# In[7]:

input_dir = '../../00_matrices'
out_dir = '../../../out_results/out_fingerprint/'


# # Upload data metadata

# In[3]:


md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=[0])
md.head()


# # Tara global 173 samples (all samples)

# In[9]:


def plot_matrix_abundance(matrix_type, sample_name):
    
    matrix_file = f'{input_dir}/Matrix_{matrix_type}_all.tsv'
    df = pd.read_csv(matrix_file, sep='\t', index_col=0)
    clr_df = clr_(df) # centered-log-ratio normalization

    # Check if the sample is in the dataframe
    if sample_name not in clr_df.index:
        raise ValueError(f"The sample '{sample_name}' is not in the DataFrame.")

    clr_df_sorted = clr_df.loc[:, clr_df.loc[sample_name].sort_values(ascending=False).index]
    clr_df_transposed = clr_df_sorted.T

    plt.figure(figsize=(14, 8))
    for sample in clr_df_transposed.columns:
        plt.plot(clr_df_transposed.index, clr_df_transposed[sample])

    plt.title(f'clr-abundance of TFs for {matrix_type} matrix \nordered by {sample_name}')
    plt.xlabel('TFs')
    plt.ylabel('clr-abundance')
    plt.xticks(rotation=90, ha='center')

    output_file = f'fingerprint_tara_173_{matrix_type}.pdf'
    plt.tight_layout()
    plt.savefig(out_dir+output_file, bbox_inches='tight')
    #plt.show()
    plt.close()

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
sample_name = 'TSC000' # Name of the sample to order

for matrix_type in matrix_types:
    plot_matrix_abundance(matrix_type, sample_name)


# # Tara global 10 surface samples selected

# In[12]:


def plot_matrix_abundance_selected(matrix_type, sample_name, selected_indices):
    
    matrix_file = f'{input_dir}/Matrix_{matrix_type}_srf.tsv'
    df = pd.read_csv(matrix_file, sep='\t', index_col=0)
    clr_df = clr_(df)  # centered-log-ratio normalization

    if sample_name not in clr_df.index:
        raise ValueError(f"The sample '{sample_name}' is not in the DataFrame.")

    clr_df_sorted = clr_df.loc[:, clr_df.loc[sample_name].sort_values(ascending=False).index]
    clr_df_transposed = clr_df_sorted.T

    clr_df_transposed_filtered = clr_df_transposed[selected_indices]

    plt.figure(figsize=(14, 8))
    for sample in clr_df_transposed_filtered.columns:
        plt.plot(clr_df_transposed_filtered.index, clr_df_transposed_filtered[sample])

    plt.title(f'clr-abundance of TFs for {matrix_type} matrix \nordered by {sample_name}')
    plt.xlabel('TFs')
    plt.ylabel('clr-abundance')
    plt.xticks(rotation=90, ha='center')

    output_file = f'fingerprint_tara_10_selected_{matrix_type}.pdf'
    plt.tight_layout()
    plt.savefig(out_dir+output_file, bbox_inches='tight')
    # plt.show()
    plt.close()

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
sample_name = 'TSC013' # Name of the sample to order

# Selected samples
selected_indices = ['TSC272', 'TSC065', 'TSC254', 'TSC013', 'TSC242', 'TSC216', 'TSC027', 'TSC135', 'TSC141', 'TSC167']

for matrix_type in matrix_types:
    plot_matrix_abundance_selected(matrix_type, sample_name, selected_indices)

