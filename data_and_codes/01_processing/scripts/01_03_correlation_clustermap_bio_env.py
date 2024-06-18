import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

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

input_dir = '../../00_matrices'
output_dir = '../../../out_results/out_correlation/correlation_bio_env'

os.makedirs(output_dir, exist_ok=True)

md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=0)

categorical_cols = ['PANGAEA sample id', 'Station.label', 'lower.size.fraction','upper.size.fraction','Event.date',
                    'Depth.nominal', 'Depth.Mixed.Layer','Layer','Layer2', 'polar', 'Ocean', 'Province', 'NP']
numeric_md = md.drop(columns=categorical_cols)

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
escenarios = ['all', 'polar', 'nonpolar', 'srf', 'epi', 'epi-nonpolar', 'srf-polar', 'srf-nonpolar']

for matrix in matrix_types:
    for escenario in escenarios:
        try:
            # Read the specific matrix file
            file_path = f'{input_dir}/Matrix_{matrix}_{escenario}.tsv'
            output_file = f'{output_dir}/corr_bio_env_clustermap_{matrix}_{escenario}'
            df = pd.read_csv(file_path, sep='\t', index_col=0)
            clr_df = clr_(df)

            
            aligned_md = numeric_md.loc[clr_df.index] # Align md df
            combined_df = pd.concat([aligned_md, clr_df], axis=1)
            corr_matrix = combined_df.corr(method='spearman') # Compute correlation matrix of the combined dataframe
            corr_md_clr = corr_matrix.loc[aligned_md.columns, clr_df.columns]

            # Remove columns and rows with all constant values which result in NaN correlations
            corr_md_clr = corr_md_clr.dropna(axis=0, how='all').dropna(axis=1, how='all')

            # Perform hierarchical clustering
            row_linkage = linkage(pdist(corr_md_clr), method='average')
            col_linkage = linkage(pdist(corr_md_clr.T), method='average')
            row_dendro = dendrogram(row_linkage, no_plot=True)
            col_dendro = dendrogram(col_linkage, no_plot=True)

            # Reorder the matrix based on the clustering
            reordered_corr = corr_md_clr.iloc[row_dendro['leaves'], col_dendro['leaves']]
            reordered_corr.to_csv(f'{output_file}.tsv', sep='\t')

            # Plot
            fig, ax = plt.subplots(figsize=(30, 20))
            ax.set_facecolor('white')
            ax.imshow(np.ones_like(reordered_corr), cmap='gray_r', interpolation='nearest')

            ax.set_xticks(np.arange(len(reordered_corr.columns)))
            ax.set_yticks(np.arange(len(reordered_corr.index)))
            ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)
            ax.set_yticklabels(reordered_corr.index, fontsize=8, color="black")
            ax.set_xticklabels(reordered_corr.columns, fontsize=8, color="black", rotation=90)

            ax.set_xticks(np.arange(len(reordered_corr.columns) + 1) - .5, minor=True)
            ax.set_yticks(np.arange(len(reordered_corr.index) + 1) - .5, minor=True)
            ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1)
            rect = Rectangle((-.5, -.5), len(reordered_corr.columns), len(reordered_corr.index), linewidth=2, edgecolor='lightgray', facecolor='none')
            ax.add_patch(rect)

            # squares proportional to the absolute value of correlation
            norm = plt.Normalize(-1, 1)
            sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
            for i in range(len(reordered_corr.columns)):
                for j in range(len(reordered_corr.index)):
                    correlation = reordered_corr.iat[j, i]
                    color = sm.to_rgba(correlation)
                    size = abs(correlation) * 1  # Adjust size factor as needed
                    rect = Rectangle(xy=(i - size / 2, j - size / 2), width=size, height=size, facecolor=color)
                    ax.add_patch(rect)
            # Add colorbar
            plt.colorbar(sm, ax=ax, shrink=0.4, aspect=20, pad=0.04).set_label('Correlation')
            plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
            plt.close()

        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping...")
