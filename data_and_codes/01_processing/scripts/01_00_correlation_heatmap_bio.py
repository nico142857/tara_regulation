import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

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
output_dir = '../../../out_results/out_correlation/correlation_bio'

os.makedirs(output_dir, exist_ok=True)

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
escenarios = ['all', 'polar', 'nonpolar', 'srf', 'epi', 'epi-nonpolar', 'srf-polar', 'srf-nonpolar']

for matrix_type in matrix_types:
    for escenario in escenarios:
        file_path = f'{input_dir}/Matrix_{matrix_type}_{escenario}.tsv'
        output_file = f'{output_dir}/corr_bio_heatmap_{matrix_type}_{escenario}'

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\t', index_col=0)
            clr_df = clr_(df)
            corr_df = clr_df.corr(method='spearman').T
            corr_df.to_csv(f'{output_file}.tsv', sep='\t')

            # Creación de la figura
            fig, ax = plt.subplots(figsize=(30, 30))
            ax.set_facecolor('white')
            ax.imshow(np.ones_like(corr_df), cmap='gray_r', interpolation='nearest')

            # Set the tick labels and rotation for the x and y axes
            ax.set_xticks(np.arange(len(corr_df.columns)))
            ax.set_yticks(np.arange(len(corr_df.index)))
            ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)
            ax.set_yticklabels(corr_df.index, fontsize=13, color="black")
            ax.set_xticklabels(corr_df.columns, fontsize=13, color="black", rotation=90)

            # Create grid lines between the tick labels
            ax.set_xticks(np.arange(len(corr_df.columns) + 1) - .5, minor=True)
            ax.set_yticks(np.arange(len(corr_df.index) + 1) - .5, minor=True)
            ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1)

            # Add rectangle around the grid
            rect = plt.Rectangle((-.5, -.5), len(corr_df.columns), len(corr_df.index), linewidth=2, edgecolor='lightgray', facecolor='none')
            ax.add_patch(rect)

            # Create squares with radius proportional to the absolute value of correlation
            norm = plt.Normalize(-1, 1)
            sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm_r')
            for i in range(len(corr_df.columns)):
                for j in range(len(corr_df.index)):
                    correlation = corr_df.iat[j, i]
                    color = sm.to_rgba(correlation)
                    size = abs(correlation) * 1  # Adjust size factor as needed
                    rect = Rectangle(xy=(i - size / 2, j - size / 2), width=size, height=size, facecolor=color)
                    ax.add_patch(rect)

            # Add color bar
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.04)
            cbar.set_label('Correlation', fontsize=12)
            plt.title(f'Correlation Matrix for {matrix_type} {escenario}', fontsize=16)

            plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
            plt.close()
        else:
            print(f"File not found: {file_path}")

