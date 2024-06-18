import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# CLR implementation (from your provided code)
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
input_dir_r2 = '../../../out_results/out_regressions/'

output_dir = '../../../out_results/out_regressions'
os.makedirs(output_dir, exist_ok=True)

# Load metadata for joining
md = pd.read_csv(f'{input_dir_matrices}/metadata.tsv', sep='\t', index_col=0)

# Iterate over each file in the directory
for filename in os.listdir(input_dir_r2):
    if filename.startswith("R2_") and filename.endswith(".tsv"):
        matrix_type, subsample = filename.split("_")[1], filename.split("_")[2].replace(".tsv", "")
        r2_df = pd.read_csv(os.path.join(input_dir_r2, filename), sep='\t', index_col=0)

        # Flatten the DataFrame and filter pairs with R2 > 0.5
        high_r2_pairs = [(row, col) for row, col in (r2_df[r2_df > 0.5]).stack().index]
        
        # Load the corresponding matrix file
        matrix_filename = f'Matrix_{matrix_type}_{subsample}.tsv'
        matrix_filepath = os.path.join(input_dir_matrices, matrix_filename)
        df = pd.read_csv(matrix_filepath, sep='\t', index_col=0)

        clr_df = clr_(df)
        aligned_md = md.loc[clr_df.index]
        
        # Generate jointplots for each pair
        for (row, column) in high_r2_pairs:
            if row in clr_df.columns and column in aligned_md.columns:
                sns.regplot(
                    x=aligned_md[column],
                    y=clr_df[row],
                    scatter_kws={'alpha': 0.8},
                    line_kws={'color': 'red', 'alpha' : 0.5}
                )
                plt.title(f'Scatter Plot for {matrix_type}, {subsample}: {row} vs {column}')
                plt.xlabel(column)
                plt.ylabel(row)
                output_file_path = os.path.join(output_dir, f'Regfit_{matrix_type}_{subsample}_{row}_{column}.pdf')
                plt.savefig(output_file_path, bbox_inches='tight')
                plt.close()
            else:
                print(f"Column {row} not found in clr_df or column {column} not found in md")
