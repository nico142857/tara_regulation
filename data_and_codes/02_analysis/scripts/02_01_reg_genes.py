import pandas as pd
import os
import numpy as np

# Step 1: Load the dataframe
input_list_path = '../../../out_results/out_shap_values/list_from_shap_correlated_env.tsv'
input_matrices_dir = '../../00_matrices/'
output_dir = '../../../out_results/out_reg_genes'

df = pd.read_csv(input_list_path, sep='\t')

# Create a function to process the TFs from a column
def process_tfs(tfs_column):
    results = []
    for index, row in df.iterrows():
        matrix_type = row['matrix_type']
        # Process only matrix types M0 and M1
        if matrix_type in ['M0', 'M1']:
            if pd.notna(row[tfs_column]):  # Check column is non NaN
                subsample = row['subsample']
                target_variable = row['target_variable']
                matrix_info = f"{matrix_type}, {subsample}, {target_variable}"
                tfs = row[tfs_column].split(',') # TFs list is a string separated by ', '
                for tf in tfs:
                    tf = tf.strip()  # Remove extra spaces
                    # Find the regulated genes for the matrix type and TF
                    file_path = f"{input_matrices_dir}/reg_genes_{matrix_type}_by_TF/reg_genes_{matrix_type}_{tf}.tsv"
                    if os.path.exists(file_path):
                        # Read the gene matrix
                        matrix = pd.read_csv(file_path, sep='\t', index_col=0)
                        non_zero_counts = (matrix != 0).sum(axis=0) # Count non-zero values
                        sorted_genes = non_zero_counts.sort_values(ascending=False) # Sort from most regulated to least regulated
                        top_1_percent = sorted_genes.head(len(sorted_genes) // 100) # Find top 1% of regulated genes
                        results.append((matrix_info, tf, list(top_1_percent.index)))
                    else:
                        print(f"File not found {file_path}")
    return results

# Function to write results to a plain text file
def write_results_to_file(results, filename):
    full_path = os.path.join(output_dir, filename)
    with open(full_path, 'w') as file:
        for matrix_type, tf, genes in results:
            file.write(f"{matrix_type}\n")
            file.write(f"{tf}: {genes}\n")

# Example of how to process one of the columns
results_shap_top_tfs = process_tfs('shap_top_tfs')
results_shap_top_tfs_correlated = process_tfs('shap_top_tfs_correlated')
results_top_tfs_correlated = process_tfs('top_tfs_correlated_05')

write_results_to_file(results_shap_top_tfs, 'results_shap_top_tfs.txt')
write_results_to_file(results_shap_top_tfs_correlated, 'results_shap_top_tfs_correlated.txt')
write_results_to_file(results_top_tfs_correlated, 'results_top_tfs_correlated_05.txt')