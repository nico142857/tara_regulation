import pandas as pd
import os

# Paths
correlation_dir = '../../../out_results/out_correlation/correlation_bio_env'
input_list_path = '../../../out_results/out_shap_values/list_from_shap_correlated.tsv'
output_file_path = '../../../out_results/out_shap_values/list_from_shap_correlated_env.tsv'

# Read list from shap correlated
shap_correlated_df = pd.read_csv(input_list_path, sep='\t')

# Initialize an empty list to store the top_tfs_correlated lists
top_tfs_correlated_lists = []

# Create a mapping for the column names
column_mapping = {
    'CarbonFlux': 'Mean Flux at 150m',
    'NPP': 'NPP 8d VGPM (mgC/m2/day)'
}

threshold = 0.5

# Iterate over each row in the DataFrame
for index, row in shap_correlated_df.iterrows():
    # Extract variables from the current row
    matrix_type = row['matrix_type']
    subsample = row['subsample']
    target_variable = row['target_variable']
    
    # Check if the target variable is one of the continuous variables
    if target_variable in ['Temperature', 'NO3', 'CarbonFlux', 'NPP']:
        # Map the target variable to the column name in the correlation file
        if target_variable in column_mapping:
            target_variable_name = column_mapping[target_variable]
        else:
            target_variable_name = target_variable
        
        # Read the correlation
        file_path = os.path.join(correlation_dir, f'corr_bio_env_heatmap_{matrix_type}_{subsample}.tsv')
        correlation_df = pd.read_csv(file_path, sep='\t', index_col=0)
        # Filter by abs correlation treshold
        filtered_indices = correlation_df[abs(correlation_df[target_variable_name]) >= threshold][target_variable_name].index.tolist()
        
        # Join the filtered indices into a comma-separated list
        lista = ', '.join(filtered_indices)
    else:
        # For non-continuous variables, leave the list empty
        lista = ''
    
    # Append the result to the list
    top_tfs_correlated_lists.append(lista)

# Save the updated DataFrame to a TSV file
shap_correlated_df[f'top_tfs_correlated(>{threshold})'] = top_tfs_correlated_lists
shap_correlated_df.to_csv(output_file_path, sep='\t', index=False)