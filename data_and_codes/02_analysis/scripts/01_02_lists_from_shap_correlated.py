import pandas as pd
import glob
import os

# Paths
input_list_path = '../../../out_results/out_shap_values/list_from_shap.tsv'
correlation_dir = '../../../out_results/out_correlation/correlation_bio'
output_file_path = '../../../out_results/out_shap_values/list_from_shap_correlated.tsv'

# Load the initial DataFrame
shap_df = pd.read_csv(input_list_path, sep='\t')

# Initialize a list to store DataFrames
dfs = []

# Iterate over each row in the DataFrame
for index, row in shap_df.iterrows():
    matrix_type = row['matrix_type']
    subsample = row['subsample']
    target_variable = row['target_variable']
    shap_top_tfs = row['shap_top_tfs'].split(', ')

    correlation_file = f'corr_bio_heatmap_{matrix_type}_{subsample}.tsv'
    correlation_path = os.path.join(correlation_dir, correlation_file)

    # Load the correlation DataFrame
    if os.path.exists(correlation_path):
        corr_df = pd.read_csv(correlation_path, sep='\t', index_col=0)

        # Collect all correlated TFs in a single list
        all_correlated_tfs = []
        for tf in shap_top_tfs:
            if tf in corr_df.index:
                # Sort by the absolute values of correlations, take the top 5, exclude self-correlation
                top_correlations = corr_df.loc[tf].abs().sort_values(ascending=False).head(6)
                #top_correlations = top_correlations.drop(tf, errors='ignore')
                all_correlated_tfs.extend(top_correlations.index.tolist())

        # Remove duplicates and convert to string
        #all_correlated_tfs = ', '.join(sorted(set(all_correlated_tfs)))

        # Create a DataFrame for the current row and append to list
        row_df = pd.DataFrame([{
            'matrix_type': matrix_type,
            'subsample': subsample,
            'target_variable': target_variable,
            'shap_top_tfs': ', '.join(shap_top_tfs),
            'shap_top_tfs_correlated': ', '.join(all_correlated_tfs)  # All correlated TFs as a single string
        }])
        dfs.append(row_df)

# Concatenate all DataFrame rows into a single DataFrame
results_df = pd.concat(dfs)

# Save the results DataFrame to a new TSV file
results_df.to_csv(output_file_path, sep='\t', index=False)
