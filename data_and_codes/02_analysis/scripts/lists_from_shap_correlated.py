import pandas as pd
import glob
import os

# Function to determine the subsample based on the feature category
def determine_subsample(feature_category):
    if feature_category in ['Polar', 'Temperature', 'Province']:
        return 'srf'
    elif feature_category in ['Layer', 'Layer2']:
        return 'nonpolar'
    elif feature_category in ['NO3', 'NPP', 'CarbonFlux']:
        return 'epi-nonpolar'

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
    target_variable = row['target_variable']
    list_top_tfs = row['list_top_tfs'].split(', ')

    subsample = determine_subsample(target_variable)
    correlation_file = f'corr_bio_heatmap_{matrix_type}_{subsample}.tsv'
    correlation_path = os.path.join(correlation_dir, correlation_file)

    # Load the correlation DataFrame
    if os.path.exists(correlation_path):
        corr_df = pd.read_csv(correlation_path, sep='\t', index_col=0)

        # Find the top 5 correlated TFs for each TF in list_top_tfs
        correlated_tfs = {}
        for tf in list_top_tfs:
            if tf in corr_df.index:
                # Sort by the absolute values of correlations, take the top 5
                top_correlations = corr_df.loc[tf].abs().sort_values(ascending=False).head(6)  # includes self-correlation
                top_correlations = top_correlations.drop(tf, errors='ignore')  # drop self-correlation if exists
                correlated_tfs[tf] = top_correlations.index.tolist()  # Correctly convert to list

        # Create a DataFrame for the current row and append to list
        row_df = pd.DataFrame([{
            'matrix_type': matrix_type,
            'target_variable': target_variable,
            'list_top_tfs': ', '.join(list_top_tfs),
            'correlated_tfs': str(correlated_tfs)
        }])
        dfs.append(row_df)

# Concatenate all DataFrame rows into a single DataFrame
results_df = pd.concat(dfs)

# Save the results DataFrame to a new TSV file
results_df.to_csv(output_file_path, sep='\t', index=False)
