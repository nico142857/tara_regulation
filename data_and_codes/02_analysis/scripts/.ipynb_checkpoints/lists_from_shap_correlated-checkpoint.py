import pandas as pd
import glob
import os

# Path to the file containing matrix type and features
input_list_path = '../../../out_results/out_shap_values/list_from_shap.txt'

# Directory where correlation files are stored
correlation_dir = '../../../out_results/out_correlation/correlation_bio'

# Output file path
output_file_path = '../../../out_results/out_shap_values/list_from_shap_correlated.txt'

# Function to determine the subsample based on the feature category
def determine_subsample(feature_category):
    if feature_category in ['Polar', 'Temperature', 'Province']:
        return 'srf'
    elif feature_category in ['Layer', 'Layer2']:
        return 'nonpolar'
    elif feature_category in ['NO3', 'NPP', 'CarbonFlux']:
        return 'epi-nonpolar'

# Read the list from shap file
with open(input_list_path, 'r') as file:
    lines = file.readlines()

# Output dictionary to hold results
output_results = {}

# Process each line
for line in lines:
    parts = line.split('->')
    matrix_type = parts[0].strip()
    feature_category = parts[1].split()[0].strip()
    features = eval(parts[1].split('[')[1].split(']')[0])

    # Determine the subsample
    subsample = determine_subsample(feature_category)

    # Path to the correlation file
    correlation_file_path = f'{correlation_dir}/corr_bio_heatmap_{matrix_type}_{subsample}.tsv'
    
    # Read the correlation file
    if os.path.exists(correlation_file_path):
        df = pd.read_csv(correlation_file_path, sep='\t', index_col=0)
        df_abs = df.abs()

        # Extract top 5 correlated features for each feature in the list
        result = {}
        for feature in features:
            if feature in df_abs.columns:
                top5 = df_abs[feature].nlargest(5).index.tolist()
                result[feature] = top5

        output_results[f'{matrix_type} -> {feature_category}'] = result

# Write the results to a file
with open(output_file_path, 'w') as output_file:
    for key, value in output_results.items():
        output_file.write(f'{key}:\n')
        for sub_key, sub_value in value.items():
            output_file.write(f'  {sub_key}: {sub_value}\n')
        output_file.write('\n')

print(f"Process finished and results are stored in '{output_file_path}'")
