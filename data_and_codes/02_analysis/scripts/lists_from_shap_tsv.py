import numpy as np
import pandas as pd
import glob

# Define the input directory
input_dir = '../../../out_results/out_shap_values'
files = glob.glob(f'{input_dir}/shap_*_best_tfs.tsv')

# Initialize an empty list to store DataFrames
dfs = []

# Process each file
for file in files:
    # Read the file
    df = pd.read_csv(file, sep='\t', index_col=0)
    
    # Convert the features list into a comma-separated string
    df['list_top_tfs'] = df.apply(lambda row: ', '.join(row.sort_values(ascending=False).index.tolist()[:5]), axis=1)
    
    # Extract matrix_type and target_variable from the index
    df['matrix_type'] = df.index.map(lambda x: x.split(' -> ')[0])
    df['target_variable'] = df.index.map(lambda x: x.split(' -> ')[1])
    
    # Select and order columns
    ordered_df = df[['matrix_type', 'target_variable', 'list_top_tfs']]
    
    # Append the ordered DataFrame to the list
    dfs.append(ordered_df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs)

# Save the concatenated DataFrame to a tsv file
final_df.to_csv(f'{input_dir}/list_from_shap.tsv', sep='\t', index=False)
