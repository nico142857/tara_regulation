import pandas as pd
import os

# Define input and output directories
input_dir = '../../../out_results/out_correlation/correlation_bio_env'
output_dir = '../../../out_results/out_correlation/corr_metrics'
os.makedirs(output_dir, exist_ok=True)

matrix_types = ['MX', 'M0', 'M1']
subsamples = ['srf']
key_env_var = ["Temperature", "Oxygen", "ChlorophyllA", "Carbon.total", "Fluorescence"]

for matrix in matrix_types:
    for subsample in subsamples:
        file_path = os.path.join(input_dir, f'corr_bio_env_heatmap_{matrix}_{subsample}.tsv')
        df_corr = pd.read_csv(file_path, sep='\t', index_col=0)
        df_corr['abs_sum'] = df_corr[key_env_var].abs().sum(axis=1) # Calculate the absolute sum across key columns
        sorted_df = df_corr.sort_values(by='abs_sum', ascending=False) # Sort descending order
        top_25_df = sorted_df.head(25)
        #top_25_df['Alphabetical_Index'] = sorted(top_25_df.index)
        
        output_summary_file = f'corr_top_bio_by_abs_sum_{matrix}_{subsample}.tsv'
        output_path = os.path.join(output_dir, output_summary_file)
        
        top_25_df.to_csv(output_path, sep='\t', columns=['abs_sum'])
        
        print(f"Results for matrix {matrix}_{subsample} have been exported to {output_path}")
