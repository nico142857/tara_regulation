import pandas as pd
import os

# Define input and output directories
input_dir = '../../../out_results/out_correlation/correlation_bio_env'
output_subdir = 'corr_metrics'
output_dir = os.path.join(input_dir, output_subdir)
os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

# Output filename
output_summary_file = 'corr_top5_env.tsv'

summary_data = []

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Filter files to process only .tsv files that contain 'heatmap' in the filename
    if filename.endswith(".tsv") and 'heatmap' in filename:
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        # Calculate the absolute sums of each column
        abs_sums = df.abs().sum()
        sorted_columns = abs_sums.sort_values(ascending=False).index # Sort values
        top_5_columns = sorted_columns[:5].tolist() # Get the top 5 correlated columns
        top_5_max_values = [df[col].max() for col in top_5_columns]  # Get max value from each column
        # Format column names with max values
        top_5_columns_with_values = [f"{col} (max: {max_val:.2f})" for col, max_val in zip(top_5_columns, top_5_max_values)]
        
        # Extract matrix and subsample identifiers from the filename
        parts = filename.split('_')
        matrix = parts[4]
        subsample = parts[5].split('.')[0]
        
        # Append the results to the summary data list
        matrix_subsample = f"{matrix}_{subsample}"
        summary_data.append([matrix_subsample] + top_5_columns_with_values)

# Summary data to DataFrame
summary_df = pd.DataFrame(summary_data, columns=['Matrix_subsample', 'Top1 (max, avg top 15)', 'Top2 (max, avg top 15)', 'Top3 (max, avg top 15)', 'Top4 (max, avg top 15)', 'Top5 (max, avg top 15)'])
summary_df = summary_df.sort_values(by='Matrix_subsample')
summary_df.to_csv(os.path.join(output_dir, output_summary_file), sep='\t', index=False)
