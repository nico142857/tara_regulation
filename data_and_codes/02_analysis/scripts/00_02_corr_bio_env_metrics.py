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
        
        # Calculate the absolute sums of each column to determine the top 5 columns
        abs_sums = df.abs().sum()
        sorted_columns = abs_sums.sort_values(ascending=False).index[:5]  # Get the top 5 correlated column names
        
        top_5_info = []
        for col in sorted_columns:
            max_val = df[col].max()  # Maximum correlation value
            # Compute the average of the top 15 most correlated rows by absolute value
            avg_top_15 = df[col].abs().nlargest(15).mean()
            
            # Format column data with max and average values
            top_5_info.append(f"{col} (max: {max_val:.2f}, avg15: {avg_top_15:.2f})")
        
        # Extract matrix and subsample identifiers from the filename
        parts = filename.split('_')
        matrix = parts[4]
        subsample = parts[5].split('.')[0]
        
        # Append the results to the summary data list
        matrix_subsample = f"{matrix}_{subsample}"
        summary_data.append([matrix_subsample] + top_5_info)

# Convert summary data to DataFrame
summary_df = pd.DataFrame(summary_data, columns=['Matrix_subsample', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5'])
summary_df = summary_df.sort_values(by='Matrix_subsample')  # Sort DataFrame by 'Matrix_subsample'

# Save the DataFrame to a .tsv file in the output directory
summary_df.to_csv(os.path.join(output_dir, output_summary_file), sep='\t', index=False)
