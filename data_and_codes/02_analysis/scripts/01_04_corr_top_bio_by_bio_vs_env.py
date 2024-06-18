import pandas as pd
import os

# Define input and output directories
input_dir = '../../../out_results/out_correlation/correlation_bio_env'
output_dir = '../../../out_results/out_correlation/corr_metrics'
os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

# Output filename
output_summary_file = 'corr_top8_bio_by_bio_vs_env.tsv'

summary_data = []

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Filter files to process only .tsv files that contain 'heatmap' in the filename
    if filename.endswith(".tsv") and 'heatmap' in filename:
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        
        # Calculate the absolute sums of each row to determine the top 5 rows
        abs_sums = df.abs().sum(axis=1)
        top_rows = abs_sums.sort_values(ascending=False).index[:8]  # Get the top 5 correlated row names
        
        top_8_info = []
        for row in top_rows:
            # Find the maximum absolute value in the row and keep the original sign
            row_data = df.loc[row]
            max_val = row_data.max() if abs(row_data.max()) > abs(row_data.min()) else row_data.min()
            # Compute the average of the top 15 most correlated columns by absolute value
            avg_top_5 = row_data.abs().nlargest(5).mean()
            
            # Format row data with max and average values
            top_8_info.append(f"{row} (max: {max_val:.2f}, avg5: {avg_top_5:.2f})")
        
        # Extract matrix and subsample identifiers from the filename
        parts = filename.split('_')
        matrix = parts[4]
        subsample = parts[5].split('.')[0]
        
        # Append the results to the summary data list
        matrix_subsample = f"{matrix}_{subsample}"
        summary_data.append([matrix_subsample] + top_8_info)

# Convert summary data to DataFrame
summary_df = pd.DataFrame(summary_data, columns=['Matrix_subsample', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7', 'Top8'])
summary_df = summary_df.sort_values(by='Matrix_subsample')  # Sort DataFrame by 'Matrix_subsample'
summary_df.to_csv(os.path.join(output_dir, output_summary_file), sep='\t', index=False)
