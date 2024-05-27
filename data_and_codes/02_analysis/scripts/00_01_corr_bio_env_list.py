import pandas as pd
import os

input_dir = '../../../out_results/out_correlation/correlation_bio_env'
output_subdir = 'sorted_corr_list'
output_dir = os.path.join(input_dir, output_subdir)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tsv") and 'heatmap' in filename:
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        cells = []
        
        # Use index (TFs) and column (env variables) from the DataFrame
        for col_name in df.columns:
            for row_name in df.index:
                value = df.at[row_name, col_name]
                # Store absolute value (for sorting) and actual value (for output)
                cells.append((abs(value), value, row_name, col_name))
                
        cells_sorted = sorted(cells, reverse=True, key=lambda x: x[0]) # Sort cells based on the absolute value in descending order

        # Extract matrix_type and subsample from filename
        parts = filename.split('_')
        matrix_type = parts[4]
        subsample = parts[5].replace('.tsv', '')  # Removing the file extension

        # Define output file name based on the input file name
        output_filename = f'sorted_corr_list_{matrix_type}_{subsample}.txt'
        output_file = os.path.join(output_dir, output_filename)

        # Write the results to a file
        with open(output_file, 'w') as f:
            for _, actual_value, row_name, col_name in cells_sorted:
                f.write(f'({row_name}, {col_name}): {actual_value}\n')

        print(f"Results written to {output_file}")