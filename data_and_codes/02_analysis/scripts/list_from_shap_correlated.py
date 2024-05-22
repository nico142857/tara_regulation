import numpy as np
import pandas as pd

import glob
import os

# CLR implementation
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.

    Parameters:
    data (pandas.DataFrame): A DataFrame with samples as rows and components as columns.

    Returns:
    pandas.DataFrame: A clr-normalized DataFrame.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for clr normalization.")

    # Add small amount to cells with a value of 0
    if (data <= 0).any().any():
        data = data.replace(0, eps)

    # Calculate the geometric mean of each row
    gm = np.exp(data.apply(np.log).mean(axis=1))

    # Perform clr transformation
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)

    return clr_data

input_dir = '../../../out_results/out_correlation/correlation_bio'
output_dir = '../../../out_results/out_shap_values'
output_file_path = '../../../out_results/out_shap_values/list_from_shap_correlated.txt'

correlation_files = glob.glob(f'{input_dir}/corr_bio_heatmap_*.tsv')

with open(output_file_path, 'w') as output_file:
	    for file_path in correlation_files:
        # Asumimos que los nombres de los archivos siguen el formato 'correlation_tf_{matrix_type}_{subsample}.tsv'
        parts = os.path.basename(file_path).split('_')
        matrix_type = parts[2]
        subsample = parts[3]

        # Leer el DataFrame
        df = pd.read_csv(file_path, sep='\t', index_col=0)

        # Calcular el valor absoluto de las correlaciones
        df_abs = df.abs()

        # Encontrar las 5 variables con mayor correlación absoluta para cada fila
        top_correlations = df_abs.apply(lambda row: row.nlargest(5).index.tolist(), axis=1)

        # Escribir los resultados en el archivo de salida
        output_file.write(f'{matrix_type} -> {subsample}\n')
        output_file.write(top_correlations.to_string())
        output_file.write('\n\n')