import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# CLR implementation
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.
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

input_dir = '../../00_matrices'
output_dir = '../../../out_results/out_regressions'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=0)

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
escenarios = ['all', 'polar', 'nonpolar', 'srf', 'epi', 'epi-nonpolar', 'srf-nonpolar']
env_vars = ['Temperature', 'Oxygen', 'ChlorophyllA', 'Salinity', 'Carbon.total', 'NO2', 'NO3', 'PO4',
            'Ammonium.5m', 'Iron.5m', 'Alkalinity.total', 'CO3', 'HCO3', 'Si', 'Mean Flux at 150m', 'NPP 8d VGPM (mgC/m2/day)']

# Process each matrix type and scenario
for matrix in matrix_types:
    for scenario in escenarios:
        file_path = f'{input_dir}/Matrix_{matrix}_{scenario}.tsv'
        output_file_path = f'{output_dir}/R2_{matrix}_{scenario}'

        try:
            df = pd.read_csv(file_path, sep='\t', index_col=0)
            clr_df = clr_(df)
            md_aligned = md.loc[clr_df.index]

            # Prepare DataFrame to store R^2 values
            r2_df = pd.DataFrame(index=clr_df.columns, columns=env_vars)

            # Loop over each environmental variable and each component in clr_df
            for env_var in env_vars:
                if env_var in md_aligned.columns:
                    for comp in clr_df.columns:
                        combined = pd.concat([md_aligned[env_var], clr_df[comp]], axis=1).dropna()
                        if not combined.empty:
                            Y = combined[env_var]
                            X = sm.add_constant(combined[comp])  # Predictor with constant

                            model = sm.OLS(Y, X)
                            results = model.fit()

                            r2_df.loc[comp, env_var] = round(results.rsquared, 2)
                        else:
                            r2_df.loc[comp, env_var] = np.nan
                else:
                    print(f"{env_var} is not in the metadata DataFrame.")

            # Save the R^2 values to CSV
            r2_df.to_csv(f'{output_file_path}.tsv', sep='\t')

            # Generate and save a heatmap
            plt.figure(figsize=(20, 25))
            sns.heatmap(r2_df.astype(float), cmap='coolwarm', cbar=True)
            plt.title(f'R^2 Heatmap for {matrix} - {scenario} \n Regression fit')
            plt.xlabel('Environmental Variables')
            plt.ylabel('TFs')
            plt.savefig(f'{output_file_path}_heatmap.pdf', bbox_inches='tight')
            plt.close()

        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")