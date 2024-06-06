import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# CLR implementation
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (clr) normalization on a dataset.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for clr normalization.")
    if (data <= 0).any().any():
        data = data.replace(0, eps)
    gm = np.exp(data.apply(np.log).mean(axis=1))
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)
    return clr_data

input_dir = '../../00_matrices'
output_dir = '../../../out_results/out_regressions'
os.makedirs(output_dir, exist_ok=True)

md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=0)

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
escenarios = ['all', 'polar', 'nonpolar', 'srf', 'epi', 'epi-nonpolar']
env_vars = ['Temperature', 'Oxygen', 'ChlorophyllA', 'Salinity', 'Carbon.total', 'NO2', 'NO3', 'PO4',
            'Ammonium.5m', 'Iron.5m', 'Alkalinity.total', 'CO3', 'HCO3', 'Si', 'Mean Flux at 150m', 'NPP 8d VGPM (mgC/m2/day)']

for matrix in matrix_types:
    for scenario in escenarios:
        file_path = f'{input_dir}/Matrix_{matrix}_{scenario}.tsv'
        output_file_path = f'{output_dir}/R2_{matrix}_{scenario}'
        try:
            df = pd.read_csv(file_path, sep='\t', index_col=0)
            clr_df = clr_(df)
            md_aligned = md.loc[clr_df.index]
            r2_df = pd.DataFrame(index=clr_df.columns, columns=env_vars)

            for env_var in env_vars:
                if env_var in md_aligned.columns:
                    for comp in clr_df.columns:
                        combined = pd.concat([md_aligned[env_var], clr_df[comp]], axis=1).dropna()
                        if not combined.empty:
                            Y = combined[env_var]
                            X = sm.add_constant(combined[comp])
                            model = sm.OLS(Y, X)
                            results = model.fit()
                            r2_df.loc[comp, env_var] = round(results.rsquared, 2)
                        else:
                            r2_df.loc[comp, env_var] = np.nan

            r2_df.to_csv(f'{output_file_path}.tsv', sep='\t')

            # Generate and save a custom heatmap
            fig, ax = plt.subplots(figsize=(20, 25))
            ax.set_facecolor('white')
            norm = Normalize(vmin=0, vmax=1)
            cmap = plt.cm.viridis
            scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)

            background_matrix = np.ones((len(r2_df.index), len(r2_df.columns)))
            ax.imshow(background_matrix, cmap='gray_r', interpolation='nearest')

            ax.set_xticks(np.arange(len(r2_df.columns)))
            ax.set_yticks(np.arange(len(r2_df.index)))
            ax.set_xticklabels(r2_df.columns, fontsize=8, rotation=90)
            ax.set_yticklabels(r2_df.index, fontsize=8)

            ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=1)
            ax.set_xticks(np.arange(len(r2_df.columns) + 1) - .5, minor=True)
            ax.set_yticks(np.arange(len(r2_df.index) + 1) - .5, minor=True)

            for i in range(len(r2_df.columns)):
                for j in range(len(r2_df.index)):
                    r_squared = r2_df.iat[j, i]
                    if pd.notna(r_squared):
                        size = r_squared * 1
                        color = cmap(norm(r_squared))
                        rect = Rectangle((i - size / 2, j - size / 2), width=size, height=size, facecolor=color, edgecolor=color)
                        ax.add_patch(rect)

            cbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.5, aspect=20, pad=0.04)
            cbar.set_label('R^2')
            plt.title(f'R^2 Heatmap for {matrix} - {scenario} \n Regression fit')
            plt.xlabel('Environmental Variables')
            plt.ylabel('TFs')
            plt.savefig(f'{output_file_path}_heatmap.pdf', bbox_inches='tight')
            plt.close()

        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
