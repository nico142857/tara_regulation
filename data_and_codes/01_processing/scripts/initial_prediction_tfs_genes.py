#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score

from tqdm import tqdm


# # Upload metadata

# In[7]:


md = pd.read_csv('../../00_matrices/metadata.tsv', sep='\t', index_col=0)
md.head()


# # Bins of environmental variables

# In[10]:

# Temperature
bins_temperature = [-np.inf, 10, 22, np.inf]
labels_temperature = ['<10', '10-22', '>22']
temp_series = md['Temperature'].dropna()  # Drop NaN values before binning
md.loc[temp_series.index, 'Temperature_binned'] = pd.cut(temp_series, bins=bins_temperature, labels=labels_temperature)

# Oxygen
bins_oxygen = [-np.inf, 185, 250, np.inf]
labels_oxygen = ['<185', '185-250', '>250']
oxygen_series = md['Oxygen'].dropna()
md.loc[oxygen_series.index, 'Oxygen_binned'] = pd.cut(oxygen_series, bins=bins_oxygen, labels=labels_oxygen)

# ChlorophyllA
chloro_series = md['ChlorophyllA'].dropna()
md.loc[chloro_series.index, 'ChlorophyllA_binned'] = np.where(chloro_series <= 0.28, '<=0.28', '>0.28')

# Fluorescence
fluor_series = md['Fluorescence'].dropna()
md.loc[fluor_series.index, 'Fluorescence_binned'] = np.where(fluor_series <= 2.3, '<=2.3', '>2.3')

# Salinity
bins_salinity = [-np.inf, 34, 37, np.inf]
labels_salinity = ['<=34', '34-37', '>37']
salinity_series = md['Salinity'].dropna()
md.loc[salinity_series.index, 'Salinity_binned'] = pd.cut(salinity_series, bins=bins_salinity, labels=labels_salinity)

# NO3
no3_series = md['NO3'].dropna()
md.loc[no3_series.index, 'NO3_binned'] = np.where(no3_series <= 7, '<=7', '>7')

# Mean Flux at 150m (Carbon Export)
bins_flux = [-np.inf, 0.7, 3, np.inf]
labels_flux = ['<=0.7', '0.7-3', '>3']
flux_series = md['Mean Flux at 150m'].dropna()
md.loc[flux_series.index, 'Mean_Flux_150m_binned'] = pd.cut(flux_series, bins=bins_flux, labels=labels_flux)

# NPP 8d VGPM (mgC/m2/day)
bins_npp = [-np.inf, 275, 540, np.inf]
labels_npp = ['<=275', '275-540', '>540']
npp_series = md['NPP 8d VGPM (mgC/m2/day)'].dropna()
md.loc[npp_series.index, 'NPP_binned'] = pd.cut(npp_series, bins=bins_npp, labels=labels_npp)

# # CLR implementation

# In[11]:


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


# # Model

# In[64]:


matrices = ['Matrix_MX_all', 'Matrix_M0_all', 'Matrix_M1_all', 'Matrix_guidi_all', 'Matrix_salazar_all', 'Matrix_stress_all',
           #'Matrix_GEN_M4_all', 'Matrix_GEN_M0_all','Matrix_GEN_M1_all', 'Matrix_GEN_guidi_all', 'Matrix_GEN_salazar_all', 'Matrix_GEN_stress_all'
           ]
variables = ['polar', 'Layer', 'Layer2', 'Province', 'Temperature_binned', 'Oxygen_binned', 'ChlorophyllA_binned', 'Fluorescence_binned', 'Salinity_binned', 'NO3_binned', 'Mean_Flux_150m_binned', 'NPP_binned']
#variables = ['Temperature_binned', 'Oxygen_binned']
num_cycles = 100  # Number of cycles to run

results = []

# Label encoder instance
label_encoder = LabelEncoder()

# Iterate over each matrix and variable to train the model
for matrix_name in tqdm(matrices, desc='Processing matrices'):
    matrix = pd.read_csv(f'../../00_matrices/{matrix_name}.tsv', sep='\t', index_col=0)
    matrix = clr_(matrix)  # Apply clr transformation

    for variable in tqdm(variables, desc=f'Variables in {matrix_name}', leave=False):
        valid_indices = md[variable].dropna().index
        X = matrix.loc[valid_indices]
        y = label_encoder.fit_transform(md.loc[valid_indices, variable]) # Encode labels

        # Store scores for each cycle
        scores = []

        for cycle in tqdm(range(num_cycles), desc='Simulation cycles', leave=False):
            # Split the data into training and testing sets with a different random state each time
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=cycle, stratify=y)

            # Create and train the model
            model = XGBClassifier(n_estimators=250)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)

            # Calculate metrics
            scores.append({
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted')
            })

        # Compute average scores over all cycles
        avg_scores = {key: np.mean([score[key] for score in scores]) for key in scores[0]}
        results.append({'matrix_type': '_'.join(matrix_name.split('_')[1:]), 'variable': variable, **avg_scores})

# Save results
output_file = 'initial_prediction_tf_vs_gen_new'
out_dir = '../../../out_results/out_initial_predictions/'
os.makedirs(out_dir, exist_ok=True)

df_results = pd.DataFrame(results)
df_results.to_csv(f'{out_dir}{output_file}.tsv', sep='\t', index=False)

# Display the results
for result in results:
    print(f"Matrix: {result['matrix_type']}, Variable: {result['variable']}, "
          f"Average F1 Score: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}, "
          f"Recall: {result['recall']:.4f}, Precision: {result['precision']:.4f} over {num_cycles} cycles")

