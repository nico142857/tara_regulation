#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from tqdm import tqdm


# # Upload metadata

# In[7]:


md = pd.read_csv('../../00_matrices/metadata.tsv', sep='\t', index_col=0)
md.head()


# # Bins of environmental variables

# In[10]:


bins_temperature = [-np.inf, 10, 22, np.inf]
labels_temperature = ['<10', '10-22', '>22']
md['Temperature_binned'] = pd.cut(md['Temperature'], bins=bins_temperature, labels=labels_temperature)

bins_oxygen = [-np.inf, 185, 250, np.inf]
labels_oxygen = ['<185', '185-250', '>250']
md['Oxygen_binned'] = pd.cut(md['Oxygen'], bins=bins_oxygen, labels=labels_oxygen)

md['ChlorophyllA_binned'] = np.where(md['ChlorophyllA'] <= 0.28, '<=0.28', '>0.28')

md['Fluorescence_binned'] = np.where(md['Fluorescence'] <= 2.3, '<=2.3', '>2.3')

bins_salinity = [-np.inf, 34, 37, np.inf]
labels_salinity = ['<=34', '34-37', '>37']
md['Salinity_binned'] = pd.cut(md['Salinity'], bins=bins_salinity, labels=labels_salinity)

md['NO3_binned'] = np.where(md['NO3'] <= 7, '<=7', '>7')

bins_flux = [-np.inf, 0.7, 3, np.inf]
labels_flux = ['<=0.7', '0.7-3', '>3']
md['Mean_Flux_150m_binned'] = pd.cut(md['Mean Flux at 150m'], bins=bins_flux, labels=labels_flux)

bins_npp = [-np.inf, 275, 540, np.inf]
labels_npp = ['<=275', '275-540', '>540']
md['NPP_binned'] = pd.cut(md['NPP 8d VGPM (mgC/m2/day)'], bins=bins_npp, labels=labels_npp)


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


matrices = ['Matrix_MX_all', 'Matrix_M1_all', 'Matrix_M0_all', 'Matrix_guidi_all', 'Matrix_salazar_all', 'Matrix_stress_all',
           'Matrix_GEN_M4_all', 'Matrix_GEN_M0_all', 'Matrix_GEN_guidi_all', 'Matrix_GEN_salazar_all', 'Matrix_GEN_stress_all'
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
        X = matrix
        y = label_encoder.fit_transform(md[variable])  # Encode labels
         
        # Store f1 scores for each cycle
        f1_scores = []

        for cycle in tqdm(range(num_cycles), desc='Simulation cycles', leave=False):
            # Split the data into training and testing sets with a different random state each time
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=cycle)

            # Create and train the model
            model = XGBClassifier(n_estimators=250)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(score)

        # Compute average F1 score over all cycles
        avg_f1_score = np.mean(f1_scores)
        results.append(('_'.join(matrix_name.split('_')[1:]), variable, avg_f1_score))

# Filename
output_file = 'initial_prediction_tf_vs_gen'
# Directory
out_dir = '../../../out_results/out_initial_predictions/'
        
df_results = pd.DataFrame(results, columns=['matrix_type', 'target_variable', 'f1_score'])
df_results.to_csv(out_dir+output_file+'.tsv', sep='\t', index=False)

# Display the results
for result in results:
    print(f'Matrix: {result[0]}, Variable: {result[1]}, Average F1 Score over {num_cycles} cycles: {result[2]:.4f}')
