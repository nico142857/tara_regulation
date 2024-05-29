#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import xgboost as xgb # Extreme Gradient Boosting Algorithm
import pickle # Saving the model
import json # json file w/r

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

import shap # Shap values

import os


# # Upload data

# In[2]:

input_dir_matrices = '../../00_matrices'
input_dir_models = '../../../out_results/out_xgb_models'
output_dir = '../../../out_results/out_shap_values'

# metadata
md = pd.read_csv(f'{input_dir_matrices}/metadata.tsv', sep='\t', index_col=[0])

# Define mappings for each cat 
layer_mapping = {'SRF': 0, 'DCM': 1, 'MES': 2}
layer2_mapping = {'EPI': 0, 'MES': 1}
polar_mapping = {'non polar': 0, 'polar': 1}
#province_mapping = {'B2': 0, 'B3': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5}

# Copy of metadata to encode
md_encoded = md.copy()
# Map categorical values
md_encoded['Layer'] = md_encoded['Layer'].map(layer_mapping)
md_encoded['Layer2'] = md_encoded['Layer2'].map(layer2_mapping)
md_encoded['polar'] = md_encoded['polar'].map(polar_mapping)
#md_encoded['Province'] = md_encoded['Province'].map(province_mapping)


# # CLR implementation

# In[3]:


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


# # SHAP (SRF) polar / non polar

# In[4]:


private_list = ['TSC021', # TARA_031 non polar
                'TSC060', # TARA_042 non polar
                'TSC085', # TARA_065 non polar
                'TSC102', # TARA_068 non polar
                'TSC141', # TARA_084 polar south
                'TSC145', # TARA_085 polar south
                'TSC173', # TARA_112 non polar
                'TSC213', # TARA_132 non polar
                'TSC216', # TARA_133 non polar
                'TSC237', # TARA_150 non polar
                'TSC261', # TARA_189 polar
                'TSC276', # TARA_208 polar
                'TSC285', # TARA_163 polar
               ] 
# 8 non polar, 5 polar

def shap_polar_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_srf.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_srf_polar'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices
    y_total = aligned_md['polar']  # POLAR (0/1)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])

    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]

    # Reverse polar mapping, assuming polar_mapping is predefined
    reverse_polar_mapping = {v: k for k, v in polar_mapping.items()}

    # Create DataFrame for easier plotting
    feature_names = clr_df.columns
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    class_names = ['Class ' + reverse_polar_mapping[label] for label in unique_labels]
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)

    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)

    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)

    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)

    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_polar = {matrix_type: shap_polar_(matrix_type) for matrix_type in matrix_types}
results_polar_total = {matrix_type: shap_polar_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_polar_total


# # SHAP (SRF) temperature

# In[5]:


private_list = ['TSC021', # TARA_031 non polar, HIGH
                'TSC060', # TARA_042 non polar, HIGH
                'TSC085', # TARA_065 non polar, MID
                'TSC102', # TARA_068 non polar, MID
                'TSC141', # TARA_084 polar south, LOW
                'TSC145', # TARA_085 polar south, LOW
                'TSC173', # TARA_112 non polar, HIGH
                'TSC213', # TARA_132 non polar, HIGH
                'TSC216', # TARA_133 non polar, MID
                'TSC237', # TARA_150 non polar, MID
                'TSC261', # TARA_189 polar, LOW
                'TSC276', # TARA_208 polar, LOW
                'TSC285', # TARA_163 polar, LOW
               ]
# 4 HIGH, 4 MID, 5 LOW

def shap_temperature_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_srf.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_srf_temperature'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices

    # Create the target variable
    continuous_y = aligned_md['Temperature'].copy()

    # Drop NaN values from continuous_y and corresponding rows in clr_df
    valid_indices = continuous_y.dropna().index
    continuous_y = continuous_y.loc[valid_indices]
    clr_df = clr_df.loc[valid_indices]

    # Bin the Temperature to obtain a binary target variable
    y_total = continuous_y.apply(lambda temp: 0 if temp < 10 else (1 if temp <= 22 else 2))
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])
    
    temperature_bin_labels = {
        0: 'Class <10°C',
        1: 'Class 10-22°C',
        2: 'Class >22°C'
    }
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Creating a DataFrame for easier plotting with the correct class names
    feature_names = clr_df.columns
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    class_names = [temperature_bin_labels[label] for label in unique_labels]
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_temperature = {matrix_type: shap_temperature_(matrix_type) for matrix_type in matrix_types}
results_temperature_total = {matrix_type: shap_temperature_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_temperature_total


# # SHAP (SRF) province

# In[6]:


private_list = ['TSC021', # TARA_031 non polar
                'TSC060', # TARA_042 non polar
                'TSC085', # TARA_065 non polar
                'TSC102', # TARA_068 non polar
                #'TSC141', # TARA_084 polar south
                'TSC145', # TARA_085 polar south
                'TSC173', # TARA_112 non polar
                'TSC184', # TARA_122 non polar
                'TSC213', # TARA_132 non polar
                'TSC216', # TARA_133 non polar
                'TSC237', # TARA_150 non polar
                'TSC261', # TARA_189 polar
                'TSC276', # TARA_208 polar
                #'TSC285', # TARA_163 polar
               ] 

def shap_province_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_srf.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_srf_province'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices
    y_total = aligned_md['Province']

    # Extract and sort the unique labels
    unique_labels = sorted(y_total.unique())
    le = LabelEncoder()
    le.fit(unique_labels)
    y_encoded = le.transform(y_total)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])  # clr_df.loc[private_list] should be your feature matrix
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Creating a DataFrame for easier plotting
    feature_names = clr_df.columns  # Ensure this is the correct list of features
    
    # Correct class names for the model
    model_classes = model.classes_
    original_labels = le.inverse_transform(model_classes)
    class_names = ['Class ' + label for label in original_labels]
    
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_temperature = {matrix_type: shap_province_(matrix_type) for matrix_type in matrix_types}
results_province_total = {matrix_type: shap_province_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_province_total


# # SHAP (Non Polar) layer

# In[7]:


private_list = ['TSC021', # TARA_031 SRF
                
                'TSC058', # TARA_042 DCM
                'TSC060', # TARA_042 SRF
                
                'TSC081', # TARA_065 DCM
                'TSC083', # TARA_065 MES
                'TSC085', # TARA_065 SRF
                
                'TSC096', # TARA_068 DCM
                'TSC099', # TARA_068 MES
                'TSC102', # TARA_068 SRF
                
                #'TSC142', # TARA_085 DCM
                #'TSC144', # TARA_085 MES
                #'TSC145', # TARA_085 SRF
                
                'TSC171', # TARA_112 DCM
                'TSC172', # TARA_112 MES
                'TSC173', # TARA_112 SRF
                
                'TSC211', # TARA_132 DCM
                'TSC212', # TARA_132 MES
                'TSC213', # TARA_132 SRF
                
                'TSC214', # TARA_133 DCM
                'TSC215', # TARA_133 MES
                'TSC216', # TARA_133 SRF
                
                'TSC236', # TARA_150 DCM
                'TSC237', # TARA_150 SRF
                
                #'TSC261', # TARA_189 SRF
                #'TSC262', # TARA_189 DCM
                #'TSC263', # TARA_189 MES
               ] 
# 8 SRF, 7 DCM, 5 MES

def shap_layer_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_nonpolar.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_nonpolar_layer'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices
    y_total = aligned_md['Layer']
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])  # clr_df.loc[private_list] should be your feature matrix
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Reverse layer mapping to get the original labels
    reverse_layer_mapping = {v: k for k, v in layer_mapping.items()}
    
    # Creating a DataFrame for easier plotting
    feature_names = clr_df.columns
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    # Update class names using the reverse mapping
    class_names = ['Class ' + reverse_layer_mapping[label] for label in unique_labels]
    
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_temperature = {matrix_type: shap_layer_(matrix_type) for matrix_type in matrix_types}
results_layer_total = {matrix_type: shap_layer_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_layer_total


# # SHAP (Non Polar) layer2

# In[8]:


private_list = ['TSC021', # TARA_031 SRF
                
                'TSC058', # TARA_042 DCM
                'TSC060', # TARA_042 SRF
                
                'TSC081', # TARA_065 DCM
                'TSC083', # TARA_065 MES
                'TSC085', # TARA_065 SRF
                
                'TSC096', # TARA_068 DCM
                'TSC099', # TARA_068 MES
                'TSC102', # TARA_068 SRF
                
                #'TSC142', # TARA_085 DCM
                #'TSC144', # TARA_085 MES
                #'TSC145', # TARA_085 SRF
                
                'TSC171', # TARA_112 DCM
                'TSC172', # TARA_112 MES
                'TSC173', # TARA_112 SRF
                
                'TSC211', # TARA_132 DCM
                'TSC212', # TARA_132 MES
                'TSC213', # TARA_132 SRF
                
                'TSC214', # TARA_133 DCM
                'TSC215', # TARA_133 MES
                'TSC216', # TARA_133 SRF
                
                'TSC236', # TARA_150 DCM
                'TSC237', # TARA_150 SRF
                
                #'TSC261', # TARA_189 SRF
                #'TSC262', # TARA_189 DCM
                #'TSC263', # TARA_189 MES
               ] 
# 8 SRF, 7 DCM, 5 MES

def shap_layer2_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_nonpolar.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_nonpolar_layer2'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices
    y_total = aligned_md['Layer2']
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])  # clr_df.loc[private_list] should be your feature matrix
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Reverse layer mapping to get the original labels
    reverse_layer2_mapping = {v: k for k, v in layer2_mapping.items()}
    
    # Creating a DataFrame for easier plotting
    feature_names = clr_df.columns
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    # Update class names using the reverse mapping
    class_names = ['Class ' + reverse_layer2_mapping[label] for label in unique_labels]
    
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_temperature = {matrix_type: shap_layer_(matrix_type) for matrix_type in matrix_types}
results_layer2_total = {matrix_type: shap_layer2_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_layer2_total


# # SHAP (EPI - NonPolar) NO3

# In[9]:


private_list = ['TSC018',
                'TSC068',
                'TSC126',
                'TSC184',
                'TSC233',
                
    
                'TSC138',
                'TSC159',
                'TSC217',
               ]

def shap_no3_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_epi-nonpolar.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_epi-nonpolar_no3'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]

    # Create the target variable
    continuous_y = aligned_md['NO3'].copy()

    # Drop NaN values from continuous_y and corresponding rows in clr_df
    valid_indices = continuous_y.dropna().index
    continuous_y = continuous_y.loc[valid_indices]
    clr_df = clr_df.loc[valid_indices]

    # Bin the NO3 to obtain a binary target variable
    y_total = continuous_y.apply(lambda temp: 0 if temp < 7 else 1)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])  # clr_df.loc[private_list] is the predictor (abundance of TFs)
    
    no3_bin_labels = {
    0: 'Class <=7 NO3',
    1: 'Class > NO3'
    }
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Creating a DataFrame for easier plotting
    feature_names = clr_df.columns  # Ensure this is the correct list of features
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    class_names = [no3_bin_labels[label] for label in unique_labels]
    
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_temperature = {matrix_type: shap_no3_(matrix_type) for matrix_type in matrix_types}
results_no3_total = {matrix_type: shap_no3_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_no3_total


# # SHAP (EPI - NonPolar) NPP

# In[10]:


private_list = ['TSC021', # TARA_031 SRF
                
                'TSC058', # TARA_042 DCM
                'TSC060', # TARA_042 SRF
                
                'TSC081', # TARA_065 DCM
                'TSC085', # TARA_065 SRF
                
                'TSC096', # TARA_068 DCM
                'TSC102', # TARA_068 SRF
                
                'TSC171', # TARA_112 DCM
                'TSC173', # TARA_112 SRF
                
                'TSC211', # TARA_132 DCM
                'TSC213', # TARA_132 SRF
                
                'TSC214', # TARA_133 DCM
                'TSC216', # TARA_133 SRF
                
                'TSC236', # TARA_150 DCM
                'TSC237', # TARA_150 SRF
               ] 

def shap_npp_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_epi-nonpolar.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_epi-nonpolar_npp'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices

    # Create the target variable
    continuous_y = aligned_md['NPP 8d VGPM (mgC/m2/day)'].copy()

    # Drop NaN values from continuous_y and corresponding rows in clr_df
    valid_indices = continuous_y.dropna().index
    continuous_y = continuous_y.loc[valid_indices]
    clr_df = clr_df.loc[valid_indices]

    # Bin the NPP to obtain a binary target variable
    y_total = continuous_y.apply(lambda temp: 0 if temp < 275 else (1 if temp <= 540 else 2))
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])  # clr_df.loc[private_list] should be your feature matrix
        
    npp_bin_labels = {
        0: 'Class <275 NPP',
        1: 'Class 275-540 NPP',
        2: 'Class >540 NPP'
    }
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Creating a DataFrame for easier plotting
    feature_names = clr_df.columns  # Ensure this is the correct list of features
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    class_names = [npp_bin_labels[label] for label in unique_labels]
    
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_npp = {matrix_type: shap_npp_(matrix_type) for matrix_type in matrix_types}
results_npp_total = {matrix_type: shap_npp_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_npp_total


# # SHAP (EPI - NonPolar) Carbon FLUX

# In[11]:


private_list = ['TSC058', # TARA_042 DCM LOW
                'TSC060', # TARA_042 SRF LOA
                
                'TSC085', # TARA_065 SRF MID
                
                'TSC096', # TARA_068 DCM LOW
                'TSC102', # TARA_068 SRF LOW

                'TSC191', # TARA_123 SRF HIGH
                
                'TSC214', # TARA_133 DCM MID
                'TSC216', # TARA_133 SRF HIGH

                'TSC224', # TARA_141 SRF HIGH

                'TSC229', # TARA_145 SRF MID
                
                'TSC236', # TARA_150 DCM LOW
                'TSC237', # TARA_150 SRF LOW
               ]

def shap_cflux_(matrix_type):
    # Load the matrix data
    df = pd.read_csv(f'{input_dir_matrices}/Matrix_{matrix_type}_epi-nonpolar.tsv', sep='\t', index_col=[0])
    
    # Load the corresponding model
    filename = f'model_{matrix_type}_epi-nonpolar_cflux'
    with open(f'{input_dir_models}/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # clr normalization
    clr_df = clr_(df)
    aligned_md = md_encoded.loc[clr_df.index]

    # Create the target variable
    continuous_y = aligned_md['Mean Flux at 150m'].copy()

    # Drop NaN values from continuous_y and corresponding rows in clr_df
    valid_indices = continuous_y.dropna().index
    continuous_y = continuous_y.loc[valid_indices]
    clr_df = clr_df.loc[valid_indices]

    # Bin the CarbonExport to obtain a binary target variable
    y_total = continuous_y.apply(lambda temp: 0 if temp < 0.7 else (1 if temp <= 3 else 2))
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(clr_df.loc[private_list])  # clr_df.loc[private_list] should be your feature matrix
        
    cflux_bin_labels = {
        0: 'Class <0.7 Mean Flux',
        1: 'Class 0.7-3 Mean Flux',
        2: 'Class >3 Mean Flux'
    }
    
    # Calculate mean SHAP values for each feature for each class
    mean_shap_values_per_class = [np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))]
    
    # Creating a DataFrame for easier plotting
    feature_names = clr_df.columns  # Ensure this is the correct list of features
    unique_labels = y_total.unique()
    unique_labels.sort()  # Sort the labels if they are not in order
    class_names = [cflux_bin_labels[label] for label in unique_labels]
    
    shap_df = pd.DataFrame(np.array(mean_shap_values_per_class).T, columns=class_names, index=feature_names)
    
    # Calculate total mean SHAP value for each feature
    shap_df['Total Mean SHAP'] = shap_df.sum(axis=1)
    
    # Sort the DataFrame based on Total Mean SHAP value
    shap_df_sorted = shap_df.sort_values(by='Total Mean SHAP', ascending=True)
    
    # Select only the top n features
    top_n = 15
    shap_df_top = shap_df_sorted.tail(top_n)
    
    # Drop the total column for plotting the top n features
    shap_df_top = shap_df_top.drop(columns=['Total Mean SHAP'])
    
    return shap_df_top

# Usage example for different matrix types
matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']
#results_cflux = {matrix_type: shap_cflux_(matrix_type) for matrix_type in matrix_types}
results_cflux_total = {matrix_type: shap_cflux_(matrix_type).sum(axis=1) for matrix_type in matrix_types}
#results_cflux_total


# # Visualization

# In[36]:


results_dict = {
    'results_polar_total': "Polar",
    'results_temperature_total': "Temperature",
    'results_province_total': "Province",
    'results_layer_total': "Layer",
    'results_layer2_total': "Layer2",
    'results_no3_total': "NO3",
    'results_cflux_total': "CarbonFlux",
    'results_npp_total': "NPP"
}

matrix_types = ['MX', 'M0', 'M1', 'guidi', 'salazar', 'stress']  # List all matrix types you have
data_tests = {f"data_test_{type}": {} for type in matrix_types}

for dict_name, analysis_type in results_dict.items():
    current_dict = eval(dict_name)  # Get the dictionary
    for matrix_type, series in current_dict.items():
        if matrix_type in matrix_types:
            key = f"{matrix_type} -> {analysis_type}"
            data_tests[f"data_test_{matrix_type}"][key] = series

def plot_shap_clustermap(matrix_type, data_tests, feature_subset):
    # Convert data_test for the current matrix type to DataFrame and reindex
    shap_summary_df = pd.DataFrame(data_tests[f'data_test_{matrix_type}']).reindex(feature_subset, axis=0).T
    shap_summary_df.to_csv(f'{output_dir}/shap_{matrix_type}_best_tfs.tsv', sep='\t')
    shap_summary_df_filled = shap_summary_df.fillna(-1)
    
    # Create and close clustermap
    clustermap = sns.clustermap(shap_summary_df_filled,
                                method='average', metric='euclidean',
                                cmap='viridis', figsize=(25, 10),
                                row_cluster=False, col_cluster=True)
    plt.close(clustermap.fig)
    
    # Reorder columns based on the dendrogram
    ordered_cols = clustermap.dendrogram_col.reordered_ind
    shap_summary_df_ordered = shap_summary_df.iloc[:, ordered_cols].dropna(axis=1, how='all')
    shap_summary_df_ordered.to_csv(f'{output_dir}/shap_{matrix_type}_best_tfs_clustermap.tsv', sep='\t')
    
    # Visualization
    max_abs_shap_value = np.nanmax(np.abs(shap_summary_df_ordered.values))
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.set_facecolor('white')
    ax.imshow(np.ones_like(shap_summary_df_ordered), cmap='gray_r', interpolation='nearest', aspect='equal')
    
    ax.set_xticks(np.arange(len(shap_summary_df_ordered.columns)))
    ax.set_yticks(np.arange(len(shap_summary_df_ordered.index)))
    ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True, bottom=False, top=True, length=0)
    ax.set_yticklabels(shap_summary_df_ordered.index, fontsize=8, color="black")
    ax.set_xticklabels(shap_summary_df_ordered.columns, fontsize=8, color="black", rotation=90)
    
    ax.set_xticks(np.arange(len(shap_summary_df_ordered.columns) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(shap_summary_df_ordered.index) + 1) - .5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    rect = plt.Rectangle((-.5, -.5), len(shap_summary_df_ordered.columns), len(shap_summary_df_ordered.index), linewidth=2, edgecolor='lightgray', facecolor='none')
    ax.add_patch(rect)
    
    norm = plt.Normalize(shap_summary_df_ordered.min().min(), shap_summary_df_ordered.max().max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    for i in range(len(shap_summary_df_ordered.columns)):
        for j in range(len(shap_summary_df_ordered.index)):
            value = shap_summary_df_ordered.iat[j, i]
            color = sm.to_rgba(value)
            size = np.abs(value) / max_abs_shap_value
            rect = Rectangle(xy=(i - size / 2, j - size / 2), width=size, height=size, facecolor=color)
            ax.add_patch(rect)

    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.5, aspect=50, pad=0.02)
    cbar.set_label('SHAP Value Magnitude')
    
    plt.savefig(f'{output_dir}/shap_{matrix_type}_best_tfs_clustermap.pdf', bbox_inches='tight')
    #plt.show()
    plt.close()


# In[37]:


df = pd.read_csv(f'{input_dir_matrices}/Matrix_MX_all.tsv', sep='\t', index_col=[0])
feature_subset = df.columns  # All the matrices have the same features

for matrix_type in matrix_types:
    plot_shap_clustermap(matrix_type, data_tests, feature_subset)
