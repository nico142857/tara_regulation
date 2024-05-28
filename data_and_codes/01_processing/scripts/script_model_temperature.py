#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb # Extreme Gradient Boosting Algorthim
from hyperopt import fmin, tpe, Trials, hp # Tree based hyperparameter optimizer
import pickle # Saving the model
import json # json file w/r

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from scipy.stats import uniform

import shap

import sys

from tqdm import tqdm


# # Upload data

# In[ ]:

input_dir = '../../00_matrices'
out_dir = '../../../out_results/out_xgb_models'
filename = sys.argv[1]
parts = filename.split('_')
matrix_type = parts[1] if len(parts) > 1 else None
subsample = parts[2].split('.')[0] if len(parts) > 2 else None
target = 'temperature'

# Biological matrix
df = pd.read_csv(f'{input_dir}/{filename}', sep='\t', index_col=[0])
# ENvironment matrix
md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=[0])

# Define mappings for each column
layer_mapping = {'SRF': 0, 'DCM': 1, 'MES': 2}
layer2_mapping = {'EPI': 0, 'MES': 1}
polar_mapping = {'non polar': 0, 'polar': 1}
#province_mapping = {'B2': 0, 'B3': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5}

md_encoded = md.copy()
# Apply mappings to each respective column in the 'md_encoded' DataFrame
md_encoded['Layer'] = md_encoded['Layer'].map(layer_mapping)
md_encoded['Layer2'] = md_encoded['Layer2'].map(layer2_mapping)
md_encoded['polar'] = md_encoded['polar'].map(polar_mapping)


# # CLR implementation

# In[ ]:


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


# # Define validation list

# In[ ]:


# Exclude these SRF samples to validate
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

# Filter the DataFrame to include only the desired samples
filtered_metadata = md.loc[private_list]


# # Model optimization

# In[ ]:


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

# Exclude the private list from training data
X = clr_df.drop(private_list, errors='ignore')
y = y_total.drop(private_list, errors='ignore')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Define parameter space
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 100, 200, 350),
    'max_depth': hp.quniform('max_depth', 2, 3, 6),
    'min_child_weight': hp.quniform('min_child_weight', 1, 3, 6),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'subsample': hp.uniform('subsample', 0.75, 1),
    'gamma': hp.uniform('gamma', 0.0, 3.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 5.0),
}

def objective(params):
    # Make sure to cast max_depth and min_child_weight to int as they are supposed to be integer values
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'min_child_weight': int(params['min_child_weight']),
        'learning_rate': params['learning_rate'],
        'subsample': params['subsample'],
        'gamma': params['gamma'],
        'reg_lambda': params['reg_lambda'],
    }
    clf = xgb.XGBClassifier(**params)
    # Using StratifiedKFold for cross-validation
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=25, random_state=42)
    score = cross_val_score(clf, X, y, scoring='f1_weighted', cv=rskf).mean()
    return -score  # Negating the score because fmin() tries to minimize the objective


best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=Trials())

print("Best parameters:", best)

# Retrieve the best parameters and cast them to the correct type
number_of_classes = y.nunique()
final_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_child_weight': int(best['min_child_weight']),
    'learning_rate': best['learning_rate'],
    'subsample': best['subsample'],
    'gamma': best['gamma'],
    'reg_lambda': best['reg_lambda'],
    'objective': 'multi:softmax',
    'num_class': number_of_classes,
}

# Instantiate the model with the best parameters
final_model = xgb.XGBClassifier(**final_params)

# Train the model on the dataset excluding the private list
final_model.fit(X, y)

# Extract data for private list for evaluation
private_data = clr_df.loc[private_list]
private_labels = y_total.loc[private_list]

# Make predictions on private data
private_predictions = final_model.predict(private_data)

# Calculate accuracy and F1 score for the data on the private list
private_accuracy = accuracy_score(private_labels, private_predictions)
private_f1 = f1_score(private_labels, private_predictions, average='weighted')  # Adjust the 'average' parameter as needed

# Output the performance metrics
print("Accuracy on Private List:", private_accuracy)
print("F1 Score on Private List:", private_f1)


escenario_target = f'{matrix_type}_{subsample}_{target}'

# Pickle the trained model
model_filename = f'model_{escenario_target}.pkl'
with open(f'{out_dir}/{model_filename}', 'wb') as file:
    pickle.dump(final_model, file)

# Save best hyperparameters
params_filename = f'best_hyperparameters_{escenario_target}.json'
with open(f'{out_dir}/metrics/{params_filename}', 'w') as file:
    json.dump(final_params, file)

# Save evaluation metrics
metrics_filename = f'evaluation_metrics_{escenario_target}.json'
with open(f'{out_dir}/metrics/{metrics_filename}', 'w') as file:
    json.dump({'Private Accuracy': private_accuracy, 'Private F1 Score': private_f1}, file)

