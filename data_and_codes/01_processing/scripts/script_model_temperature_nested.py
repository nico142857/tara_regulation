#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb  # Extreme Gradient Boosting Algorithm
from hyperopt import fmin, tpe, Trials, hp  # Tree-based hyperparameter optimizer
import pickle  # Saving the model
import json  # JSON file write/read

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from scipy.stats import uniform

import shap
import sys
from tqdm import tqdm

# --------------------------
# # Upload data
# --------------------------

input_dir = '../../00_matrices'
out_dir = '../../../out_results/out_xgb_models'
filename = sys.argv[1]
parts = filename.split('_')
matrix_type = parts[1] if len(parts) > 1 else None
subsample = parts[2].split('.')[0] if len(parts) > 2 else None
target = 'temperature'

# Biological matrix
df = pd.read_csv(f'{input_dir}/{filename}', sep='\t', index_col=[0])
# Environment matrix
md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=[0])

# Define mappings for each column
layer_mapping = {'SRF': 0, 'DCM': 1, 'MES': 2}
layer2_mapping = {'EPI': 0, 'MES': 1}
polar_mapping = {'non polar': 0, 'polar': 1}
# province_mapping = {'B2': 0, 'B3': 1, 'B5': 2, 'B6': 3, 'B7': 4, 'B8': 5}

md_encoded = md.copy()
# Apply mappings to each respective column in the 'md_encoded' DataFrame
md_encoded['Layer'] = md_encoded['Layer'].map(layer_mapping)
md_encoded['Layer2'] = md_encoded['Layer2'].map(layer2_mapping)
md_encoded['polar'] = md_encoded['polar'].map(polar_mapping)

# --------------------------
# # CLR implementation
# --------------------------

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

    # Add a small amount to cells with a value of 0
    if (data <= 0).any().any():
        data = data.replace(0, eps)

    # Calculate the geometric mean of each row
    gm = np.exp(data.apply(np.log).mean(axis=1))

    # Perform clr transformation
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)

    return clr_data

# --------------------------
# # Define validation list
# --------------------------

# Exclude these SRF samples to validate
private_list = ['TSC021',  # TARA_031 non polar, HIGH
                'TSC060',  # TARA_042 non polar, HIGH
                'TSC085',  # TARA_065 non polar, MID
                'TSC102',  # TARA_068 non polar, MID
                'TSC141',  # TARA_084 polar south, LOW
                'TSC145',  # TARA_085 polar south, LOW
                'TSC173',  # TARA_112 non polar, HIGH
                'TSC213',  # TARA_132 non polar, HIGH
                'TSC216',  # TARA_133 non polar, MID
                'TSC237',  # TARA_150 non polar, MID
                'TSC261',  # TARA_189 polar, LOW
                'TSC276',  # TARA_208 polar, LOW
                'TSC285',  # TARA_163 polar, LOW
               ]
# 4 HIGH, 4 MID, 5 LOW

# Filter the DataFrame to include only the desired samples (if needed later)
filtered_metadata = md.loc[private_list]

# --------------------------
# # Model optimization data prep
# --------------------------

clr_df = clr_(df)
aligned_md = md_encoded.loc[clr_df.index]  # Align the target metadata with the current dataframe's indices

# Create the target variable
continuous_y = aligned_md['Temperature'].copy()
# Drop NaN values from continuous_y and corresponding rows in clr_df
valid_indices = continuous_y.dropna().index
continuous_y = continuous_y.loc[valid_indices]
clr_df = clr_df.loc[valid_indices]

# Bin the Temperature to obtain a three-class target variable
y_total = continuous_y.apply(lambda temp: 0 if temp < 10 else (1 if temp <= 22 else 2))

# Exclude the private list from training data
X = clr_df.drop(private_list, errors='ignore')
y = y_total.drop(private_list, errors='ignore')

# --------------------------
# # Nested Cross-Validation
# --------------------------

# Define a function to tune hyperparameters using Hyperopt with inner cross-validation.
def tune_hyperparameters(X_train, y_train, max_evals=100):
    # Define parameter space with updated distributions:
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 350, 50),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, -2),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.uniform('gamma', 0.0, 5.0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 0),   # from exp(-5) ~ 0.0067 to 1
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2),  # from exp(-5) ~ 0.0067 to exp(2) ~ 7.39
    }
    
    def objective(params):
        # Cast parameters that should be integers and add the rest:
        params_casted = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'min_child_weight': int(params['min_child_weight']),
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
        }
        clf = xgb.XGBClassifier(**params_casted)
        # Use inner cross-validation for hyperparameter evaluation
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=25, random_state=42)
        score = cross_val_score(clf, X_train, y_train, scoring='f1_weighted', cv=inner_cv).mean()
        return -score  # Negative because fmin minimizes the objective
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    return best

# Outer cross-validation to estimate performance
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

print("Starting Nested Cross-Validation...")
for train_idx, test_idx in outer_cv.split(X, y):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Tune hyperparameters on the training fold (inner CV)
    best_fold = tune_hyperparameters(X_train_fold, y_train_fold, max_evals=100)
    best_params_fold = {
        'n_estimators': int(best_fold['n_estimators']),
        'max_depth': int(best_fold['max_depth']),
        'min_child_weight': int(best_fold['min_child_weight']),
        'learning_rate': best_fold['learning_rate'],
        'subsample': best_fold['subsample'],
        'colsample_bytree': best_fold['colsample_bytree'],
        'gamma': best_fold['gamma'],
        'reg_alpha': best_fold['reg_alpha'],
        'reg_lambda': best_fold['reg_lambda'],
    }
    # Train model on the current training fold with the tuned hyperparameters
    clf = xgb.XGBClassifier(**best_params_fold, 
                              objective='multi:softmax', 
                              num_class=y.nunique())
    clf.fit(X_train_fold, y_train_fold)
    
    # Evaluate on the outer test fold
    y_pred_fold = clf.predict(X_test_fold)
    fold_score = f1_score(y_test_fold, y_pred_fold, average='weighted')
    outer_scores.append(fold_score)
    print(f"Outer Fold F1 Score: {fold_score:.4f}")

avg_outer_score = np.mean(outer_scores)
print("Nested CV average F1 Score on outer folds:", avg_outer_score)

# --------------------------
# # Final Model Training and Evaluation on Private List
# --------------------------

# Now, run hyperparameter tuning on the entire training set (excluding the private list)
print("Tuning final hyperparameters on all training data...")
best_final = tune_hyperparameters(X, y, max_evals=1000)
final_params = {
    'n_estimators': int(best_final['n_estimators']),
    'max_depth': int(best_final['max_depth']),
    'min_child_weight': int(best_final['min_child_weight']),
    'learning_rate': best_final['learning_rate'],
    'subsample': best_final['subsample'],
    'colsample_bytree': best_final['colsample_bytree'],
    'gamma': best_final['gamma'],
    'reg_alpha': best_final['reg_alpha'],
    'reg_lambda': best_final['reg_lambda'],
    'objective': 'multi:softmax',
    'num_class': y.nunique(),
}

print("Best final parameters:", final_params)

# Instantiate the final model with the best parameters
final_model = xgb.XGBClassifier(**final_params)

# Train the final model on the entire training set (excluding private list)
final_model.fit(X, y)

# Evaluate on the private hold-out list
private_data = clr_df.loc[private_list]
private_labels = y_total.loc[private_list]
private_predictions = final_model.predict(private_data)
private_accuracy = accuracy_score(private_labels, private_predictions)
private_f1 = f1_score(private_labels, private_predictions, average='weighted')

print("Accuracy on Private List:", private_accuracy)
print("F1 Score on Private List:", private_f1)

# --------------------------
# # Save the Model and Metrics
# --------------------------

escenario_target = f'{matrix_type}_{subsample}_{target}'

# Pickle the trained model
model_filename = f'model_{escenario_target}.pkl'
with open(f'{out_dir}/{model_filename}', 'wb') as file:
    pickle.dump(final_model, file)

# Save best hyperparameters
params_filename = f'best_hyperparameters_{escenario_target}.json'
with open(f'{out_dir}/metrics/{params_filename}', 'w') as file:
    json.dump(final_params, file)

# Save evaluation metrics (including private list performance and nested CV average)
metrics_filename = f'evaluation_metrics_{escenario_target}.json'
with open(f'{out_dir}/metrics/{metrics_filename}', 'w') as file:
    json.dump({'Nested CV Average F1 Score': avg_outer_score,
               'Private Accuracy': private_accuracy,
               'Private F1 Score': private_f1}, file)
