#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb  # Extreme Gradient Boosting Algorithm
from hyperopt import fmin, tpe, Trials, hp  # Tree-based hyperparameter optimizer
import pickle  # For saving the model
import json    # For saving hyperparameters and metrics

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import sys
import shap
from tqdm import tqdm

# --------------------------
# Upload data
# --------------------------
input_dir = '../../00_matrices'
out_dir   = '../../../out_results/out_xgb_models'
filename  = sys.argv[1]
parts     = filename.split('_')
matrix_type = parts[1] if len(parts) > 1 else None
subsample   = parts[2].split('.')[0] if len(parts) > 2 else None
target      = 'polar'

# Biological matrix
df = pd.read_csv(f'{input_dir}/{filename}', sep='\t', index_col=[0])
# Environment matrix
md = pd.read_csv(f'{input_dir}/metadata.tsv', sep='\t', index_col=[0])

# Define mappings for each column
layer_mapping  = {'SRF': 0, 'DCM': 1, 'MES': 2}
layer2_mapping = {'EPI': 0, 'MES': 1}
polar_mapping  = {'non polar': 0, 'polar': 1}

md_encoded = md.copy()
md_encoded['Layer']  = md_encoded['Layer'].map(layer_mapping)
md_encoded['Layer2'] = md_encoded['Layer2'].map(layer2_mapping)
md_encoded['polar']  = md_encoded['polar'].map(polar_mapping)

# --------------------------
# CLR implementation
# --------------------------
def clr_(data, eps=1e-6):
    """
    Perform centered log-ratio (CLR) normalization on a dataset.
    
    Parameters:
        data (pandas.DataFrame): Samples as rows and components as columns.
    Returns:
        pandas.DataFrame: CLR-normalized data.
    """
    if (data < 0).any().any():
        raise ValueError("Data should be strictly positive for CLR normalization.")
    if (data <= 0).any().any():
        data = data.replace(0, eps)
    gm = np.exp(data.apply(np.log).mean(axis=1))
    clr_data = data.apply(np.log).subtract(np.log(gm), axis=0)
    return clr_data

# --------------------------
# Define validation list (private hold-out samples)
# --------------------------
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
                'TSC285']  # TARA_163 polar, LOW

# --------------------------
# Model optimization data prep
# --------------------------
clr_df = clr_(df)
aligned_md = md_encoded.loc[clr_df.index]  # Align metadata with CLR data

# Create the target variable from the "polar" column
y_total = aligned_md['polar'].copy()

# Drop rows with NaN in the target
valid_indices = y_total.dropna().index
y_total = y_total.loc[valid_indices]
clr_df   = clr_df.loc[valid_indices]

# For binary classification, ensure target is integer type (0/1)
y_total = y_total.astype(int)

# Exclude private samples from training data and store as X_full_train and y_full_train
mask = ~clr_df.index.isin(private_list)
X_full_train = clr_df.loc[mask]
y_full_train = y_total.loc[mask]

# --------------------------
# Nested Cross-Validation Setup
# --------------------------
def tune_hyperparameters(X_train_cv, y_train_cv, max_evals=100):
    # Define hyperparameter search space
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 350, 50),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -5, -2),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.uniform('gamma', 0.0, 5.0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 0),
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2),
    }
    
    def objective(params):
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
        # Use binary:logistic for binary classification (polar)
        clf = xgb.XGBClassifier(**params_casted, objective='binary:logistic')
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=25, random_state=42)
        score = cross_val_score(clf, X_train_cv, y_train_cv, scoring='f1', cv=inner_cv).mean()
        return -score  # Negative because fmin minimizes the objective
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best

# Outer CV loop for unbiased performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

print("Starting Nested Cross-Validation for 'polar'...")
for train_idx, test_idx in outer_cv.split(X_full_train, y_full_train):
    X_train_fold = X_full_train.iloc[train_idx]
    y_train_fold = y_full_train.iloc[train_idx]
    X_test_fold  = X_full_train.iloc[test_idx]
    y_test_fold  = y_full_train.iloc[test_idx]
    
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
    
    clf = xgb.XGBClassifier(**best_params_fold, objective='binary:logistic')
    clf.fit(X_train_fold, y_train_fold)
    y_pred_fold = clf.predict(X_test_fold)
    fold_score = f1_score(y_test_fold, y_pred_fold, average='binary')
    outer_scores.append(fold_score)
    print(f"Outer Fold F1 Score: {fold_score:.4f}")

avg_outer_score = np.mean(outer_scores)
print("Nested CV average F1 Score on outer folds:", avg_outer_score)

# --------------------------
# Final Model Training and Evaluation on Private List
# --------------------------
print("Tuning final hyperparameters on all training data...")
best_final = tune_hyperparameters(X_full_train, y_full_train, max_evals=1000)
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
    'objective': 'binary:logistic',
}

print("Best final parameters:", final_params)

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X_full_train, y_full_train)

# Evaluate on the private hold-out set
private_data = clr_df.loc[private_list]
private_labels = y_total.loc[private_list]
private_accuracy = accuracy_score(private_labels, final_model.predict(private_data))
private_f1 = f1_score(private_labels, final_model.predict(private_data), average='binary')

print("Accuracy on Private List:", private_accuracy)
print("F1 Score on Private List:", private_f1)

# --------------------------
# Save the Model and Metrics
# --------------------------
escenario_target = f'{matrix_type}_{subsample}_{target}'

model_filename = f'model_{escenario_target}_nested.pkl'
with open(f'{out_dir}/{model_filename}', 'wb') as file:
    pickle.dump(final_model, file)

params_filename = f'best_hyperparameters_{escenario_target}_nested.json'
with open(f'{out_dir}/metrics/{params_filename}', 'w') as file:
    json.dump(final_params, file)

metrics_filename = f'evaluation_metrics_{escenario_target}_nested.json'
with open(f'{out_dir}/metrics/{metrics_filename}', 'w') as file:
    json.dump({'Nested CV Average F1 Score': avg_outer_score,
               'Private Accuracy': private_accuracy,
               'Private F1 Score': private_f1}, file)
