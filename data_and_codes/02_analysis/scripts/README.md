This folder contains scripts to make analysis on prior obtained results.
The scripts are labeled according to a linear run for replication.

- '01_00_Obtain_shap_values.py': Script to, given the trained XGBoost classification models, obtain the TFs SHAP values for each (matrix_type, subsample) -> target_variable
- '01_01_lists_from_shap.py': Script to, given the computed SHAP values for each TF, obtain a list of top 15 TFs that better impact the classification models
- '01_02_lists_from_shap_correlated.py': Script to, given the list of SHAP values, obtain the most correlated TFs associated to them
- '01_03_lists_from_shap_correlated_env.py': Script to obtain the most correlated TFs associated to an environmental variable

### Credits
Developed by: **Nicolas Toro-Llanco**
- **Email**: [ntoro@dim.uchile.cl](mailto:ntoro@dim.uchile.cl)
- **GitHub**: [github.com/nico142857](https://github.com/nico142857)