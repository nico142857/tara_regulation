- `Ocean_fingerprint`: This script is designed to analyze and visualize clr (centered log-ratio) normalized abundance of transcription factors (TFs) for different matrix types (MX, M0, M1, guidi, salazar, stress) and different metagenomic sources (Tara Oceans, Human Gut, Lake Water and Farmland Soil).

- `correlation_*`: A collection of scripts to compute correlations between the variables used in this study:
	- correlation_heatmap_bio.py: Computes the auto-correlation between transcription factors (TFs).
	- correlation_clustermap_bio.py: Computes the auto-correlation between TFs and reorders the labels using hierarchical clustering for improved readability.
	- correlation_heatmap_bio_env.py: Computes the correlation between TF abundance and non-categorical environmental variables.

- `initial_prediction_tfs_genes.py`: This script performs clr normalization on different matrix types (`MX`, `M0`, `M1`, `guidi`, `salazar`, `stress`) for both TFs and Genes, then trains and evaluates XGBoost classification models to predict various binned environmental variables, outputting the results as average F1 scores over multiple simulation cycles (n=100). This is the first approach to evaluate the capacity of transcription factors and genes to predict the environment and also comparing their results.
	- 'standar_initial_predictions.sh': A bash script designed to execute the `initial_prediction_tfs_genes.py` script for use with SLURM clusters.

- `script_model_*.py`: A collection of scripts designed to optimize the parameters of XGBoost models with the objective of maximizing the F1-score. These scripts fit a final XGBoost model with the optimized parameters using TF abundance as predictors and the specified environmental variable as the target (indicated in the script name) using a public/private samples approach [1]. For example:
	- 'script_model_polar': Reads the input matrix provided by the user and searches for the best hyperparameters (n_estimators, max_depth, min_child_weight, learning_rate, subsample, gamma, reg_lambda) that maximize the F1-score using a repeated cross-validation approach. Finally, it fits an XGBoost model using the optimized hyperparameters, with TF abundance as predictors and the polar/non-polar classification as the target variable on the public samples.
	
	[1] To optimize the parameters, equally distributed samples for each target variable class are selected and excluded from the list of samples; these are denoted as "private samples," while the remaining are referred to as "public samples."

## How to run script_model_*
We use run_jobs_model.sh, which is a bash that runs `standar_models.sh` for each target variable (in the name of the scripts) of interest and each matrix_type used as predictor.
Keep in mind that the target variables were classified in specific samples subsets (denoted later as subsample)
- Surface samples | polar/non polar, temperature (Low <=10°C, Mid 10-22°C, High >22°C), province (Fremont et. al. 2022)
- Nonpolar samples | Layer (Surface, DCM, Mesopelagic), Layer2 (Epipelagic, Mesopelagic)
- Epi-nonpolar samples | NO3 (Low <=7, High > 7), NPP (Low <=275, Mid 275-540, High >540), CarbonExport Mean Flux at 150m (Low <=0.7, Mid 0.7-3, High >3)

### Credits
Developed by: **Nicolas Toro-Llanco**
- **Email**: [ntoro@dim.uchile.cl](mailto:ntoro@dim.uchile.cl)
- **GitHub**: [github.com/nico142857](https://github.com/nico142857)