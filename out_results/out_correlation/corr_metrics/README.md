This folder contains tables with useful information regarding the correlations of transcription factors (TFs) with environmental variables.

- **`corr_top8_bio_by_bio_vs_env.tsv`**:
  - Contains the top 8 TFs correlated with environmental variables across all matrices and subsamples.
  - Displays the maximum correlation value of these TFs to any environmental variable.
  - Includes the average correlation of these TFs to the top 5 most correlated environmental variables.

- **`corr_top8_env_by_bio_vs_env.tsv`**:
  - Contains the top 8 environmental variables correlated with TFs across all matrices and subsamples.
  - Displays the maximum correlation value of these environmental variables to any TF.
  - Includes the average correlation of these environmental variables to the top 5 most correlated TFs.

- **`corr_top_bio_by_abs_sum_<matrix_type>_<subsample>.tsv`**:
  - Lists the TFs most correlated with Temperature, Oxygen, Fluorescence, Chlorophyll-a, and Carbon total.
  - `<matrix_type>` and `<subsample>` specify the type of matrix and subsample used.
