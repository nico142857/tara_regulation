# Linear Regression Results

This folder contains results regarding linear regressions:

- **`R2_<matrix_type>_<subsample>.tsv`**
  - Contains the R² values of the regressions fit between each TF and environmental variable.

- **`R2_<matrix_type>_<subsample>.pdf`**
  - A visualization of the R² values from the above `.tsv` file.

- **`Regfit_<matrix_type>_<subsample>_<TF>_<env>.pdf`**
  - A visualization of the actual regression line between the TF and environmental variable for those (TF, env) pairs with an R² above 0.5.
