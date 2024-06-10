Models were trained on specific subsamples tailored to different target variables using XGBoost:

- Polar / Non-Polar: Surface samples
- Temperature: Surface samples
- Province: Surface samples
- Layer: NonPolar samples
- Layer2: NonPolar samples
- NO3: EPI-NonPolar samples
- Mean Flux at 150m (Carbon Export): EPI-NonPolar samples
- NPP (Net Primary Production): EPI-NonPolar samples

Each category utilizes a data matrix type indicated as `<matrix_type> `(e.g., `M0`, `MX`) reflecting the SHAP values relevant to that model.

Files are named following the pattern `shap_<matrix_type>_best_tfs_clustermap.pdf`. Additionally, .tsv files are provided for enhanced data accessibility.
