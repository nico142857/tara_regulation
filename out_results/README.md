## Description of the Folders

For each matrix type (`MX`, `M0`, `M1`, `guidi`, `salazar`, `stress`):

### out_fingerprint:

- **`fingerprint_tara_173_<matrix_type>.pdf`**: 
  Line plots of clr-abundance values for TFs ordered by the sample `TSC000` for each specified matrix type.
  
- **`fingerprint_tara_10_selected_<matrix_type>.pdf`**: 
  Line plots of clr-abundance values for TFs ordered by the sample `TSC013` and filtered by selected samples [1] for each specified matrix type.
  
- **`fingerprint_tara_10_selected_avg_<matrix_type>.pdf`**: 
  Line plots of the mean clr-abundance values of TFs, along with shaded areas representing standard deviation, ordered by the sample `TSC013` and filtered by selected samples [1] for each specified matrix type.
  
- **`fingerprint_tara_10_comparison1_<matrix_type>.pdf`**: 
  Line plots of the mean clr-abundance values of TFs for Tara Oceans and Human Gut samples. The plots include shaded areas representing standard deviation and annotated Pearson and Spearman correlation coefficients, ordered by the sample `TSC013` and filtered by selected samples [1] for each specified matrix type.
  
- **`fingerprint_regplot_tara_gut_<matrix_type>.pdf`**: 
  Scatter plots with linear regression lines of the mean clr-abundance values between Tara Oceans and Human Gut samples. The plots include annotations of the R² value and the linear regression equation, ordered by the sample `TSC013` and filtered by selected samples for each specified matrix type.
  
- **`fingerprint_tara_10_comparison2_<matrix_type>.pdf`**: 
  Line plots of the mean clr-abundance values of TFs for Tara Oceans, Human Gut, and Lake Water samples. The plots include shaded areas representing standard deviation and a correlation table showing the correlation coefficients between the mean clr-abundance values of the three sample types, ordered by the sample `TSC013` and filtered by selected samples [1] for each specified matrix type.
  
- **`fingerprint_tara_10_comparison3_<matrix_type>.pdf`**: 
  Line plots of the mean clr-abundance values of TFs for Tara Oceans, Human Gut, Lake Water, and Farmland Soil samples. The plots include shaded areas representing standard deviation and a correlation table showing the correlation coefficients between the mean clr-abundance values of the four sample types, ordered by the sample `TSC013` and filtered by selected samples [1] for each specified matrix type.
  
- **`fingerprint_FabR_Temperature_<matrix_type>.pdf`**: 
  Dual-axis plots of the temperature and FabR values for samples in each specified matrix type. The plots show the temperature values ordered by sample along the primary y-axis and the FabR values on the secondary y-axis, both plotted against the samples ordered by temperature.

[1] The selected samples are `['TSC272', 'TSC065', 'TSC254', 'TSC013', 'TSC242', 'TSC216', 'TSC027', 'TSC135', 'TSC141', 'TSC167']`













