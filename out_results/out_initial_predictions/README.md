### Overview
This folder contains metric files for two types of classification tasks using transcription factors (TFs) as predictors for environmental variables. These files evaluate the effectiveness of classification models under different conditions and configurations.

### Files and Descriptions

1. **`initial_prediction_tf_vs_gen`**
   - **Description**: This file provides metrics for classification tasks using both transcription factors (TFs) abundance and gene abundance as predictors. The results offer insight into the model's performance across a variety of environmental variables.
   - **Format**: Each entry includes:
     - **Matrix_Type**: The model matrix used in the classification.
     - **Variable**: The specific environmental variable being classified.
     -**Metrics**: F1, Accuracy, Recall and Precision

2. **`initial_prediction_tf_by_context`**
   - **Description**: This file contains classification metrics for environmental variables using only TFs abundance as predictors. The metrics are organized by different subsamples of the ocean, and each row names the subsample and the corresponding classification metrics.
   - **Format**: Structured with:
     - **Target**: The environmental variable being classified (e.g., Temperature, Oxygen, Fluorescence).
     - **F1_Score**: The F1 score achieved for the classification of the respective target.
     - **Matrix_Type**: The model matrix type used in the classification (e.g., MX, M0, M1, guidi, salazar).

### Credits
Developed by: **Nicolas Toro-Llanco**
- **Email**: [ntoro@dim.uchile.cl](mailto:ntoro@dim.uchile.cl)
- **GitHub**: [github.com/nico142857](https://github.com/nico142857)