### Overview
This folder contains all trained XGBoost models used for the classification of different environmental variables. These models are designed to address specific classification tasks using environmental data.

### File Naming Convention
The files are named according to the following pattern:
- **model_`matrix_type`_`subsample`_`environmental_variable_to_classify`.pkl**

Where:
- **`matrix_type`**: Refers to the regulatory matrix used as a predictor.
- **`subsample`**: Indicates the locations from which the samples were taken.
- **`environmental_variable_to_classify`**: Specifies the target variables that the models are designed to classify.

### Description
Each model file is a serialized Python object that can be deserialized using the `pickle` library. These models are trained using the XGBoost algorithm, which is optimized for performance and accuracy in classification tasks.

### Usage
To utilize these models, ensure that Python and the XGBoost library are installed on your system. Load a model using Python’s `pickle` module to make predictions or further analyze the model's performance.

### Credits
Developed by: **Nicolas Toro-Llanco**
- **Email**: [ntoro@dim.uchile.cl](mailto:ntoro@dim.uchile.cl)
- **GitHub**: [github.com/nico142857](https://github.com/nico142857)
