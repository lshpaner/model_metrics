from model_tuner import *
from eda_toolkit import ensure_directory
import pandas as pd
import numpy as np
import os
import sys

print(os.path.join(os.pardir, ".."))
sys.path.append("..")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from model_metrics import ModelCalculator

import shap

## Define base paths
## `base_path`` represents the parent directory of your current working directory
base_path = os.path.join(os.pardir)
## Go up one level from 'notebooks' to the parent directory, then into the
## 'results' folder

model_path = os.path.join(os.pardir, "model_files")

# Use the function to ensure the 'data' directory exists
ensure_directory(model_path)

csv_filename_coeff = "results_coefficients.csv"
csv_filename_per_row = "results_per_row_shap.csv"
csv_global_shap = "results_global_shap.csv"

##################### Hard-coded paths for debugging #######################

# model_path = "../../models/results/orig_models/"
# data_path = "../../data/processed"

############################################################################
######################### Read in Model Object #############################
############################################################################

model_name = loadObjects(
    os.path.join(
        model_path,
        "logistic_regression_model.pkl",
    )
)

model = model_name.estimator.named_steps["lg"]

############################################################################
############################# Read in Data #################################
############################################################################

X_valid = pd.read_parquet(os.path.join(model_path, "X_valid.parquet"))
y_valid = pd.read_parquet(os.path.join(model_path, "y_valid.parquet"))

outcome = y_valid.columns[0]

X_valid = {outcome: X_valid}

################################################################################
############################# Model Calculator #################################
################################################################################
outcome = "target"
model_dict = {"model": {outcome: model_name}}


## Initialize the ModelCalculator
calculator = ModelCalculator(
    model_dict=model_dict,
    outcomes=[outcome],
    top_n=5,
)

############################# Model Coefficients ###############################

## Generate the predictions and model coefficients
results_df_coeff = calculator.generate_predictions(
    X_test=X_valid,
    y_test=y_valid,
    calculate_shap=False,
    use_coefficients=True,
    include_contributions=False,
    subset_results=True,
)
print(os.path.join(model_path, csv_filename_coeff))
results_df_coeff.to_csv(os.path.join(model_path, csv_filename_coeff))

################################ Row-wise SHAP #################################
## Generate the predictions and SHAP contributions
results_df_per_row = calculator.generate_predictions(
    X_test=X_valid,
    y_test=y_valid,
    calculate_shap=True,
    use_coefficients=False,
    include_contributions=False,
    subset_results=True,
)
print(os.path.join(model_path, csv_filename_per_row))

results_df_per_row.to_csv(os.path.join(model_path, csv_filename_per_row))


################################# Global SHAP ##################################
## Generate the predictions and SHAP contributions
results_df_per_row = calculator.generate_predictions(
    X_test=X_valid,
    y_test=y_valid,
    calculate_shap=False,
    global_shap=True,
    use_coefficients=False,
    include_contributions=False,
    subset_results=False,
)

print(os.path.join(model_path, csv_global_shap))
results_df_per_row.to_csv(os.path.join(model_path, csv_global_shap))
