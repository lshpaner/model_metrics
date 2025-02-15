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

# from py_scripts.model_classification import model


import shap

## Define base paths
## `base_path`` represents the parent directory of your current working directory
base_path = os.path.join(os.pardir)
## Go up one level from 'notebooks' to the parent directory, then into the
## 'results' folder

model_path = os.path.join(os.pardir, "model_files")

# Use the function to ensure the 'data' directory exists
ensure_directory(model_path)

csv_filename_per_row = "per_row_shap.csv"

model_arg_name = "model"

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

print(model_name.estimator.named_steps["lg"])
model = model_name.estimator.named_steps["lg"]

print("Loaded model object:", model_name)
print("Original model argument:", model_arg_name)

############################################################################
############################# Read in Data #################################
############################################################################

X_valid = pd.read_parquet(os.path.join(model_path, "X_valid.parquet"))
y_valid = pd.read_parquet(os.path.join(model_path, "y_valid.parquet"))

################################################################################
############################# Model Calculator #################################
################################################################################

## Initialize the ModelCalculator
calculator = ModelCalculator(
    model_dict=model,
    outcomes=y_valid,
    top_n=5,
)

print(calculator)
print(X_valid)


############################### Row-wise SHAP ##################################

## Generate the predictions and SHAP contributions
results_df_per_row = calculator.generate_predictions(
    X_test=X_valid,
    y_test=y_valid,
    calculate_shap=False,
    use_coefficients=True,
    include_contributions=False,
    subset_results=True,
)
print(os.path.join(model_path, csv_filename_per_row))
results_df_per_row.to_csv(os.path.join(model_path, csv_filename_per_row))

print(results_df_per_row)
quit()
########################## Overall Coefficients/SHAP ###########################

# Use the original model argument name to check for logistic regression
print(model_arg_name)
if model_arg_name == "model":
    print()
    print("Calculating global coefficients for Logistic Regression...")
    print()
    results_df = calculator.generate_predictions(
        X_test=X_valid,
        y_test=y_valid,
        global_coefficients=True,  # Only calculate coefficients
    )
else:
    print()
    print("Calculating SHAP values for other models...")
    print()
    results_df = calculator.generate_predictions(
        X_test=X_valid,
        y_test=y_valid,
        global_shap=True,  # Only calculate SHAP values
    )
# Save the results
results_df.to_csv(os.path.join(model_path, csv_filename_per_row))
