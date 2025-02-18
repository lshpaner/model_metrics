import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from eda_toolkit import ensure_directory
from model_tuner import Model, dumpObjects
import model_tuner

# Add the parent directory to sys.path to access 'functions.py'
print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__} \n")


if __name__ == "__main__":

    argv = sys.argv[1:]
    model_path = argv[0]

    ## Define base paths
    ## `base_path`` represents the parent directory of your current working directory
    base_path = os.path.join(os.pardir)
    ## Go up one level from 'notebooks' to the parent directory, then into the
    ## 'results' folder

    model_path = os.path.join(os.pardir, "model_files/single_model_regression_results")
    image_path_png = os.path.join(base_path, "images", "png_images")
    image_path_svg = os.path.join(base_path, "images", "svg_images")

    # Use the function to ensure the directories exist
    ensure_directory(model_path)
    ensure_directory(image_path_png)
    ensure_directory(image_path_svg)

    # Ensure model_path is an absolute Path object
    model_path = Path(os.path.abspath(argv[0]))

    # Ensure the directory exists before saving the model
    model_path.mkdir(parents=True, exist_ok=True)

    # Define the path to save the pickled model
    model_filename = model_path / "lasso_regression_model.pkl"

    # Load the dataset with frame=True to get a DataFrame
    diabetes = load_diabetes(as_frame=True)["frame"]

    print(diabetes)

    X = diabetes[[col for col in diabetes.columns if "target" not in col]]

    y = diabetes["target"]

    lasso_name = "lasso"
    lasso = Lasso(random_state=3)
    tuned_parameters_lasso = [
        {
            f"{lasso_name}__fit_intercept": [True, False],
            f"{lasso_name}__precompute": [True, False],
            f"{lasso_name}__copy_X": [True, False],
            f"{lasso_name}__max_iter": [100, 500, 1000, 2000],
            f"{lasso_name}__tol": [1e-4, 1e-3],
            f"{lasso_name}__warm_start": [True, False],
            f"{lasso_name}__positive": [True, False],
        }
    ]
    lasso_definition = {
        "clc": lasso,
        "estimator_name": lasso_name,
        "tuned_parameters": tuned_parameters_lasso,
        "randomized_grid": False,
        "early": False,
    }
    model_definitions = {lasso_name: lasso_definition}

    model_type = "lasso"
    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]

    kfold = False
    calibrate = False

    model = Model(
        name="Logistic Regression",
        estimator_name=estimator_name,
        model_type="regression",
        calibrate=calibrate,
        estimator=clc,
        kfold=kfold,
        stratify_y=False,
        grid=tuned_parameters_lasso,
        randomized_grid=False,
        scoring=["r2"],
        n_jobs=-2,
        random_state=42,
    )

    model.grid_search_param_tuning(X, y)

    X_train, y_train = model.get_train_data(X, y)
    X_test, y_test = model.get_test_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)

    ### Parquet the validation and test data
    X_test.to_parquet(os.path.join(model_path, "X_test.parquet"))
    y_test.to_frame().to_parquet(os.path.join(model_path, "y_test.parquet"))
    X_valid.to_parquet(os.path.join(model_path, "X_valid.parquet"))
    y_valid.to_frame().to_parquet(os.path.join(model_path, "y_valid.parquet"))

    model.fit(X_train, y_train)

    print("Validation Metrics")
    model.return_metrics(
        X_valid,
        y_valid,
        model_metrics=True,
    )
    print()
    print("Test Metrics")
    model.return_metrics(
        X_test,
        y_test,
        model_metrics=True,
    )

    dumpObjects(model, model_filename)

    print(f"Model saved to {model_filename}")
