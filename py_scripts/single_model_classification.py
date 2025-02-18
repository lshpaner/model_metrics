import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
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

    model_path = os.path.join(
        os.pardir, "model_files/single_model_classification_results"
    )
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
    model_filename = model_path / "logistic_regression_model.pkl"

    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=100000,  # Number of samples
        n_features=50,  # Number of features
        n_informative=25,  # Number of informative features
        n_classes=2,  # Number of classes (binary classification)
        random_state=42,  # Reproducibility
    )

    # Convert to a DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(50)])
    df["target"] = y

    # Convert y into a DataFrame
    y = pd.DataFrame(y, columns=["target"]).squeeze()

    # Convert X into a DataFrame
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)

    estimator_name = "lg"
    # Set the parameters by cross-validation
    tuned_parameters = [
        {
            estimator_name + "__C": np.logspace(-4, 0, 3),
        }
    ]
    kfold = False
    calibrate = False

    model = Model(
        name="Logistic Regression",
        estimator_name=estimator_name,
        model_type="classification",
        calibrate=calibrate,
        estimator=lr,
        kfold=kfold,
        stratify_y=False,
        grid=tuned_parameters,
        randomized_grid=False,
        scoring=["roc_auc"],
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
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )
    print()
    print("Test Metrics")
    model.return_metrics(
        X_test,
        y_test,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )

    y_prob = model.predict_proba(X_test)

    ### F1 Weighted
    y_pred = model.predict(X_test, optimal_threshold=True)

    ### Report Model Metrics

    model.return_metrics(
        X_test,
        y_test,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
        return_dict=False,
    )

    dumpObjects(model, model_filename)

    print(f"Model saved to {model_filename}")
