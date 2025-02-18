import typer
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import os
import sys
import model_tuner
from model_tuner import Model, dumpObjects
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.base import clone
from pathlib import Path

# Add the parent directory to sys.path to access 'functions.py'
print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

from py_scripts.model_params import model_definitions

app = typer.Typer()

PROCESSED_DATA_DIR = Path("model_files")
MODELS_DIR = Path("model_files")
RESULTS_DIR = Path("model_files/results")


from pathlib import Path

# Define RESULTS_DIR correctly
RESULTS_DIR = Path(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "model_files/results")
    )
)

# Ensure the directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    model_type: str = "lr",
    exp_name: str = "logistic_regression",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    results: Path = MODELS_DIR / "results",
):

    print()
    print(f"Model Tuner version: {model_tuner.__version__}")
    print(f"Model Tuner authors: {model_tuner.__author__}")
    print()

    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    print("-" * 80)
    print("X")
    print("-" * 80)

    print(X.head())  # inspect first 5 rows of X

    print("-" * 80)
    print("y")
    print("-" * 80)

    print(y.head())  # inspect first 5 rows of y

    X = X.select_dtypes(include=np.number)

    y.loc[:, "income"] = y["income"].str.rstrip(".")  # Remove trailing periods

    # Check the updated value counts
    print(y["income"].value_counts())

    y.value_counts()

    y = y["income"].map({"<=50K": 0, ">50K": 1})

    rstate = 222

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]

    # Set the parameters
    tuned_parameters = model_definitions[model_type]["tuned_parameters"]
    rand_grid = model_definitions[model_type]["randomized_grid"]
    early_stop = model_definitions[model_type]["early"]

    best_model = {}
    best_score = 0

    model_dict = {}
    metrics = {}

    for sampler in [
        None,
        SMOTE(random_state=rstate),
    ]:
        print()
        print("Sampler", sampler)

        pipeline = [
            ("StandardScalar", StandardScaler()),
            ("Preprocessor", SimpleImputer()),
        ]

        print()
        print("=" * 60)

        model_dict = Model(
            pipeline_steps=pipeline,
            name=estimator_name,
            model_type="classification",
            estimator_name=estimator_name,
            calibrate=True,
            estimator=clone(clc),
            kfold=False,
            grid=tuned_parameters,
            n_jobs=2,
            randomized_grid=False,
            scoring=["roc_auc"],
            random_state=rstate,
            stratify_y=True,
            boost_early=early_stop,
            imbalance_sampler=sampler,
        )

        ####################################################################
        #################### Extract Split Data Subsets ####################
        ####################################################################

        model_dict.grid_search_param_tuning(X, y, f1_beta_tune=True)

        X_train, y_train = model_dict.get_train_data(X, y)
        X_test, y_test = model_dict.get_test_data(X, y)
        X_valid, y_valid = model_dict.get_valid_data(X, y)

        ### Parquet the validation and test data
        X_test.to_parquet(
            os.path.join(PROCESSED_DATA_DIR, "X_test.parquet"),
        )
        y_test.to_frame().to_parquet(
            os.path.join(PROCESSED_DATA_DIR, "y_test.parquet"),
        )
        X_valid.to_parquet(
            os.path.join(PROCESSED_DATA_DIR, "X_valid.parquet"),
        )
        y_valid.to_frame().to_parquet(
            os.path.join(PROCESSED_DATA_DIR, "y_valid.parquet"),
        )

        ####################################################################

        model_dict.fit(X, y, score="roc_auc")

        if model_dict.calibrate:
            model_dict.calibrateModel(X, y, score="roc_auc")

        return_metrics_dict = model_dict.return_metrics(
            X,
            y,
            optimal_threshold=True,
            print_threshold=True,
            model_metrics=True,
            return_dict=True,
        )

        metrics = pd.Series(return_metrics_dict).to_frame(estimator_name)
        metrics = round(metrics, 3)
        print("=" * 80)

        ####################################################################
        print("=" * 80)
        cur_model = {}
        cur_model[estimator_name] = model_dict

        if metrics.loc["AUC ROC", estimator_name] > best_score:
            best_score = metrics.loc["AUC ROC", estimator_name]
            best_model = model_dict

        dumpObjects(
            {
                "model": best_model,  # Trained model
            },
            RESULTS_DIR / f"{str(clc).split('(')[0]}.pkl",
        )


if __name__ == "__main__":
    app()
