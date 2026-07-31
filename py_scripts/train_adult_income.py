import typer
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
import re
import shutil
import sys
import model_tuner
from model_tuner import Model, dumpObjects
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.base import clone
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Add the parent directory to sys.path to access 'functions.py'
print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")

from py_scripts.model_params import model_definitions

app = typer.Typer()

PROCESSED_DATA_DIR = Path("model_files")
MODELS_DIR = Path("model_files")

# Define RESULTS_DIR correctly
RESULTS_DIR = Path(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "model_files/results")
    )
)

# Ensure the directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

################################################################################
# MLflow configuration
################################################################################
# The registry reads a local FileStore off disk rather than talking to a
# tracking server, so runs go to an mlruns/ tree at the repository root. Two
# conventions matter for it to find anything: the model artifact must be named
# model.pkl, and it must sit in a subdirectory named "<algo>_<TARGET>" so the
# algorithm can be recovered from the folder while the run name carries the
# variant.

TARGET = "income"  # outcome column; also the token stripped from artifact names

# Anchored to this file rather than the working directory, so the tree lands
# inside the repository no matter where the script is invoked from.
MLRUNS_DIR = Path(__file__).resolve().parents[1] / "mlruns"

# MLflow 3.x puts the filesystem tracking backend in maintenance mode and
# raises unless this is set. Harmless on versions that predate the check.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def _clean_metric_key(key):
    """Make a metric name safe for MLflow's FileStore.

    Keys become filenames, so a '/' in something like 'Precision/PPV' would
    create a subdirectory the registry then skips.
    """
    return re.sub(r"[^\w\-. ]+", "_", str(key).replace("/", "_")).strip()


def _find_run_id(experiment_name, run_name):
    """Locate an existing run by name, or None if it has not been created.

    Reruns reuse the run rather than piling up a new run_id each time, so the
    experiment holds one run per variant and the registry is not left choosing
    between near-identical candidates.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0].info.run_id if runs else None


def _log_param(key, value):
    """Log a parameter, replacing a stale value left by an earlier run.

    MLflow treats params as immutable within a run, so resuming a run and
    re-logging a hyperparameter that has since changed raises. Because the run
    is deliberately reused, the FileStore entry is rewritten directly in that
    case. That is backend-specific, but FileStore is the backend the registry
    reads anyway. Falls back to a tag if the directory is not where expected.
    """
    try:
        mlflow.log_param(key, value)
        return
    except MlflowException:
        pass

    run = mlflow.active_run()
    param_path = (
        MLRUNS_DIR
        / run.info.experiment_id
        / run.info.run_id
        / "params"
        / str(key)
    )
    if param_path.parent.is_dir():
        param_path.write_text(str(value))
    else:
        mlflow.set_tag(f"param.{key}", value)


def _log_params(params):
    """Log a dict of parameters, tolerating changed values on a reused run."""
    for key, value in params.items():
        _log_param(key, value)


def _log_metrics(metrics, prefix):
    """Log a metric dict under a split prefix, skipping non-numeric values."""
    if not isinstance(metrics, dict):
        return
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            mlflow.log_metric(f"{prefix} {_clean_metric_key(key)}", numeric)


def _log_model_artifact(model, estimator_name, staging_dir):
    """Write the bare model as model.pkl and log it where the registry looks.

    The pickle has to be the estimator itself, not a wrapper dict: the registry
    calls predict_proba on whatever it loads.
    """
    artifact_dir = f"{estimator_name}_{TARGET}"
    staged = Path(staging_dir) / "model.pkl"
    dumpObjects(model, staged)
    mlflow.log_artifact(str(staged), artifact_path=artifact_dir)
    staged.unlink(missing_ok=True)
    return artifact_dir


@app.command()
def main(
    model_type: str = "lr",
    pipeline_type: str = "orig",
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
    early_stop = model_definitions[model_type]["early"]

    best_model = {}
    best_score = 0

    model_dict = {}
    metrics = {}

    print()

    pipeline = [
        ("StandardScalar", StandardScaler()),
        ("Preprocessor", SimpleImputer()),
    ]

    print()
    print("=" * 60)

    ############################################################################
    # Point MLflow at the repository's FileStore
    ############################################################################
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    experiment_name = f"{TARGET}_model"
    mlflow.set_experiment(experiment_name)
    run_name = f"{estimator_name}_{pipeline_type}_training"

    existing_run_id = _find_run_id(experiment_name, run_name)
    if existing_run_id:
        print(f"Reusing MLflow run '{run_name}' ({existing_run_id[:8]})")
    else:
        print(f"Starting new MLflow run '{run_name}'")

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
    )

    with mlflow.start_run(
        run_id=existing_run_id,
        run_name=None if existing_run_id else run_name,
    ):

        ####################################################################
        #################### Extract Split Data Subsets ####################
        ####################################################################

        model_dict.grid_search_param_tuning(X, y, f1_beta_tune=True)

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

        ####################################################################
        ################## Log parameters, metrics, model ##################
        ####################################################################

        mlflow.set_tags(
            {
                "estimator_name": estimator_name,
                "model_type": "classification",
                "pipeline_type": pipeline_type,
                "target": TARGET,
                "model_tuner_version": model_tuner.__version__,
                "kfold": "False",
                "calibrate": str(model_dict.calibrate),
                "boost_early": str(early_stop),
            }
        )

        _log_params(
            {
                "model_type": model_type,
                "estimator": str(clc).split("(")[0],
                "scoring": model_dict.scoring[0],
                "randomized_grid": False,
                "stratify_y": True,
                "random_state": rstate,
                "n_samples": len(y),
                "n_features": X.shape[1],
                "prevalence": round(float(y.mean()), 4),
            }
        )

        best = model_dict.best_params_per_score.get(model_dict.scoring[0], {})
        for key, value in (best.get("params") or {}).items():
            _log_param(key, value)

        # Metrics on the full frame, matching what the script already computes.
        _log_metrics(return_metrics_dict, "full")

        # Held-out splits, so the registry's verify_entry() has something
        # split-specific to compare a reloaded model against.
        _log_metrics(
            model_dict.return_metrics(
                X_valid, y_valid, optimal_threshold=True, model_metrics=True,
                return_dict=True,
            ),
            "valid",
        )
        _log_metrics(
            model_dict.return_metrics(
                X_test, y_test, optimal_threshold=True, model_metrics=True,
                return_dict=True,
            ),
            "test",
        )

        for score_name, threshold in (
            getattr(model_dict, "threshold", {}) or {}
        ).items():
            mlflow.log_metric(
                f"threshold {_clean_metric_key(score_name)}", threshold
            )

        artifact_dir = _log_model_artifact(
            model_dict, estimator_name, RESULTS_DIR
        )

        run = mlflow.active_run()
        print()
        print(f"Logged to MLflow: {MLRUNS_DIR}")
        print(f"  experiment : {experiment_name}")
        print(f"  run name   : {run_name}")
        print(f"  run id     : {run.info.run_id}")
        print(f"  artifact   : {artifact_dir}/model.pkl")


if __name__ == "__main__":
    app()