import typer
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import sys
from pathlib import Path

import model_tuner
from model_tuner import loadObjects

import model_metrics
from model_metrics import (
    summarize_model_performance,
    show_calibration_curve,
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
)

import mlflow
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

plt.ion()  # enables interactive mode

################################################################################
# Paths
################################################################################
# Anchored to this file rather than the working directory, matching the training
# script's RESULTS_DIR, so everything lands inside the repository no matter
# where the script is invoked from.

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = REPO_ROOT / "model_files"
RESULTS_DIR = REPO_ROOT / "model_files" / "results"
MLRUNS_DIR = REPO_ROOT / "mlruns"

IMAGE_PATH_PNG = RESULTS_DIR / "images" / "png_images"
IMAGE_PATH_SVG = RESULTS_DIR / "images" / "svg_images"

################################################################################
# MLflow configuration
################################################################################
# Evaluation attaches to the run the training script created rather than opening
# a second one, so the metrics and figures land alongside the model artifact and
# the registry sees a single coherent run.

TARGET = "income"  # outcome column; matches the training script

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
    """Locate an existing run by name, or None if it has not been created."""
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


def _log_summary(summary, prefix):
    """Log a summarize_model_performance frame as flat MLflow metrics.

    The frame carries metric names in a 'Metrics' column rather than the index,
    with one further column per model title. Non-numeric cells are skipped
    rather than coerced. With several models the title is folded into the metric
    name so the columns stay distinguishable.
    """
    if not isinstance(summary, pd.DataFrame):
        return

    label_col = next(
        (c for c in summary.columns if str(c).strip().lower() == "metrics"), None
    )
    if label_col is None:
        labels = [str(i) for i in summary.index]
        value_cols = list(summary.columns)
    else:
        labels = [str(v) for v in summary[label_col]]
        value_cols = [c for c in summary.columns if c != label_col]

    multi_model = len(value_cols) > 1

    for column in value_cols:
        for label, value in zip(labels, summary[column]):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            name = f"{column} {label}" if multi_model else label
            mlflow.log_metric(f"{prefix} {_clean_metric_key(name)}", numeric)


def _resolve(*candidates):
    """First path in the list that exists, else None."""
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    return None


def _load_model(estimator_name, clc):
    """Load the fitted model, preferring the MLflow artifact.

    The training script logs the bare estimator as model.pkl and separately
    dumps a {"model": ...} wrapper to RESULTS_DIR. The registry-backed artifact
    is preferred because it is the same object the run was scored on; the local
    wrapper is the fallback and gets unwrapped.
    """
    artifact = _resolve(
        *MLRUNS_DIR.glob(f"*/*/artifacts/{estimator_name}_{TARGET}/model.pkl")
    )
    if artifact is not None:
        print(f"Loading model from MLflow artifact: {artifact}")
        return loadObjects(str(artifact))

    local = RESULTS_DIR / f"{str(clc).split('(')[0]}.pkl"
    print(f"Loading model from local pickle: {local}")
    loaded = loadObjects(str(local))
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    return loaded


@app.command()
def main(
    model_type: str = "lr",
    pipeline_type: str = "orig",
):

    print()
    print(f"Model Tuner version: {model_tuner.__version__}")
    print(f"Model Metrics version: {model_metrics.__version__}")
    print()

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]
    model_title = str(clc).split("(")[0]

    ############################################################################
    # Ensure output directories exist
    ############################################################################
    IMAGE_PATH_PNG.mkdir(parents=True, exist_ok=True)
    IMAGE_PATH_SVG.mkdir(parents=True, exist_ok=True)

    print(f"Results path exists: {RESULTS_DIR.exists()}")
    print(f"PNG image path exists: {IMAGE_PATH_PNG.exists()}")
    print(f"SVG image path exists: {IMAGE_PATH_SVG.exists()}")

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
        print(f"\nResuming MLflow run '{run_name}' ({existing_run_id[:8]})")
    else:
        print(f"\nNo existing run named '{run_name}'; starting a new one")

    ############################################################################
    # Load the model object and test data
    ############################################################################
    # The training script writes the parquet files relative to its working
    # directory, so check the anchored location first and fall back to the
    # cwd-relative one.

    model = _load_model(estimator_name, clc)

    X_test_path = _resolve(
        PROCESSED_DATA_DIR / "X_test.parquet", Path("model_files/X_test.parquet")
    )
    y_test_path = _resolve(
        PROCESSED_DATA_DIR / "y_test.parquet", Path("model_files/y_test.parquet")
    )
    if X_test_path is None or y_test_path is None:
        raise FileNotFoundError(
            "Could not find X_test.parquet / y_test.parquet. Run the training "
            f"script first, or check {PROCESSED_DATA_DIR}."
        )

    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)

    print(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    ############################################################################
    # Naming conventions
    ############################################################################
    pipelines_or_models = [
        model,
    ]

    model_titles = [
        model_title,
    ]

    with mlflow.start_run(
        run_id=existing_run_id,
        run_name=None if existing_run_id else run_name,
    ):

        ########################################################################
        # Summarize model performance
        ########################################################################
        model_summary = summarize_model_performance(
            model=pipelines_or_models,
            X=X_test,
            y=y_test,
            model_title=model_titles,
            return_df=True,
        )

        print(f"\n{model_summary}")

        ########################################################################
        # Calibration curve
        ########################################################################
        show_calibration_curve(
            model=pipelines_or_models,
            X=X_test,
            y=y_test,
            model_title=model_titles,
            overlay=False,
            title="Calibration Curves",
            text_wrap=40,
            figsize=(4, 4),
            label_fontsize=14,
            tick_fontsize=9,
            bins=10,
            show_brier_score=True,
            save_plot=True,
            image_path_png=IMAGE_PATH_PNG,
            image_path_svg=IMAGE_PATH_SVG,
            linestyle_kwgs={"color": "black"},
        )

        ########################################################################
        # Confusion matrix
        ########################################################################
        show_confusion_matrix(
            model=pipelines_or_models,
            X=X_test,
            y=y_test,
            model_title=model_titles,
            cmap="Blues",
            text_wrap=40,
            save_plot=True,
            image_path_png=IMAGE_PATH_PNG,
            image_path_svg=IMAGE_PATH_SVG,
            subplots=False,
            n_cols=3,
            n_rows=1,
            figsize=(4, 4),
            show_colorbar=False,
            inner_fontsize=10,
            class_report=True,
        )

        ########################################################################
        # ROC curve
        ########################################################################
        show_roc_curve(
            model=pipelines_or_models,
            X=X_test,
            y=y_test,
            overlay=False,
            model_title=model_titles,
            decimal_places=3,
            save_plot=True,
            subplots=False,
            figsize=(4, 4),
            image_path_png=IMAGE_PATH_PNG,
            image_path_svg=IMAGE_PATH_SVG,
        )

        ########################################################################
        # Precision-recall curve
        ########################################################################
        show_pr_curve(
            model=pipelines_or_models,
            X=X_test,
            y=y_test,
            model_title=model_titles,
            decimal_places=3,
            overlay=False,
            subplots=False,
            save_plot=True,
            image_path_png=IMAGE_PATH_PNG,
            image_path_svg=IMAGE_PATH_SVG,
            figsize=(4, 4),
        )

        ########################################################################
        # Log the evaluation metrics and figures
        ########################################################################

        _log_summary(model_summary, "test")

        for score_name, threshold in (
            getattr(model, "threshold", {}) or {}
        ).items():
            mlflow.log_metric(
                f"threshold {_clean_metric_key(score_name)}", threshold
            )

        for directory, artifact_path in (
            (IMAGE_PATH_PNG, "figures/png"),
            (IMAGE_PATH_SVG, "figures/svg"),
        ):
            if any(Path(directory).iterdir()):
                mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

        mlflow.set_tags(
            {
                "evaluated": "true",
                "model_metrics_version": model_metrics.__version__,
            }
        )

        run = mlflow.active_run()
        print()
        print(f"Logged to MLflow: {MLRUNS_DIR}")
        print(f"  experiment : {experiment_name}")
        print(f"  run name   : {run_name}")
        print(f"  run id     : {run.info.run_id}")


if __name__ == "__main__":
    app()