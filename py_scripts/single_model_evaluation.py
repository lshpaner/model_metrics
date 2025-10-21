## Step 1. Load the requisite libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

import model_metrics

print(model_metrics.__version__)

from model_tuner import Model, loadObjects

from eda_toolkit import ensure_directory
from model_metrics import (
    summarize_model_performance,
    show_calibration_curve,
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
)

plt.ion()  # enables interactive mode

## Step 2. Append the correct paths
# Add the parent directory to sys.path to access relevant .py scripts

print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

print(f"Model Metrics version: {model_metrics.__version__}")
print(f"Model Metrics authors: {model_metrics.__author__} \n")

if __name__ == "__main__":

    argv = sys.argv[1:]
    model_path = argv[0]

    ## Define base paths
    ## `base_path`` represents the parent directory of your current working directory
    base_path = os.path.join(os.pardir)
    ## Go up one level from 'notebooks' to the parent directory, then into the
    ## 'results' folder

    # Define paths as Path objects
    model_path = Path(
        os.path.abspath(
            os.path.join(
                os.pardir,
                "model_metrics/model_files/single_model_classification_results",
            )
        )
    )
    image_path_png = model_path / "images" / "png_images"
    image_path_svg = model_path / "images" / "svg_images"

    # Ensure directories exist
    image_path_png.mkdir(parents=True, exist_ok=True)
    image_path_svg.mkdir(parents=True, exist_ok=True)

    # Use the function to ensure the 'data' directory exists
    ensure_directory(model_path)
    ensure_directory(image_path_png)
    ensure_directory(image_path_svg)

    print(f"Model path exists: {Path(model_path).exists()}")
    print(f"PNG image path exists: {Path(image_path_png).exists()}")
    print(f"SVG image path exists: {Path(image_path_svg).exists()}")

    ## Step 3. Load the model object and test data
    model = loadObjects(os.path.join(model_path, "logistic_regression_model.pkl"))

    X_test = pd.read_parquet(os.path.join(model_path, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(model_path, "y_test.parquet"))

    ## Step 4. Set the desired naming conventions
    pipelines_or_models = [
        model,
    ]

    # Model titles
    model_titles = [
        "Logistic Regression",
    ]

    ## Step 5. Summarize Model Performance
    model_summary = summarize_model_performance(
        model=pipelines_or_models,
        X=X_test,
        y=y_test,
        model_title=model_titles,
        # model_threshold=thresholds,
        return_df=True,
        # custom_threshold=0.7,
    )

    print(f"\n{model_summary}")

    ## Step 6. Plot the calibration curve

    # Plot calibration curves in overlay mode
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
        # grid=True,
        # gridlines=False,
        save_plot=True,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        linestyle_kwgs={"color": "black"},
    )

    ## Step 7. Plot the confusion matrix

    show_confusion_matrix(
        model=pipelines_or_models,
        X=X_test,
        y=y_test,
        model_title=model_titles,
        cmap="Blues",
        text_wrap=40,
        save_plot=True,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        subplots=False,
        n_cols=3,
        n_rows=1,
        figsize=(4, 4),
        show_colorbar=False,
        # label_fontsize=14,
        # tick_fontsize=12,
        inner_fontsize=10,
        class_report=True,
        # custom_threshold=0.5,
        # labels=False,
    )

    # Plot ROC curves
    show_roc_curve(
        model=pipelines_or_models,
        X=X_test,
        y=y_test,
        overlay=False,
        model_title=model_titles,
        decimal_places=3,
        # n_cols=3,
        # n_rows=1,
        # curve_kwgs={
        #     "Logistic Regression": {"color": "blue", "linewidth": 2},
        #     "SVM": {"color": "red", "linestyle": "--", "linewidth": 1.5},
        # },
        # linestyle_kwgs={"color": "grey", "linestyle": "--"},
        save_plot=True,
        subplots=False,
        figsize=(4, 4),
        # label_fontsize=16,
        # tick_fontsize=16,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        # gridlines=False,
    )

    # Plot PR curves
    show_pr_curve(
        model=pipelines_or_models,
        X=X_test,
        y=y_test,
        # x_label="Hello",
        model_title=model_titles,
        decimal_places=3,
        overlay=False,
        subplots=False,
        save_plot=True,
        image_path_png=image_path_png,
        image_path_svg=image_path_svg,
        figsize=(4, 4),
        # tick_fontsize=16,
        # label_fontsize=16,
        # grid=True,
        # gridlines=False,
    )

    input("Press ENTER to quit...")
