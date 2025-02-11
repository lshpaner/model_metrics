## Step 1. Load the requisite libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import model_metrics

print(model_metrics.__version__)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from model_tuner import Model, loadObjects
import model_tuner

from eda_toolkit import ensure_directory
from model_metrics import (
    summarize_model_performance,
    plot_calibration_curve,
    plot_conf_matrix,
)

plt.ion()  # enables interactive mode
plt.rcParams["figure.max_open_warning"] = 50  # or some other threshold

## Step 2. Append the correct paths
# Add the parent directory to sys.path to access relevant .py scripts

print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

print(f"Model Metrics version: {model_metrics.__version__}")
print(f"Model Metrics authors: {model_metrics.__author__} \n")

## Define base paths
## `base_path`` represents the parent directory of your current working directory
base_path = os.path.join(os.pardir)
## Go up one level from 'notebooks' to the parent directory, then into the
## 'results' folder

model_path = os.path.join(os.pardir, "model_files")
image_path_png = os.path.join(base_path, "images", "png_images")
image_path_svg = os.path.join(base_path, "images", "svg_images")

# Use the function to ensure the 'data' directory exists
ensure_directory(model_path)
ensure_directory(image_path_png)
ensure_directory(image_path_svg)


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
    pipelines_or_models=pipelines_or_models,
    X=X_test,
    y=y_test,
    model_titles=model_titles,
    # model_threshold=thresholds,
    return_df=True,
    # custom_threshold=0.7,
)

print(f"\n{model_summary}")


## Step 6. Plot the calibration curve

# Plot calibration curves in overlay mode
plot_calibration_curve(
    pipelines_or_models=pipelines_or_models,
    X=X_test,
    y=y_test,
    model_titles=model_titles,
    overlay=False,
    title="Calibration Curves",
    text_wrap=40,
    figsize=(12, 8),
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

plot_conf_matrix(
    pipelines_or_models=pipelines_or_models,
    X=X_test,
    y=y_test,
    model_titles=model_titles,
    cmap="Blues",
    text_wrap=40,
    save_plot=True,
    image_path_png=image_path_png,
    image_path_svg=image_path_svg,
    grid=False,
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

input("Press ENTER to quit...")
