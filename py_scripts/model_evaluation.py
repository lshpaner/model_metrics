## Step 1. Load the requisite libraries
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to sys.path to access 'functions.py'
print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from model_tuner import Model, loadObjects
import model_tuner

from model_metrics import ModelEvaluationMetrics

## Step 2. Append the correct paths
# Add the parent directory to sys.path to access 'functions.py'
print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

print(f"Model Tuner version: {model_tuner.__version__}")
print(f"Model Tuner authors: {model_tuner.__author__} \n")

model_path = "../results/"


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

## Step 5. Set the ModelEvaluationMetrics to new variable and run evaluator

model_summary = evaluator.summarize_model_performance(
    pipelines_or_models=pipelines_or_models,
    X=X_test,
    y=y_test,
    model_titles=model_titles,
    # model_threshold=thresholds,
    return_df=True,
    # custom_threshold=0.7,
)

print(model_summary)
