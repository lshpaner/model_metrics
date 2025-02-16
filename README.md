[![PyPI](https://img.shields.io/pypi/v/model_metris.svg)](https://pypi.org/project/model_metrics/)
[![Downloads](https://pepy.tech/badge/model_metrics)](https://pepy.tech/project/model_metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/lshpaner/model_metrics/blob/main/LICENSE.md)

<br>

<img src="https://raw.githubusercontent.com/lshpaner/model_metrics/refs/heads/main/assets/mm_logo.svg" width="300" style="border: none; outline: none; box-shadow: none;" oncontextmenu="return false;">

<br> 

# Overview
The Model Metrics library is designed to facilitate the evaluation and interpretation of machine learning models. It provides functionality for generating predictions, computing model metrics, extracting coefficients, SHAP values, analyzing feature importance, and visualizing performance metrics through confusion matrices, ROC curves, precision-recall curves, and calibration plots.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Initializing the Model Metrics](#initializing-the-model-metrics)
4. [Generating Predictions](#generating-predictions)
5. [Extracting Global SHAP Values](#extracting-global-shap-values)
6. [Plotting Performance Metrics](#plotting-performance-metrics)
    - [Customizing Plot Styling](#customizing-plot-styling)
7. [Summarizing Model Performance](#summarizing-model-performance)
8. [Notes](#notes)
9. [Contributing](#contributing)
10. [License](#license)


## Features

* Prediction & Model Evaluation:
    * Supports multiple models and outcomes.
    * Computes metrics like precision, recall, F1-score, specificity, AUC-ROC, Brier Score, and Average Precision.
    * Allows for k-fold cross-validation evaluation.

* SHAP & Feature Importance:
    * Calculates SHAP values at both global and per-row levels.
    * Supports extraction of top-N most important features per row.
    * Computes model coefficients for feature importance assessment.

* Visualization & Reporting:
    * Confusion Matrix visualization with enhanced labeling.
    * ROC Curve, Precision-Recall Curve, and Calibration Curve plotting.
    * Grid-based layout for multi-model comparisons.
    * Customizable styling for plots, including overlays.

## Installation


```
pip install model_metrics
```

## Initializing the Model Metrics

```python
from model_metrics import ModelCalculator

model_dict = {"model": {"outcome1": trained_model1, "outcome2": trained_model2}}
outcomes = ["outcome1", "outcome2"]
calculator = ModelCalculator(model_dict, outcomes)
```

## Generating Predictions

```python
results_df = calculator.generate_predictions(
    X_test, y_test, calculate_shap=True, use_coefficients=True
)
```

## Extracting Global SHAP Values

```python
global_shap_df = calculator.generate_predictions(X_test, y_test, global_shap=True)
```

## Plotting Performance Metrics

```python
from model_evaluator import (
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
    show_calibration_curve,
)

# Confusion Matrix
show_confusion_matrix(model, X_test, y_test)

# ROC Curve
show_roc_curve(model, X_test, y_test, overlay=True)

# Precision-Recall Curve
show_pr_curve(model, X_test, y_test, overlay=True)

# Calibration Curve
show_calibration_curve(model, X_test, y_test)
```

### Customizing Plot Styling

```python
curve_style = {"color": "blue", "linestyle": "--"}
show_roc_curve(model, X_test, y_test, curve_kwgs=curve_style)
```

## Summarizing Model Performance

```python
from model_evaluator import summarize_model_performance

metrics_df = summarize_model_performance(model, X_test, y_test, return_df=True)
print(metrics_df)
```

## Notes

- Ensure models support `predict_proba` for probability-based metrics.

- SHAP calculations may be computationally expensive on large datasets.

- Supports both standalone models and scikit-learn pipelines.

## Contributing
We welcome contributions! If you have suggestions or improvements, please submit an issue or pull request. Follow the standard GitHub flow for contributing.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/lshpaner/model_metrics/blob/main/LICENSE.md) file for details.

For more detailed documentation, refer to the docstrings within each function.
