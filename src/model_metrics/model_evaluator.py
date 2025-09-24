import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import os
from tqdm import tqdm
import textwrap

import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)

################################################################################
############################## Helper Functions ################################
################################################################################


def save_plot_images(filename, save_plot, image_path_png, image_path_svg):
    """
    Save the plot to specified directories.
    """
    if save_plot:
        if not (image_path_png or image_path_svg):
            raise ValueError(
                "save_plot is set to True, but no image path is provided. "
                "Please specify at least one of `image_path_png` or `image_path_svg`."
            )
        if image_path_png:
            os.makedirs(image_path_png, exist_ok=True)
            plt.savefig(
                os.path.join(image_path_png, f"{filename}.png"),
                bbox_inches="tight",
            )
        if image_path_svg:
            os.makedirs(image_path_svg, exist_ok=True)
            plt.savefig(
                os.path.join(image_path_svg, f"{filename}.svg"),
                bbox_inches="tight",
            )


def get_predictions(model, X, y, model_threshold, custom_threshold, score):
    """
    Get predictions and threshold-adjusted predictions for a given model.
    Handles both single-model and k-fold cross-validation scenarios.

    Parameters:
    - model: The model or pipeline object to use for predictions.
    - X: Features for prediction.
    - y: True labels.
    - model_threshold: Predefined threshold for the model.
    - custom_threshold: User-defined custom threshold (overrides model_threshold).
    - score: The scoring metric to determine the threshold.

    Returns:
    - aggregated_y_true: Ground truth labels.
    - aggregated_y_prob: Predicted probabilities.
    - aggregated_y_pred: Threshold-adjusted predictions.
    - threshold: The threshold used for predictions.
    """
    # Determine the model to use for predictions
    test_model = model.test_model if hasattr(model, "test_model") else model

    # Default threshold
    threshold = 0.5

    # Set the threshold based on custom_threshold, model_threshold, or model scoring
    if custom_threshold:
        threshold = custom_threshold
    elif model_threshold:
        if score is not None:
            threshold = getattr(model, "threshold", {}).get(score, 0.5)
        else:
            threshold = getattr(model, "threshold", {}).get(
                getattr(model, "scoring", [0])[0], 0.5
            )

    # Handle k-fold logic if the model uses cross-validation
    if hasattr(model, "kfold") and model.kfold:
        print("\nRunning k-fold model metrics...\n")
        aggregated_y_true = []
        aggregated_y_pred = []
        aggregated_y_prob = []

        for fold_idx, (train, test) in tqdm(
            enumerate(model.kf.split(X, y), start=1),
            total=model.kf.get_n_splits(),
            desc="Processing Folds",
        ):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            # Fit and predict for this fold
            test_model.fit(X_train, y_train.values.ravel())

            if hasattr(test_model, "predict_proba"):
                y_pred_proba = test_model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > threshold).astype(int)
            else:
                # Fallback if predict_proba is not available
                y_pred_proba = test_model.predict(X_test)
                y_pred = (y_pred > threshold).astype(int)

            aggregated_y_true.extend(y_test.values.tolist())
            aggregated_y_pred.extend(y_pred.tolist())
            aggregated_y_prob.extend(y_pred_proba.tolist())
    else:
        # Single-model scenario
        aggregated_y_true = y

        if hasattr(test_model, "predict_proba"):
            aggregated_y_prob = test_model.predict_proba(X)[:, 1]
            aggregated_y_pred = (aggregated_y_prob > threshold).astype(int)
        else:
            # Fallback if predict_proba is not available
            aggregated_y_prob = test_model.predict(X)
            aggregated_y_pred = (aggregated_y_prob > threshold).astype(int)

    return aggregated_y_true, aggregated_y_prob, aggregated_y_pred, threshold


# Helper function
def extract_model_name(pipeline_or_model):
    """Extracts the final model name from a pipeline or standalone model."""
    if hasattr(pipeline_or_model, "steps"):  # It's a pipeline
        return pipeline_or_model.steps[-1][
            1
        ].__class__.__name__  # Final estimator's class name
    return pipeline_or_model.__class__.__name__  # Individual model class name


################################################################################
######################### Summarize Model Performance ##########################
################################################################################


def has_feature_importances(model):
    """Check if the model has a feature_importances_ attribute."""
    return hasattr(model, "feature_importances_")


def summarize_model_performance(
    model=None,
    X=None,
    y_prob=None,
    y_pred=None,
    y=None,
    model_type="classification",
    model_threshold=None,
    model_title=None,
    custom_threshold=None,
    score=None,
    return_df=False,
    overall_only=False,
    decimal_places=3,
):
    """
    Summarize performance metrics for classification or regression models.

    This function computes and displays (or returns) key performance metrics
    for one or more models. It supports classification and regression tasks,
    and can operate either from model objects with feature data, or directly
    from provided predictions/probabilities. Outputs include classification
    metrics, regression metrics, model thresholds, coefficients, and feature
    importances where applicable.

    Parameters
    ----------
    model : estimator, list of estimators, or None, default=None
        A single trained model or a list of models/pipelines.
        If None, `y_prob` or `y_pred` must be provided.

    X : array-like or None, default=None
        Feature matrix used for evaluation. Required if `model` is given
        without `y_prob`/`y_pred`. Must have column names if feature
        importance or coefficients are extracted.

    y_prob : array-like, list of arrays, or None, default=None
        Predicted probabilities for classification models. Can be used
        instead of `model` and `X`.

    y_pred : array-like, list of arrays, or None, default=None
        Predicted values for regression models or predicted class labels for
        classification. Can be used instead of `model` and `X`.

    y : array-like
        Ground truth labels or target values.

    model_type : {"classification", "regression"}, default="classification"
        Type of task. Determines which metrics are computed.

    model_threshold : float, dict, or None, default=None
        Decision threshold(s) for classification models. Can be a single float
        applied to all models or a dict keyed by model title or class name.
        Ignored if `custom_threshold` is set.

    model_title : str, list of str, pandas.Series, or None, default=None
        Custom model names for display. If None, defaults to "Model_1",
        "Model_2", etc.

    custom_threshold : float or None, default=None
        Overrides `model_threshold` and applies a fixed threshold across all
        classification models. When set, the "Model Threshold" row is excluded.

    score : str or None, default=None
        Optional scoring metric used when thresholds are resolved from
        `get_predictions`.

    return_df : bool, default=False
        If True, returns results as a pandas DataFrame.
        If False, prints a formatted table.

    overall_only : bool, default=False
        For regression models only, if True, only overall metrics are returned
        without coefficients or feature importances.

    decimal_places : int, default=3
        Number of decimal places to round results.

    Returns
    -------
    pandas.DataFrame or None
        - If `return_df=True`, returns a DataFrame of metrics.
        - Otherwise, prints metrics to stdout and returns None.

    Raises
    ------
    ValueError
        - If `model_type` is not "classification" or "regression".
        - If `overall_only=True` is used with classification models.
        - If neither (`model` and `X`) nor (`y_prob` or `y_pred`) are provided.
    """

    if not (
        (model is not None and X is not None)
        or y_prob is not None
        or y_pred is not None
    ):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if model_type == "classification":
        if isinstance(y_prob, list) and isinstance(y_prob[0], float):
            y_probs = [y_prob]
        else:
            y_probs = y_prob
    else:
        if isinstance(y_pred, list) and isinstance(y_pred[0], float):
            y_preds = [y_pred]
        else:
            y_preds = y_pred

    if model:
        num_models = len(model)
    elif y_prob:
        num_models = len(y_probs)
    else:
        num_models = len(y_preds)

    if y_prob is not None or y_pred is not None:
        model = [None] * num_models

    # Check if model is iterable; if not, wrap it in a list
    try:
        iter(model)
        models = model if isinstance(model, list) else [model]
    except TypeError:
        models = [model]  # Handle non-iterable objects like a single model

    model_type = model_type.lower()
    if model_type not in ["classification", "regression"]:
        raise ValueError(
            "Invalid model_type. Must be 'classification' or 'regression'."
        )

    if model_type == "classification" and overall_only:
        raise ValueError(
            "The 'overall_only' option is only valid for regression models. "
            "It cannot be used with classification."
        )

    metrics_data = []

    # Normalize model_title input
    if model_title is None:
        model_title = [f"Model_{i+1}" for i in range(len(models))]
    elif isinstance(model_title, str):
        model_title = [model_title]
    elif isinstance(model_title, pd.Series):
        model_title = model_title.tolist()
    elif not isinstance(model_title, list):
        raise TypeError(
            "model_title must be a string, list of strings, Series, or None."
        )

    for i, model in enumerate(models):
        name = model_title[i]
        if model_type == "classification":

            if X is None:
                y_true = y
                y_prob = y_probs[i]
                threshold = 0.5
                if custom_threshold:
                    threshold = custom_threshold
                if model_threshold:
                    threshold = model_threshold[name]
            else:
                # get y_true and y_prob from your helper, but ignore its threshold
                y_true, y_prob, _, threshold = get_predictions(
                    model, X, y, model_threshold, custom_threshold, score
                )

            # always derive predictions from the resolved threshold
            y_pred = (np.asarray(y_prob) > float(threshold)).astype(int)

            # Compute metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)  # Sensitivity
            specificity = recall_score(y_true, y_pred, pos_label=0)
            auc_roc = roc_auc_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            f1 = f1_score(y_true, y_pred)

            # Append metrics for this model
            model_metrics = {
                "Model": name,
                "Precision/PPV": round(precision, decimal_places),
                "Average Precision": round(avg_precision, decimal_places),
                "Sensitivity/Recall": round(recall, decimal_places),
                "Specificity": round(specificity, decimal_places),
                "F1-Score": round(f1, decimal_places),
                "AUC ROC": round(auc_roc, decimal_places),
                "Brier Score": round(brier, decimal_places),
                "Model Threshold": (
                    round(threshold, decimal_places) if threshold is not None else None
                ),
            }

            metrics_data.append(model_metrics)

        elif model_type == "regression":

            if model and isinstance(model, sm.OLS):
                # Always add a constant term for all regression models
                X_with_intercept = sm.add_constant(X)
                # For statsmodels OLS, predict and extract coefficients directly
                y_pred = model.predict(X_with_intercept)
                coefficients = pd.Series(model.params.round(decimal_places)).to_dict()
            else:
                # For scikit-learn models, predict on X_with_intercept, adjusting for intercept
                if X is None:
                    y_pred = y_preds[i]
                else:
                    try:
                        # Always add a constant term for all regression models
                        X_with_intercept = sm.add_constant(X)
                        y_pred = model.predict(X_with_intercept)
                    except ValueError:
                        # If the model doesn’t accept the constant, predict on original X
                        y_pred = model.predict(X)

                # Extract coefficients for scikit-learn models
                if hasattr(model, "coef_") or (
                    type(model) is Pipeline and hasattr(model[-1], "coef_")
                ):

                    if hasattr(model, "coef_"):
                        coef_ = model.coef_
                        intercept_ = model.intercept_
                    else:
                        coef_ = model[-1].coef_
                        intercept_ = model[-1].intercept_

                    # Get feature names from X (excluding const for coef_)
                    feature_names = (
                        X.columns if isinstance(X, pd.DataFrame) else range(len(coef_))
                    )
                    coefficients = (
                        pd.Series(coef_, index=feature_names)
                        .round(decimal_places)
                        .to_dict()
                    )
                    # Add intercept if it exists
                    coefficients["const"] = round(intercept_, decimal_places)
                else:
                    coefficients = {}

            # Ensure y and y_pred are 1D NumPy arrays
            y = np.asarray(y).ravel()
            y_pred = np.asarray(y_pred).ravel()

            # Compute regression metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            exp_var = explained_variance_score(y, y_pred)
            r2 = r2_score(y, y_pred)
            # Handle MAPE safely to avoid division errors
            nonzero_mask = y != 0  # Avoid division by zero
            if np.any(nonzero_mask):  # Ensure at least one valid value
                mape = (
                    np.mean(
                        np.abs(
                            (y[nonzero_mask] - y_pred[nonzero_mask]) / y[nonzero_mask]
                        )
                    )
                    * 100
                )
            else:
                mape = np.nan  # If all y-values are zero, return NaN

            # Base columns for all regression models (without Feat. Imp. initially)
            base_columns = {
                "Model": name,
                "Metric": "Overall Metrics",
                "Variable": "",  # Empty since this isn’t a coefficient
                "Coefficient": "",
                "MAE": round(mae, decimal_places),
                "MAPE": round(mape, decimal_places) if not pd.isna(mape) else "NaN",
                "MSE": round(mse, decimal_places),
                "RMSE": round(rmse, decimal_places),
                "Expl. Var.": round(exp_var, decimal_places),
                "R^2 Score": round(r2, decimal_places),
            }

            # Append overall metrics
            metrics_data.append(base_columns)

            # Only append coefficients and Feat. Imp. if not overall_only
            if not overall_only:
                # Determine feature importances
                feature_importance = {}
                if has_feature_importances(model):
                    # Use built-in feature importances for tree-based models (only on original features, not const)
                    feature_importance = (
                        pd.Series(model.feature_importances_, index=X.columns)
                        .round(decimal_places)
                        .to_dict()
                    )

                # Append coefficient rows for models with coefficients (no P-value)
                if coefficients:
                    # Ensure 'const' is first if it exists
                    if "const" in coefficients:
                        metrics_data.append(
                            {
                                "Model": name,
                                "Metric": "Coefficient",
                                "Variable": "const",
                                "Coefficient": coefficients["const"],
                                "MAE": "",
                                "MAPE": "",
                                "MSE": "",
                                "RMSE": "",
                                "Expl. Var.": "",
                                "R^2 Score": "",
                            }
                        )
                    # Then append the remaining features
                    for feature in [f for f in coefficients if f != "const"]:
                        metrics_data.append(
                            {
                                "Model": name,
                                "Metric": "Coefficient",
                                "Variable": feature,
                                "Coefficient": coefficients[feature],
                                "MAE": "",
                                "MAPE": "",
                                "MSE": "",
                                "RMSE": "",
                                "Expl. Var.": "",
                                "R^2 Score": "",
                            }
                        )

                # Append feature importance only if the model has feature importances
                if feature_importance and has_feature_importances(model):
                    for feature in feature_importance:
                        metrics_data.append(
                            {
                                "Model": name,
                                "Metric": "Feat. Imp.",
                                "Variable": feature,
                                "Coefficient": "",
                                "MAE": "",
                                "MAPE": "",
                                "MSE": "",
                                "RMSE": "",
                                "Expl. Var.": "",
                                "R^2 Score": "",
                                "Feat. Imp.": feature_importance[feature],
                            }
                        )

    # Check if any model in the list has feature importances
    has_feature_importances_any = any(has_feature_importances(m) for m in models)

    # Define base columns for classification and regression separately
    if model_type == "classification":
        base_columns = [
            "Model",
            "Precision/PPV",
            "Average Precision",
            "Sensitivity/Recall",
            "Specificity",
            "F1-Score",
            "AUC ROC",
            "Brier Score",
            "Model Threshold",
        ]
    if model_type == "regression":
        base_columns = [
            "Model",
            "Metric",
            "Variable",
            "Coefficient",
            "MAE",
            "MAPE",
            "MSE",
            "RMSE",
            "Expl. Var.",
            "R^2 Score",
        ]

    # Add "Feature Importance" column only if any model has feature importances
    # and it's regression
    columns = base_columns.copy()
    if has_feature_importances_any and model_type == "regression" and not overall_only:
        columns.insert(4, "Feat. Imp.")  # Insert after "Coefficient"

    # Create DataFrame with the determined columns
    metrics_df = pd.DataFrame(metrics_data, columns=columns)

    # Fill empty values for better printing
    metrics_df = metrics_df.fillna("").astype(str)

    if model_type == "regression":
        # Remove "Feature Importance" column if no model has feature importances
        if not has_feature_importances_any:
            if "Feat. Imp." in metrics_df.columns:
                metrics_df = metrics_df.drop(
                    columns=["Feat. Imp."],
                    errors="ignore",
                )

        # Check if any model in the list (or its pipeline's final step) has coef_
        has_any_coef = any(
            hasattr(m, "coef_") or (type(m) is Pipeline and hasattr(m[-1], "coef_"))
            for m in models
        )
        # Remove "Coefficient" and "Variable" columns if no model has coef_
        if not has_any_coef:
            if "Coefficient" in metrics_df.columns:
                metrics_df = metrics_df.drop(
                    columns=["Coefficient", "Variable"],
                    errors="ignore",
                )

    if overall_only:
        if model_type == "regression":
            metrics_df = (
                metrics_df[metrics_df["Metric"] == "Overall Metrics"]
                .drop(columns=["Variable", "Coefficient"], errors="ignore")
                .reset_index(drop=True)
            )

            if not has_feature_importances_any:
                if "Feat. Imp." in metrics_df.columns:
                    metrics_df = metrics_df.drop(
                        columns=["Feat. Imp."],
                        errors="ignore",
                    )
            metrics_df.index = [""] * len(metrics_df)

    # **Manual formatting**
    if not return_df:
        if model_type == "classification":
            # Transpose for classification: metrics as rows, models as columns
            print("Model Performance Metrics (Transposed):")
            metrics_df = metrics_df.set_index(
                "Model"
            ).T  # Transpose the DataFrame for printing
            col_widths = {
                col: max(metrics_df[col].astype(str).map(len).max(), len(str(col))) + 2
                for col in metrics_df.columns
            }
            col_widths["Metrics"] = (
                max(metrics_df.index.astype(str).map(len).max(), len("Metrics")) + 2
            )
            separator = "-" * (sum(col_widths.values()) + len(col_widths) * 3)

            # Print header (all columns right-aligned, including "Metrics")
            print(separator)
            header = (
                "Metrics".rjust(col_widths["Metrics"])
                + " | "
                + " | ".join(
                    f"{str(col).rjust(col_widths[col])}" for col in metrics_df.columns
                )
            )
            print(header)
            print(separator)

            # Print each metric row with all values right-aligned
            for metric, row_data in metrics_df.iterrows():
                row = f"{metric.rjust(col_widths['Metrics'])} | " + " | ".join(
                    f"{str(row_data[col]).rjust(col_widths[col])}"
                    for col in metrics_df.columns
                )
                print(row)
            print(separator)

        else:
            # Regression formatting with all columns right-aligned
            col_widths = {
                col: max(metrics_df[col].astype(str).map(len).max(), len(col)) + 2
                for col in metrics_df.columns
            }
            separator = "-" * (sum(col_widths.values()) + len(col_widths) * 3)

            # Print header (all columns right-aligned)
            print("Model Performance Metrics")
            print(separator)
            print(
                " | ".join(
                    f"{col.rjust(col_widths[col])}" for col in metrics_df.columns
                ),
            )
            print(separator)

            # Track the previous model name for regression models only
            prev_model = None
            for i, (_, row_data) in enumerate(metrics_df.iterrows()):
                current_model = row_data["Model"] if "Model" in row_data else None
                if (
                    model_type == "regression"
                    and current_model
                    and current_model != prev_model
                    and i > 0
                ):
                    print(separator)
                row = " | ".join(
                    f"{str(row_data[col]).rjust(col_widths[col])}"
                    for col in metrics_df.columns
                )
                print(row)
                prev_model = (
                    current_model
                    if model_type == "regression" and current_model
                    else prev_model
                )
            print(separator)
    else:
        if model_type == "classification":
            metrics_df = metrics_df.set_index("Model").T.reset_index()
            metrics_df.columns.name = None
            metrics_df.rename(columns={"index": "Metrics"}, inplace=True)
        metrics_df.index = [""] * len(metrics_df)
        return metrics_df


################################################################################
############################## Confusion Matrix ################################
################################################################################


def show_confusion_matrix(
    model=None,
    X=None,
    y_prob=None,
    y=None,
    model_title=None,
    title=None,
    model_threshold=None,
    custom_threshold=None,
    class_labels=None,
    cmap="Blues",
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    figsize=(8, 6),
    labels=True,
    label_fontsize=12,
    tick_fontsize=10,
    inner_fontsize=10,
    subplots=False,
    score=None,
    class_report=False,
    **kwargs,
):
    """
    Generate and display confusion matrices for one or multiple models.

    This function computes confusion matrices for classifiers and visualizes
    them with customizable formatting, threshold adjustments, and optional
    classification reports. Supports both individual and subplot-based plots.

    Parameters:
    - model (estimator or list, optional): A single model or a list of
      models or pipelines.
    - X (array-like, optional): Feature matrix for predictions. Required
      when `model` is provided and `y_prob` is not.
    - y_prob (array-like or list of array-like, optional): Predicted
      probabilities for the positive class. Can be provided instead of
      `model` and `X`.
    - y (array-like): True labels.
    - model_title (str or list of str, optional): Custom titles for models.
      If a single string is provided it is converted to a list. If None,
      defaults to "Model 1", "Model 2", etc.
    - title (str or None, optional): Plot title. If None, a default title
      including the applied threshold is used. If "", no title is shown.
    - model_threshold (float or dict, optional): Decision threshold to apply
      when converting probabilities to class labels. If a dict is provided,
      it may map by model title or model class name. Ignored if
      `custom_threshold` is provided.
    - custom_threshold (float or None, optional): Explicit threshold override
      applied to all models. If set, it takes precedence over
      `model_threshold`.
    - class_labels (list, optional): Custom class names for axis tick labels
      and display labels.
    - cmap (str, default="Blues"): Colormap for visualization.
    - save_plot (bool, default=False): Whether to save plots to disk.
    - image_path_png (str, optional): Path to save PNG images.
    - image_path_svg (str, optional): Path to save SVG images.
    - text_wrap (int, optional): Maximum width for wrapping long titles.
    - figsize (tuple, default=(8, 6)): Figure size for each confusion matrix.
    - labels (bool, default=True): Whether to show TN, FP, FN, TP text
      labels inside the cells.
    - label_fontsize (int, default=12): Font size for axis labels and titles.
    - tick_fontsize (int, default=10): Font size for tick labels.
    - inner_fontsize (int, default=10): Font size for numbers inside cells.
    - subplots (bool, default=False): If True, display multiple plots in a
      subplot layout.
    - score (str, optional): Metric name used by threshold selection when
      predictions are derived via `get_predictions`.
    - class_report (bool, default=False): If True, print the scikit-learn
      classification report for each model.
    - **kwargs: Additional options.
        - n_cols (int, optional): Number of columns when `subplots=True`.
        - show_colorbar (bool, optional): Whether to show the colorbar.

    Returns:
    - None

    Raises:
    - ValueError: If neither (`model` and `X`) nor `y_prob` is provided.
    """

    if not ((model is not None and X is not None) or y_prob is not None):
        raise ValueError("You need to provide model and X or y_prob")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if isinstance(y_prob, list) and isinstance(y_prob[0], float):
        y_probs = [y_prob]
    else:
        y_probs = y_prob

    num_models = len(model) if model else len(y_probs)

    if y_prob is not None:
        model = [None] * num_models

    # Normalize model_title input
    if model_title is None:
        model_title = [f"Model {i+1}" for i in range(len(model))]
    elif isinstance(model_title, str):
        model_title = [model_title]
    elif not isinstance(model_title, list):
        raise TypeError("model_title must be a string, a list of strings, or None.")

    # Setup subplots if enabled
    if subplots:
        n_cols = kwargs.get("n_cols", 2)
        n_rows = (len(model) + n_cols - 1) // n_cols
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
        )
        axes = axes.flatten()
    else:
        axes = [None] * len(model)

    for idx, (m, ax) in enumerate(zip(model, axes)):
        # Determine the model name
        if model_title:
            name = model_title[idx]
        else:
            name = extract_model_name(m)

        if X is None:
            y_true = y
            y_prob = y_probs[idx]
            threshold = 0.5
            if custom_threshold:
                threshold = custom_threshold
            if model_threshold:
                threshold = model_threshold[name]
            y_pred = (np.asarray(y_prob) > float(threshold)).astype(int)
        else:
            y_true, _, y_pred, threshold = get_predictions(
                m,
                X,
                y,
                model_threshold,
                custom_threshold,
                score,
            )
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create confusion matrix DataFrame
        conf_matrix_df = pd.DataFrame(
            cm,
            index=(
                [f"Actual {label}" for label in class_labels]
                if class_labels
                else ["Actual 0", "Actual 1"]
            ),
            columns=(
                [f"Predicted {label}" for label in class_labels]
                if class_labels
                else ["Predicted 0", "Predicted 1"]
            ),
        )

        print(f"Confusion Matrix for {name}: \n")
        print(f"{conf_matrix_df}\n")
        if class_report:
            print(f"Classification Report for {name}: \n")
            print(classification_report(y_true, y_pred))

        # Plot the confusion matrix
        # Use ConfusionMatrixDisplay with custom class_labels
        if class_labels:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[f"{label}" for label in class_labels],
            )
        else:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["0", "1"]
            )
        show_colorbar = kwargs.get("show_colorbar", True)
        if subplots:
            if "colorbar" in disp.plot.__code__.co_varnames:
                disp.plot(cmap=cmap, ax=ax, colorbar=show_colorbar)
            else:
                disp.plot(cmap=cmap, ax=ax)
        else:
            _, ax = plt.subplots(figsize=figsize)
            if "colorbar" in disp.plot.__code__.co_varnames:
                disp.plot(cmap=cmap, ax=ax, colorbar=show_colorbar)
            else:
                disp.plot(cmap=cmap, ax=ax)

        # Ensure text annotations are not duplicated
        if hasattr(disp, "text_") and disp.text_ is not None:
            unique_texts = set()
            for text_obj in disp.text_.ravel():
                text_value = text_obj.get_text()
                if text_value in unique_texts:
                    text_obj.set_text("")  # Clear duplicate text
                else:
                    unique_texts.add(text_value)

        for i in range(disp.text_.shape[0]):
            for j in range(disp.text_.shape[1]):
                new_value = disp.confusion_matrix[i, j]
                disp.text_[i, j].set_text(f"{new_value:,}")

        # **Forcefully Remove the Colorbar If It Exists**
        if not show_colorbar:
            # Locate colorbar within the figure and remove it
            for cb in ax.figure.get_axes():
                if isinstance(cb, mcolorbar.Colorbar):
                    cb.remove()

            # Additional safeguard: clear colorbar from the ConfusionMatrixDisplay
            if hasattr(disp, "im_") and disp.im_ is not None:
                if hasattr(disp.im_, "colorbar") and disp.im_.colorbar is not None:
                    try:
                        disp.im_.colorbar.remove()
                    except Exception as e:
                        print(f"Warning: Failed to remove colorbar: {e}")

        if title is None:
            final_title = f"Confusion Matrix: {name} (Threshold = {threshold:.2f})"
        elif title == "":
            final_title = None  # Explicitly set no title
        else:
            final_title = title  # Use provided custom title

        # Apply text wrapping if needed
        if (
            final_title is not None
            and text_wrap is not None
            and isinstance(text_wrap, int)
        ):
            final_title = "\n".join(textwrap.wrap(final_title, width=text_wrap))

        if final_title is not None:
            ax.set_title(final_title, fontsize=label_fontsize)

        # Adjust font sizes for axis labels and tick labels
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

        # Adjust the font size for the numeric values directly
        if disp.text_ is not None:
            for text in disp.text_.ravel():
                text.set_fontsize(inner_fontsize)  # Apply inner_fontsize here

        # Add labels (TN, FP, FN, TP) only if `labels` is True
        if labels:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    label_text = (
                        "TN"
                        if i == 0 and j == 0
                        else (
                            "FP"
                            if i == 0 and j == 1
                            else "FN" if i == 1 and j == 0 else "TP"
                        )
                    )
                    rgba_color = disp.im_.cmap(disp.im_.norm(cm[i, j]))
                    luminance = (
                        0.2126 * rgba_color[0]
                        + 0.7152 * rgba_color[1]
                        + 0.0722 * rgba_color[2]
                    )
                    ax.text(
                        j,
                        i - 0.3,  # Slight offset above numeric value
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=inner_fontsize,
                        color="white" if luminance < 0.5 else "black",
                    )

        # Always display numeric values (confusion matrix counts)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                rgba_color = disp.im_.cmap(disp.im_.norm(cm[i, j]))
                luminance = (
                    0.2126 * rgba_color[0]
                    + 0.7152 * rgba_color[1]
                    + 0.0722 * rgba_color[2]
                )

                ax.text(
                    j,
                    i,  # Exact position for numeric value
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=inner_fontsize,
                    color="white" if luminance < 0.5 else "black",
                )

        if not subplots:
            save_plot_images(
                f"confusion_matrix_{name}",
                save_plot,
                image_path_png,
                image_path_svg,
            )
            plt.show()

    if subplots:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "grid_confusion_matrix",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


################################################################################
##################### ROC AUC and Precision Recall Curves ######################
################################################################################


def show_roc_curve(
    model=None,
    X=None,
    y_prob=None,
    y=None,
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    model_title=None,
    decimal_places=2,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    subplots=False,
    n_rows=None,
    n_cols=2,
    figsize=(8, 6),
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    group_category=None,
):
    """
    Plot Receiver Operating Characteristic (ROC) curves for models or pipelines
    with optional styling, subplot layout, and grouping by categories, including
    class counts in the legend.

    Parameters:
    - model: estimator or list of estimators
        A single model or a list of models/pipelines to plot ROC curves for.
        The model(s) must implement either `predict_proba()` or
        `decision_function()`.
    - X: array-like
        Feature data for prediction, typically a pandas DataFrame or NumPy array.
    - y_prob (array-like or list of array-like, optional): Predicted probabilities.
      Can be provided instead of model and X.
    - y: array-like
        True binary labels for evaluation,(e.g., a pandas Series or NumPy array).
    - model_title: str or list of str, optional
        Title or list of titles for the models. If a single string is provided,
        it is automatically converted to a one-element list. If None, defaults to
        "Model 1", "Model 2", etc. Required when using a nested dictionary for
        `curve_kwgs`.
    - xlabel: str, optional
        Label for the x-axis (default: "False Positive Rate").
    - ylabel: str, optional
        Label for the y-axis (default: "True Positive Rate").
    - decimal_places: int, optional
        Number of decimal places to round AUC values in the legend and print
        output (default: 2).
    - overlay: bool, optional
        Whether to overlay multiple models on a single plot (default: False).
    - title: str, optional
        Custom title for the plot when `overlay=True` or per-model title when
        `subplots=True`. If None, uses a default title; if "", disables the title.
    - save_plot: bool, optional
        Whether to save the plot to the specified paths (default: False).
    - image_path_png: str, optional
        Path to save the plot as a PNG image.
    - image_path_svg: str, optional
        Path to save the plot as an SVG image.
    - text_wrap: int, optional
        Maximum width for wrapping titles if they are too long (default: None).
    - curve_kwgs: list or dict, optional
        Styling for individual model curves. If `model_title` is specified as a
        list, `curve_kwgs` must be a nested dictionary with model titles as keys
        and their respective style dictionaries (e.g., {'color': 'red',
        'linestyle': '--'}) as values. Otherwise, `curve_kwgs` must be a list of
        style dictionaries corresponding to the models.
    - linestyle_kwgs: dict, optional
        Styling for the random guess diagonal line (default: {'color': 'gray',
        'linestyle': '--', 'linewidth': 2}).
    - subplots: bool, optional
        Whether to organize plots in a subplot layout (default: False). Cannot be
        True if `overlay=True`.
    - n_rows: int, optional
        Number of rows in the subplot layout. If not specified, calculated
        automatically based on the number of models and `n_cols`.
    - n_cols: int, optional
        Number of columns in the subplot layout (default: 2).
    - figsize: tuple, optional
        Custom figure size (width, height) for the plot(s) (default: None, uses
        (8, 6) for overlay or calculated size for subplots).
    - label_fontsize: int, optional
        Font size for titles and axis labels (default: 12).
    - tick_fontsize: int, optional
        Font size for tick labels and legend (default: 10).
    - gridlines: bool, optional
        Whether to display grid lines on the plot (default: True).
    - group_category: array-like, optional
        Categorical data (e.g., pandas Series or NumPy array) to group ROC curves
        by unique values. Cannot be used with `subplots=True` or `overlay=True`.
        If provided, separate ROC curves are plotted for each group, with AUC
        and class counts (Total, Pos, Neg) shown in the legend.

    Raises:
        - ValueError: If `subplots=True` and `overlay=True` are both set, if
            `subplots=True` and `group_category` is provided, if `overlay=True`
            and `group_category` is provided, or if `overlay=True` and only one
            model is provided.
    """

    if not ((model is not None and X is not None) or y_prob is not None):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if isinstance(y_prob, list) and isinstance(y_prob[0], float):
        y_probs = [y_prob]
    else:
        y_probs = y_prob

    num_models = len(model) if model else len(y_probs)

    if y_prob is not None:
        model = [None] * num_models

    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    if subplots and group_category is not None:
        raise ValueError(
            f"`subplots` cannot be set to True when `group_category` is provided. "
            f"When selecting `group_category`, make sure `subplots` and `overlay` "
            f"are set to `False`."
        )

    if overlay and len(model) == 1:
        raise ValueError(
            f"Cannot use `overlay=True` with only one model. "
            f"Use `overlay=False` to plot a single model, or provide multiple "
            f"models for overlay."
        )

    if overlay and group_category is not None:
        raise ValueError(
            f"`overlay` cannot be set to True when `group_category` is "
            f"provided. When selecting `group_category`, make sure `subplots` and "
            f"`overlay` are set to `False`."
        )

    # Ensure models is a list
    if not isinstance(model, list):
        model = [model]

    # Normalize model_title input
    if model_title is None:
        model_title = [f"Model {i+1}" for i in range(len(model))]
    elif isinstance(model_title, str):
        model_title = [model_title]
    elif not isinstance(model_title, list):
        raise TypeError("model_title must be a string, a list of strings, or None.")

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_title]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(model)

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(model) / n_cols)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (mod, name, curve_style) in enumerate(
        zip(model, model_title, curve_styles)
    ):

        if X is None:
            y_true = y
            y_prob = y_probs[idx]
        else:
            y_true, y_prob, _, _ = get_predictions(
                mod,
                X,
                y,
                None,
                None,
                None,
            )

        if group_category is not None:
            fpr = {}
            tpr = {}
            auc_str = {}
            counts = {}
            for gr in group_category.unique():
                idx = group_category.values == gr
                counts[gr] = [
                    idx.sum(),
                    y_true.values[idx].sum(),
                    (1 - y_true.values[idx]).sum(),
                ]
                fpr[gr], tpr[gr], _ = roc_curve(y_true[idx], y_prob[idx])
                roc_auc = roc_auc_score(y_true[idx], y_prob[idx])
                # Format AUC with decimal_places for print and legend
                auc_str[gr] = f"{roc_auc:.{decimal_places}f}"

        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            # Format AUC with decimal_places for print and legend
            auc_str = f"{roc_auc:.{decimal_places}f}"

        print(f"AUC for {name}: {roc_auc:.{decimal_places}f}")

        if overlay:
            plt.plot(
                fpr,
                tpr,
                label=f"{name} (AUC = {auc_str})",
                **curve_style,
            )
        elif subplots:
            ax = axes[idx]
            if group_category is not None:
                for gr in tpr:
                    ax.plot(
                        fpr[gr],
                        tpr[gr],
                        label=f"AUC for {gr} = {auc_str[gr]:{decimal_places}}, "
                        f"Count: {counts[gr][0]:,}, "
                        f"Pos: {counts[gr][1]:,}, "
                        f"Neg: {counts[gr][2]:,}",
                        **curve_style,
                    )
            else:
                ax.plot(fpr, tpr, label=f"AUC = {auc_str}", **curve_style)
            ax.plot([0, 1], [0, 1], label="Random Guess", **linestyle_kwgs)
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            if title is None:
                grid_title = f"ROC Curve: {name}"  # Default title
            elif title == "":
                grid_title = None  # Disable the title
            else:
                grid_title = title  # Use provided custom title

            if grid_title and text_wrap:
                grid_title = "\n".join(
                    textwrap.wrap(grid_title, width=text_wrap),
                )

            if grid_title:  # Only set title if not explicitly disabled
                ax.set_title(grid_title, fontsize=label_fontsize)
            if group_category is not None:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    fontsize=tick_fontsize,
                    ncol=1,
                )
            else:
                ax.legend(loc="lower right", fontsize=tick_fontsize)
            ax.grid(visible=gridlines)
        else:
            plt.figure(figsize=figsize)
            if group_category is not None:
                for gr in group_category.unique():
                    plt.plot(
                        fpr[gr],
                        tpr[gr],
                        label=f"AUC for {gr} = {auc_str[gr]:{decimal_places}}, "
                        f"Count: {counts[gr][0]:,}, "
                        f"Pos: {counts[gr][1]:,}, "
                        f"Neg: {counts[gr][2]:,}",
                        **curve_style,
                    )

            else:
                plt.plot(fpr, tpr, label=f"AUC = {auc_str}", **curve_style)
            plt.plot([0, 1], [0, 1], label="Random Guess", **linestyle_kwgs)
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            if title is None:
                plot_title = f"ROC Curve: {name}"  # Default title
            elif title == "":
                plot_title = None  # Disable the title
            else:
                plot_title = title  # Use provided custom title

            if plot_title and text_wrap:
                plot_title = "\n".join(
                    textwrap.wrap(plot_title, width=text_wrap),
                )

            if plot_title:  # Only set title if not explicitly disabled
                plt.title(plot_title, fontsize=label_fontsize)

            plt.title(plot_title, fontsize=label_fontsize)
            if group_category is not None:
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    fontsize=tick_fontsize,
                    ncol=1,
                )
            else:
                plt.legend(loc="lower right", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            name_clean = name.lower().replace(" ", "_")
            if group_category is not None:
                save_plot_images(
                    f"{name_clean}_{group_category.name}_roc_auc",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )
            else:
                save_plot_images(
                    f"{name_clean}_roc_auc",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )

            plt.show()

    if overlay:
        plt.plot([0, 1], [0, 1], label="Random Guess", **linestyle_kwgs)
        plt.xlabel(xlabel, fontsize=label_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        plt.tick_params(axis="both", labelsize=tick_fontsize)
        if title is None:
            overlay_title = "ROC Curves: Overlay"  # Default title
        elif title == "":
            overlay_title = None  # Disable the title
        else:
            overlay_title = title  # Use provided custom title

        if overlay_title and text_wrap:
            overlay_title = "\n".join(
                textwrap.wrap(overlay_title, width=text_wrap),
            )

        if overlay_title:  # Only set title if not explicitly disabled
            plt.title(overlay_title, fontsize=label_fontsize)

        if group_category is not None:
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fontsize=tick_fontsize,
                ncol=1,
            )
        else:
            plt.legend(loc="lower right", fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "overlay_roc_auc_plot",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif subplots:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "grid_roc_auc_plot",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


def show_pr_curve(
    model=None,
    X=None,
    y_prob=None,
    y=None,
    xlabel="Recall",
    ylabel="Precision",
    model_title=None,
    decimal_places=2,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    subplots=False,
    n_rows=None,
    n_cols=2,
    figsize=(8, 6),
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    group_category=None,
    legend_metric="ap",
):
    """
    Plot Precision-Recall (PR) curves for models or pipelines with optional
    styling, subplot grid layout, and grouping by categories, including class
    counts and selected legend metrics.

    Parameters:
    - model: estimator or list of estimators
        A single model or a list of models/pipelines to plot PR curves for.
        The model(s) must implement either `predict_proba()` or
        `decision_function()`.
    - X: array-like
        Feature data for prediction, typically a pandas DataFrame or NumPy array.
    - y_prob (array-like or list of array-like, optional): Predicted probabilities.
      Can be provided instead of model and X.
    - y: array-like
        True binary labels for evaluation (e.g., a pandas Series or NumPy array).
    - group_category: array-like, optional
        Categorical data (e.g., pandas Series or NumPy array) to group PR curves
        by unique values. If provided, plots separate PR curves for each group
        with metric values and class counts (Total, Pos, Neg) in the legend.
    - model_title: str or list of str, optional
        Title or list of titles for the models. If a single string is provided,
        it is automatically converted to a one-element list. If None, defaults to
        "Model 1", "Model 2", etc. Required when using a nested dictionary for
        `curve_kwgs`.
    - xlabel: str, optional
        Label for the x-axis. Defaults to "Recall".
    - ylabel: str, optional
        Label for the y-axis. Defaults to "Precision".
    - decimal_places: int, optional
        Number of decimal places to display in the legend. Defaults to 3.
    - overlay: bool, optional
        Whether to overlay multiple models on a single plot. Defaults to False.
    - title: str, optional
        Custom title for the plot. If None, a default title is used.
        If "", the title is omitted.
    - save_plot: bool, optional
        Whether to save the plot(s) to file. Defaults to False.
    - image_path_png: str, optional
        File path to save the plot(s) as PNG.
    - image_path_svg: str, optional
        File path to save the plot(s) as SVG.
    - text_wrap: int, optional
        Maximum width (in characters) to wrap long titles. If None, no wrapping.
    - curve_kwgs: list or dict, optional
        Styling for PR curves. Can be a list of dicts or a nested dict
        keyed by model title.
    - subplots: bool, optional
        Whether to organize the PR plots in a subplot grid layout. Cannot be
        used with `overlay=True` or `group_category`.
    - n_rows: int, optional
        Number of rows in the subplot layout. If not specified, calculated
        automatically.
    - n_cols: int, optional
        Number of columns in the subplot layout. Defaults to 2.
    - figsize: tuple, optional
        Figure size in inches (width, height). Defaults to (8, 6).
    - label_fontsize: int, optional
        Font size for axis labels and titles. Defaults to 12.
    - tick_fontsize: int, optional
        Font size for tick labels and legend text. Defaults to 10.
    - gridlines: bool, optional
        Whether to display grid lines on plots. Defaults to True.
    - legend_metric: str, optional
        Metric to show in the legend: either "ap" (Average Precision, default)
        or "aucpr" (area under the PR curve).

    Raises:
    - ValueError:
        - If `subplots=True` and `overlay=True` are both set.
        - If `group_category` is used with `subplots=True` or `overlay=True`.
        - If `legend_metric` is not one of {"ap", "aucpr"}.
    - TypeError:
        - If `model_title` is not a string, list of strings, or None.
    """

    if not ((model is not None and X is not None) or y_prob is not None):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if isinstance(y_prob, list) and isinstance(y_prob[0], float):
        y_probs = [y_prob]
    else:
        y_probs = y_prob

    num_models = len(model) if model else len(y_probs)

    if y_prob is not None:
        model = [None] * num_models

    # Validate legend_metric
    valid_metrics = ["ap", "aucpr"]
    if legend_metric not in valid_metrics:
        raise ValueError(
            f"`legend_metric` must be one of {valid_metrics}, got {legend_metric}"
        )

    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    if subplots and group_category is not None:
        raise ValueError(
            f"`subplots` cannot be set to True when `group_category` is provided. "
            f"When selecting `group_category`, make sure `subplots` and `overlay` "
            f"are set to `False`."
        )

    if overlay and group_category is not None:
        raise ValueError(
            f"`overlay` cannot be set to True when `group_category` is "
            f"provided. When selecting `group_category`, make sure `subplots` and "
            f"`overlay` are set to `False`."
        )

    if not isinstance(model, list):
        model = [model]

    # Normalize model_title input
    if model_title is None:
        model_title = [f"Model {i+1}" for i in range(len(model))]
    elif isinstance(model_title, str):
        model_title = [model_title]
    elif not isinstance(model_title, list):
        raise TypeError("model_title must be a string, a list of strings, or None.")

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_title]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(model)

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(model) / n_cols)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (mod, name, curve_style) in enumerate(
        zip(model, model_title, curve_styles)
    ):

        if X is None:
            y_true = y
            y_prob = y_probs[idx]
        else:
            y_true, y_prob, _, _ = get_predictions(
                mod,
                X,
                y,
                None,
                None,
                None,
            )

        counts = {}
        if group_category is not None:
            precision = {}
            recall = {}
            ap_str = {}
            aucpr_str = {}
            for gr in group_category.unique():
                idx = group_category.values == gr
                counts[gr] = [
                    idx.sum(),  # Total count for this group
                    y_true.values[idx].sum(),  # Positive class count (y_true = 1)
                    (1 - y_true.values[idx]).sum(),  # Negative class count (y_true = 0)
                ]
                precision[gr], recall[gr], _ = precision_recall_curve(
                    y_true[idx], y_prob[idx]
                )
                avg_precision = average_precision_score(y_true[idx], y_prob[idx])
                auc_val = auc(recall[gr], precision[gr])
                # Format Average Precision with decimal_places for print and legend
                ap_str[gr] = f"{avg_precision:.{decimal_places}f}"
                aucpr_str[gr] = f"{auc_val:.{decimal_places}f}"

        else:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            auc_val = auc(recall, precision)
            # Format Average Precision with decimal_places for print and legend
            ap_str = f"{avg_precision:.{decimal_places}f}"
            aucpr_str = f"{auc_val:.{decimal_places}f}"

        if legend_metric == "aucpr":
            print(f"AUCPR for {name}: {auc_val:.{decimal_places}f}")
        else:
            print(f"Average Precision for {name}: {avg_precision:.{decimal_places}f}")

        # Determine the metric label and value based on legend_metric
        metric_label = "AP" if legend_metric == "ap" else "AUCPR"
        metric_str = ap_str if legend_metric == "ap" else aucpr_str

        if overlay:
            plt.plot(
                recall,
                precision,
                label=f"{name} ({metric_label} = {metric_str})",
                **curve_style,
            )
        elif subplots:
            ax = axes[idx]
            if group_category is not None:
                for gr in group_category.unique():
                    ax.plot(
                        recall[gr],
                        precision[gr],
                        label=f"{metric_label} for {gr} = {metric_str[gr]}, "
                        f"Count: {counts[gr][0]:,}, "
                        f"Pos: {counts[gr][1]:,}, Neg: {counts[gr][2]:,}",
                        **curve_style,
                    )
            else:
                ax.plot(
                    recall,
                    precision,
                    label=f"{metric_label} = {metric_str}",
                    **curve_style,
                )

            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)

            if title is None:
                grid_title = f"PR Curve: {name}"  # Default title
            elif title == "":
                grid_title = None  # Disable the title
            else:
                grid_title = title  # Use provided custom title

            if grid_title and text_wrap:
                grid_title = "\n".join(
                    textwrap.wrap(grid_title, width=text_wrap),
                )

            if grid_title:  # Only set title if not explicitly disabled
                ax.set_title(grid_title, fontsize=label_fontsize)
            if group_category is not None:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    fontsize=tick_fontsize,
                    ncol=1,
                )
            else:
                ax.legend(loc="lower left", fontsize=tick_fontsize)
            ax.grid(visible=gridlines)

        else:
            plt.figure(figsize=figsize or (8, 6))
            if group_category is not None:
                for gr in group_category.unique():
                    plt.plot(
                        recall[gr],
                        precision[gr],
                        label=f"{metric_label} for {gr} = {metric_str[gr]}, "
                        f"Count: {counts[gr][0]:,}, "
                        f"Pos: {counts[gr][1]:,}, Neg: {counts[gr][2]:,}",
                        **curve_style,
                    )
            else:
                plt.plot(
                    recall,
                    precision,
                    label=f"{metric_label} = {metric_str}",
                    **curve_style,
                )

            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)

            if title is None:
                plot_title = f"PR Curve: {name}"  # Default title
            elif title == "":
                plot_title = None  # Disable the title
            else:
                plot_title = title  # Use provided custom title

            if plot_title and text_wrap:
                plot_title = "\n".join(
                    textwrap.wrap(plot_title, width=text_wrap),
                )

            if plot_title:  # Only set title if not explicitly disabled
                plt.title(plot_title, fontsize=label_fontsize)

            if group_category is not None:
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    fontsize=tick_fontsize,
                    ncol=1,
                )
            else:
                plt.legend(loc="lower left", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            name_clean = name.lower().replace(" ", "_")
            if group_category is not None:
                save_plot_images(
                    f"{name_clean}_{group_category.name}_precision_recall",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )
            else:
                save_plot_images(
                    f"{name_clean}_precision_recall",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )
            plt.show()

    if overlay:
        plt.xlabel(xlabel, fontsize=label_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        plt.tick_params(axis="both", labelsize=tick_fontsize)
        if title is None:
            overlay_title = "PR Curves: Overlay"  # Default title
        elif title == "":
            overlay_title = None  # Disable the title
        else:
            overlay_title = title  # Use provided custom title

        if overlay_title and text_wrap:
            overlay_title = "\n".join(
                textwrap.wrap(overlay_title, width=text_wrap),
            )

        if overlay_title:  # Only set title if not explicitly disabled
            plt.title(overlay_title, fontsize=label_fontsize)

        if group_category is not None:
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fontsize=tick_fontsize,
                ncol=1,
            )
        else:
            plt.legend(loc="lower left", fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "overlay_pr_plot",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif subplots:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "grid_pr_plot",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


################################################################################
########################## Lift Charts and Gain Charts #########################
################################################################################


def show_lift_chart(
    model=None,
    X=None,
    y_prob=None,
    y=None,
    xlabel="Percentage of Sample",
    ylabel="Lift",
    model_title=None,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    subplots=False,
    n_cols=2,
    n_rows=None,
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
):
    """
    Generate and display Lift charts for one or multiple models.

    A Lift chart measures the effectiveness of a predictive model by comparing
    the lift of positive instances in sorted predictions versus random selection.
    Supports multiple models with overlay or subplots layouts and customizable
    styling.

    Parameters:
    - model (list or estimator): One or more trained models.
    - X (array-like): Feature matrix.
    - y_prob (array-like or list of array-like, optional): Predicted probabilities.
      Can be provided instead of model and X.
    - y (array-like): True labels.
    - xlabel (str, default="Percentage of Sample"): Label for the x-axis.
    - ylabel (str, default="Lift"): Label for the y-axis.
    - model_title (list, optional): Custom titles for models.
    - overlay (bool, default=False): Whether to overlay multiple models in one plot.
    - title (str, optional): Custom title; set to `""` to disable.
    - save_plot (bool, default=False): Whether to save the plot.
    - image_path_png (str, optional): Path to save PNG image.
    - image_path_svg (str, optional): Path to save SVG image.
    - text_wrap (int, optional): Maximum title width before wrapping.
    - curve_kwgs (dict or list, optional): Styling options for model curves.
    - linestyle_kwgs (dict, optional): Styling options for the baseline.
    - subplots (bool, default=False): Display multiple plots in a subplot layout.
    - n_cols (int, default=2): Number of columns in the subplot layout.
    - n_rows (int, optional): Number of rows in the subplot layout.
    - figsize (tuple, optional): Figure size.
    - label_fontsize (int, default=12): Font size for axis labels.
    - tick_fontsize (int, default=10): Font size for tick labels.
    - gridlines (bool, default=True): Whether to show grid lines.

    Raises:
    - ValueError: If `subplots=True` and `overlay=True` are both set.

    Returns:
    - None
    """

    if not ((model is not None and X is not None) or y_prob is not None):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if isinstance(y_prob, list) and isinstance(y_prob[0], float):
        y_probs = [y_prob]
    else:
        y_probs = y_prob

    num_models = len(model) if model else len(y_probs)

    if y_prob is not None:
        model = [None] * num_models

    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    if not isinstance(model, list):
        model = [model]

    if model_title is None:
        model_title = [f"Model {i+1}" for i in range(len(model))]

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_title]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(model)

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(model) / n_cols)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (mod, name, curve_style) in enumerate(
        zip(model, model_title, curve_styles)
    ):

        if X is None:
            y_prob = y_probs[idx]
        else:
            y_prob = mod.predict_proba(X)[:, 1]

        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = np.array(y)[sorted_indices]

        cumulative_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        percentages = np.linspace(
            1 / len(y_true_sorted),
            1,
            len(y_true_sorted),
        )

        lift_values = cumulative_gains / percentages

        if overlay:
            plt.plot(
                percentages,
                lift_values,
                label=f"{name}",
                **curve_style,
            )
        elif subplots:
            ax = axes[idx]
            ax.plot(
                percentages,
                lift_values,
                label=f"Lift Curve",
                **curve_style,
            )
            ax.plot([0, 1], [1, 1], label="Baseline", **linestyle_kwgs)
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            if title is None:
                grid_title = f"Lift Chart: {name}"  # Default title
            elif title == "":
                grid_title = None  # Disable the title
            else:
                grid_title = title  # Use provided custom title

            if grid_title and text_wrap:
                grid_title = "\n".join(
                    textwrap.wrap(grid_title, width=text_wrap),
                )

            if grid_title:  # Only set title if not explicitly disabled
                ax.set_title(grid_title, fontsize=label_fontsize)
                ax.legend(loc="best", fontsize=tick_fontsize)
            ax.grid(visible=gridlines)
        else:
            plt.figure(figsize=figsize or (8, 6))
            plt.plot(
                percentages,
                lift_values,
                label=f"Lift Curve",
                **curve_style,
            )
            plt.plot([0, 1], [1, 1], label="Baseline", **linestyle_kwgs)
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            if title is None:
                plot_title = f"Lift Chart: {name}"  # Default title
            elif title == "":
                plot_title = None  # Disable the title
            else:
                plot_title = title  # Use provided custom title

            if plot_title and text_wrap:
                plot_title = "\n".join(
                    textwrap.wrap(plot_title, width=text_wrap),
                )

            if plot_title:  # Only set title if not explicitly disabled
                plt.title(plot_title, fontsize=label_fontsize)

            plt.title(plot_title, fontsize=label_fontsize)
            plt.legend(loc="best", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            save_plot_images(
                f"{name}_lift",
                save_plot,
                image_path_png,
                image_path_svg,
            )
            plt.show()

    if overlay:
        plt.plot([0, 1], [1, 1], label="Baseline", **linestyle_kwgs)
        plt.xlabel(xlabel, fontsize=label_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        plt.tick_params(axis="both", labelsize=tick_fontsize)
        if title is None:
            overlay_title = "Lift Charts: Overlay"  # Default title
        elif title == "":
            overlay_title = None  # Disable the title
        else:
            overlay_title = title  # Use provided custom title

        if overlay_title and text_wrap:
            overlay_title = "\n".join(
                textwrap.wrap(overlay_title, width=text_wrap),
            )

        if overlay_title:  # Only set title if not explicitly disabled
            plt.title(overlay_title, fontsize=label_fontsize)
        plt.legend(loc="best", fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "overlay_lift",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif subplots:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "grid_lift",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


def show_gain_chart(
    model=None,
    X=None,
    y_prob=None,
    y=None,
    xlabel="Percentage of Sample",
    ylabel="Cumulative Gain",
    model_title=None,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    subplots=False,
    n_cols=2,
    n_rows=None,
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
):
    """
    Generate and display Gain charts for one or multiple models.

    A Gain chart evaluates model effectiveness by comparing the cumulative gain
    of positive instances in sorted predictions versus random selection.
    Supports multiple models with overlay or subplots layouts and customizable styling.

    Parameters:
    - model (list or estimator): One or more trained models.
    - X (array-like): Feature matrix.
    - y_prob (array-like or list of array-like, optional): Predicted probabilities.
      Can be provided instead of model and X.
    - y (array-like): True labels.
    - xlabel (str, default="Percentage of Sample"): Label for the x-axis.
    - ylabel (str, default="Cumulative Gain"): Label for the y-axis.
    - model_title (list, optional): Custom titles for models.
    - overlay (bool, default=False): Whether to overlay multiple models in one plot.
    - title (str, optional): Custom title; set to `""` to disable.
    - save_plot (bool, default=False): Whether to save the plot.
    - image_path_png (str, optional): Path to save PNG image.
    - image_path_svg (str, optional): Path to save SVG image.
    - text_wrap (int, optional): Maximum title width before wrapping.
    - curve_kwgs (dict or list, optional): Styling options for model curves.
    - linestyle_kwgs (dict, optional): Styling options for the baseline.
    - subplots (bool, default=False): Display multiple plots in a subplot layout.
    - n_cols (int, default=2): Number of columns in the subplot layout.
    - n_rows (int, optional): Number of rows in the subplot layout.
    - figsize (tuple, optional): Figure size.
    - label_fontsize (int, default=12): Font size for axis labels.
    - tick_fontsize (int, default=10): Font size for tick labels.
    - gridlines (bool, default=True): Whether to show grid lines.

    Raises:
    - ValueError: If `subplots=True` and `overlay=True` are both set.

    Returns:
    - None
    """

    if not ((model is not None and X is not None) or y_prob is not None):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if isinstance(y_prob, list) and isinstance(y_prob[0], float):
        y_probs = [y_prob]
    else:
        y_probs = y_prob

    num_models = len(model) if model else len(y_probs)

    if y_prob is not None:
        model = [None] * num_models

    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    if not isinstance(model, list):
        model = [model]

    if model_title is None:
        model_title = [f"Model {i+1}" for i in range(len(model))]

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_title]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(model)

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(model) / n_cols)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (mod, name, curve_style) in enumerate(
        zip(model, model_title, curve_styles)
    ):
        if X is None:
            y_prob = y_probs[idx]
        else:
            y_prob = mod.predict_proba(X)[:, 1]

        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = np.array(y)[sorted_indices]

        cumulative_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        percentages = np.linspace(0, 1, len(y_true_sorted))

        if overlay:
            plt.plot(
                percentages,
                cumulative_gains,
                label=f"{name}",
                **curve_style,
            )
        elif subplots:
            ax = axes[idx]
            ax.plot(
                percentages,
                cumulative_gains,
                label=f"Gain Curve",
                **curve_style,
            )
            ax.plot([0, 1], [0, 1], label="Baseline", **linestyle_kwgs)
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            if title is None:
                grid_title = f"Gain Chart: {name}"  # Default title
            elif title == "":
                grid_title = None  # Disable the title
            else:
                grid_title = title  # Use provided custom title

            if grid_title and text_wrap:
                grid_title = "\n".join(
                    textwrap.wrap(grid_title, width=text_wrap),
                )

            if grid_title:  # Only set title if not explicitly disabled
                ax.set_title(grid_title, fontsize=label_fontsize)

            ax.legend(loc="best", fontsize=tick_fontsize)
            ax.grid(visible=gridlines)
        else:
            plt.figure(figsize=figsize or (8, 6))
            plt.plot(
                percentages,
                cumulative_gains,
                label=f"Gain Curve",
                **curve_style,
            )
            plt.plot([0, 1], [0, 1], label="Baseline", **linestyle_kwgs)
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            if title is None:
                plot_title = f"Gain Chart: {name}"  # Default title
            elif title == "":
                plot_title = None  # Disable the title
            else:
                plot_title = title  # Use provided custom title

            if plot_title and text_wrap:
                plot_title = "\n".join(
                    textwrap.wrap(plot_title, width=text_wrap),
                )

            if plot_title:  # Only set title if not explicitly disabled
                plt.title(plot_title, fontsize=label_fontsize)

            plt.title(plot_title, fontsize=label_fontsize)
            plt.legend(loc="best", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            save_plot_images(
                f"{name}_gain",
                save_plot,
                image_path_png,
                image_path_svg,
            )
            plt.show()

    if overlay:
        plt.plot([0, 1], [0, 1], label="Baseline", **linestyle_kwgs)
        plt.xlabel(xlabel, fontsize=label_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        plt.tick_params(axis="both", labelsize=tick_fontsize)
        if title is None:
            overlay_title = "Gain Charts: Overlay"  # Default title
        elif title == "":
            overlay_title = None  # Disable the title
        else:
            overlay_title = title  # Use provided custom title

        if overlay_title and text_wrap:
            overlay_title = "\n".join(
                textwrap.wrap(overlay_title, width=text_wrap),
            )

        if overlay_title:  # Only set title if not explicitly disabled
            plt.title(overlay_title, fontsize=label_fontsize)

        plt.legend(loc="best", fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "overlay_gain",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif subplots:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "grid_gain",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


################################################################################
############################## Calibration Curve ###############################
################################################################################


def show_calibration_curve(
    model=None,
    X=None,
    y_prob=None,
    y=None,
    xlabel="Mean Predicted Probability",
    ylabel="Fraction of Positives",
    model_title=None,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    subplots=False,
    n_cols=2,
    n_rows=None,
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    bins=10,
    marker="o",
    show_brier_score=True,
    gridlines=True,
    linestyle_kwgs=None,
    group_category=None,
    **kwargs,
):
    """
    Plot calibration curves for one or more classification models.

    A calibration curve compares the predicted probabilities of a classifier
    to the actual observed outcomes. This function supports individual, overlay,
    and subplot grid-based visualization modes, and optionally plots separate
    curves per subgroup defined by a categorical variable (e.g., race, age group).

    Parameters:
    - model (estimator or list): One or more trained classification models.
    - X (array-like): Feature matrix used for prediction.
    - y_prob (array-like or list of array-like, optional): Predicted probabilities.
      Can be provided instead of model and X.
    - y (array-like): True binary target labels.
    - xlabel (str, default="Mean Predicted Probability"): Label for the x-axis.
    - ylabel (str, default="Fraction of Positives"): Label for the y-axis.
    - model_title (str or list, optional): Custom name(s) for model(s). Must
      match number of models.
    - overlay (bool, default=False): Whether to overlay models in a single plot.
    - title (str, optional): Custom plot title. If `None`, a default title is
      used; if `""`, the title is completely suppressed.
    - save_plot (bool, default=False): Whether to save the generated plot(s).
    - image_path_png (str, optional): Path to save the PNG image.
    - image_path_svg (str, optional): Path to save the SVG image.
    - text_wrap (int, optional): Maximum # of characters before wrapping title.
    - curve_kwgs (dict or list, optional): Styling options for model curves.
    - subplots (bool, default=False): Display models in a grid of subplots.
    - n_cols (int, default=2): Number of columns for the subplot layout.
    - n_rows (int, optional): Number of rows for the subplot layout.
    - figsize (tuple, optional): Custom figure size (width, height).
    - label_fontsize (int, default=12): Font size for axis labels and title.
    - tick_fontsize (int, default=10): Font size for tick marks and legend.
    - bins (int, default=10): # of bins to use for computing calibration curve.
    - marker (str, default="o"): Marker used for each calibration point.
    - show_brier_score (bool, default=True): Whether to show Brier score in legend.
    - gridlines (bool, default=True): Whether to display gridlines on the plot.
    - linestyle_kwgs (dict, optional): Styling options for the diagonal
      "perfectly calibrated" line.
    - group_category (array-like, optional): A categorical series to plot
      subgroup calibration curves.

    Raises:
    - ValueError: If both `subplots=True` and `overlay=True` are set (incompatible).
    - ValueError: If `group_category` used w/ either `overlay=True` or `subplots=True`.
    - ValueError: If length of `curve_kwgs` does not match number of models.
    - TypeError: If `model_title` is not a string, list, pandas Series, or None.

    Returns:
    - None
    """

    if not ((model is not None and X is not None) or y_prob is not None):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    # Ensure y_prob is always a list of arrays:
    # if a single array/Series is passed, wrap it in a list so y_prob[0] works
    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    if isinstance(y_prob, list) and isinstance(y_prob[0], float):
        y_probs = [y_prob]
    else:
        y_probs = y_prob

    num_models = len(model) if model else len(y_probs)

    if y_prob is not None:
        model = [None] * num_models

    # Error checks for incompatible display modes
    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    if group_category is not None and (overlay or subplots):
        raise ValueError(
            "`group_category` requires `overlay=False` and `subplots=False`."
        )

    # Ensure model is a list
    if not isinstance(model, list):
        model = [model]

    # Handle model titles
    if model_title is None:
        model_title = [f"Model_{i+1}" for i in range(len(model))]
    elif isinstance(model_title, str):
        model_title = [model_title]
    elif isinstance(model_title, pd.Series):
        model_title = model_title.tolist()

    if not isinstance(model_title, list):
        raise TypeError("`model_title` must be str, list, Series, or None.")

    # Handle style settings for each model
    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_title]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(model)

    if len(curve_styles) != len(model):
        raise ValueError("Length of `curve_kwgs` must match the number of models.")

    # Grid layout setup if requested
    if subplots:
        if n_rows is None:
            n_rows = math.ceil(len(model) / n_cols)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    # Initialize overlay figure
    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    # Loop over each model
    for idx, (mod, name, curve_style) in enumerate(
        zip(model, model_title, curve_styles)
    ):
        if X is None:
            y_true = y
            y_prob = y_probs[idx]
        else:
            y_true, y_prob, _, _ = get_predictions(
                mod,
                X,
                y,
                None,
                None,
                None,
            )
        # Handle single-column (y_true) DataFrame
        if isinstance(y_true, pd.DataFrame) and y_true.shape[1] == 1:
            y_true = y_true.iloc[:, 0]

        # GROUPED CALIBRATION BY CATEGORY
        if group_category is not None:
            group_series = pd.Series(group_category)
            unique_groups = group_series.dropna().unique()
            plt.figure(figsize=figsize or (8, 6))

            for group_val in unique_groups:
                group_idx = group_series == group_val
                y_group = y_true[group_idx]
                prob_group = y_prob[group_idx]

                # Skip group if not enough data or only one class
                if len(y_group) < bins or len(set(y_group)) < 2:
                    print(
                        f"Skipping group {group_val} (len={len(y_group)}, "
                        f"unique={set(y_group)})"
                    )
                    continue

                # Calibration computation
                prob_true, prob_pred = calibration_curve(
                    y_group, prob_group, n_bins=bins
                )
                brier = (
                    brier_score_loss(y_group, prob_group) if show_brier_score else None
                )
                legend_label = f"{group_val}"
                if show_brier_score:
                    legend_label += f" (Brier: {brier:.4f})"

                # Plot curve for group
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker=marker,
                    label=legend_label,
                    **curve_style,
                    **kwargs,
                )

            # Add diagonal reference line
            linestyle_kwgs = linestyle_kwgs or {}
            linestyle_kwgs.setdefault("color", "gray")
            linestyle_kwgs.setdefault("linestyle", "--")
            plt.plot(
                [0, 1],
                [0, 1],
                label="Perfectly Calibrated",
                **linestyle_kwgs,
            )

            # Plot formatting
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            if title is None:
                group_title = f"Calibration Curve: {name}"
            elif title == "":
                group_title = None
            else:
                group_title = title

            if text_wrap and group_title:
                group_title = "\n".join(
                    textwrap.wrap(
                        group_title,
                        width=text_wrap,
                    )
                )

            if group_title:
                plt.title(group_title, fontsize=label_fontsize)
            plt.legend(loc="best", fontsize=tick_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            plt.grid(visible=gridlines)

            # Save grouped plot
            name_clean = name.lower().replace(" ", "_")
            if save_plot:
                filename = f"{name_clean}_by_{group_category.name}_calibration"
                save_plot_images(
                    filename,
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )

            plt.show()
            continue  # Skip standard rendering

        # STANDARD CALIBRATION
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=bins,
        )
        brier_score = brier_score_loss(y_true, y_prob) if show_brier_score else None
        legend_label = f"{name}"
        if show_brier_score:
            legend_label += f" (Brier: {brier_score:.4f})"

        # PLOT IN OVERLAY
        if overlay:
            plt.plot(
                prob_pred,
                prob_true,
                marker=marker,
                label=legend_label,
                **curve_style,
                **kwargs,
            )

        # PLOT IN SUBPLOTS
        elif subplots:
            ax = axes[idx]
            ax.plot(
                prob_pred,
                prob_true,
                marker=marker,
                label=legend_label,
                **curve_style,
                **kwargs,
            )
            linestyle_kwgs = linestyle_kwgs or {}
            linestyle_kwgs.setdefault("color", "gray")
            linestyle_kwgs.setdefault("linestyle", "--")
            ax.plot(
                [0, 1],
                [0, 1],
                label="Perfectly Calibrated",
                **linestyle_kwgs,
            )
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            if title is None:
                grid_title = f"Calibration Curve: {name}"
            elif title == "":
                grid_title = None
            else:
                grid_title = title

            if text_wrap and grid_title:
                grid_title = "\n".join(
                    textwrap.wrap(
                        grid_title,
                        width=text_wrap,
                    )
                )

            if grid_title:
                ax.set_title(grid_title, fontsize=label_fontsize)
            ax.legend(loc="best", fontsize=tick_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            if gridlines:
                ax.grid(True, which="both", axis="both")

        # STANDARD SINGLE PLOT
        else:
            plt.figure(figsize=figsize or (8, 6))
            plt.plot(
                prob_pred,
                prob_true,
                marker=marker,
                label=legend_label,
                **curve_style,
                **kwargs,
            )
            linestyle_kwgs = linestyle_kwgs or {}
            linestyle_kwgs.setdefault("color", "gray")
            linestyle_kwgs.setdefault("linestyle", "--")
            plt.plot(
                [0, 1],
                [0, 1],
                label="Perfectly Calibrated",
                **linestyle_kwgs,
            )
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            if title is None:
                plot_title = f"Calibration Curve: {name}"
            elif title == "":
                plot_title = None
            else:
                plot_title = title

            if text_wrap and plot_title:
                plot_title = "\n".join(
                    textwrap.wrap(
                        plot_title,
                        width=text_wrap,
                    )
                )

            if plot_title:
                plt.title(plot_title, fontsize=label_fontsize)
            plt.legend(loc="best", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            save_plot_images(
                f"{name}_Calibration",
                save_plot,
                image_path_png,
                image_path_svg,
            )
            plt.show()

    # Final overlay post-processing
    if overlay:
        linestyle_kwgs = linestyle_kwgs or {}
        linestyle_kwgs.setdefault("color", "gray")
        linestyle_kwgs.setdefault("linestyle", "--")
        plt.plot([0, 1], [0, 1], label="Perfectly Calibrated", **linestyle_kwgs)
        plt.xlabel(xlabel, fontsize=label_fontsize)
        plt.ylabel(ylabel, fontsize=label_fontsize)
        if title is None:
            overlay_title = "Calibration Curves: Overlay"
        elif title == "":
            overlay_title = None
        else:
            overlay_title = title

        if text_wrap and overlay_title:
            overlay_title = "\n".join(
                textwrap.wrap(
                    overlay_title,
                    width=text_wrap,
                )
            )

        if overlay_title:
            plt.title(overlay_title, fontsize=label_fontsize)
        plt.legend(loc="best", fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "overlay_calibration",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    # Final subplot grid cleanup (hide unused axes)
    elif subplots:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "grid_calibration",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


################################################################################
################## Classification Metrics Threshold Trade-Off ##################
################################################################################


def plot_threshold_metrics(
    model=None,
    X_test=None,
    y_test=None,
    y_prob=None,
    title=None,
    text_wrap=None,
    figsize=(8, 6),
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    baseline_thresh=True,
    curve_kwgs=None,
    baseline_kwgs=None,
    threshold_kwgs=None,
    lookup_kwgs=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    lookup_metric=None,
    lookup_value=None,
    decimal_places=4,
    model_threshold=None,
):
    """
    Plot Precision, Recall, F1 Score, and Specificity vs. decision thresholds.

    This function evaluates threshold-dependent classification metrics
    (Precision, Recall, F1 Score, Specificity) across the range of possible
    thresholds, optionally highlighting baseline, model-specified, or custom
    lookup thresholds.

    Parameters
    ----------
    model : estimator, optional
        A trained model that supports `predict_proba`. Required if `y_prob`
        is not provided.

    X_test : array-like, optional
        Feature matrix for testing. Required if `model` is provided.

    y_test : array-like of shape (n_samples,)
        True binary labels. Required.

    y_prob : array-like, optional
        Predicted probabilities for the positive class. Can be provided
        instead of `model` and `X_test`.

    title : str or None, default=None
        Title for the plot. If None, a default title is shown.
        If `""`, no title is displayed.

    text_wrap : int, optional
        Maximum width of the title before wrapping onto multiple lines.

    figsize : tuple, default=(8, 6)
        Size of the matplotlib figure.

    label_fontsize : int, default=12
        Font size for axis labels and title.

    tick_fontsize : int, default=10
        Font size for tick labels.

    gridlines : bool, default=True
        Whether to display gridlines.

    baseline_thresh : bool, default=True
        If True, draws a vertical reference line at threshold = 0.5.

    curve_kwgs : dict, optional
        Keyword arguments passed to all metric curves
        (e.g., {"linestyle": "-", "linewidth": 1}).

    baseline_kwgs : dict, optional
        Keyword arguments for styling the baseline threshold line
        (default: black dotted line).

    threshold_kwgs : dict, optional
        Keyword arguments for styling the model threshold line when
        `model_threshold` is provided (default: black dotted line).

    lookup_kwgs : dict, optional
        Keyword arguments for styling the lookup threshold line when
        `lookup_metric` and `lookup_value` are provided
        (default: gray dashed line).

    save_plot : bool, default=False
        If True, saves the plot to disk.

    image_path_png : str, optional
        Path to save the plot as a PNG image.

    image_path_svg : str, optional
        Path to save the plot as an SVG image.

    lookup_metric : {"precision", "recall", "f1", "specificity"}, optional
        Metric for which to locate the threshold closest to `lookup_value`.

    lookup_value : float, optional
        Target value of the lookup metric. Must be provided together with
        `lookup_metric`.

    decimal_places : int, default=4
        Number of decimal places for reported thresholds.

    model_threshold : float, optional
        A model-specific threshold to highlight with a vertical line. Useful
        if the model uses a threshold other than 0.5 for predictions.

    Raises
    ------
    ValueError
        If `y_test` is not provided.
    ValueError
        If neither (`model` and `X_test`) nor `y_prob` is provided.
    ValueError
        If only one of `lookup_metric` or `lookup_value` is provided.

    Returns
    -------
    None
        Displays the threshold metrics plot and optionally saves it to disk.
    """

    curve_kwgs = curve_kwgs or {"linestyle": "-", "linewidth": 1}
    baseline_kwgs = baseline_kwgs or {
        "linestyle": ":",
        "linewidth": 1.5,
        "color": "black",
        "alpha": 0.7,
    }

    threshold_kwgs = threshold_kwgs or {
        "linestyle": ":",
        "linewidth": 1.5,
        "color": "black",
        "alpha": 0.7,
    }

    lookup_kwgs = lookup_kwgs or {
        "linestyle": "--",
        "linewidth": 1.5,
        "color": "gray",
        "alpha": 0.7,
    }

    if y_test is None:
        raise ValueError("y_test is required.")
    if not (((model is not None) and (X_test is not None)) or (y_prob is not None)):
        raise ValueError(
            "Provide model and X_test with y_test, or provide y_prob with y_test."
        )

    if y_prob is None:
        _, y_pred_probs, _, _ = get_predictions(
            model,
            X_test,
            y_test,
            None,
            None,
            None,
        )
    else:
        y_pred_probs = y_prob

    # Calculate Precision, Recall, and F1 Score for various thresholds
    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_pred_probs,
    )
    f1_scores = (
        2 * (precision * recall) / (precision + recall + 1e-9)
    )  # Avoid division by zero

    # Calculate Specificity for various thresholds
    fpr, _, roc_thresholds = roc_curve(y_test, y_pred_probs)
    specificity = 1 - fpr

    # Find the best threshold for a given metric (if requested)
    best_threshold = None
    if (lookup_metric is not None and lookup_value is None) or (
        lookup_value is not None and lookup_metric is None
    ):
        raise ValueError(
            "Both `lookup_metric` and `lookup_value` must be provided together."
        )
    if lookup_metric and lookup_value is not None:
        metric_dict = {
            "precision": (precision[:-1], thresholds),
            "recall": (recall[:-1], thresholds),
            "f1": (f1_scores[:-1], thresholds),
            "specificity": (specificity, roc_thresholds),
        }

        if lookup_metric in metric_dict:
            metric_values, metric_thresholds = metric_dict[lookup_metric]

            # Find the closest threshold to the requested metric value
            closest_idx = (np.abs(metric_values - lookup_value)).argmin()
            best_threshold = metric_thresholds[closest_idx]

            # Print the result
            print(
                f"Best threshold for target {lookup_metric} of "
                f"{round(lookup_value, decimal_places)} is "
                f"{round(best_threshold, decimal_places)}"
            )
        else:
            print(
                f"Invalid lookup metric: {lookup_metric}. Choose from "
                f"'precision', 'recall', 'f1', 'specificity'."
            )

    # Create the plot
    _, ax = plt.subplots(figsize=figsize)

    # Plot Precision, Recall, F1 Score vs. Thresholds
    ax.plot(
        thresholds,
        f1_scores[:-1],
        label="F1 Score",
        color="red",
        **curve_kwgs,
    )
    ax.plot(
        thresholds,
        recall[:-1],
        label="Recall",
        color="green",
        **curve_kwgs,
    )
    ax.plot(
        thresholds,
        precision[:-1],
        label="Precision",
        color="blue",
        **curve_kwgs,
    )

    # Plot Specificity (adjust to match the corresponding thresholds)
    ax.plot(
        roc_thresholds,
        specificity,
        label="Specificity",
        color="purple",
        **curve_kwgs,
    )

    if baseline_thresh:
        # Draw baseline lines at 0.5 for thresholds and metrics
        ax.axvline(x=0.5, **baseline_kwgs, label="Threshold = 0.5")

    # Highlight the best threshold found
    if best_threshold is not None:
        ax.axvline(
            x=best_threshold,
            label=f"Best Threshold: {round(best_threshold, decimal_places)}",
            **lookup_kwgs,
        )

    if model_threshold is not None:
        ax.axvline(
            x=model_threshold,
            **threshold_kwgs,
            label=f"Model Threshold: {round(model_threshold, decimal_places)}",
        )

    # Apply labels, legend, and formatting
    ax.set_xlabel("Thresholds", fontsize=label_fontsize)
    ax.set_ylabel("Metrics", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(visible=gridlines)

    # Apply title with text wrapping if provided
    if title is None:
        ax.set_title(
            "Precision, Recall, F1 Score, Specificity vs. Thresholds",
            fontsize=label_fontsize,
        )
    elif title != "":
        if text_wrap:
            title_wrapped = "\n".join(textwrap.wrap(title, width=text_wrap))
            ax.set_title(title_wrapped, fontsize=label_fontsize)
        else:
            ax.set_title(title, fontsize=label_fontsize)
    # if title == "", no title at all

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # Push further down to ensure it's outside
        ncol=3,
        fontsize=tick_fontsize,
        frameon=False,
    )

    if lookup_metric:
        save_plot_images(
            filename=f"threshold_metrics_{lookup_metric}",
            save_plot=save_plot,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
        )
    else:
        save_plot_images(
            filename="threshold_metrics",
            save_plot=save_plot,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
        )

    # Display the plot
    plt.show()
