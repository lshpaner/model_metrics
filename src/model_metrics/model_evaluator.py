import pandas as pd
import numpy as np
import math
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import seaborn as sns
import sys
import os
import re
from tqdm import tqdm
import textwrap

import statsmodels.api as sm
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
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


def get_model_probabilities(model, X, name):
    """
    Extract probabilities for the positive class from the model.
    """
    if hasattr(model, "predict_proba"):  # Direct model with predict_proba
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "named_steps"):  # Pipeline
        final_model = list(model.named_steps.values())[-1]
        if hasattr(final_model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(final_model, "decision_function"):
            y_scores = final_model.decision_function(X)
            return 1 / (1 + np.exp(-y_scores))  # Convert to probabilities
    elif hasattr(model, "decision_function"):  # Standalone model with decision_function
        y_scores = model.decision_function(X)
        return 1 / (1 + np.exp(-y_scores))  # Convert to probabilities
    else:
        raise ValueError(f"Model {name} does not support probability-based prediction.")


# Helper function
def extract_model_titles(models_or_pipelines, model_titles=None):
    """
    Extract titles from models or pipelines using an optional external list of
    model titles.

    Parameters:
    -----------
    models_or_pipelines : list
        A list of model or pipeline objects.

    model_titles : list, optional
        A list of human-readable model titles to map class names to.

    Returns:
    --------
    list
        A list of extracted or matched model titles.
    """
    titles = []
    for model in models_or_pipelines:
        try:
            if hasattr(model, "estimator"):
                model = model.estimator
            if hasattr(model, "named_steps"):  # Check if it's a pipeline
                final_model = list(model.named_steps.values())[-1]
                class_name = final_model.__class__.__name__
            else:
                class_name = model.__class__.__name__

            # If model_titles is provided, attempt to map class_name
            if model_titles:
                matched_title = next(
                    (title for title in model_titles if class_name in title),
                    class_name,
                )
            else:
                matched_title = (
                    class_name  # Default to class_name if no titles are provided
                )

            titles.append(matched_title)
        except AttributeError:
            titles.append("Unknown Model")

    return titles


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
    model,
    X,
    y,
    model_type="classification",
    model_threshold=None,
    model_titles=None,
    custom_threshold=None,
    score=None,
    return_df=False,
    overall_only=False,
    decimal_places=3,
):
    """
    Summarizes model performance metrics, including overall metrics and model
    coefficients or feature importances based on model type and attributes.

    Parameters:
    -----------
    model : list or object
        - A single trained model or a list of trained models.
        - Supports classification and regression models.

    X : pd.DataFrame
        Feature matrix used for evaluation. Must have column names for feature
        importance calculations if using tree-based models.

    y : pd.Series or np.array
        Target variable.

    model_type : str, default="classification"
        Specifies whether the models are classification or regression.
        - Must be either "classification" or "regression".

    model_threshold : dict or None, default=None
        - If provided, contains threshold values for classification models.
        - Used when custom_threshold is not set.

    model_titles : str, list of str, pd.Series, or None, default=None
        Custom model names for display.
        - If a single string is provided, it is automatically wrapped in a list.
        - If a Series is provided, it is converted to a list.
        - If None, default names like "Model_1", "Model_2", etc. are used.

    custom_threshold : float or None, default=None
        - If set, overrides model_threshold and applies a fixed threshold for
          classification.
        - When set, the "Model Threshold" row is excluded.

    score : str or None, default=None
        - Custom scoring metric for classification models.

    return_df : bool, default=False
        - If True, returns a DataFrame instead of printing results.

    overall_only : bool, default=False
        - If True, returns only the "Overall Metrics" row for regression models.
        - Removes "Variable", "Coefficient", and "Feat. Imp." columns.
        - Ensures index removal for a clean DataFrame display.

    decimal_places : int, default=3
        Number of decimal places to round metrics.

    Returns:
    --------
    pd.DataFrame or None
        - If return_df=True, returns a DataFrame containing model performance
          metrics.
        - Otherwise, prints the formatted table.

    Raises:
    -------
    ValueError:
        - If model_type="classification" and overall_only=True.
        - If model_type is not "classification" or "regression".

    Notes:
    ------
    - For classification models:
        - Computes precision, recall, specificity, AUC ROC, F1-score,
          Brier score, etc.
        - Requires models supporting predict_proba or decision_function.

    - For regression models:
        - Computes MAE, MAPE, MSE, RMSE, Expl. Var., and R² Score.
        - Uses statsmodels.OLS to extract coefficients for models with a `coef_`
          attribute (e.g., linear models like Lasso, LinearRegression).
        - Uses feature importances for models with a `feature_importances_`
          attribute (e.g., tree-based models like RandomForestRegressor), displayed
          in a "Feat. Imp." column.

    - If overall_only=True, the DataFrame will:
        - Contain only "Overall Metrics".
        - Drop unnecessary coefficient-related columns ("Variable", "Coefficient",
          "Feat. Imp.").
        - Have an empty index to remove the leading row number.
    """

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

    # Normalize model_titles input
    if model_titles is None:
        model_titles = [f"Model_{i+1}" for i in range(len(models))]
    elif isinstance(model_titles, str):
        model_titles = [model_titles]
    elif isinstance(model_titles, pd.Series):
        model_titles = model_titles.tolist()
    elif not isinstance(model_titles, list):
        raise TypeError(
            "model_titles must be a string, list of strings, Series, or None."
        )

    for i, model in enumerate(models):
        name = model_titles[i]
        if model_type == "classification":
            y_true, y_prob, y_pred, threshold = get_predictions(
                model, X, y, model_threshold, custom_threshold, score
            )

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
            # Always add a constant term for all regression models
            X_with_intercept = sm.add_constant(X)

            if isinstance(model, sm.OLS):
                # For statsmodels OLS, predict and extract coefficients directly
                y_pred = model.predict(X_with_intercept)
                coefficients = pd.Series(model.params.round(decimal_places)).to_dict()
            else:
                # For scikit-learn models, predict on X_with_intercept, adjusting for intercept
                try:
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
############################## COnfusion Matrix ################################
################################################################################


def show_confusion_matrix(
    model,
    X,
    y,
    model_titles=None,
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
    grid=False,  # Added grid option
    score=None,
    class_report=False,
    **kwargs,
):
    """
    Generate and display confusion matrices for one or multiple models.

    This function computes confusion matrices for classifiers and visualizes
    them with customizable formatting, threshold adjustments, and optional
    classification reports. Supports both individual and grid-based plots.

    Parameters:
    - model (estimator): A single model (string) or a list of models/pipelines.
    - X (array-like): Feature matrix for predictions.
    - y (array-like): True labels.
    - model_titles (list, optional): Custom titles for models.
    - model_threshold (float, optional): Threshold for predictions.
    - custom_threshold (float, optional): User-defined threshold override.
    - class_labels (list, optional): Custom labels for the confusion matrix.
    - cmap (str, default="Blues"): Colormap for visualization.
    - save_plot (bool, default=False): Whether to save plots.
    - image_path_png (str, optional): Path to save PNG images.
    - image_path_svg (str, optional): Path to save SVG images.
    - text_wrap (int, optional): Max title width before wrapping.
    - figsize (tuple, default=(8,6)): Figure size for each confusion matrix.
    - labels (bool, default=True): Whether to display TN, FP, FN, TP labels.
    - label_fontsize (int, default=12): Font size for axis labels.
    - tick_fontsize (int, default=10): Font size for tick labels.
    - inner_fontsize (int, default=10): Font size for matrix values.
    - grid (bool, default=False): Whether to display multiple plots in a grid.
    - score (str, optional): Metric to optimize when selecting a threshold.
    - class_report (bool, default=False): Whether to print classification reports.
    - **kwargs: Additional options for plot customization.

    Returns:
    - None
    """
    if not isinstance(model, list):
        model = [model]

    # Normalize model_titles input
    if model_titles is None:
        model_titles = [f"Model {i+1}" for i in range(len(model))]
    elif isinstance(model_titles, str):
        model_titles = [model_titles]
    elif not isinstance(model_titles, list):
        raise TypeError("model_titles must be a string, a list of strings, or None.")

    # Setup grid if enabled
    if grid:
        n_cols = kwargs.get("n_cols", 2)
        n_rows = (len(model) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
        )
        axes = axes.flatten()
    else:
        axes = [None] * len(model)

    for idx, (m, ax) in enumerate(zip(model, axes)):
        # Determine the model name
        if model_titles:
            name = model_titles[idx]
        else:
            name = extract_model_name(m)

        y_true, y_prob, y_pred, threshold = get_predictions(
            m, X, y, model_threshold, custom_threshold, score
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
        if grid:
            if "colorbar" in disp.plot.__code__.co_varnames:
                disp.plot(cmap=cmap, ax=ax, colorbar=show_colorbar)
            else:
                disp.plot(cmap=cmap, ax=ax)
        else:
            fig, ax = plt.subplots(figsize=figsize)
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

        # Re-annotate correctly
        fmt = ".0f"  # Adjust format if needed
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

        if not grid:
            save_plot_images(
                f"Confusion_Matrix_{name}",
                save_plot,
                image_path_png,
                image_path_svg,
            )
            plt.show()

    if grid:
        for ax in axes[len(model) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "Grid_Confusion_Matrix", save_plot, image_path_png, image_path_svg
        )
        plt.show()


################################################################################
##################### ROC AUC and Precision Recall Curves ######################
################################################################################


def show_roc_curve(
    models,
    X,
    y,
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    model_titles=None,
    decimal_places=2,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    grid=False,  # Grid layout option
    n_rows=None,
    n_cols=2,  # Number of columns for the grid
    figsize=(8, 6),  # User-defined figure size
    label_fontsize=12,  # Font size for title and axis labels
    tick_fontsize=10,  # Font size for tick labels and legend
    gridlines=True,
    group_category=None,
):
    """
    Plot Receiver Operating Characteristic (ROC) curves for models or pipelines
    with optional styling, grid layout, and grouping by categories, including
    class counts in the legend.

    Parameters:
    - models: list
        List of models or pipelines to plot ROC curves for.
    - X: array-like
        Feature data for prediction, typically a pandas DataFrame or NumPy array.
    - y: array-like
        True binary labels for evaluation,(e.g., a pandas Series or NumPy array).
    - model_titles: str or list of str, optional
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
        `grid=True`. If None, uses a default title; if "", disables the title.
    - save_plot: bool, optional
        Whether to save the plot to the specified paths (default: False).
    - image_path_png: str, optional
        Path to save the plot as a PNG image.
    - image_path_svg: str, optional
        Path to save the plot as an SVG image.
    - text_wrap: int, optional
        Maximum width for wrapping titles if they are too long (default: None).
    - curve_kwgs: list or dict, optional
        Styling for individual model curves. If `model_titles` is specified as a
        list, `curve_kwgs` must be a nested dictionary with model titles as keys
        and their respective style dictionaries (e.g., {'color': 'red',
        'linestyle': '--'}) as values. Otherwise, `curve_kwgs` must be a list of
        style dictionaries corresponding to the models.
    - linestyle_kwgs: dict, optional
        Styling for the random guess diagonal line (default: {'color': 'gray',
        'linestyle': '--', 'linewidth': 2}).
    - grid: bool, optional
        Whether to organize plots in a grid layout (default: False). Cannot be
        True if `overlay=True`.
    - n_rows: int, optional
        Number of rows in the grid layout. If not specified, calculated
        automatically based on the number of models and `n_cols`.
    - n_cols: int, optional
        Number of columns in the grid layout (default: 2).
    - figsize: tuple, optional
        Custom figure size (width, height) for the plot(s) (default: None, uses
        (8, 6) for overlay or calculated size for grid).
    - label_fontsize: int, optional
        Font size for titles and axis labels (default: 12).
    - tick_fontsize: int, optional
        Font size for tick labels and legend (default: 10).
    - gridlines: bool, optional
        Whether to display grid lines on the plot (default: True).
    - group_category: array-like, optional
        Categorical data (e.g., pandas Series or NumPy array) to group ROC curves
        by unique values. Cannot be used with `grid=True` or `overlay=True`.
        If provided, separate ROC curves are plotted for each group, with AUC
        and class counts (Total, Pos, Neg) shown in the legend.

    Raises:
        - ValueError: If `grid=True` and `overlay=True` are both set, if
            `grid=True` and `group_category` is provided, if `overlay=True` and
            `group_category` is provided, or if `overlay=True` and only one
            model is provided.
    Notes:
    - When `group_category` is provided, the legend includes AUC, total count,
      and positive/negative class counts for each group (e.g., "AUC = 0.XX,
      Count: Total: X, Pos: Y, Neg: Z").
    - The random guess line (diagonal line at y=x) is plotted for reference.
    - Titles can be customized, disabled with `title=""`, or default to
      "ROC Curve: Model Name" or "ROC Curves: Overlay" if not specified.
    """

    if overlay and grid:
        raise ValueError("`grid` cannot be set to True when `overlay` is True.")

    if grid and group_category is not None:
        raise ValueError(
            f"`grid` cannot be set to True when `group_category` is provided. "
            f"When selecting `group_category`, make sure `grid` and `overlay` "
            f"are set to `False`."
        )

    if overlay and len(models) == 1:
        raise ValueError(
            f"Cannot use `overlay=True` with only one model. "
            f"Use `overlay=False` to plot a single model, or provide multiple "
            f"models for overlay."
        )

    if overlay and group_category is not None:
        raise ValueError(
            f"`overlay` cannot be set to True when `group_category` is "
            f"provided. When selecting `group_category`, make sure `grid` and "
            f"`overlay` are set to `False`."
        )

    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]

    # Normalize model_titles input
    if model_titles is None:
        model_titles = [f"Model {i+1}" for i in range(len(models))]
    elif isinstance(model_titles, str):
        model_titles = [model_titles]
    elif not isinstance(model_titles, list):
        raise TypeError("model_titles must be a string, a list of strings, or None.")

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(models)

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if grid and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(models) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (model, name, curve_style) in enumerate(
        zip(models, model_titles, curve_styles)
    ):
        y_true, y_prob, y_pred, threshold = get_predictions(
            model, X, y, None, None, None
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
        elif grid:
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
        plt.grid()
        save_plot_images(
            "overlay_roc_auc_plot",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif grid:
        for ax in axes[len(models) :]:
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
    models,
    X,
    y,
    xlabel="Recall",
    ylabel="Precision",
    model_titles=None,
    decimal_places=2,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    grid=False,
    n_rows=None,
    n_cols=2,
    figsize=(8, 6),
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    group_category=None,
):
    """
    Plot Precision-Recall (PR) curves for models or pipelines with optional
    styling, grid layout, and grouping by categories, including class counts in
    the legend.

    Parameters:
    - models: list
        List of models or pipelines to plot PR curves for.
    - X: array-like
        Feature data for prediction, typically a pandas DataFrame or NumPy array.
    - y: array-like
        True binary labels for evaluation (e.g., a pandas Series or NumPy array).
    - group_category: array-like, optional
        Categorical data (e.g., pandas Series or NumPy array) to group PR curves
        by unique values. If provided, plots separate PR curves for each group
        with Average Precision (AP) and class counts (Total, Pos, Neg) in the
        legend.
    - model_titles: str or list of str, optional
        Title or list of titles for the models. If a single string is provided,
        it is automatically converted to a one-element list. If None, defaults to
        "Model 1", "Model 2", etc. Required when using a nested dictionary for
        `curve_kwgs`.
    - xlabel: str, optional
        Label for the x-axis (default: "Recall").
    - ylabel: str, optional
        Label for the y-axis (default: "Precision").
    - decimal_places: int, optional
        Number of decimal places to round Average Precision (AP) values in the
        legend and print output (default: 3).
    - overlay: bool, optional
        Whether to overlay multiple models on a single plot (default: False).
    - title: str, optional
        Custom title for the plot when `overlay=True` or per-model title when
        `grid=True`. If None, uses a default title; if "", disables the title.
    - save_plot: bool, optional
        Whether to save the plot to the specified paths (default: False).
    - image_path_png: str, optional
        Path to save the plot as a PNG image.
    - image_path_svg: str, optional
        Path to save the plot as an SVG image.
    - text_wrap: int, optional
        Maximum width for wrapping titles if they are too long (default: None).
    - curve_styles: list or dict, optional
        Styling for individual model curves. If `model_titles` is specified as a
        list, `curve_styles` must be a nested dictionary with model titles as
        keys and their respective style dictionaries (e.g., {'color': 'red',
        'linestyle': '--'}) as values. Otherwise, `curve_styles` must be a list
        of style dictionaries corresponding to the models.
    - linestyle_kwgs: dict, optional
        Styling for the random classifier baseline line (default: {'color':
        'gray', 'linestyle': '--', 'linewidth': 2}).
    - grid: bool, optional
        Whether to organize plots in a grid layout (default: False). Cannot be
        True if `overlay=True`.
    - n_rows: int, optional
        Number of rows in the grid layout. If not specified, calculated
        automatically based on the number of models and `n_cols`.
    - n_cols: int, optional
        Number of columns in the grid layout (default: 2).
    - figsize: tuple, optional
        Custom figure size (width, height) for the plot(s) (default: None, uses
        (8, 6) for overlay or calculated size for grid).
    - label_fontsize: int, optional
        Font size for titles and axis labels (default: 12).
    - tick_fontsize: int, optional
        Font size for tick labels and legend (default: 10).
    - gridlines: bool, optional
        Whether to display grid lines on the plot (default: True).
    - group_category: array-like, optional
        Categorical data (e.g., pandas Series or NumPy array) to group ROC curves
        by unique values. Cannot be used with `grid=True` or `overlay=True`.
        If provided, separate ROC curves are plotted for each group, with AUC
        and class counts (Total, Pos, Neg) shown in the legend.


    Raises:
    - ValueError: If `grid=True` and `overlay=True` are both set, if
        `grid=True` and `group_category` is provided, or if `overlay=True` and
        `group_category` is provided.

    Notes:
    - When `group_category` is provided, the legend includes Average Precision
      (AP), total count, and positive/negative class counts for each group
      (e.g., "Average Precision for Male = 0.___, Count: Total: X, Pos: Y,
      Neg: Z").
    - The random classifier baseline (precision = fraction of positive samples)
      is plotted for reference.
    - Titles can be customized, disabled with `title=""`, or default to "PR
      Curve: Model Name" or "PR Curves: Overlay" if not specified.
    """

    if overlay and grid:
        raise ValueError("`grid` cannot be set to True when `overlay` is True.")

    if grid and group_category is not None:
        raise ValueError(
            f"`grid` cannot be set to True when `group_category` is provided. "
            f"When selecting `group_category`, make sure `grid` and `overlay` "
            f"are set to `False`."
        )

    if overlay and group_category is not None:
        raise ValueError(
            f"`overlay` cannot be set to True when `group_category` is "
            f"provided. When selecting `group_category`, make sure `grid` and "
            f"`overlay` are set to `False`."
        )

    if not isinstance(models, list):
        models = [models]

    # Normalize model_titles input
    if model_titles is None:
        model_titles = [f"Model {i+1}" for i in range(len(models))]
    elif isinstance(model_titles, str):
        model_titles = [model_titles]
    elif not isinstance(model_titles, list):
        raise TypeError("model_titles must be a string, a list of strings, or None.")

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(models)

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if grid and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(models) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (model, name, curve_style) in enumerate(
        zip(models, model_titles, curve_styles)
    ):
        y_true, y_prob, y_pred, threshold = get_predictions(
            model, X, y, None, None, None
        )

        counts = {}
        if group_category is not None:
            precision = {}
            recall = {}
            ap_str = {}  # Use ap_str for Average Precision
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
                # Format Average Precision with decimal_places for print and legend
                ap_str[gr] = f"{avg_precision:.{decimal_places}f}"

        else:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            # Format Average Precision with decimal_places for print and legend
            ap_str = f"{avg_precision:.{decimal_places}f}"

        print(f"Average Precision for {name}: {avg_precision:.{decimal_places}f}")

        if overlay:
            plt.plot(
                recall,
                precision,
                label=f"{name} (AP = {avg_precision:.{decimal_places}f})",
                **curve_style,
            )
        elif grid:
            ax = axes[idx]
            if group_category is not None:
                for gr in group_category.unique():
                    ax.plot(
                        recall[gr],
                        precision[gr],
                        label=f"AP for {gr} = {ap_str[gr]}, "
                        f"Count: {counts[gr][0]:,}, "
                        f"Pos: {counts[gr][1]:,}, Neg: {counts[gr][2]:,}",
                        **curve_style,
                    )
            else:
                ax.plot(
                    recall,
                    precision,
                    label=f"Average Precision = {ap_str}",
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
                        label=f"AP for {gr} = {ap_str[gr]}, "
                        f"Count: {counts[gr][0]:,}, "
                        f"Pos: {counts[gr][1]:,}, Neg: {counts[gr][2]:,}",
                        **curve_style,
                    )
            else:
                plt.plot(
                    recall,
                    precision,
                    label=f"Average Precision = {ap_str}",
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
        plt.grid()
        save_plot_images(
            "overlay_pr_plot",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif grid:
        for ax in axes[len(models) :]:
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
    models,
    X,
    y,
    xlabel="Percentage of Sample",
    ylabel="Lift",
    model_titles=None,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    grid=False,
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
    Supports multiple models with overlay or grid layouts and customizable styling.

    Parameters:
    - models (list or estimator): One or more trained models.
    - X (array-like): Feature matrix.
    - y (array-like): True labels.
    - xlabel (str, default="Percentage of Sample"): Label for the x-axis.
    - ylabel (str, default="Lift"): Label for the y-axis.
    - model_titles (list, optional): Custom titles for models.
    - overlay (bool, default=False): Whether to overlay multiple models in one plot.
    - title (str, optional): Custom title; set to `""` to disable.
    - save_plot (bool, default=False): Whether to save the plot.
    - image_path_png (str, optional): Path to save PNG image.
    - image_path_svg (str, optional): Path to save SVG image.
    - text_wrap (int, optional): Maximum title width before wrapping.
    - curve_kwgs (dict or list, optional): Styling options for model curves.
    - linestyle_kwgs (dict, optional): Styling options for the baseline.
    - grid (bool, default=False): Display multiple plots in a grid layout.
    - n_cols (int, default=2): Number of columns in the grid layout.
    - n_rows (int, optional): Number of rows in the grid layout.
    - figsize (tuple, optional): Figure size.
    - label_fontsize (int, default=12): Font size for axis labels.
    - tick_fontsize (int, default=10): Font size for tick labels.
    - gridlines (bool, default=True): Whether to show grid lines.

    Raises:
    - ValueError: If `grid=True` and `overlay=True` are both set.

    Returns:
    - None
    """

    if overlay and grid:
        raise ValueError("`grid` cannot be set to True when `overlay` is True.")

    if not isinstance(models, list):
        models = [models]

    if model_titles is None:
        model_titles = [f"Model {i+1}" for i in range(len(models))]

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(models)

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if grid and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(models) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (model, name, curve_style) in enumerate(
        zip(models, model_titles, curve_styles)
    ):
        y_probs = model.predict_proba(X)[:, 1]
        sorted_indices = np.argsort(y_probs)[::-1]
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
        elif grid:
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
                f"{name}_Lift",
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
            "Overlay_Lift",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif grid:
        for ax in axes[len(models) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images("Grid_Lift", save_plot, image_path_png, image_path_svg)
        plt.show()


def show_gain_chart(
    models,
    X,
    y,
    xlabel="Percentage of Sample",
    ylabel="Cumulative Gain",
    model_titles=None,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    grid=False,
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
    Supports multiple models with overlay or grid layouts and customizable styling.

    Parameters:
    - models (list or estimator): One or more trained models.
    - X (array-like): Feature matrix.
    - y (array-like): True labels.
    - xlabel (str, default="Percentage of Sample"): Label for the x-axis.
    - ylabel (str, default="Cumulative Gain"): Label for the y-axis.
    - model_titles (list, optional): Custom titles for models.
    - overlay (bool, default=False): Whether to overlay multiple models in one plot.
    - title (str, optional): Custom title; set to `""` to disable.
    - save_plot (bool, default=False): Whether to save the plot.
    - image_path_png (str, optional): Path to save PNG image.
    - image_path_svg (str, optional): Path to save SVG image.
    - text_wrap (int, optional): Maximum title width before wrapping.
    - curve_kwgs (dict or list, optional): Styling options for model curves.
    - linestyle_kwgs (dict, optional): Styling options for the baseline.
    - grid (bool, default=False): Display multiple plots in a grid layout.
    - n_cols (int, default=2): Number of columns in the grid layout.
    - n_rows (int, optional): Number of rows in the grid layout.
    - figsize (tuple, optional): Figure size.
    - label_fontsize (int, default=12): Font size for axis labels.
    - tick_fontsize (int, default=10): Font size for tick labels.
    - gridlines (bool, default=True): Whether to show grid lines.

    Raises:
    - ValueError: If `grid=True` and `overlay=True` are both set.

    Returns:
    - None
    """

    if overlay and grid:
        raise ValueError("`grid` cannot be set to True when `overlay` is True.")

    if not isinstance(models, list):
        models = [models]

    if model_titles is None:
        model_titles = [f"Model {i+1}" for i in range(len(models))]

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(models)

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if grid and not overlay:
        if n_rows is None:
            n_rows = math.ceil(len(models) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (model, name, curve_style) in enumerate(
        zip(models, model_titles, curve_styles)
    ):
        y_probs = model.predict_proba(X)[:, 1]
        sorted_indices = np.argsort(y_probs)[::-1]
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
        elif grid:
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
                f"{name}_Gain",
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
            "Overlay_Gain",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif grid:
        for ax in axes[len(models) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images("Grid_Gain", save_plot, image_path_png, image_path_svg)
        plt.show()


################################################################################
############################## Calibration Curve ###############################
################################################################################


def show_calibration_curve(
    model,
    X,
    y,
    xlabel="Mean Predicted Probability",
    ylabel="Fraction of Positives",
    model_titles=None,
    overlay=False,
    title=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    grid=False,  # Grid layout option
    n_cols=2,  # Number of columns for the grid
    figsize=None,  # User-defined figure size
    label_fontsize=12,
    tick_fontsize=10,
    bins=10,  # Number of bins for calibration curve
    marker="o",  # Marker style for the calibration points
    show_brier_score=True,
    gridlines=True,
    linestyle_kwgs=None,
    **kwargs,
):
    """
    Plot calibration curves for models or pipelines with optional styling and
    grid layout.

    Parameters:
    - model: list
        List of models or pipelines to plot.
    - X: array-like
        Features for prediction.
    - y: array-like
        True labels.
    - model_titles: list or str, optional
        Titles for individual models.
    - overlay: bool
        Whether to overlay multiple models on a single plot.
    - title: str, optional
        Custom title for the plot when `overlay=True`.
    - save_plot: bool
        Whether to save the plot.
    - image_path_png: str, optional
        Path to save PNG images.
    - image_path_svg: str, optional
        Path to save SVG images.
    - text_wrap: int, optional
        Max width for wrapping titles.
    - curve_kwgs: list or dict, optional
        Styling for individual model curves.
    - grid: bool, optional
        Whether to organize plots in a grid layout (default: False).
    - n_cols: int, optional
        Number of columns in the grid layout (default: 2).
    - figsize: tuple, optional
        Custom figure size (width, height) for the plot(s).
    - label_fontsize: int, optional
        Font size for axis labels and title.
    - tick_fontsize: int, optional
        Font size for tick labels and legend.
    - bins: int, optional
        Number of bins for the calibration curve (default: 10).
    - marker: str, optional
        Marker style for calibration curve points (default: "o").

    Raises:
    - ValueError: If `grid=True` and `overlay=True` are both set.
    """
    if overlay and grid:
        raise ValueError("`grid` cannot be set to True when `overlay` is True.")

    if not isinstance(model, list):
        model = [model]

    # Normalize model_titles input
    if model_titles is None:
        model_titles = [f"Model_{i+1}" for i in range(len(model))]
    elif isinstance(model_titles, str):
        model_titles = [model_titles]
    elif isinstance(model_titles, pd.Series):
        model_titles = model_titles.tolist()
    elif not isinstance(model_titles, list):
        raise TypeError(
            "model_titles must be a string, list of strings, Series, or None."
        )

    if isinstance(curve_kwgs, dict):
        curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
    elif isinstance(curve_kwgs, list):
        curve_styles = curve_kwgs
    else:
        curve_styles = [{}] * len(model)

    if len(curve_styles) != len(model):
        raise ValueError("The length of `curve_kwgs` must match the number of models.")

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if grid and not overlay:
        import math

        n_rows = math.ceil(len(model) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
        )
        axes = axes.flatten()

    for idx, (model, name, curve_style) in enumerate(
        zip(model, model_titles, curve_styles)
    ):
        y_true, y_prob, y_pred, threshold = get_predictions(
            model, X, y, None, None, None
        )
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins)

        # Calculate Brier score if enabled
        brier_score = brier_score_loss(y_true, y_prob) if show_brier_score else None

        legend_label = f"{name}"
        if show_brier_score:
            legend_label += f" $\Rightarrow$ (Brier score: {brier_score:.4f})"

        if overlay:
            plt.plot(
                prob_pred,
                prob_true,
                marker=marker,
                label=legend_label,
                **curve_style,
                **kwargs,
            )
        elif grid:
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
                grid_title = f"Calibration Curve: {name}"  # Default title
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
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.grid(visible=gridlines)
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
                plot_title = f"Calibration Curve: {name}"  # Default title
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

            plt.legend(loc="best", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            save_plot_images(
                f"{name}_Calibration",
                save_plot,
                image_path_png,
                image_path_svg,
            )
            plt.show()

    if overlay:
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
            overlay_title = "Calibration Curves: Overlay"  # Default title
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
            "Overlay_Calibration",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()
    elif grid:
        num_models = len(model) if isinstance(model, list) else 1
        for ax in axes[num_models:]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images(
            "Grid_Calibration",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


################################################################################
######################### Kolgomorov-Smirnoff Curve ############################
################################################################################


def show_ks_curve(
    models,
    X,
    y,
    xlabel="Cumulative Probability",
    ylabel="Empirical CDF",
    model_titles=None,
    title=None,
    decimal_places=2,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    text_wrap=None,
    curve_kwgs=None,
    linestyle_kwgs=None,
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    model_threshold=None,
    custom_threshold=None,
    score=None,
):
    """
    Plot the Kolmogorov-Smirnov (KS) statistic curve using model predictions.

    Parameters:
    - models: List of trained models or a single model.
    - X: Features for prediction.
    - y: True binary labels.
    - model_titles: str or list of model names for labeling.
    - xlabel, ylabel: Axis labels.
    - title: Title for the plot.
    - save_plot: Whether to save the plot.
    - image_path_png, image_path_svg: Paths to save PNG and SVG images.
    - text_wrap: Max width for wrapping titles.
    - curve_kwgs: Styling for the CDF curves.
    - linestyle_kwgs: Styling for the KS statistic vertical line.
    - figsize: Custom figure size.
    - label_fontsize, tick_fontsize: Font sizes.
    - gridlines: Whether to show grid lines.
    - model_threshold, custom_threshold: Thresholds for probability-based
      classification.
    - score: Scoring metric used for selecting a threshold.
    """

    if not isinstance(models, list):
        models = [models]

    # Normalize model_titles input
    if model_titles is None:
        model_titles = [f"Model_{i+1}" for i in range(len(models))]
    elif isinstance(model_titles, str):
        model_titles = [model_titles]
    elif isinstance(model_titles, pd.Series):
        model_titles = model_titles.tolist()
    elif not isinstance(model_titles, list):
        raise TypeError(
            "model_titles must be a string, list of strings, Series, or None."
        )

    curve_kwgs = curve_kwgs or {}
    linestyle_kwgs = linestyle_kwgs or {"linestyle": "--", "linewidth": 2}

    colors = itertools.cycle(
        [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
    )
    plt.figure(figsize=figsize or (8, 6))

    for model, name, color in zip(models, model_titles, colors):
        print(f"Processing Model: {name}")  # Debugging statement

        y_true, y_prob, _, _ = get_predictions(
            model, X, y, model_threshold, custom_threshold, score
        )

        y_true = np.array(y_true).flatten()
        y_prob = np.array(y_prob).flatten()

        positives = y_prob[y_true == 1] if np.any(y_true == 1) else np.array([])
        negatives = y_prob[y_true == 0] if np.any(y_true == 0) else np.array([])

        if len(positives) == 0 or len(negatives) == 0:
            print(
                f"Warning: {name} has an empty group. "
                f"KS test may not be meaningful."
            )
            continue  # Skip this model

        pos_sorted = np.sort(positives)
        neg_sorted = np.sort(negatives)

        pos_cdf = np.linspace(0, 1, len(pos_sorted), endpoint=True)
        neg_cdf = np.linspace(0, 1, len(neg_sorted), endpoint=True)

        # Interpolate CDFs onto a common grid
        common_grid = np.sort(np.concatenate([pos_sorted, neg_sorted]))
        pos_interp_cdf = np.interp(
            common_grid,
            pos_sorted,
            pos_cdf,
            left=0,
            right=1,
        )
        neg_interp_cdf = np.interp(
            common_grid,
            neg_sorted,
            neg_cdf,
            left=0,
            right=1,
        )

        # Compute KS statistic and find the correct split point
        ks_stat, ks_p_value = ks_2samp(positives, negatives)
        ks_x = common_grid[np.argmax(np.abs(pos_interp_cdf - neg_interp_cdf))]

        sns.lineplot(
            x=pos_sorted, y=pos_cdf, label=f"{name} Positive CDF", **curve_kwgs
        )
        sns.lineplot(
            x=neg_sorted, y=neg_cdf, label=f"{name} Negative CDF", **curve_kwgs
        )
        plt.axvline(
            x=ks_x,
            **{**linestyle_kwgs, "color": color},
            label=f"{name} KS Stat: {ks_stat:.{decimal_places}f}",
        )

        if ks_p_value <= 0.01:
            p_value_str = "p < 0.01"
        else:
            p_value_str = f"p-value = {ks_p_value:.{decimal_places}f}"
        print(
            f"{name}: Kolmogorov-Smirnov Statistic = "
            f"{ks_stat:.{decimal_places}f}, "
            f"{p_value_str}"
        )

        # Warn if sample size is too small
        if len(positives) < 10 or len(negatives) < 10:
            print(f"Warning: Small sample size for {name}. KS test may be unreliable.")

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.tick_params(axis="both", labelsize=tick_fontsize)
    plt.legend(fontsize=tick_fontsize)
    plt.grid(visible=gridlines)

    if title:
        if text_wrap:
            title = "\n".join(textwrap.wrap(title, width=text_wrap))
        plt.title(title, fontsize=label_fontsize)

    if save_plot:
        if image_path_png:
            plt.savefig(image_path_png, format="png")
        if image_path_svg:
            plt.savefig(image_path_svg, format="svg")

    plt.show()


################################################################################
################## Classification Metrics Threshold Trade-Off ##################
################################################################################


def plot_threshold_metrics(
    model,
    X_test,
    y_test,
    title=None,
    text_wrap=None,
    figsize=(8, 6),
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    baseline_thresh=True,
    curve_kwgs=None,
    baseline_kwgs=None,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    lookup_metric=None,
    lookup_value=None,
    decimal_places=4,
):
    """
    Plot Precision, Recall, F1 Score, and Specificity vs. Thresholds for a model.
    Allows finding the best threshold for a given precision, recall, F1-score,
    or specificity.

    Parameters:
    - model: A trained model that supports `predict_proba`.
    - X_test: Feature matrix for testing.
    - y_test: True binary labels.
    - title: Custom title for the plot (default: None).
    - text_wrap: Maximum width for title text before wrapping (default: None).
    - figsize: Tuple specifying figure size (default: (10, 6)).
    - label_fontsize: Font size for axis labels and title (default: 12).
    - tick_fontsize: Font size for tick labels (default: 10).
    - gridlines: Boolean flag to display gridlines (default: True).
    - baseline_thresh: Boolean flag to display baseline threshold on plot.
    - curve_kwgs: Dictionary of keyword arguments for curve styling
      (default: None).
    - baseline_kwgs: Dictionary of keyword arguments for baseline styling
      (default: None).
    - save_plot: Boolean flag to save the plot (default: False).
    - image_path_png: Path to save the plot as a PNG image (default: None).
    - image_path_svg: Path to save the plot as an SVG image (default: None).
    - lookup_metric: Metric to find the best threshold for ("precision",
      "recall", "f1", "specificity").
    - lookup_value: Desired value for the chosen lookup metric (default: None).
    - decimal_places: Number of decimal places for numerical outputs except axes
      (default: 4).
    """

    curve_kwgs = curve_kwgs or {"linestyle": "-", "linewidth": 1}
    baseline_kwgs = baseline_kwgs or {
        "linestyle": ":",
        "linewidth": 1.5,
        "color": "black",
        "alpha": 0.7,
    }

    # Use get_predictions instead of model.predict_proba
    _, y_pred_probs, _, _ = get_predictions(
        model,
        X_test,
        y_test,
        None,
        None,
        None,
    )

    # Calculate Precision, Recall, and F1 Score for various thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1_scores = (
        2 * (precision * recall) / (precision + recall + 1e-9)
    )  # Avoid division by zero

    # Calculate Specificity for various thresholds
    fpr, _, roc_thresholds = roc_curve(y_test, y_pred_probs)
    specificity = 1 - fpr

    # Find the best threshold for a given metric (if requested)
    best_threshold = None
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
                f"Best threshold for {lookup_metric} = "
                f"{round(lookup_value, decimal_places)} is: "
                f"{round(best_threshold, decimal_places)}"
            )
        else:
            print(
                f"Invalid lookup metric: {lookup_metric}. Choose from "
                f"'precision', 'recall', 'f1', 'specificity'."
            )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

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
            linestyle="--",
            color="gray",
            label=f"Best Threshold: {round(best_threshold, decimal_places)}",
        )

    # Apply labels, legend, and formatting
    ax.set_xlabel("Thresholds", fontsize=label_fontsize)
    ax.set_ylabel("Metrics", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(visible=gridlines)

    # Apply title with text wrapping if provided
    if title:
        if text_wrap:
            title = "\n".join(textwrap.wrap(title, width=text_wrap))
        ax.set_title(title, fontsize=label_fontsize)
    else:
        ax.set_title(
            "Precision, Recall, F1 Score, Specificity vs. Thresholds",
            fontsize=label_fontsize,
        )

    # Move the legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15 * (figsize[0] / 10)),  # Scales with width
        ncol=int(figsize[0] // 2),  # Adjust columns dynamically
        fontsize=label_fontsize,
        columnspacing=1.5,
        handletextpad=1.2,
    )
    # Save plot if required
    if save_plot:
        if image_path_png:
            fig.savefig(image_path_png, format="png", bbox_inches="tight")
        if image_path_svg:
            fig.savefig(image_path_svg, format="svg", bbox_inches="tight")

    # Display the plot
    plt.show()
