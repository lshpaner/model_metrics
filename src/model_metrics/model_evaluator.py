import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import sys
import os
import re
from tqdm import tqdm
import textwrap

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


def summarize_model_performance(
    model,
    X,
    y,
    model_threshold=None,
    model_titles=None,
    custom_threshold=None,
    score=None,
    return_df=False,
):
    """
    Summarize key performance metrics for multiple models.

    Parameters:
    - model: list
        A list of models or pipelines to evaluate.
        Each pipeline should either end with a classifier or contain one.
    - X: array-like
        The input features for generating predictions.
    - y_true: array-like
        The true labels corresponding to the input features.
    - model_threshold: dict or None, optional
        A dictionary mapping model names to predefined thresholds for binary
        classification. If provided, these thresholds will be displayed in
        the table but not used for metric recalculations when `custom_threshold`
        is set.
    - model_titles: list or None, optional
        A list of custom titles for individual models. If not provided, the
        names of the models will be extracted automatically.
    - custom_threshold: float or None, optional
        A custom threshold to apply for recalculating metrics. If set, this
        threshold will override the default threshold of 0.5 and any thresholds
        from `model_threshold` for all models.
        When specified, the "Model Threshold" row is omitted from the table.
    - return_df: bool, optional
        Whether to return the metrics as a pandas DataFrame instead of printing
        them to the console. Default is False.

    Returns:
    - pd.DataFrame or None
        If `return_df` is True, returns a DataFrame summarizing model performance
        metrics, including precision, recall, specificity, F1-Score, AUC ROC,
        and Brier Score. Otherwise, prints the metrics in a formatted table.

    Notes:
    - If `model_threshold` is provided and `custom_threshold` is not set, the
    "Model Threshold" row will display the values from `model_threshold`.
    - If `custom_threshold` is set, it applies to all models for metric
    recalculations, and the "Model Threshold" row is excluded from the table.
    - Automatically extracts model names if `model_titles` is not provided.
    - Models must support `predict_proba` or `decision_function` for predictions.
    """

    if not isinstance(model, list):
        model = [model]

    metrics_data = []

    for i, model in enumerate(model):
        # Determine the model name
        if model_titles:
            name = model_titles[i]
        else:
            name = extract_model_name(model)  # Extract detailed name

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
            "Precision/PPV": precision,
            "Average Precision": avg_precision,
            "Sensitivity/Recall": recall,
            "Specificity": specificity,
            "F1-Score": f1,
            "AUC ROC": auc_roc,
            "Brier Score": brier,
            "Model Threshold": threshold,
        }
        metrics_data.append(model_metrics)

    # Create a DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Model", inplace=True)
    metrics_df = metrics_df.T

    # Return the DataFrame if requested
    if return_df:
        return metrics_df

    # Adjust column widths for center alignment
    col_widths = {col: max(len(col), 8) + 2 for col in metrics_df.columns}
    row_name_width = max(len(row) for row in metrics_df.index) + 2

    # Center-align headers
    headers = [
        f"{'Metric'.center(row_name_width)}"
        + "".join(f"{col.center(col_widths[col])}" for col in metrics_df.columns)
    ]

    # Separator line
    separator = "-" * (row_name_width + sum(col_widths.values()))

    # Print table header
    print("Model Performance Metrics:")
    print("\n".join(headers))
    print(separator)

    # Center-align rows
    for row_name, row_data in metrics_df.iterrows():
        row = f"{row_name.center(row_name_width)}" + "".join(
            (
                f"{f'{value:.4f}'.center(col_widths[col])}"
                if isinstance(value, float)
                else f"{str(value).center(col_widths[col])}"
            )
            for col, value in zip(metrics_df.columns, row_data)
        )
        print(row)


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
    - model (list or estimator): A single model or a list of models/pipelines.
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

    if model_titles is None:
        model_titles = [extract_model_name(model) for model in model]

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
    n_cols=2,  # Number of columns for the grid
    figsize=None,  # User-defined figure size
    label_fontsize=12,  # Font size for title and axis labels
    tick_fontsize=10,  # Font size for tick labels and legend
    gridlines=True,
):
    """
    Plot ROC curves for models or pipelines with optional styling and grid layout.

    Parameters:
    - model: list
        List of models or pipelines to plot.
    - X: array-like
        Features for prediction.
    - y: array-like
        True labels.
    - model_titles: list of str, optional
        Titles for individual models. Required when providing a nested
        dictionary for `curve_kwgs`.
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
        Styling for individual model curves. If `model_titles` is specified as a
        list of titles, `curve_kwgs` must be a nested dictionary with model
        titles as keys and their respective style dictionaries as values.
        Otherwise, `curve_kwgs` must be a list of style dictionaries
        corresponding to the models.
    - linestyle_kwgs: dict, optional
        Styling for the random guess diagonal line.
    - grid: bool, optional
        Whether to organize plots in a grid layout (default: False).
    - n_cols: int, optional
        Number of columns in the grid layout (default: 2).
    - figsize: tuple, optional
        Custom figure size (width, height) for the plot(s).
    - label_fontsize: int, optional
        Font size for title and axis labels.
    - tick_fontsize: int, optional
        Font size for tick labels and legend.

    Raises:
    - ValueError: If `grid=True` and `overlay=True` are both set.
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
        y_true, y_prob, y_pred, threshold = get_predictions(
            model, X, y, None, None, None
        )

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        print(f"AUC for {name}: {roc_auc:.3f}")

        if overlay:
            plt.plot(
                fpr,
                tpr,
                label=f"{name} (AUC = {roc_auc:.3f})",
                **curve_style,
            )
        elif grid:
            ax = axes[idx]
            ax.plot(fpr, tpr, label=f"ROC Curve", **curve_style)
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
            ax.legend(loc="lower right", fontsize=tick_fontsize)
            ax.grid(visible=gridlines)
        else:
            plt.figure(figsize=figsize or (8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve", **curve_style)
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
            plt.legend(loc="lower right", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            save_plot_images(
                f"{name}_ROC",
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

        plt.legend(loc="lower right", fontsize=tick_fontsize)
        plt.grid()
        save_plot_images(
            "Overlay_ROC",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif grid:
        for ax in axes[len(models) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images("Grid_ROC", save_plot, image_path_png, image_path_svg)
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
    grid=False,  # Grid layout option
    n_cols=2,  # Number of columns for the grid
    figsize=None,  # User-defined figure size
    label_fontsize=12,  # Font size for title and axis labels
    tick_fontsize=10,  # Font size for tick labels and legend
    gridlines=True,
):
    """
    Plot PR curves for models or pipelines with optional styling and grid layout.

    Parameters:
    - model: list
        List of models or pipelines to plot.
    - X: array-like
        Features for prediction.
    - y: array-like
        True labels.
    - model_titles: list of str, optional
        Titles for individual models.
    - overlay: bool
        Whether to overlay multiple models on a single plot.
    - title: str, optional
        Custom title for the plot.
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
        Font size for title and axis labels.
    - tick_fontsize: int, optional
        Font size for tick labels and legend.

    Raises:
    - ValueError: If `grid=True` and `overlay=True` are both set.
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

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

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
            ax.plot(recall, precision, label=f"PR Curve", **curve_style)
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

            ax.legend(loc="lower left", fontsize=tick_fontsize)
            ax.grid(visible=gridlines)
        else:
            plt.figure(figsize=figsize or (8, 6))
            plt.plot(recall, precision, label=f"PR Curve", **curve_style)
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

            plt.legend(loc="lower left", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            save_plot_images(
                f"{name}_PR",
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

        plt.legend(loc="lower left", fontsize=tick_fontsize)
        plt.grid()
        save_plot_images(
            "Overlay_PR",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()

    elif grid:
        for ax in axes[len(models) :]:
            ax.axis("off")
        plt.tight_layout()
        save_plot_images("Grid_PR", save_plot, image_path_png, image_path_svg)
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

            ax.legend(loc="upper right", fontsize=tick_fontsize)
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
            plt.legend(loc="upper right", fontsize=tick_fontsize)
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

        plt.legend(loc="upper right", fontsize=tick_fontsize)
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

            ax.legend(loc="lower right", fontsize=tick_fontsize)
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
            plt.legend(loc="lower right", fontsize=tick_fontsize)
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

        plt.legend(loc="lower right", fontsize=tick_fontsize)
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
    - model_titles: list of str, optional
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

    if model_titles is None:
        model_titles = extract_model_titles(model)

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

            ax.legend(loc="upper left", fontsize=tick_fontsize)
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

            plt.legend(loc="upper left", fontsize=tick_fontsize)
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

        plt.legend(loc="upper left", fontsize=tick_fontsize)
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
