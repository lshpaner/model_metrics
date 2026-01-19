import pandas as pd
import numpy as np
import math
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as mcolorbar
from matplotlib.lines import Line2D
import textwrap

from sklearn.cluster import KMeans
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    average_precision_score,
)

from model_metrics.metrics_utils import (
    save_plot_images,
    normalize_model_titles,
    get_predictions,
    extract_model_name,
    validate_and_normalize_inputs,
    hanley_mcneil_auc_test,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_leverage_and_cooks_distance,
    compute_residual_diagnostics,
    print_resid_diagnostics_table,
    check_heteroskedasticity,
    has_feature_importances,
    get_feature_importances,
    get_coef_and_intercept,
)

from model_metrics.plot_utils import (
    apply_plot_title,
    apply_legend,
    setup_subplots,
    normalize_curve_styles,
)


################################################################################
######################### Summarize Model Performance ##########################
################################################################################


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
    group_category=None,
    include_adjusted_r2=False,
):
    """
    Summarize performance metrics for classification or regression models.

    Computes and displays or returns key metrics for one or more models.
    Supports classification and regression, operating from trained model
    objects or directly from prediction/probability inputs. Optionally
    computes group-specific metrics for classification tasks.

    Parameters
    ----------
    model : estimator, list of estimators, or None, default=None
        Trained model or list of models. If None, `y_prob` or `y_pred` must be provided.
    X : array-like or None, default=None
        Feature matrix. Required if `model` is provided without predictions.
    y_prob : array-like or list, default=None
        Predicted probabilities for classification models.
    y_pred : array-like or list, default=None
        Predicted class labels or regression outputs.
    y : array-like
        True target values.
    model_type : {"classification", "regression"}, default="classification"
        Specifies which type of metrics to compute.
    model_threshold : float, dict, or None, default=None
        Classification decision threshold(s). Ignored if `custom_threshold` is set.
    custom_threshold : float or None, default=None
        Overrides `model_threshold` across all models.
    model_title : str, list, or None, default=None
        Custom model names. Defaults to "Model_1", "Model_2", etc.
    score : str or None, default=None
        Optional scoring metric for threshold tuning.
    return_df : bool, default=False
        If True, returns a pandas DataFrame; otherwise prints a summary table.
    overall_only : bool, default=False
        For regression only. If True, returns only overall metrics.
    decimal_places : int, default=3
        Number of decimal places to round displayed metrics.
    group_category : str, array-like, or None, default=None
        Optional grouping variable for classification metrics.
        Can be a column name in `X` or an array matching `y` in length.
    include_adjusted_r2 : bool, default=False
        For regression only. If True, computes and includes adjusted R-squared score.
        Requires both model and X to be provided for proper calculation.

    Returns
    -------
    pandas.DataFrame or None
        If `return_df=True`, returns metrics as a DataFrame:
        - Classification (no groups): metrics as rows, models as columns.
        - Classification (grouped): metrics as rows, groups as columns.
        - Regression: rows for metrics, coefficients, and/or feature importances.
        If `return_df=False`, prints a formatted summary table.

    Raises
    ------
    ValueError
        - If `model_type` is not "classification" or "regression".
        - If `overall_only=True` is used with classification models.
        - If neither (`model` and `X`) nor (`y_prob` or `y_pred`) are provided.
    """

    # --- Input validation ---
    if not (
        (model is not None and X is not None)
        or y_prob is not None
        or y_pred is not None
    ):
        raise ValueError("You need to provide model and X or y_pred")

    if model is not None and not isinstance(model, list):
        model = [model]

    if isinstance(y_prob, np.ndarray):
        y_prob = [y_prob]

    model_type = model_type.lower()
    if model_type not in ["classification", "regression"]:
        raise ValueError("model_type must be 'classification' or 'regression'")

    if model_type == "classification" and overall_only:
        raise ValueError("'overall_only' only applies to regression models")

    models = model if isinstance(model, list) else [model]
    metrics_data = []

    # --- Normalize model titles ---
    model_title = normalize_model_titles(
        model_title, len(models), format_template="Model_{i}"
    )

    # --- Main loop ---
    for i, m in enumerate(models):
        name = model_title[i]

        if model_type == "classification":
            if X is None:
                y_true = y
                y_prob_m = y_prob[i]
                threshold = custom_threshold or (
                    model_threshold[name]
                    if (
                        model_threshold
                        and isinstance(model_threshold, dict)
                        and name in model_threshold
                    )
                    else 0.5
                )
            else:
                y_true, y_prob_m, _, threshold = get_predictions(
                    m, X, y, model_threshold, custom_threshold, score
                )
            y_pred_m = (np.asarray(y_prob_m) > float(threshold)).astype(int)

            # overall row
            overall_row = {"Model": name, "Group": "Overall"}
            overall_row.update(
                compute_classification_metrics(
                    y_true,
                    y_pred_m,
                    y_prob_m,
                    threshold,
                )
            )
            metrics_data.append(overall_row)

            # group rows
            if group_category is not None:
                group_series = (
                    X[group_category].reset_index(drop=True)
                    if isinstance(group_category, str) and isinstance(X, pd.DataFrame)
                    else pd.Series(group_category).reset_index(drop=True)
                )
                if len(group_series) != len(y_true):
                    raise ValueError(
                        "Length mismatch between group_category and y_true"
                    )

                for g in group_series.unique():
                    idx = group_series[group_series == g].index.to_numpy()
                    y_true_g = np.asarray(y_true)[idx]
                    # must skip groups with single-class y to avoid AUC/AP errors
                    if len(np.unique(y_true_g)) < 2:
                        continue
                    y_prob_g = np.asarray(y_prob_m)[idx]
                    y_pred_g = np.asarray(y_pred_m)[idx]
                    g_row = {"Model": name, "Group": str(g)}
                    g_row.update(
                        compute_classification_metrics(
                            y_true_g, y_pred_g, y_prob_g, threshold
                        )
                    )
                    metrics_data.append(g_row)

        else:  # regression
            # Get number of features for adjusted r-squared
            n_features = None
            if include_adjusted_r2 and X is not None:
                n_features = (
                    X.shape[1]
                    if hasattr(X, "shape")
                    else len(X[0]) if len(X) > 0 else None
                )

            # predictions
            if (m is not None) and isinstance(m, sm.OLS):
                Xc = sm.add_constant(X)
                y_pred_m = m.predict(Xc)
                coef_series = pd.Series(m.params.round(decimal_places))
                coefficients = coef_series.to_dict()
            else:
                if X is None:
                    y_pred_m = y_pred[i]
                else:
                    try:
                        Xc = sm.add_constant(X)
                        y_pred_m = m.predict(Xc)
                    except Exception:
                        y_pred_m = m.predict(X)

                coef_, intercept_ = get_coef_and_intercept(m)
                if coef_ is not None:
                    feature_names = (
                        X.columns if isinstance(X, pd.DataFrame) else range(len(coef_))
                    )
                    coefficients = (
                        pd.Series(coef_, index=feature_names)
                        .round(decimal_places)
                        .to_dict()
                    )
                    if intercept_ is not None:
                        coefficients["const"] = round(float(intercept_), decimal_places)
                else:
                    coefficients = {}

            # metrics row
            y_arr = np.asarray(y).ravel()
            y_pred_arr = np.asarray(y_pred_m).ravel()
            reg_metrics = compute_regression_metrics(
                y_arr, y_pred_arr, n_features, include_adjusted_r2, decimal_places
            )

            base_row = {
                "Model": name,
                "Metric": "Overall Metrics",
                "Variable": "",
                "Coefficient": "",
                **reg_metrics,
            }
            metrics_data.append(base_row)

            if not overall_only:
                # Define empty metrics dictionary FIRST
                empty_metrics = {
                    "MAE": "",
                    "MAPE": "",
                    "MSE": "",
                    "RMSE": "",
                    "Expl. Var.": "",
                    "R^2": "",
                }
                if include_adjusted_r2:
                    empty_metrics["Adj. R^2"] = ""

                # coefficients rows (const first if present)
                if "const" in coefficients:
                    metrics_data.append(
                        {
                            "Model": name,
                            "Metric": "Coefficient",
                            "Variable": "const",
                            "Coefficient": coefficients["const"],
                            **empty_metrics,
                        }
                    )
                for var, val in coefficients.items():
                    if var == "const":
                        continue
                    metrics_data.append(
                        {
                            "Model": name,
                            "Metric": "Coefficient",
                            "Variable": var,
                            "Coefficient": val,
                            **empty_metrics,
                        }
                    )

                # feature importances (tree models)
                if has_feature_importances(m) and isinstance(X, pd.DataFrame):
                    fi = get_feature_importances(m, X.columns, decimal_places)
                    for var, val in fi.items():
                        metrics_data.append(
                            {
                                "Model": name,
                                "Metric": "Feat. Imp.",
                                "Variable": var,
                                "Coefficient": "",
                                "Feat. Imp.": val,
                                **empty_metrics,
                            }
                        )
    # --- Build DataFrame ---
    metrics_df = pd.DataFrame(metrics_data)

    # --- Handle regression column ordering and NaN feature importances ---
    if model_type == "regression":
        # Reorder columns to put Feat. Imp. after Coefficient
        desired_cols = [
            "Model",
            "Metric",
            "Variable",
            "Coefficient",
            "Feat. Imp.",
            "MAE",
            "MAPE",
            "MSE",
            "RMSE",
            "Expl. Var.",
            "R^2",
        ]
        if include_adjusted_r2:
            desired_cols.append("Adj. R^2")
        existing_cols = [col for col in desired_cols if col in metrics_df.columns]
        metrics_df = metrics_df[existing_cols]

        # Replace NaN with "-" in Feat. Imp. column if it exists
        if "Feat. Imp." in metrics_df.columns:
            metrics_df["Feat. Imp."] = metrics_df["Feat. Imp."].fillna("")

    # --- Shape classification output (ordering + grouped vs non-grouped) ---
    if model_type == "classification":
        metric_order = [
            "Precision/PPV",
            "Average Precision",
            "Sensitivity/Recall",
            "Specificity",
            "F1-Score",
            "AUC ROC",
            "Brier Score",
            "Model Threshold",
        ]
        if group_category is not None:
            # keep only one model's group table for header clarity
            if "Model" in metrics_df.columns and len(metrics_df["Model"].unique()) > 1:
                first_model = metrics_df["Model"].iloc[0]
                metrics_df = metrics_df[metrics_df["Model"] == first_model].copy()

            # drop Overall row for grouped display
            metrics_df = metrics_df[metrics_df["Group"] != "Overall"].copy()

            metrics_df = (
                metrics_df.melt(
                    id_vars=["Group"],
                    value_vars=metric_order,
                    var_name="Metrics",
                    value_name="Value",
                )
                .pivot(index="Metrics", columns="Group", values="Value")
                .reset_index()
            )
            # enforce order
            metrics_df["Metrics"] = pd.Categorical(
                metrics_df["Metrics"], categories=metric_order, ordered=True
            )
            metrics_df = metrics_df.sort_values("Metrics").reset_index(drop=True)
            metrics_df.columns.name = None
        else:
            # non-grouped: drop Group col if present and transpose (models as columns)
            if "Group" in metrics_df.columns:
                metrics_df = metrics_df.drop(columns=["Group"], errors="ignore")
            metrics_df = metrics_df.set_index("Model").T.reset_index()
            metrics_df.rename(columns={"index": "Metrics"}, inplace=True)
            metrics_df.columns.name = None
            metrics_df.index = [""] * len(metrics_df)

    # --- Regression: nothing to reshape; preserve rows (incl. Coeff/Feat. Imp.) ---
    if model_type == "regression":
        # Keep as-is. Do not wipe coefficient rows.
        pass

    # **Manual formatting**
    if not return_df:
        if model_type == "classification":
            # Handle grouped vs non-grouped separately
            if "Model" not in metrics_df.columns:
                # Grouped case — skip transposed printing, just print the table as-is
                print("Grouped Model Performance Metrics:")
                print(metrics_df.to_string(index=False))
                return

            # Non-grouped classification: safe to transpose
            print("Model Performance Metrics:")
            metrics_print = metrics_df.set_index("Model").T
            col_widths = {
                col: max(metrics_print[col].astype(str).map(len).max(), len(str(col)))
                + 2
                for col in metrics_print.columns
            }
            col_widths["Metrics"] = (
                max(metrics_print.index.astype(str).map(len).max(), len("Metrics")) + 2
            )
            separator = "-" * (sum(col_widths.values()) + len(col_widths) * 3)

            print(separator)
            header = (
                "Metrics".rjust(col_widths["Metrics"])
                + " | "
                + " | ".join(
                    f"{str(col).rjust(col_widths[col])}"
                    for col in metrics_print.columns
                )
            )
            print(header)
            print(separator)

            for metric, row_data in metrics_print.iterrows():
                row = f"{metric.rjust(col_widths['Metrics'])} | " + " | ".join(
                    f"{str(row_data[col]).rjust(col_widths[col])}"
                    for col in metrics_print.columns
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

            print("Model Performance Metrics")
            print(separator)
            print(
                " | ".join(
                    f"{col.rjust(col_widths[col])}" for col in metrics_df.columns
                ),
            )
            print(separator)

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
                prev_model = current_model if model_type == "regression" else prev_model
            print(separator)
    else:
        if model_type == "classification":
            if "Model" in metrics_df.columns:
                # non-grouped already shaped above
                metrics_df.index = [""] * len(metrics_df)
                return metrics_df
            else:
                # grouped already pivoted to (Metrics + group columns)
                metrics_df.index = [""] * len(metrics_df)
                return metrics_df
        else:
            # regression: return as-is (keep coefficient / feat. imp. rows)
            return metrics_df.reset_index(drop=True)


################################################################################
############################## Confusion Matrix ################################
################################################################################


def show_confusion_matrix(
    model=None,
    X=None,
    y=None,
    y_prob=None,
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
    show_colorbar=False,
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
    - show_colorbar (bool, default=False): Whether to show the colorbar.
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
    model_title = normalize_model_titles(model_title, len(model))

    # Setup subplots if enabled
    if subplots:
        n_cols = kwargs.get("n_cols", 2)
        n_rows = math.ceil(len(model) / n_cols)
        _, axes = setup_subplots(
            num_models=len(model),
            n_cols=n_cols,
            n_rows=n_rows,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        )
    else:
        axes = [None] * len(model)

    for idx, (m, ax) in enumerate(zip(model, axes)):
        # Determine the model name
        if model_title:
            name = model_title[idx]
        else:
            name = extract_model_name(m)  # Fallback to model class name

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
        # show_colorbar = kwargs.get("show_colorbar", True)
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

        apply_plot_title(
            title,
            default_title=f"Confusion Matrix: {name} (Threshold = {threshold:.2f})",
            text_wrap=text_wrap,
            fontsize=label_fontsize,
            ax=ax,
        )

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
            "confusion_matrix_subplots",
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
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    group_category=None,
    delong=None,
    show_operating_point=False,
    operating_point_method="youden",
    operating_point_kwgs=None,
    legend_loc="lower right",
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
    - delong: tuple or list of array-like, optional
        Two predicted probability arrays (e.g., [y_prob_model1, y_prob_model2]) to
        perform a Hanley & McNeil AUC comparison, a parametric approximation of
        DeLong's test for correlated ROC curves. The test compares two models
        evaluated on the same samples to determine whether the difference in AUC
        is statistically significant. Cannot be used when `group_category` is
        specified, since AUCs are computed on separate subsets of patients.
    - legend_loc: str, optional
        Location for the legend. Standard matplotlib locations like 'lower right',
        'upper left', etc., or 'bottom' to place legend below the plot
        (default: 'lower right').

    Raises:
        ..,kj- ValueError: If `subplots=True` and `overlay=True` are both set, if
          `subplots=True` and `group_category` is provided, if `overlay=True`
          and `group_category` is provided, or if `overlay=True` and only one
          model is provided.
        - ValueError: If `delong` is provided while `group_category` is specified,
          since AUCs from different groups cannot be compared using this test.
    """

    # Validate and normalize inputs
    model, y_probs, _ = validate_and_normalize_inputs(model, X, y_prob)

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

    if delong is not None and group_category is not None:
        raise ValueError(
            f"Cannot run DeLong's (Hanley & McNeil comparison) when `group_category` "
            f"is specified, because AUCs are computed on separate subsets of patients."
        )

    # Normalize model_title input
    model_title = normalize_model_titles(model_title, len(model))

    # Normalize curve_kwgs input
    curve_styles = normalize_curve_styles(curve_kwgs, model_title, len(model))

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        _, axes = setup_subplots(
            num_models=len(model), n_cols=n_cols, n_rows=n_rows, figsize=figsize
        )

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
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            # Format AUC with decimal_places for print and legend
            auc_str = f"{roc_auc:.{decimal_places}f}"

            if show_operating_point:
                if operating_point_method == "youden":
                    scores = tpr - fpr
                    idx_opt = np.argmax(scores)
                elif operating_point_method == "closest_topleft":
                    idx_opt = np.argmin((1 - tpr) ** 2 + fpr**2)
                else:
                    raise ValueError(
                        "operating_point_method must be 'youden' or 'closest_topleft'"
                    )

                op_fpr = fpr[idx_opt]
                op_tpr = tpr[idx_opt]
                op_thresh = thresholds[idx_opt]

                # Format operating point label with both threshold and coordinates
                op_label = f"Op: {op_thresh:.{decimal_places}f} at ({op_fpr:.{decimal_places}f}, {op_tpr:.{decimal_places}f})"

        print(f"AUC for {name}: {roc_auc:.{decimal_places}f}")

        # Optional: Hanley & McNeil AUC comparison if two probability arrays are provided
        if delong is not None and idx == 0:
            try:
                if not isinstance(delong, (tuple, list)) or len(delong) != 2:
                    raise ValueError(
                        "`delong` must be a tuple or list containing two y_prob arrays."
                    )

                y_prob_1, y_prob_2 = delong

                # Resolve model names if available
                if model_title is not None and len(model_title) >= 2:
                    name1, name2 = model_title[0], model_title[1]
                else:
                    name1, name2 = "Model 1", "Model 2"

                # Call the helper with verbose=False to avoid duplicate prints
                hanley_mcneil_auc_test(
                    y_true,
                    y_prob_1,
                    y_prob_2,
                    model_names=(name1, name2),
                    return_values=True,
                    verbose=True,  # prevents double printing
                )

            except Exception as e:
                print(f"Error running Hanley & McNeil AUC comparison: {e}")

        if overlay:
            (line,) = plt.plot(
                fpr,
                tpr,
                **curve_style,
            )

            if show_operating_point:
                point_kwgs = operating_point_kwgs or {
                    "s": 80,
                    "marker": "o",
                    "facecolor": "red",
                    "edgecolor": "black",
                }

                scatter = plt.scatter(
                    op_fpr,
                    op_tpr,
                    zorder=10,
                    **point_kwgs,
                )

                # Create combined legend entry with line and marker
                combined_handle = (
                    line,
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="red",
                        markeredgecolor="black",
                        markersize=8,
                        linestyle="None",
                    ),
                )
                line.set_label(f"{name} (AUC = {auc_str}, {op_label})")
            else:
                line.set_label(f"{name} (AUC = {auc_str})")

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
                if show_operating_point:
                    point_kwgs = operating_point_kwgs or {
                        "s": 80,
                        "marker": "o",
                        "facecolor": "white",
                        "edgecolor": "black",
                    }

                    ax.scatter(
                        op_fpr,
                        op_tpr,
                        label=op_label,
                        zorder=10,
                        **point_kwgs,
                    )

            ax.plot([0, 1], [0, 1], label="Random Guess", **linestyle_kwgs)
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            # Set title per subplot
            apply_plot_title(
                title,
                default_title=f"ROC Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
                ax=ax,
            )
            if group_category is not None:
                # Add legend below plot for group_category
                apply_legend("bottom", fontsize=tick_fontsize, ax=ax, ncol=1)
            else:
                # Get handles and labels for ordering
                handles, labels = ax.get_legend_handles_labels()

                # Order: AUC curves, then Random Guess, then Operating Points
                ordered_labels = []
                for l in labels:
                    if "AUC" in l:
                        ordered_labels.append(l)
                for l in labels:
                    if "Random Guess" in l:
                        ordered_labels.append(l)
                for l in labels:
                    if "Op:" in l:
                        ordered_labels.append(l)

                ordered_handles = [handles[labels.index(l)] for l in ordered_labels]
                # Apply ordered legend
                apply_legend(
                    legend_loc,
                    fontsize=tick_fontsize,
                    ax=ax,
                    handles=ordered_handles,
                    labels=ordered_labels,
                )
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
                # If you want operating point on single-plot too, it MUST be here (or outside both blocks)
                if show_operating_point:
                    point_kwgs = operating_point_kwgs or {
                        "s": 80,
                        "marker": "o",
                        "facecolor": "white",
                        "edgecolor": "black",
                    }
                    plt.scatter(
                        op_fpr,
                        op_tpr,
                        label=op_label,
                        zorder=10,
                        **point_kwgs,
                    )
            plt.plot([0, 1], [0, 1], label="Random Guess", **linestyle_kwgs)
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            # Set title for single plot
            apply_plot_title(
                title,
                default_title=f"ROC Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
            )

            if group_category is not None:
                # Add legend below plot for group_category
                apply_legend("bottom", fontsize=tick_fontsize, ncol=1)
            else:
                handles, labels = plt.gca().get_legend_handles_labels()

                # Order: AUC curves, then Random Guess, then Operating Points
                ordered_labels = []
                for l in labels:
                    if "AUC" in l:
                        ordered_labels.append(l)
                for l in labels:
                    if "Random Guess" in l:
                        ordered_labels.append(l)
                for l in labels:
                    if "Op:" in l:
                        ordered_labels.append(l)

                ordered_handles = [handles[labels.index(l)] for l in ordered_labels]
                # Apply ordered legend
                apply_legend(
                    legend_loc,
                    fontsize=tick_fontsize,
                    handles=ordered_handles,
                    labels=ordered_labels,
                )
            plt.grid(visible=gridlines)
            # Set title for single plot
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
        # Set title for overlay plot
        apply_plot_title(
            title,
            default_title="ROC Curves: Overlay",  # Generic for overlay
            text_wrap=text_wrap,
            fontsize=label_fontsize,
        )

        if group_category is not None:
            # Add legend below plot for group_category
            apply_legend("bottom", fontsize=tick_fontsize, ncol=1)
        else:
            handles, labels = plt.gca().get_legend_handles_labels()

            # Order: AUC curves, then Random Guess, then Operating Points
            ordered_labels = []
            for l in labels:
                if "AUC" in l:
                    ordered_labels.append(l)
            for l in labels:
                if "Random Guess" in l:
                    ordered_labels.append(l)
            for l in labels:
                if "Op:" in l:
                    ordered_labels.append(l)

            ordered_handles = [handles[labels.index(l)] for l in ordered_labels]
            # Apply ordered legend
            apply_legend(
                legend_loc,
                fontsize=tick_fontsize,
                handles=ordered_handles,
                labels=ordered_labels,
            )
        plt.grid(visible=gridlines)
        save_plot_images(
            "roc_auc_overlay_plot",
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
            "roc_auc_subplots",
            save_plot,
            image_path_png,
            image_path_svg,
        )
        plt.show()


def show_pr_curve(
    model=None,
    X=None,
    y=None,
    y_prob=None,
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
    n_cols=2,
    n_rows=None,
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    group_category=None,
    legend_metric="ap",
    legend_loc="lower left",
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
    - legend_loc: str, optional
        Location for the legend. Standard matplotlib locations like 'lower left',
        'upper right', etc., or 'bottom' to place legend below the plot
        (default: 'lower left').

    Raises:
    - ValueError:
        - If `subplots=True` and `overlay=True` are both set.
        - If `group_category` is used with `subplots=True` or `overlay=True`.
        - If `legend_metric` is not one of {"ap", "aucpr"}.
    - TypeError:
        - If `model_title` is not a string, list of strings, or None.
    """

    # Validate and normalize inputs
    model, y_probs, _ = validate_and_normalize_inputs(model, X, y_prob)

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

    # Normalize model_title input
    model_title = normalize_model_titles(model_title, len(model))

    # Normalize curve_kwgs input
    curve_styles = normalize_curve_styles(curve_kwgs, model_title, len(model))

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        _, axes = setup_subplots(len(model), n_cols, n_rows, figsize)

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
            # Set title per subplot
            apply_plot_title(
                title,
                default_title=f"PR Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
                ax=ax,
            )

            if group_category is not None:
                # Add legend below plot for group_category
                apply_legend("bottom", fontsize=tick_fontsize, ax=ax, ncol=1)
            else:
                apply_legend(legend_loc, fontsize=tick_fontsize, ax=ax)
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
            # Set title for single plot
            apply_plot_title(
                title,
                default_title=f"PR Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
            )

            if group_category is not None:
                # Add legend below plot for group_category
                apply_legend("bottom", fontsize=tick_fontsize, ncol=1)

            else:
                apply_legend(legend_loc, fontsize=tick_fontsize)
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
        # Set title for overlay plot
        apply_plot_title(
            title,
            default_title="PR Curves: Overlay",
            text_wrap=text_wrap,
            fontsize=label_fontsize,
        )

        if group_category is not None:
            # Add legend below plot for group_category
            apply_legend("bottom", fontsize=tick_fontsize, ncol=1)
        else:
            apply_legend(legend_loc, fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "pr_overlay_plot",
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
            "pr_subplots",
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
    y=None,
    y_prob=None,
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
    legend_loc="best",
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
    - y (array-like): True labels.
    - y_prob (array-like or list of array-like, optional): Predicted probabilities.
      Can be provided instead of model and X.
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
    - legend_loc (str, optional): Location for the legend. Standard matplotlib
      locations like 'best', 'upper right', 'lower left', etc., or 'bottom' to
      place legend below the plot (default: 'best').

    Raises:
    - ValueError: If `subplots=True` and `overlay=True` are both set.

    Returns:
    - None
    """

    # Validate and normalize inputs
    model, y_probs, _ = validate_and_normalize_inputs(model, X, y_prob)

    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    # Normalize model_title input
    model_title = normalize_model_titles(model_title, len(model))

    # Normalize curve styles
    curve_styles = normalize_curve_styles(curve_kwgs, model_title, len(model))

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        _, axes = setup_subplots(len(model), n_cols, n_rows, figsize)

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
            # Set title per subplot
            apply_plot_title(
                title,
                default_title=f"Lift Chart: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
                ax=ax,
            )
            # Add legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize, ax=ax)
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
            # Set title for single plot
            apply_plot_title(
                title,
                default_title=f"Lift Chart: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
            )
            # Add legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize)
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
        # Set title for overlay plot
        apply_plot_title(
            title,
            default_title="Lift Charts: Overlay",
            text_wrap=text_wrap,
            fontsize=label_fontsize,
        )
        # Add legend below plot for group_category
        apply_legend(legend_loc, fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "lift_overlay",
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
            "lift_subplots",
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
    legend_loc="best",
    show_gini=False,
    decimal_places=3,
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
    - legend_loc (str, optional): Location for the legend. Standard matplotlib
      locations like 'best', 'upper right', 'lower left', etc., or 'bottom' to
      place legend below the plot (default: 'best').
    - show_gini (bool, default=True): Whether to show Gini coefficient in the legend.
    - decimal_places (int, default=3): Number of decimal places for Gini coefficient.

    Raises:
    - ValueError: If `subplots=True` and `overlay=True` are both set.

    Returns:
    - None
    """

    # Validate and normalize inputs
    model, y_probs, _ = validate_and_normalize_inputs(model, X, y_prob)

    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    # Normalize model titles
    model_title = normalize_model_titles(model_title, len(model))

    # Normalize curve styles
    curve_styles = normalize_curve_styles(curve_kwgs, model_title, len(model))

    linestyle_kwgs = linestyle_kwgs or {
        "color": "gray",
        "linestyle": "--",
        "linewidth": 2,
    }

    if overlay:
        plt.figure(figsize=figsize or (8, 6))

    if subplots and not overlay:
        _, axes = setup_subplots(len(model), n_cols, n_rows, figsize)

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

        # Calculate Gini coefficient
        augc = auc(percentages, cumulative_gains)
        gini = 2 * augc - 1

        # Print Gini coefficient only if show_gini is True
        if show_gini:
            print(f"Gini coefficient for {name}: {gini:.{decimal_places}f}")

        # Create label with optional Gini
        if show_gini:
            model_label = f"{name} (Gini = {gini:.{decimal_places}f})"
        else:
            model_label = f"{name}"

        if overlay:
            plt.plot(
                percentages,
                cumulative_gains,
                label=model_label,
                **curve_style,
            )
        elif subplots:
            ax = axes[idx]

            # For subplots, use simpler label or full label based on preference
            if show_gini:
                subplot_label = f"Gini = {gini:.{decimal_places}f}"
            else:
                subplot_label = "Gain Curve"

            ax.plot(
                percentages,
                cumulative_gains,
                label=subplot_label,
                **curve_style,
            )
            ax.plot([0, 1], [0, 1], label="Baseline", **linestyle_kwgs)
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            # Set title per subplot
            apply_plot_title(
                title,
                default_title=f"Gain Chart: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
                ax=ax,
            )
            # Add legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize, ax=ax)
            ax.grid(visible=gridlines)
        else:
            plt.figure(figsize=figsize or (8, 6))

            # For single plots, show Gini in legend
            if show_gini:
                single_label = f"Gini = {gini:.{decimal_places}f}"
            else:
                single_label = "Gain Curve"

            plt.plot(
                percentages,
                cumulative_gains,
                label=single_label,  # ← Use the conditional label
                **curve_style,
            )
            plt.plot([0, 1], [0, 1], label="Baseline", **linestyle_kwgs)
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            # Set title for single plot
            apply_plot_title(
                title,
                default_title=f"Gain Chart: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
            )
            # Add legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize)
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
        # Set title for overlay plot
        apply_plot_title(
            title,
            default_title="Gain Charts: Overlay",
            text_wrap=text_wrap,
            fontsize=label_fontsize,
        )
        # Add legend below plot for group_category
        apply_legend(legend_loc, fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "gain_overlay",
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
            "gain_subplots",
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
    brier_decimals=3,
    gridlines=True,
    linestyle_kwgs=None,
    group_category=None,
    legend_loc="best",
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
    - brier_decimals (int, default=3): Number of decimal places to display for
      the Brier score values in the legend.
    - gridlines (bool, default=True): Whether to display gridlines on the plot.
    - linestyle_kwgs (dict, optional): Styling options for the diagonal
      "perfectly calibrated" line.
    - group_category (array-like, optional): A categorical series to plot
      subgroup calibration curves.
    - legend_loc (str, optional): Location for the legend. Standard matplotlib
      locations like 'best', 'upper right', 'lower left', etc., or 'bottom' to
      place legend below the plot (default: 'best').

    Raises:
    - ValueError: If both `subplots=True` and `overlay=True` are set (incompatible).
    - ValueError: If `group_category` used w/ either `overlay=True` or `subplots=True`.
    - ValueError: If length of `curve_kwgs` does not match number of models.
    - TypeError: If `model_title` is not a string, list, pandas Series, or None.

    Returns:
    - None
    """

    # Validate and normalize inputs
    model, y_probs, _ = validate_and_normalize_inputs(model, X, y_prob)

    # Error checks for incompatible display modes
    if overlay and subplots:
        raise ValueError("`subplots` cannot be set to True when `overlay` is True.")

    if group_category is not None and (overlay or subplots):
        raise ValueError(
            "`group_category` requires `overlay=False` and `subplots=False`."
        )

    # Normalize model titles
    model_title = normalize_model_titles(model_title, len(model))

    # Handle style settings for each model
    curve_styles = normalize_curve_styles(curve_kwgs, model_title, len(model))

    # Subplot layout setup if requested
    if subplots:
        _, axes = setup_subplots(len(model), n_cols, n_rows, figsize)

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
                    legend_label += f" (Brier: {brier:.{brier_decimals}f})"

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
            # Set title for single grouped plot
            apply_plot_title(
                title,
                default_title=f"Calibration Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
            )
            # Add legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize)
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
            legend_label += f" (Brier: {brier_score:.{brier_decimals}f})"

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
            # Set title per subplot
            apply_plot_title(
                title,
                default_title=f"Calibration Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
                ax=ax,
            )
            # apply_legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize, ax=ax)
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
            # Set title for single plot
            apply_plot_title(
                title,
                default_title=f"Calibration Curve: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize,
            )
            # Add legend below plot for group_category
            apply_legend(legend_loc, fontsize=tick_fontsize)
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
        # Set title for overlay plot
        apply_plot_title(
            title,
            default_title="Calibration Curves: Overlay",
            text_wrap=text_wrap,
            fontsize=label_fontsize,
        )
        # Add legend below plot for group_category
        apply_legend(legend_loc, fontsize=tick_fontsize)
        plt.grid(visible=gridlines)
        save_plot_images(
            "calibration_overlay",
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
            "calibration_subplots",
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


################################################################################
# Regression Residuals Plotting
################################################################################


def show_residual_diagnostics(
    model=None,
    X=None,
    y=None,
    y_pred=None,
    model_title=None,
    plot_type="all",
    figsize=None,
    label_fontsize=12,
    tick_fontsize=10,
    gridlines=True,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    show_outliers=False,
    n_outliers=3,
    suptitle=None,
    suptitle_y=0.995,
    text_wrap=None,
    point_kwgs=None,
    line_kwgs=None,
    show_lowess=False,
    lowess_kwgs=None,
    group_category=None,
    show_centroids=False,
    n_clusters=None,
    centroid_kwgs=None,
    legend_loc="best",
    n_cols=None,
    n_rows=None,
    heteroskedasticity_test=None,
    decimal_places=4,
    show_plots=True,
    show_diagnostics_table=False,
    return_diagnostics=False,
    histogram_type="frequency",
    kmeans_rstate=42,
):
    """
    Plot diagnostic residual plots for regression models.

    Creates comprehensive residual diagnostic plots to validate regression model
    assumptions including linearity, normality, homoscedasticity, and to identify
    influential observations.

    Parameters
    ----------
    model : estimator or list of estimators, optional
        Trained regression model(s). If None, y_pred must be provided.
    X : array-like, optional
        Feature matrix. Required if model is provided.
    y_pred : array-like or list, optional
        Predicted values. Can be provided instead of model and X.
    y : array-like
        True target values.
    model_title : str or list of str, optional
        Custom name(s) for model(s). Defaults to "Model 1", "Model 2", etc.
    plot_type : str or list, default="all"
        Which diagnostic plot(s) to display. Options:
        - "all": All diagnostic plots in a 2x3 grid
        - "fitted": Residuals vs Fitted Values
        - "qq": Q-Q plot for normality
        - "scale_location": Scale-Location plot (sqrt standardized residuals)
        - "leverage": Residuals vs Leverage (Cook's distance)
        - "histogram": Histogram of residuals
        - "predictors": Residuals vs each predictor (creates multiple plots)
        Can pass a list like ["fitted", "qq"] for specific plots.
    figsize : tuple, optional
        Figure size (width, height). Default varies by plot_type:
        - "all": (15, 10)
        - single plot: (8, 6)
    label_fontsize : int, default=12
        Font size for axis labels and titles.
    tick_fontsize : int, default=10
        Font size for tick labels.
    gridlines : bool, default=True
        Whether to display grid lines.
    save_plot : bool, default=False
        Whether to save the plot(s) to disk.
    image_path_png : str, optional
        Path to save PNG image.
    image_path_svg : str, optional
        Path to save SVG image.
    show_outliers : bool, default=False
        Whether to label outlier points on plots.
    n_outliers : int, default=3
        Number of most extreme outliers to label.
    suptitle : str, optional
        Custom title for the overall figure. If None, uses default
        "Residual Diagnostics: {model_name}"; if "", no suptitle is displayed.
    suptitle_y : float, default=0.995
        Vertical position of the figure suptitle (0-1 range). Adjust when using
        custom figsize to prevent title overlap with subplots.
    text_wrap : int, optional
        Maximum width for wrapping titles.
    point_kwgs : dict, optional
        Styling for scatter points (e.g., {'alpha': 0.6, 'color': 'blue'}).
    line_kwgs : dict, optional
        Styling for reference lines (e.g., {'color': 'red', 'linestyle': '--'}).
    show_lowess : bool, default=True
        Whether to show the lowess smoothing trend line on residual plots.
    lowess_kwgs : dict, optional
        Styling for lowess smoothing line
        (e.g., {'color': 'blue', 'linewidth': 2, 'label': 'Trend'}).
    group_category : str or array-like, optional
        Categorical variable for grouping observations. Can be a column name
        in X or an array matching y in length.
    show_centroids : bool, default=False
        Whether to plot centroids for each group defined by group_category.
        Only applies when group_category is provided.
    centroid_kwgs : dict, optional
        Styling for centroid markers (e.g., {'marker': 'X', 's': 50, 'color': 'red'}).
    legend_loc : str, default="best"
        Location for the legend. Standard matplotlib locations like 'best',
        'upper right', 'lower left', etc. (default: 'best').
    n_cols : int, optional
        Number of columns for predictor plots layout. If None, uses automatic
        layout (1-3 predictors: n_cols=n_predictors, 4+: n_cols=3).
    n_rows : int, optional
        Number of rows for predictor plots layout. If None, automatically
        calculated based on n_predictors and n_cols.
    heteroskedasticity_test : str, optional
        Test for heteroskedasticity. Options: "breusch_pagan", "white",
        "goldfeld_quandt", "spearman", "all", or None (default: None, no test).
        If specified, test results will be displayed on relevant plots.
    decimal_places : int, default=4
        Number of decimal places for all numeric values in diagnostics table.
    show_plots : bool, default=True
        Whether to display diagnostic plots.
    show_diagnostics_table : bool, default=False
        Whether to print a formatted table of diagnostic statistics.
    return_diagnostics : bool, default=False
        If True, return a dictionary containing diagnostic statistics for
        programmatic access.
    histogram_type : str, default="frequency"
        Type of histogram to display. Options:
        - "frequency": Raw counts without overlay (simple, intuitive)
        - "density": Probability density scale with normal distribution overlay
          (better for assessing normality)
    kmeans_rstate : int, default=42
        Random state for reproducibility in clustering (when n_clusters is used).

    Returns
    -------
    None
        Displays the diagnostic plots.

    Raises
    ------
    ValueError
        If neither (model and X) nor y_pred is provided.
        If plot_type is not recognized.
        If both group_category and custom n_clusters are specified.
        If heteroskedasticity_test is not a valid test type.
        If histogram_type is not 'frequency' or 'density'.
    """

    # Validate and normalize inputs
    model, y_pred, num_models = validate_and_normalize_inputs(model, X, y_pred)

    # Validate centroid parameters
    if group_category is not None and n_clusters is not None:
        raise ValueError(
            "Cannot specify both `group_category` and `n_clusters`. "
            "Use `group_category` for user-defined groups OR `n_clusters` for "
            "automatic clustering."
        )

    # Validate heteroskedasticity_test parameter
    if heteroskedasticity_test is not None:
        valid_het_tests = [
            "breusch_pagan",
            "white",
            "goldfeld_quandt",
            "spearman",
            "all",
        ]
        if heteroskedasticity_test not in valid_het_tests:
            raise ValueError(
                f"heteroskedasticity_test must be one of {valid_het_tests} or None, "
                f"got '{heteroskedasticity_test}'"
            )

    if histogram_type not in ["frequency", "density"]:
        raise ValueError(
            f"histogram_type must be either 'frequency' or 'density', "
            f"got '{histogram_type}'"
        )

    # Normalize model titles
    model_title = normalize_model_titles(model_title, num_models)

    # Validate plot_type
    valid_plot_types = [
        "all",
        "fitted",
        "qq",
        "scale_location",
        "leverage",
        "influence",
        "histogram",
        "predictors",
    ]
    if isinstance(plot_type, str):
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"plot_type must be one of {valid_plot_types}, got '{plot_type}'"
            )
        plot_types = [plot_type]
    else:
        plot_types = plot_type
        for pt in plot_types:
            if pt not in valid_plot_types:
                raise ValueError(
                    f"Invalid plot_type '{pt}'. Must be one of {valid_plot_types}"
                )

    # Set default styling
    point_kwgs = point_kwgs or {"alpha": 0.6, "s": 50}
    line_kwgs = line_kwgs or {"color": "red", "linestyle": "--", "linewidth": 2}
    lowess_kwgs = lowess_kwgs or {"color": "blue", "linewidth": 2}
    centroid_kwgs = centroid_kwgs or {  # ADD THIS
        "marker": "X",
        "s": 50,
        "edgecolors": "black",
        "linewidths": 2,
        "zorder": 10,
    }

    # Process each model
    for idx, (mod, name) in enumerate(zip(model, model_title)):
        # Get predictions
        if y_pred is None:
            y_pred_m = mod.predict(X)
        else:
            y_pred_m = y_pred[idx]

        # Calculate residuals
        y_true = np.asarray(y).ravel()
        y_pred_arr = np.asarray(y_pred_m).ravel()
        residuals = y_true - y_pred_arr

        # Standardized residuals (using sqrt of MSE for proper scaling)
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        standardized_residuals = residuals / rmse

        # Calculate leverage and Cook's distance
        leverage, cooks_d, n, p = None, None, None, None
        if X is not None:  # ← Remove the plot_types check
            leverage, cooks_d, n, p = compute_leverage_and_cooks_distance(
                X, standardized_residuals
            )

        # Run heteroskedasticity tests if requested
        het_results = None
        if heteroskedasticity_test:
            het_results = check_heteroskedasticity(
                residuals,
                X=X,
                y_pred=y_pred_arr,
                test_type=heteroskedasticity_test,
                decimals=decimal_places,
            )

        # Calculate number of features
        n_features = None
        if X is not None:
            if hasattr(X, "shape"):
                n_features = X.shape[1]
            elif isinstance(X, pd.DataFrame):
                n_features = len(X.columns)
            elif isinstance(X, list):
                n_features = len(X[0]) if len(X) > 0 else None

        # Compute comprehensive diagnostics
        diagnostics = compute_residual_diagnostics(
            residuals=residuals,
            y_true=y_true,
            y_pred=y_pred_arr,
            leverage=leverage,
            cooks_d=cooks_d,
            n_features=n_features,
            model_name=name,
            decimal_places=decimal_places,
        )

        # Add heteroskedasticity test results
        if het_results:
            diagnostics["heteroskedasticity_tests"] = het_results
        # Show table if requested
        if show_diagnostics_table:
            print_resid_diagnostics_table(diagnostics, decimals=decimal_places)

        # Only do plotting if show_plots is True
        if not show_plots:
            if return_diagnostics:
                return diagnostics
            continue  # Skip to next model

        # Determine subplot layout
        if "all" in plot_types:
            plots_to_make = [
                "fitted",
                "qq",
                "scale_location",
                "leverage",
                "influence",
                "histogram",
            ]
            n_plots = len(plots_to_make)

            # Use user-specified layout or default to 2x3
            if n_rows is None and n_cols is None:
                rows, cols = 2, 3  # Default layout for "all"
            elif n_rows is None:
                rows = math.ceil(n_plots / n_cols)
                cols = n_cols
            elif n_cols is None:
                cols = math.ceil(n_plots / n_rows)
                rows = n_rows
            else:
                rows, cols = n_rows, n_cols

            fig, axes = setup_subplots(
                num_models=n_plots,
                n_cols=cols,
                n_rows=rows,
                figsize=figsize or (cols * 5, rows * 5),
            )

        elif "predictors" in plot_types:
            # For predictors, skip main grid - will create separate figure later
            plots_to_make = []  # Don't process any plots in main loop
        else:
            n_plots = len(plot_types)
            cols = 2 if n_plots > 1 else 1
            rows = math.ceil(n_plots / cols) if n_plots > 1 else 1

            # Determine default figsize based on number of plots
            if figsize is None:
                if n_plots > 1:
                    default_figsize = (12, 6 * rows)
                else:
                    default_figsize = (8, 6)
            else:
                default_figsize = figsize

            fig, axes = setup_subplots(
                num_models=n_plots,
                n_cols=cols,
                n_rows=rows,
                figsize=default_figsize,
            )
            plots_to_make = plot_types

        plot_idx = 0

        # 1. Residuals vs Fitted
        if "fitted" in plots_to_make:
            ax = axes[plot_idx]
            ax.scatter(y_pred_arr, residuals, **point_kwgs)
            ax.axhline(y=0, **line_kwgs)

            # Add lowess smooth line
            if show_lowess:
                try:
                    smoothed = lowess(residuals, y_pred_arr, frac=0.3)
                    ax.plot(smoothed[:, 0], smoothed[:, 1], **lowess_kwgs)
                except:
                    pass

            # Plot centroids if requested
            if show_centroids:
                if group_category is not None:
                    # Option 1: User-provided groups
                    if isinstance(group_category, str) and isinstance(X, pd.DataFrame):
                        groups = X[group_category]
                    else:
                        groups = pd.Series(group_category)

                    unique_groups = groups.unique()
                    colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))

                    for idx, group_val in enumerate(unique_groups):
                        mask = groups == group_val
                        centroid_x = y_pred_arr[mask].mean()
                        centroid_y = residuals[mask].mean()
                        plot_kwgs = centroid_kwgs.copy()
                        plot_kwgs["label"] = f"{group_val} (n={mask.sum()})"

                        # Handle color assignment
                        if "c" in plot_kwgs:
                            if isinstance(plot_kwgs["c"], list):
                                plot_kwgs["c"] = (
                                    plot_kwgs["c"][idx]
                                    if idx < len(plot_kwgs["c"])
                                    else colors[idx]
                                )
                        elif "color" not in plot_kwgs:
                            plot_kwgs["color"] = colors[idx]

                        ax.scatter(centroid_x, centroid_y, **plot_kwgs)

                else:
                    # Option 2: Automatic K-means clustering (when no group_category provided)
                    # Set default if not provided
                    clusters = n_clusters if n_clusters is not None else 3

                    # Stack fitted values and residuals for clustering
                    data = np.column_stack([y_pred_arr, residuals])
                    kmeans = KMeans(n_clusters=clusters, random_state=kmeans_rstate)
                    cluster_labels = kmeans.fit_predict(data)

                    colors = cm.tab10(np.linspace(0, 1, clusters))

                    # Plot centroids for each cluster
                    for cluster_id in range(clusters):
                        mask = cluster_labels == cluster_id
                        centroid_x = y_pred_arr[mask].mean()
                        centroid_y = residuals[mask].mean()
                        plot_kwgs = centroid_kwgs.copy()
                        plot_kwgs["label"] = f"Cluster {cluster_id+1} (n={mask.sum()})"

                        # Handle color assignment
                        if "c" in plot_kwgs:
                            if isinstance(plot_kwgs["c"], list):
                                plot_kwgs["c"] = (
                                    plot_kwgs["c"][cluster_id]
                                    if cluster_id < len(plot_kwgs["c"])
                                    else colors[cluster_id]
                                )
                        elif "color" not in plot_kwgs:
                            plot_kwgs["color"] = colors[cluster_id]

                        ax.scatter(centroid_x, centroid_y, **plot_kwgs)

                # Only add legend if there are labeled artists
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    apply_legend(legend_loc, fontsize=tick_fontsize - 2, ax=ax)

            # Label outliers
            if show_outliers:
                outlier_indices = np.argsort(np.abs(residuals))[-n_outliers:]
                for i in outlier_indices:
                    ax.annotate(
                        str(i),
                        (y_pred_arr[i], residuals[i]),
                        fontsize=tick_fontsize - 2,
                        alpha=0.7,
                    )

            # These should always run
            ax.set_xlabel("Fitted Values", fontsize=label_fontsize)
            ax.set_ylabel("Residuals", fontsize=label_fontsize)
            apply_plot_title(
                None, "Residuals vs Fitted", fontsize=label_fontsize, ax=ax
            )
            ax.grid(visible=gridlines, alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            plot_idx += 1

        # 2. Q-Q Plot
        if "qq" in plots_to_make:
            ax = axes[plot_idx]
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor(point_kwgs.get("color", "blue"))
            ax.get_lines()[0].set_alpha(point_kwgs.get("alpha", 0.6))
            ax.get_lines()[0].set_markersize(6)
            ax.get_lines()[1].set_color("red")
            ax.get_lines()[1].set_linewidth(2)
            apply_plot_title(None, "Normal Q-Q Plot", fontsize=label_fontsize, ax=ax)
            ax.set_xlabel("Theoretical Quantiles", fontsize=label_fontsize)
            ax.set_ylabel("Sample Quantiles", fontsize=label_fontsize)
            ax.grid(visible=gridlines, alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            plot_idx += 1

        # 3. Scale-Location Plot
        if "scale_location" in plots_to_make:
            ax = axes[plot_idx]
            sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
            ax.scatter(y_pred_arr, sqrt_abs_resid, **point_kwgs)

            # Add lowess smooth line
            if show_lowess:
                try:
                    smoothed = lowess(sqrt_abs_resid, y_pred_arr, frac=0.3)
                    ax.plot(smoothed[:, 0], smoothed[:, 1], **lowess_kwgs)
                except:
                    pass

            # Test for heteroskedasticity
            if heteroskedasticity_test:
                het_results = check_heteroskedasticity(
                    residuals,
                    X=X,
                    y_pred=y_pred_arr,
                    test_type=heteroskedasticity_test,
                    decimals=decimal_places,
                )

                # Add test results to legend instead of text annotation
                for _, result in het_results.items():
                    if "error" not in result:
                        label = result["interpretation"]
                        ax.plot(
                            [],
                            [],
                            linestyle="None",
                            marker="None",
                            label=label,
                        )

                # Apply legend with test results
                apply_legend(legend_loc, fontsize=tick_fontsize - 2, ax=ax)

            # Label outliers
            if show_outliers:
                outlier_indices = np.argsort(sqrt_abs_resid)[-n_outliers:]
                for i in outlier_indices:
                    ax.annotate(
                        str(i),
                        (y_pred_arr[i], sqrt_abs_resid[i]),
                        fontsize=tick_fontsize - 2,
                        alpha=0.7,
                    )

            ax.set_xlabel("Fitted Values", fontsize=label_fontsize)
            ax.set_ylabel("Residual Spread (Standardized)", fontsize=label_fontsize)
            apply_plot_title(
                None, "Scale-Location Plot", fontsize=label_fontsize, ax=ax
            )
            ax.grid(visible=gridlines, alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            plot_idx += 1

        # 4. Residuals vs Leverage
        if "leverage" in plots_to_make:
            ax = axes[plot_idx]
            if leverage is not None and cooks_d is not None:
                # Plot everything
                ax.scatter(leverage, standardized_residuals, **point_kwgs)
                ax.axhline(y=0, **line_kwgs)

                # Add Cook's distance contours with legend
                x_range = np.linspace(0.001, max(leverage) * 1.1, 100)
                cook_levels = [0.5, 1.0]
                colors_cook = ["orange", "red"]

                for d, color in zip(cook_levels, colors_cook):
                    y_pos = np.sqrt(d * p * (1 - x_range) / x_range)
                    y_neg = -y_pos

                    # Plot contours with labels for legend
                    ax.plot(
                        x_range,
                        y_pos,
                        "--",
                        color=color,
                        alpha=0.5,
                        linewidth=1.5,
                        label=f"Cook's d = {d}",
                    )
                    ax.plot(x_range, y_neg, "--", color=color, alpha=0.5, linewidth=1.5)

                # Label high leverage points
                if show_outliers:
                    outlier_indices = np.argsort(cooks_d)[-n_outliers:]
                    for i in outlier_indices:
                        ax.annotate(
                            str(i),
                            (leverage[i], standardized_residuals[i]),
                            fontsize=tick_fontsize - 2,
                            alpha=0.7,
                        )

                # Add legend to show Cook's distance levels
                apply_legend(legend_loc, fontsize=tick_fontsize - 2, ax=ax)

            else:
                # Show error message ONLY
                ax.text(
                    0.5,
                    0.5,
                    "Leverage calculation requires\nfeature matrix X",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=label_fontsize,
                )

            ax.set_xlabel("Leverage", fontsize=label_fontsize)
            ax.set_ylabel("Standardized Residuals", fontsize=label_fontsize)
            apply_plot_title(
                None, "Residuals vs Leverage", fontsize=label_fontsize, ax=ax
            )
            ax.grid(visible=gridlines, alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            plot_idx += 1

        # 5. Influence Plot
        if "influence" in plots_to_make:
            ax = axes[plot_idx]
            if leverage is not None and cooks_d is not None:
                # Calculate studentized residuals
                studentized_resid = standardized_residuals * np.sqrt(
                    (n - p - 1) / (n - p - standardized_residuals**2)
                )

                # Bubble sizes proportional to Cook's distance
                bubble_size = cooks_d * 5000  # Scale for visibility

                # Create bubble plot
                scatter = ax.scatter(
                    leverage,
                    studentized_resid,
                    s=bubble_size,
                    alpha=0.5,
                    edgecolors="black",
                    linewidths=0.5,
                )

                ax.axhline(y=0, **line_kwgs)

                # Reference lines for studentized residuals (±2, ±3)
                for threshold in [2, -2]:
                    ax.axhline(y=threshold, color="orange", linestyle=":", alpha=0.5)
                for threshold in [3, -3]:
                    ax.axhline(y=threshold, color="red", linestyle=":", alpha=0.5)

                # Label influential points
                if show_outliers:
                    outlier_indices = np.argsort(cooks_d)[-n_outliers:]
                    for i in outlier_indices:
                        ax.annotate(
                            str(i),
                            (leverage[i], studentized_resid[i]),
                            fontsize=tick_fontsize - 2,
                            alpha=0.7,
                        )

                ax.set_xlabel("Leverage (H Leverage)", fontsize=label_fontsize)
                ax.set_ylabel("Studentized Residuals", fontsize=label_fontsize)
                apply_plot_title(None, "Influence Plot", fontsize=label_fontsize, ax=ax)

            else:
                ax.text(
                    0.5,
                    0.5,
                    "Influence plot requires\nfeature matrix X",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=label_fontsize,
                )

            ax.grid(visible=gridlines, alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            plot_idx += 1

        # 6. Histogram
        if "histogram" in plots_to_make:
            ax = axes[plot_idx]

            if histogram_type == "density":
                # Density scale with normal distribution overlay
                ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, density=True)
                ax.axvline(x=0, **line_kwgs)

                # Add normal distribution overlay
                mu, sigma = residuals.mean(), residuals.std()
                x = np.linspace(residuals.min(), residuals.max(), 100)
                ax.plot(
                    x,
                    stats.norm.pdf(x, mu, sigma),
                    color="red",
                    linewidth=2,
                    label="Normal Distribution",
                )

                ax.set_xlabel("Residuals", fontsize=label_fontsize)
                ax.set_ylabel("Density", fontsize=label_fontsize)
                ax.legend(fontsize=tick_fontsize)

            else:  # histogram_type == "frequency"
                # Frequency histogram with scaled normal overlay
                ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
                ax.axvline(x=0, **line_kwgs)

                # Add scaled normal distribution overlay
                mu, sigma = residuals.mean(), residuals.std()
                x = np.linspace(residuals.min(), residuals.max(), 100)
                bin_width = (residuals.max() - residuals.min()) / 30
                ax.plot(
                    x,
                    stats.norm.pdf(x, mu, sigma) * len(residuals) * bin_width,
                    color="red",
                    linewidth=2,
                    label="Normal Distribution",
                )

                ax.set_xlabel("Residuals", fontsize=label_fontsize)
                ax.set_ylabel("Frequency", fontsize=label_fontsize)
                ax.legend(fontsize=tick_fontsize)

            apply_plot_title(
                None, "Histogram of Residuals", fontsize=label_fontsize, ax=ax
            )
            ax.grid(visible=gridlines, alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            plot_idx += 1

        # Hide unused subplots (only if we created a grid)
        if plots_to_make:  # Only hide if we actually made plots
            for i in range(plot_idx, len(axes)):
                axes[i].axis("off")

        # Only show main grid if we created one
        if plots_to_make:
            # Overall figure title
            apply_plot_title(
                suptitle,  # Use suptitle parameter, not title
                default_title=f"Residual Diagnostics: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize + 2,
                fig=fig,
                suptitle_y=suptitle_y,
            )
            plt.tight_layout()

            # Save plot
            if save_plot:
                name_clean = name.lower().replace(" ", "_")
                plot_type_str = (
                    "_".join(plots_to_make)
                    if len(plots_to_make) > 1
                    else plots_to_make[0]
                )
                save_plot_images(
                    f"{name_clean}_residuals_{plot_type_str}",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )

            plt.show()

        # If "predictors" was specifically requested, create individual plots
        if "predictors" in plot_types and X is not None and isinstance(X, pd.DataFrame):
            # Exclude group_category column from predictors if it's a column name
            if isinstance(group_category, str) and group_category in X.columns:
                predictor_cols = [col for col in X.columns if col != group_category]
            else:
                predictor_cols = X.columns

            n_predictors = len(predictor_cols)

            # Determine layout - user override or smart defaults
            if n_cols is not None:
                cols = n_cols
            elif n_predictors <= 3:
                cols = n_predictors
            else:
                cols = 3

            rows = n_rows if n_rows is not None else math.ceil(n_predictors / cols)

            fig, axes = setup_subplots(
                num_models=n_predictors,
                n_cols=cols,
                n_rows=rows,
                figsize=figsize or (5 * cols, 5 * rows),
            )

            for i, col in enumerate(predictor_cols):
                if i < len(axes):
                    ax = axes[i]

                    # Color points by group if group_category provided
                    if group_category is not None:
                        if isinstance(group_category, str) and isinstance(
                            X, pd.DataFrame
                        ):
                            groups = X[group_category]
                        else:
                            groups = pd.Series(group_category)

                        unique_groups = groups.unique()
                        colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))

                        # Plot each group with different color
                        for gidx, group_val in enumerate(unique_groups):
                            mask = groups == group_val
                            plot_kwgs_group = point_kwgs.copy()
                            plot_kwgs_group["label"] = f"{group_val} (n={mask.sum()})"
                            plot_kwgs_group["color"] = colors[gidx]
                            ax.scatter(
                                X.loc[mask, col], residuals[mask], **plot_kwgs_group
                            )
                        if not show_centroids:
                            handles, labels_legend = ax.get_legend_handles_labels()
                            if labels_legend:
                                apply_legend(
                                    legend_loc, fontsize=tick_fontsize - 2, ax=ax
                                )
                    else:
                        # Regular scatter without grouping
                        ax.scatter(X[col], residuals, **point_kwgs)

                    ax.axhline(y=0, **line_kwgs)
                    # Add lowess smooth
                    if show_lowess:
                        try:
                            smoothed = lowess(residuals, X[col], frac=0.3)
                            ax.plot(smoothed[:, 0], smoothed[:, 1], **lowess_kwgs)
                        except:
                            pass

                    # Add centroids if requested
                    if show_centroids:
                        if group_category is not None:
                            # Option 1: User-provided groups
                            if isinstance(group_category, str) and isinstance(
                                X, pd.DataFrame
                            ):
                                groups = X[group_category]
                            else:
                                groups = pd.Series(group_category)

                            unique_groups = groups.unique()
                            colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))

                            for gidx, group_val in enumerate(unique_groups):
                                mask = groups == group_val
                                # Handle boolean indexing for both df and arrays
                                # Get centroid x-coordinate from predictor column
                                if isinstance(X, pd.DataFrame):
                                    centroid_x = X.loc[mask, col].mean()
                                else:
                                    centroid_x = X[col][mask].mean()
                                centroid_y = residuals[mask].mean()
                                plot_kwgs = centroid_kwgs.copy()

                                # Add label to ALL subplots
                                # plot_kwgs["label"] = f"{group_val} (n={mask.sum()})"

                                # Handle color assignment
                                if "c" in plot_kwgs:
                                    if isinstance(plot_kwgs["c"], list):
                                        plot_kwgs["c"] = (
                                            plot_kwgs["c"][gidx]
                                            if gidx < len(plot_kwgs["c"])
                                            else colors[gidx]
                                        )
                                elif "color" not in plot_kwgs:
                                    plot_kwgs["color"] = colors[gidx]

                                ax.scatter(centroid_x, centroid_y, **plot_kwgs)

                        else:
                            # Option 2: Automatic K-means clustering
                            clusters = n_clusters if n_clusters is not None else 3
                            data = np.column_stack([X[col], residuals])
                            kmeans = KMeans(
                                n_clusters=clusters, random_state=kmeans_rstate
                            )
                            cluster_labels = kmeans.fit_predict(data)

                            colors = cm.tab10(np.linspace(0, 1, clusters))

                            for cluster_id in range(clusters):
                                mask = cluster_labels == cluster_id
                                # Get centroid x-coordinate from predictor column
                                if isinstance(X, pd.DataFrame):
                                    centroid_x = X.loc[mask, col].mean()
                                else:
                                    centroid_x = X[col][mask].mean()

                                centroid_y = residuals[mask].mean()
                                plot_kwgs = centroid_kwgs.copy()

                                # Only add label to first subplot
                                plot_kwgs["label"] = (
                                    f"Cluster {cluster_id+1} (n={mask.sum()})"
                                )

                                # Handle color assignment
                                if "c" in plot_kwgs:
                                    if isinstance(plot_kwgs["c"], list):
                                        plot_kwgs["c"] = (
                                            plot_kwgs["c"][cluster_id]
                                            if cluster_id < len(plot_kwgs["c"])
                                            else colors[cluster_id]
                                        )
                                elif "color" not in plot_kwgs:
                                    plot_kwgs["color"] = colors[cluster_id]

                                ax.scatter(centroid_x, centroid_y, **plot_kwgs)

                        # Add legend to ALL subplots
                        handles, labels_legend = ax.get_legend_handles_labels()
                        if labels_legend:
                            apply_legend(legend_loc, fontsize=tick_fontsize - 2, ax=ax)

                    if heteroskedasticity_test:
                        # Create single-column X for this predictor
                        X_single = (
                            X[[col]].values
                            if isinstance(X, pd.DataFrame)
                            else X[:, i : i + 1]
                        )

                        het_results = check_heteroskedasticity(
                            residuals,
                            X=X_single,
                            y_pred=y_pred_arr,
                            test_type=heteroskedasticity_test,
                            decimals=decimal_places,
                        )

                        # Add test results to legend
                        for _, result in het_results.items():
                            if "error" not in result:
                                label = result["interpretation"]
                                ax.plot(
                                    [],
                                    [],
                                    linestyle="None",
                                    marker="None",
                                    label=label,
                                )

                    # Ensure legend is shown if we have labels (from groups OR het tests)
                    handles, labels_legend = ax.get_legend_handles_labels()
                    if labels_legend:
                        apply_legend(legend_loc, fontsize=tick_fontsize - 2, ax=ax)

                    ax.set_xlabel(col, fontsize=label_fontsize)
                    ax.set_ylabel("Residuals", fontsize=label_fontsize)

                    # Apply text wrap to subplot title if needed
                    apply_plot_title(
                        None,
                        default_title=f"Residuals vs {col}",
                        text_wrap=text_wrap,
                        fontsize=label_fontsize,
                        ax=ax,
                    )

                    ax.grid(visible=gridlines, alpha=0.3)
                    ax.tick_params(labelsize=tick_fontsize)

            # Hide unused subplots
            for i in range(n_predictors, len(axes)):
                axes[i].axis("off")

            # Add overall figure title for predictor plots
            apply_plot_title(
                suptitle,
                default_title=f"Residual Diagnostics: {name}",
                text_wrap=text_wrap,
                fontsize=label_fontsize + 2,
                fig=fig,
                suptitle_y=suptitle_y,
            )

            plt.tight_layout()

            if save_plot:
                name_clean = name.lower().replace(" ", "_")
                save_plot_images(
                    f"{name_clean}_residuals_by_predictor",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )
            plt.show()
