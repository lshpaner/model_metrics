import pandas as pd
import numpy as np
from scipy.stats import spearmanr, jarque_bera, norm
from statsmodels import stats
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib_venn import venn2
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    r2_score,
    accuracy_score,
)

from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt,
)

# Registry entries: name -> (display label, scorer, higher_is_better, kind)
#   kind "score" : scorer takes probabilities / continuous scores
#   kind "label" : scorer takes thresholded 0/1 predictions
#   kind "value" : scorer takes continuous predictions (regression)
_CLASSIFICATION_METRICS = {
    "roc_auc": ("ROC AUC", roc_auc_score, True, "score"),
    "average_precision": ("Average Precision", average_precision_score, True, "score"),
    "brier": ("Brier Score (lower is better)", brier_score_loss, False, "score"),
    "precision": (
        "Precision",
        lambda yt, yp: precision_score(yt, yp, zero_division=0),
        True,
        "label",
    ),
    "recall": (
        "Recall",
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        True,
        "label",
    ),
    "f1": ("F1", lambda yt, yp: f1_score(yt, yp, zero_division=0), True, "label"),
    "accuracy": ("Accuracy", accuracy_score, True, "label"),
}
_REGRESSION_METRICS = {
    "r2": ("R\u00b2", r2_score, True, "value"),
    "mae": ("MAE (lower is better)", mean_absolute_error, False, "value"),
    "rmse": (
        "RMSE (lower is better)",
        lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        False,
        "value",
    ),
    "mse": ("MSE (lower is better)", mean_squared_error, False, "value"),
}
_ALL_METRICS = {**_CLASSIFICATION_METRICS, **_REGRESSION_METRICS}
_DEFAULT_METRICS = {
    "classification": ("roc_auc", "average_precision", "brier"),
    "regression": ("r2", "mae", "rmse"),
}


def _resolve_task(task, model, metrics):
    """Decide classification vs regression for the feature-performance curve.

    A supplied model is authoritative: predict_proba means classification,
    otherwise regression. Metrics never override the model. Only in the
    predictions-only path (no model) is the task inferred from the metric
    family, and mixing families there is an error.
    """
    if task != "auto":
        return task
    if model is not None:
        if hasattr(model, "predict_proba"):
            return "classification"
        if hasattr(model, "predict"):
            return "regression"
    if metrics is not None:
        names = [m for m in metrics if isinstance(m, str)]
        has_reg = any(n in _REGRESSION_METRICS for n in names)
        has_clf = any(n in _CLASSIFICATION_METRICS for n in names)
        if has_reg and not has_clf:
            return "regression"
        if has_clf and not has_reg:
            return "classification"
        if has_reg and has_clf:
            reg_in = [n for n in names if n in _REGRESSION_METRICS]
            clf_in = [n for n in names if n in _CLASSIFICATION_METRICS]
            raise ValueError(
                "Metrics mix classification and regression families, so the "
                "task cannot be inferred without a model.\n"
                f"  Regression metric(s) in your list : {reg_in}\n"
                f"  Classification metric(s) in your list : {clf_in}\n"
                "  Fix: pass a single family, or set task= explicitly.\n"
                f"  Regression options     : {list(_REGRESSION_METRICS)}\n"
                f"  Classification options : {list(_CLASSIFICATION_METRICS)}"
            )
    return "classification"


def _validate_metrics_for_task(metrics, task):
    """Reject metrics that belong to the wrong task family.

    Callables are exempt (their family is unknown). Unknown metric strings are
    left for the caller's registry lookup to report.
    """
    reg = [m for m in metrics if isinstance(m, str) and m in _REGRESSION_METRICS]
    clf = [m for m in metrics if isinstance(m, str) and m in _CLASSIFICATION_METRICS]
    if task == "classification" and reg:
        raise ValueError(
            f"Regression metric(s) {reg} are not valid for a classification task.\n"
            f"  Regression-only metrics : {list(_REGRESSION_METRICS)}\n"
            f"  Valid classification    : {list(_CLASSIFICATION_METRICS)}"
        )
    if task == "regression" and clf:
        raise ValueError(
            f"Classification metric(s) {clf} are not valid for a regression task.\n"
            f"  Classification-only metrics : {list(_CLASSIFICATION_METRICS)}\n"
            f"  Valid regression            : {list(_REGRESSION_METRICS)}"
        )


################################################################################
############################## Helper Functions ################################
################################################################################


def normalize_model_titles(model_title, num_models, format_template="Model {i}"):
    """
    Convert model_title to a list of appropriate length.

    Parameters
    ----------
    model_title : str, list, pd.Series, or None
        Custom model names.
    num_models : int
        Number of models.
    format_template : str, default="Model {i}"
        Template for default model names. Use {i} as placeholder for index.

    Returns
    -------
    list
        List of model titles.

    Raises
    ------
    TypeError
        If model_title is not a string, list, pd.Series, or None.
    """
    if model_title is None:
        return [format_template.format(i=i + 1) for i in range(num_models)]
    elif isinstance(model_title, str):
        return [model_title]
    elif isinstance(model_title, pd.Series):
        return model_title.tolist()
    elif isinstance(model_title, list):
        return model_title
    else:
        raise TypeError("model_title must be a string, a list of strings, or None.")


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


def validate_and_normalize_inputs(model, X, y_prob_or_pred):
    """
    Validate and normalize model/probability/prediction inputs.

    Works for both classification (y_prob) and regression (y_pred).

    Parameters
    ----------
    model : estimator or list of estimators, optional
        Trained model(s).
    X : array-like, optional
        Feature matrix.
    y_prob_or_pred : array-like or list, optional
        Predicted probabilities (classification) or predictions (regression).

    Returns
    -------
    model : list
        List of models (or None placeholders).
    y_prob_or_pred : list
        List of probability/prediction arrays.
    num_models : int
        Number of models.
    """
    if not ((model is not None and X is not None) or y_prob_or_pred is not None):
        raise ValueError("You need to provide model and X or y_prob/y_pred")

    # Normalize model to list
    if model is not None and not isinstance(model, list):
        model = [model]

    # Normalize y_prob_or_pred to list of arrays
    if y_prob_or_pred is not None:
        if isinstance(y_prob_or_pred, np.ndarray):
            y_prob_or_pred = [y_prob_or_pred]
        elif isinstance(y_prob_or_pred, list):
            # Check if it's a list of scalars (convert to single array)
            if len(y_prob_or_pred) > 0 and isinstance(y_prob_or_pred[0], (int, float)):
                y_prob_or_pred = [np.array(y_prob_or_pred)]
            # Otherwise assume it's already a list of arrays

    # Determine number of models
    num_models = len(model) if model else len(y_prob_or_pred)

    # Create placeholder models if using y_prob_or_pred
    if y_prob_or_pred is not None:
        model = [None] * num_models

    return model, y_prob_or_pred, num_models


def compute_classification_metrics(y_true, y_pred, y_prob, threshold, decimal_places=3):
    """Compute classification performance metrics."""
    return {
        "Precision/PPV": round(
            precision_score(y_true, y_pred, zero_division=0), decimal_places
        ),
        "Average Precision": round(
            average_precision_score(y_true, y_prob), decimal_places
        ),
        "Sensitivity/Recall": round(
            recall_score(y_true, y_pred, zero_division=0), decimal_places
        ),
        "Specificity": round(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            decimal_places,
        ),
        "F1-Score": round(f1_score(y_true, y_pred), decimal_places),
        "AUC ROC": round(roc_auc_score(y_true, y_prob), decimal_places),
        "Brier Score": round(brier_score_loss(y_true, y_prob), decimal_places),
        "Model Threshold": round(float(threshold), decimal_places),
    }


def compute_regression_metrics(
    y_true, y_pred, n_features=None, include_adjusted_r2=False, decimal_places=3
):
    """Compute regression performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    exp_var = explained_variance_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if np.any(mask)
        else np.nan
    )
    metrics = {
        "MAE": round(mae, decimal_places),
        "MAPE": round(mape, decimal_places) if not np.isnan(mape) else "NaN",
        "MSE": round(mse, decimal_places),
        "RMSE": round(rmse, decimal_places),
        "Expl. Var.": round(exp_var, decimal_places),
        "R^2": round(r2, decimal_places),
    }

    if include_adjusted_r2 and n_features is not None:
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        metrics["Adj. R^2"] = round(adj_r2, decimal_places)

    return metrics


def compute_leverage_and_cooks_distance(X, standardized_residuals):
    """
    Compute leverage (hat values) and Cook's distance.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix
    standardized_residuals : array-like
        Standardized residuals

    Returns
    -------
    tuple
        (leverage, cooks_d, n, p) or (None, None, None, None) if calculation fails
    """
    try:
        # Convert to array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Add constant for intercept
        X_with_intercept = np.column_stack([np.ones(len(X_array)), X_array])

        # Hat matrix diagonal - use pinv for stability
        H = (
            X_with_intercept
            @ np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
            @ X_with_intercept.T
        )
        leverage = np.diag(H)

        # Cook's distance
        n = len(standardized_residuals)
        p = X_with_intercept.shape[1]
        cooks_d = (standardized_residuals**2 / p) * (leverage / (1 - leverage))

        return leverage, cooks_d, n, p

    except Exception as e:
        return None, None, None, None


################################################################################
########################## Model Venn Diagram Helpers ##########################
################################################################################

_VENN_CATEGORY_SPEC = {
    "TP": dict(
        title="True Positives  (correctly caught)",
        subpop_val=1,
        in_set_val=1,
        both_role="both catch",
        outside_label="both miss (FN)",
        subpop_name="actual positives",
    ),
    "FP": dict(
        title="False Positives  (false alarms)",
        subpop_val=0,
        in_set_val=1,
        both_role="both false-alarm",
        outside_label="both clear (TN)",
        subpop_name="actual negatives",
    ),
    "FN": dict(
        title="False Negatives  (missed positives)",
        subpop_val=1,
        in_set_val=0,
        both_role="both miss",
        outside_label="both catch (TP)",
        subpop_name="actual positives",
    ),
    "TN": dict(
        title="True Negatives  (correctly cleared)",
        subpop_val=0,
        in_set_val=0,
        both_role="both correct",
        outside_label="both false-alarm (FP)",
        subpop_name="actual negatives",
    ),
}


def _venn_blend(c1, c2):
    """Return the RGB midpoint of two matplotlib color specs."""
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    return ((r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2)


def _venn_resolve_side(
    side,
    y_pred,
    model,
    X,
    y_true=None,
    y_prob=None,
    threshold=None,
    score=None,
):
    """Return integer 1-D array of predictions for one side.

    Accepts three mutually exclusive input paths for the same side:
        * y_pred  -- binary predictions, used as-is
        * y_prob  -- positive-class probabilities, thresholded internally
                     via threshold (defaulting to 0.5 if unset)
        * model + X -- predictions generated via the library's standard
                     `get_predictions` helper. The model's stored
                     `.threshold` attribute is looked up automatically
                     (falling back to 0.5 if absent); threshold
                     overrides; score picks a non-default key from the
                     model's threshold dict.
    """
    provided = sum(v is not None for v in (y_pred, y_prob, model))
    if provided > 1:
        raise ValueError(
            f"For side {side!r}, supply only one of "
            f"y_pred_{side}, y_prob_{side}, or model_{side}."
        )
    if y_pred is not None:
        return np.asarray(y_pred).ravel().astype(int)
    if y_prob is not None:
        threshold = threshold if threshold is not None else 0.5
        return (np.asarray(y_prob).ravel() >= threshold).astype(int)
    if model is not None:
        if X is None:
            raise ValueError(
                f"model_{side} given but X_{side} is missing "
                f"(or pass X_a alone if both models share features)."
            )
        if not hasattr(model, "predict"):
            raise TypeError(f"model_{side} has no .predict() method.")
        _, _, y_pred, _ = get_predictions(
            model,
            X,
            y_true,
            model_threshold=True,
            custom_threshold=threshold,
            score=score,
        )
        return np.asarray(y_pred).ravel().astype(int)
    raise ValueError(
        f"Provide one of y_pred_{side}, y_prob_{side}, " f"or (model_{side}, X_{side})."
    )


def _venn_category_counts(y_true, y_pred_a, y_pred_b, cat):
    """Return (a_only, b_only, both, outside, n_sub) counts for one Venn category."""
    spec = _VENN_CATEGORY_SPEC[cat]
    sub = y_true == spec["subpop_val"]
    in_a = y_pred_a == spec["in_set_val"]
    in_b = y_pred_b == spec["in_set_val"]
    a_only = int((sub & in_a & ~in_b).sum())
    b_only = int((sub & ~in_a & in_b).sum())
    both = int((sub & in_a & in_b).sum())
    outside = int((sub & ~in_a & ~in_b).sum())
    return a_only, b_only, both, outside, int(sub.sum())


def _draw_one_venn(
    ax,
    cat,
    counts,
    label_a,
    label_b,
    inner_fontsize,
    outer_fontsize,
    title_fontsize,
    colors,
    alpha,
    title_override=None,
    title_pad=None,
    label_kwgs=None,
):
    """Render one category's overlap Venn into the provided axes."""
    spec = _VENN_CATEGORY_SPEC[cat]
    a_only, b_only, both, outside, n_sub = counts
    a_total = a_only + both
    b_total = b_only + both

    opts = {
        "show_title": True,
        "show_subtitle": True,
        "show_set_labels": True,
        "show_set_totals": True,
        "show_inner_count": True,
        "show_inner_role": True,
    }
    if label_kwgs:
        opts.update(label_kwgs)

    # Build set labels beneath each circle
    if opts["show_set_labels"] and opts["show_set_totals"]:
        set_label_a = f"{label_a}\n{cat} total: {a_total:,}"
        set_label_b = f"{label_b}\n{cat} total: {b_total:,}"
    elif opts["show_set_labels"]:
        set_label_a, set_label_b = label_a, label_b
    elif opts["show_set_totals"]:
        set_label_a = f"{cat} total: {a_total:,}"
        set_label_b = f"{cat} total: {b_total:,}"
    else:
        set_label_a, set_label_b = "", ""

    v = venn2(
        subsets=(1, 1, 1),
        set_labels=(set_label_a, set_label_b),
        ax=ax,
    )

    if colors is not None:
        if len(colors) == 2:
            c_a, c_b = colors
            c_both = _venn_blend(c_a, c_b)
        elif len(colors) == 3:
            c_a, c_b, c_both = colors
        else:
            raise ValueError("colors must be a 2- or 3-tuple of color specs.")
        for rid, c in (("10", c_a), ("01", c_b), ("11", c_both)):
            patch = v.get_patch_by_id(rid)
            if patch is not None:
                patch.set_color(c)
                patch.set_alpha(alpha)

    # Inner region labels (count + role, either, or neither)
    for rid, count, role in (
        ("10", a_only, f"{label_a}"),
        ("01", b_only, f"{label_b}"),
        ("11", both, spec["both_role"]),
    ):
        lab = v.get_label_by_id(rid)
        if lab is None:
            continue
        parts = []
        if opts["show_inner_count"]:
            parts.append(f"{count:,}")
        if opts["show_inner_role"]:
            parts.append(role)
        lab.set_text("\n".join(parts))
        lab.set_fontsize(inner_fontsize)

    for sl in v.set_labels:
        if sl is not None:
            sl.set_fontsize(outer_fontsize)

    # Title
    if not opts["show_title"]:
        ax.set_title("")
        return

    main_title = title_override if title_override is not None else spec["title"]
    if opts["show_subtitle"]:
        full_title = (
            f"{main_title}\n"
            f"Out of {n_sub:,} {spec['subpop_name']}\n"
            f"{spec['outside_label']}: {outside:,}"
        )
    else:
        full_title = main_title
    ax.set_title(
        full_title,
        fontsize=title_fontsize,
        weight="bold",
        pad=title_pad if title_pad is not None else 2.0,
    )


def _venn_default_figsize(
    counts_per_cat,
    categories,
    titles,
    label_kwgs,
    title_fontsize,
    ncols,
    nrows,
):
    """Size the figure so the longest rendered title fits without overflow."""
    opts = label_kwgs or {}
    show_title = opts.get("show_title", True)
    show_subtitle = opts.get("show_subtitle", True)

    max_chars = 0
    if show_title:
        for cat in categories:
            spec = _VENN_CATEGORY_SPEC[cat]
            heading = (titles or {}).get(cat, spec["title"])
            max_chars = max(max_chars, len(heading))
            if show_subtitle:
                _, _, _, outside, n_sub = counts_per_cat[cat]
                # subtitle is rendered on two lines, size to the longer one
                line1 = f"Out of {n_sub:,} {spec['subpop_name']}"
                line2 = f"{spec['outside_label']}: {outside:,}"
                max_chars = max(max_chars, len(line1), len(line2))

    char_w_inches = title_fontsize * 0.009
    panel_w = max(5.5, max_chars * char_w_inches + 0.6)
    panel_h = 5.0
    return (panel_w * ncols, panel_h * nrows)


def _overlap_table_categorize(yt, yp):
    cats = np.empty(len(yt), dtype=object)
    cats[(yt == 1) & (yp == 1)] = "TP"
    cats[(yt == 1) & (yp == 0)] = "FN"
    cats[(yt == 0) & (yp == 1)] = "FP"
    cats[(yt == 0) & (yp == 0)] = "TN"
    return cats


def _print_overlap_summary_legend(label_a="Model A", label_b="Model B"):
    """Print the column-meaning legend for overlap_summary output."""
    text = f"""\
overlap_summary columns
-----------------------
Rows: confusion-matrix category (TP, FP, FN, TN).

  n_{label_a:<14} count {label_a} placed in this category
  n_{label_b:<14} count {label_b} placed in this category
  both             both models agreed on this category for the same observation
  {label_a}_only{' ' * max(0, 11 - len(label_a))} {label_a} placed it here, {label_b} didn't
  {label_b}_only{' ' * max(0, 11 - len(label_b))} {label_b} placed it here, {label_a} didn't
  outside          in the subpop but neither model placed it here
  subpop           denominator: actual positives for TP/FN,
                   actual negatives for FP/TN

Identity:  both + {label_a}_only + {label_b}_only + outside == subpop
"""
    print(text)


def _draw_crosstab_matrix(
    ax, ct, label_a, label_b, colors, cell_fontsize, label_fontsize
):
    """Render the 4x4 crosstab grid with per-cell colors and counts."""
    from matplotlib.patches import Rectangle

    order = list(ct.index)
    pos = {"TP", "FN"}
    n = len(order)

    for i, ra in enumerate(order):
        for j, cb in enumerate(order):
            yi = n - 1 - i  # invert so row 0 (TP) is at the top
            if (ra in pos) != (cb in pos):
                fc, tc = colors["impossible"], "#9aa0a6"
            elif ra == cb:
                fc, tc = colors["agree"], "#1e7e34"
            else:
                fc, tc = colors["disagree"], "#c82333"

            ax.add_patch(
                Rectangle(
                    (j, yi),
                    1,
                    1,
                    facecolor=fc,
                    edgecolor="white",
                    linewidth=2,
                )
            )
            val = ct.iat[i, j]
            text = (
                "-" if (isinstance(val, float) and np.isnan(val)) else f"{int(val):,}"
            )
            ax.text(
                j + 0.5,
                yi + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=cell_fontsize,
                color=tc,
                weight="bold",
            )

    # row labels (label_a categories, on the left)
    for i, ra in enumerate(order):
        ax.text(
            -0.15,
            n - 1 - i + 0.5,
            ra,
            ha="right",
            va="center",
            fontsize=label_fontsize,
            weight="bold",
        )
    # column labels (label_b categories, along the top)
    for j, cb in enumerate(order):
        ax.text(
            j + 0.5,
            n + 0.15,
            cb,
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            weight="bold",
        )

    # axis-level labels (label_a vertical on left, label_b above the columns)
    ax.text(
        -0.75,
        n / 2,
        label_a,
        ha="center",
        va="center",
        fontsize=label_fontsize + 1,
        rotation=90,
        weight="bold",
    )
    ax.text(
        n / 2,
        n + 0.55,
        label_b,
        ha="center",
        va="bottom",
        fontsize=label_fontsize + 1,
        weight="bold",
    )

    ax.set_xlim(-1.0, n + 0.2)
    ax.set_ylim(-0.3, n + 0.85)
    ax.set_aspect("equal", anchor="N")
    ax.axis("off")


def _draw_crosstab_summary(
    ax, label_b, stats, fontsize, label_fontsize=12, body_fontsize=None
):
    """Render the right-hand swap-summary text block."""
    ax.axis("off")

    if body_fontsize is None:
        body_fontsize = fontsize - 1  # preserves original default exactly

    blocks = [
        (
            f"{stats['both_fn']:,} shared false negatives",
            "#a04500",
            "Both models miss the same hard positive cases.",
        ),
        (
            f"{stats['both_tn']:,} shared true negatives",
            "#1e7e34",
            "Both models correctly rule out the same easy negatives.",
        ),
        (
            f"{stats['b_extra_tp']:,} vs {stats['b_lost_tp']:,} TP swap",
            "#1f5fb0",
            f"{label_b} catches {stats['b_extra_tp']:,} extra positives, "
            f"misses {stats['b_lost_tp']:,}. Net: {stats['net_tp']:+,} TPs.",
        ),
        (
            f"{stats['b_extra_fp']:,} vs {stats['b_avoided_fp']:,} FP swap",
            "#a02838",
            f"{label_b} adds {stats['b_extra_fp']:,} false alarms, "
            f"avoids {stats['b_avoided_fp']:,}. Net: {stats['net_fp']:+,} FPs.",
        ),
    ]

    y = 0.98
    for headline, color, body in blocks:
        ax.text(
            0.0,
            y,
            headline,
            ha="left",
            va="top",
            fontsize=label_fontsize + 5,  # was: fontsize + 6
            color=color,
            weight="bold",
            transform=ax.transAxes,
        )
        y -= 0.07
        ax.text(
            0.0,
            y,
            body,
            ha="left",
            va="top",
            fontsize=body_fontsize,  # was: fontsize - 1
            color="#5a6268",
            style="italic",
            transform=ax.transAxes,
            wrap=True,
        )
        y -= 0.14


def _draw_crosstab_legend(ax, colors, fontsize):
    """Render a horizontal color-legend strip."""
    from matplotlib.patches import Patch

    ax.axis("off")
    ax.legend(
        handles=[
            Patch(facecolor=colors["agree"], edgecolor="#999", label="agree"),
            Patch(
                facecolor=colors["disagree"], edgecolor="#999", label="disagree (swap)"
            ),
            Patch(
                facecolor=colors["impossible"],
                edgecolor="#999",
                label="impossible (true label conflict)",
            ),
        ],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=fontsize,
    )


def _print_overlap_crosstab_legend(ct, label_a="Model A", label_b="Model B"):
    """Print the cell-meaning legend and derived swap summary for
    overlap_crosstab. Expects the raw integer-count crosstab."""
    total = int(ct.values.sum())
    agree = int(np.trace(ct.values))
    disagree = total - agree
    agree_pct = (agree / total * 100) if total else 0.0

    both_tp = int(ct.loc["TP", "TP"])
    both_fp = int(ct.loc["FP", "FP"])
    both_fn = int(ct.loc["FN", "FN"])
    both_tn = int(ct.loc["TN", "TN"])

    b_extra_tp = int(ct.loc["FN", "TP"])  # a missed, b caught
    b_lost_tp = int(ct.loc["TP", "FN"])  # a caught, b missed
    b_extra_fp = int(ct.loc["TN", "FP"])  # a cleared, b false-alarmed
    b_avoided_fp = int(ct.loc["FP", "TN"])  # a false-alarmed, b cleared

    net_tp = b_extra_tp - b_lost_tp
    net_fp = b_extra_fp - b_avoided_fp

    text = f"""\
overlap_crosstab cells
----------------------
Rows: {label_a}'s confusion-matrix category.
Cols: {label_b}'s confusion-matrix category.

  diagonal            agreement: both models put the observation in the
                      same category. (TP,TP), (FP,FP), (FN,FN), (TN,TN).
  (TP,FN) / (FN,TP)   TP swap: one model catches an actual positive the
                      other misses.
  (FP,TN) / (TN,FP)   FP swap: one model false-alarms on an actual
                      negative the other clears.
  impossible cells    the 8 cells mixing a positive-subpop category
                      (TP, FN) with a negative-subpop one (FP, TN). A
                      single observation cannot be, say, TP for one model
                      and FP for the other, because y_true is fixed.
                      Always 0, or NaN when mask_impossible=True.

Swap summary
------------
  Agreement:     {agree:,} / {total:,}  ({agree_pct:.1f}%)
  Disagreement:  {disagree:,}

  Shared TP:     {both_tp:,}
  Shared FN:     {both_fn:,}
  Shared TN:     {both_tn:,}
  Shared FP:     {both_fp:,}

  TP swap:       {label_b} catches {b_extra_tp:,}, misses {b_lost_tp:,} (net {net_tp:+,} TP)
  FP swap:       {label_b} adds {b_extra_fp:,}, avoids {b_avoided_fp:,} (net {net_fp:+,} FP)
"""
    print(text)


_FONT_ALIASES = {
    "arial": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "helvetica": [
        "Helvetica",
        "Helvetica Neue",
        "Nimbus Sans",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "helvetica neue": [
        "Helvetica Neue",
        "Helvetica",
        "Nimbus Sans",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "times": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "times new roman": [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "courier": [
        "Courier New",
        "Courier",
        "Liberation Mono",
        "Nimbus Mono",
        "DejaVu Sans Mono",
    ],
    "courier new": [
        "Courier New",
        "Courier",
        "Liberation Mono",
        "Nimbus Mono",
        "DejaVu Sans Mono",
    ],
    "consolas": [
        "Consolas",
        "Cascadia Code",
        "Cascadia Mono",
        "Source Code Pro",
        "Inconsolata",
        "Liberation Mono",
        "DejaVu Sans Mono",
    ],
    "georgia": ["Georgia", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "verdana": ["Verdana", "Liberation Sans", "DejaVu Sans"],
    "calibri": ["Calibri", "Carlito", "Liberation Sans", "DejaVu Sans"],
    "cambria": ["Cambria", "Caladea", "Liberation Serif", "DejaVu Serif"],
    "tahoma": ["Tahoma", "Liberation Sans", "DejaVu Sans"],
    "garamond": ["Garamond", "EB Garamond", "Liberation Serif", "DejaVu Serif"],
}


def _resolve_font_family(font):
    """Expand a font name (or list of names) to a fallback chain filtered
    to the fonts actually installed on this system.

    Returns None when the input is None (caller should leave rcParams
    untouched).

    Raises
    ------
    ValueError
        If none of the candidate fonts (after alias expansion) are
        installed on the system.
    TypeError
        If `font` is not None, a string, or a list/tuple of strings.
    """
    if font is None:
        return None

    from matplotlib import font_manager

    # build a list of candidates by expanding any aliases
    if isinstance(font, str):
        key = font.lower().strip()
        candidates = _FONT_ALIASES.get(key, [font])
    elif isinstance(font, (list, tuple)):
        candidates = []
        for f in font:
            if isinstance(f, str):
                candidates.extend(_FONT_ALIASES.get(f.lower().strip(), [f]))
            else:
                candidates.append(f)
    else:
        raise TypeError(
            f"font must be a string, a list/tuple of strings, or None. "
            f"Got {type(font).__name__}."
        )

    # filter to fonts actually installed; dedupe while preserving order
    installed, seen = [], set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            font_manager.findfont(name, fallback_to_default=False)
            installed.append(name)
        except Exception:
            continue

    if not installed:
        raise ValueError(
            f"None of the requested fonts are installed on this system: "
            f"{candidates!r}. Install one of them, pass a font from the "
            f"built-in aliases ({sorted(_FONT_ALIASES)}), or supply your "
            f"own fallback list including one font you know exists. "
            f"To see what's available: "
            f"`sorted({{f.name for f in matplotlib.font_manager.fontManager.ttflist}})`."
        )

    return installed


################################################################################
############################ Regression Helpers ################################
################################################################################


def compute_residual_diagnostics(
    residuals,
    y_true,
    y_pred,
    leverage=None,
    cooks_d=None,
    n_features=None,
    model_name="Model",
    decimal_places=4,
):
    """
    Compute comprehensive residual diagnostic statistics.

    Parameters
    ----------
    residuals : array-like
        Model residuals
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    leverage : array-like, optional
        Leverage values
    cooks_d : array-like, optional
        Cook's distance values
    n_features : int, optional
        Number of features
    model_name : str, default="Model"
        Name of the model
    decimal_places : int, default=4
        Decimal places for rounding

    Returns
    -------
    dict
        Dictionary of diagnostic statistics
    """
    rmse = np.sqrt(np.mean(residuals**2))

    diagnostics = {
        "model_name": model_name,
        "n_observations": len(residuals),
        "n_predictors": n_features if n_features is not None else "N/A",
        "mean_residual": residuals.mean(),
        "std_residual": residuals.std(),
        "min_residual": residuals.min(),
        "max_residual": residuals.max(),
        "mae": np.mean(np.abs(residuals)),
        "rmse": rmse,
    }

    # Add R^2 and Adjusted R^2
    if n_features is not None:
        try:
            reg_metrics = compute_regression_metrics(
                y_true,
                y_pred,
                n_features=n_features,
                include_adjusted_r2=True,
                decimal_places=decimal_places,
            )
            diagnostics["r2"] = reg_metrics["R^2"]
            diagnostics["adj_r2"] = reg_metrics["Adj. R^2"]
        except Exception:
            pass

    # Add normality test
    try:
        jb_stat, jb_pval = jarque_bera(residuals)
        diagnostics["jarque_bera_stat"] = jb_stat
        diagnostics["jarque_bera_pval"] = jb_pval
    except Exception:
        pass

    # Add autocorrelation test
    try:
        dw_stat = durbin_watson(residuals)
        diagnostics["durbin_watson"] = dw_stat
    except Exception:
        pass

    # Add leverage diagnostics
    if leverage is not None and cooks_d is not None:
        diagnostics["max_leverage"] = leverage.max()
        diagnostics["mean_leverage"] = leverage.mean()
        diagnostics["max_cooks_d"] = cooks_d.max()

        if n_features is not None:
            n = len(residuals)
            leverage_threshold = 2 * n_features / n
            diagnostics["leverage_threshold"] = leverage_threshold
            diagnostics["high_leverage_count"] = np.sum(leverage > leverage_threshold)
            diagnostics["influential_points_05"] = np.sum(cooks_d > 0.5)
            diagnostics["influential_points_10"] = np.sum(cooks_d > 1.0)

    return diagnostics


def print_resid_diagnostics_table(diagnostics, decimals=4):
    """
    Print a formatted table of diagnostic statistics.

    Parameters
    ----------
    diagnostics : dict
        Dictionary containing diagnostic statistics
    decimals : int, default=4
        Number of decimal places to display for numeric values
    """
    name = diagnostics.get("model_name", "Model")

    print(f"\n{'='*60}")
    print(f"Residual Diagnostics: {name}")
    print(f"{'='*60}")
    print(f"{'Statistic':<30} {'Value':>20}")
    print(f"{'-'*60}")

    # Basic statistics
    print(f"{'N Observations':<30} {diagnostics['n_observations']:>20}")
    if "n_predictors" in diagnostics and diagnostics["n_predictors"] != "N/A":
        print(f"{'N Predictors':<30} {diagnostics['n_predictors']:>20}")

    # Model fit metrics
    if "r2" in diagnostics:
        print(f"{'-'*60}")
        print(f"{'R-squared':<30} {diagnostics['r2']:>20.{decimals}f}")
    if "adj_r2" in diagnostics:
        print(f"{'Adjusted R-squared':<30} {diagnostics['adj_r2']:>20.{decimals}f}")

    print(f"{'-'*60}")
    # Error metrics
    print(f"{'RMSE':<30} {diagnostics['rmse']:>20.{decimals}f}")
    print(f"{'MAE':<30} {diagnostics['mae']:>20.{decimals}f}")

    print(f"{'-'*60}")
    # Residual statistics
    print(f"{'Mean Residual':<30} {diagnostics['mean_residual']:>20.{decimals}f}")
    print(f"{'Std Residual':<30} {diagnostics['std_residual']:>20.{decimals}f}")
    print(f"{'Min Residual':<30} {diagnostics['min_residual']:>20.{decimals}f}")
    print(f"{'Max Residual':<30} {diagnostics['max_residual']:>20.{decimals}f}")

    # Normality test
    if "jarque_bera_pval" in diagnostics:
        jb_pval = diagnostics["jarque_bera_pval"]
        jb_status = "Normal" if jb_pval > 0.05 else "Non-Normal"
        print(f"{'Jarque-Bera Test':<30} p={jb_pval:.{decimals}f} ({jb_status})")

    # Autocorrelation test
    if "durbin_watson" in diagnostics:
        print(f"{'Durbin-Watson':<30} {diagnostics['durbin_watson']:>20.{decimals}f}")

    # Influence diagnostics
    if "max_leverage" in diagnostics:
        print(f"{'-'*60}")
        print(
            f"{'Mean Leverage':<30} " f"{diagnostics['mean_leverage']:>20.{decimals}f}"
        )
        print(f"{'Max Leverage':<30} " f"{diagnostics['max_leverage']:>20.{decimals}f}")

        if "leverage_threshold" in diagnostics:
            print(
                f"{'Leverage Threshold (2p/n)':<30} "
                f"{diagnostics['leverage_threshold']:>20.{decimals}f}"
            )
        if "high_leverage_count" in diagnostics:
            print(
                f"{'High Leverage Points':<30} "
                f"{diagnostics['high_leverage_count']:>20}"
            )
    # Heteroskedasticity tests
    if "heteroskedasticity_tests" in diagnostics:
        print(f"{'-'*60}")
        print("Heteroskedasticity Tests:")
        for test_name, result in diagnostics["heteroskedasticity_tests"].items():
            if "error" not in result:
                status = (
                    "Heteroskedastic" if result["heteroskedastic"] else "Homoskedastic"
                )
                stat_val = result["statistic"]
                pval = result["pvalue"]
                print(f"{result['test_name']:<28}  p={pval:.{decimals}f} ({status})")

    print(f"{'='*60}\n")


def resid_diagnostics_to_dataframe(diagnostics, flatten_het_tests=True):
    """
    Convert diagnostics dictionary to a pandas DataFrame.

    Parameters
    ----------
    diagnostics : dict
        Dictionary containing diagnostic statistics from show_residual_diagnostics
    flatten_het_tests : bool, default=True
        Whether to flatten heteroskedasticity tests into separate rows.
        If True, creates rows like 'hetero_breusch_pagan_stat',
        'hetero_breusch_pagan_pval'.
        If False, keeps heteroskedasticity_tests as a nested structure.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Statistic', 'Value']
    """

    # Make a copy to avoid modifying original
    diag_copy = diagnostics.copy()

    # Extract and flatten heteroskedasticity tests if requested
    if flatten_het_tests and "heteroskedasticity_tests" in diag_copy:
        hetero_tests = diag_copy.pop("heteroskedasticity_tests", None)

        if hetero_tests and isinstance(hetero_tests, dict):
            for test_name, result in hetero_tests.items():
                if isinstance(result, dict):
                    if "error" in result:
                        diag_copy[f"hetero_{test_name}_error"] = result["error"]
                    else:
                        diag_copy[f"hetero_{test_name}_stat"] = result.get("statistic")
                        diag_copy[f"hetero_{test_name}_pval"] = result.get("pvalue")
                        diag_copy[f"hetero_{test_name}_heteroskedastic"] = result.get(
                            "heteroskedastic"
                        )

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(diag_copy, orient="index", columns=["Value"])
    df.index.name = "Statistic"
    df = df.reset_index()

    return df


def check_heteroskedasticity(
    residuals, X=None, y_pred=None, test_type="breusch_pagan", decimals=4
):
    """
    Test for heteroskedasticity in residuals.

    Parameters
    ----------
    residuals : array-like
        Model residuals
    X : array-like, optional
        Feature matrix (required for breusch_pagan and white tests)
    y_pred : array-like, optional
        Predicted values (required for goldfeld_quandt and simple tests)
    test_type : str, default="breusch_pagan"
        Type of test to perform. Options:
        - "breusch_pagan": Breusch-Pagan test
        - "white": White's test
        - "goldfeld_quandt": Goldfeld-Quandt test
        - "spearman": Spearman correlation test
        - "all": Run all applicable tests
    decimals : int, default=4
        Decimal places for rounding results.

    Returns
    -------
    dict
        Dictionary containing test results with keys:
        - 'test_name': Name of the test
        - 'statistic': Test statistic
        - 'pvalue': P-value
        - 'heteroskedastic': Boolean (True if heteroskedasticity detected at alpha=0.05)
        - 'interpretation': Text interpretation
    """
    results = {}

    # Prepare X by encoding categorical variables if present
    X_numeric = None
    if X is not None:
        try:
            # Convert to DataFrame if it isn't already
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
            else:
                X_df = pd.DataFrame(X)

            # Identify categorical columns
            cat_cols = X_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if cat_cols:
                # Encode categorical columns using label encoding
                X_encoded = X_df.copy()
                for col in cat_cols:
                    X_encoded[col] = pd.Categorical(X_df[col]).codes
                X_numeric = X_encoded.values
            else:
                # No categorical columns, use as-is
                X_numeric = (
                    X_df.values if isinstance(X, pd.DataFrame) else np.asarray(X)
                )
        except Exception as e:
            # If encoding fails, try to use X as-is
            X_numeric = np.asarray(X)

    if test_type in ["breusch_pagan", "all"]:
        if X_numeric is not None:
            try:
                # Add constant for intercept
                X_with_const = np.column_stack([np.ones(len(X_numeric)), X_numeric])
                lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(
                    residuals, X_with_const
                )

                het_status = "heteroskedastic" if lm_pval < 0.05 else "homoskedastic"

                results["breusch_pagan"] = {
                    "test_name": "Breusch-Pagan",
                    "statistic": round(lm_stat, decimals),
                    "pvalue": round(lm_pval, decimals),
                    "heteroskedastic": lm_pval < 0.05,
                    "interpretation": (
                        f"BP test: $\\mathit{{p}}$={round(lm_pval, decimals)} "
                        f"({het_status})"
                    ),
                }
            except Exception as e:
                results["breusch_pagan"] = {"error": str(e)}

    if test_type in ["white", "all"]:
        if X_numeric is not None:
            try:
                # Add constant for intercept
                X_with_const = np.column_stack([np.ones(len(X_numeric)), X_numeric])
                lm_stat, lm_pval, f_stat, f_pval = het_white(residuals, X_with_const)

                het_status = "heteroskedastic" if lm_pval < 0.05 else "homoskedastic"

                results["white"] = {
                    "test_name": "White",
                    "statistic": round(lm_stat, decimals),
                    "pvalue": round(lm_pval, decimals),
                    "heteroskedastic": lm_pval < 0.05,
                    "interpretation": (
                        f"White test: $\\mathit{{p}}$={round(lm_pval, decimals)} "
                        f"({het_status})"
                    ),
                }
            except Exception as e:
                results["white"] = {"error": str(e)}

    if test_type in ["goldfeld_quandt", "all"]:
        if X_numeric is not None and y_pred is not None:
            try:
                # Goldfeld-Quandt needs to sort by a predictor variable
                # Sort by predicted values (or first predictor if X has multiple columns)
                if hasattr(X_numeric, "ndim"):
                    if X_numeric.ndim == 1:
                        sort_variable = X_numeric
                    else:
                        sort_variable = y_pred
                else:
                    sort_variable = y_pred

                sort_idx = np.argsort(sort_variable)
                sorted_resid = residuals[sort_idx]

                # Need X matrix for GQ test
                # Convert to numpy array first to handle DataFrame indexing issues
                X_array = np.asarray(X_numeric)

                if X_array.ndim == 1:
                    X_sorted = X_array[sort_idx].reshape(-1, 1)
                else:
                    X_sorted = X_array[sort_idx]

                # Add constant
                X_with_const = np.column_stack([np.ones(len(X_sorted)), X_sorted])

                # het_goldfeldquandt returns (F-stat, p-value, ordering)
                f_stat, f_pval, ordering = het_goldfeldquandt(
                    sorted_resid, X_with_const
                )

                het_status = "heteroskedastic" if f_pval < 0.05 else "homoskedastic"

                results["goldfeld_quandt"] = {
                    "test_name": "Goldfeld-Quandt",
                    "statistic": round(f_stat, decimals),
                    "pvalue": round(f_pval, decimals),
                    "heteroskedastic": f_pval < 0.05,
                    "interpretation": (
                        f"GQ test: $\\mathit{{p}}$={round(f_pval, decimals)} ({het_status})"
                    ),
                }
            except Exception as e:
                results["goldfeld_quandt"] = {"error": str(e)}

    if test_type in ["spearman", "all"]:
        if y_pred is not None:
            try:
                corr, pval = spearmanr(np.abs(residuals), y_pred)

                is_het = pval < 0.05 and abs(corr) > 0.1
                het_status = "heteroskedastic" if is_het else "homoskedastic"

                results["spearman"] = {
                    "test_name": "Spearman Correlation",
                    "statistic": round(corr, decimals),
                    "pvalue": round(pval, decimals),
                    "heteroskedastic": pval < 0.05 and abs(corr) > 0.1,
                    "interpretation": (
                        f"Spearman: rho={round(corr, decimals)}, "
                        f"$\\mathit{{p}}$={round(pval, decimals)} "
                        f"({het_status})"
                    ),
                }
            except Exception as e:
                results["spearman"] = {"error": str(e)}

    return results


def has_feature_importances(model):
    """Check if the model has a feature_importances_ attribute."""
    if model is None:
        return False
    if hasattr(model, "feature_importances_"):
        return True
    return isinstance(model, Pipeline) and hasattr(model[-1], "feature_importances_")


def get_feature_importances(model, feature_names, decimal_places=3):
    """Extract feature importances from model or pipeline."""
    from sklearn.pipeline import Pipeline

    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif isinstance(model, Pipeline) and hasattr(model[-1], "feature_importances_"):
        imps = model[-1].feature_importances_
    else:
        return {}
    return pd.Series(imps, index=feature_names).round(decimal_places).to_dict()


def get_coef_and_intercept(model):
    """
    Return (coef_, intercept_) from model or final pipeline step if
    present; else (None, None).
    """
    from sklearn.pipeline import Pipeline

    if model is None:
        return None, None
    if hasattr(model, "coef_"):
        return model.coef_, getattr(model, "intercept_", None)
    if isinstance(model, Pipeline) and hasattr(model[-1], "coef_"):
        return model[-1].coef_, getattr(model[-1], "intercept_", None)
    return None, None
