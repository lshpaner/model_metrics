# model_utils.py
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, jarque_bera, norm
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
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
)

from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt,
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


def hanley_mcneil_auc_test(
    y_true,
    y_scores_1,
    y_scores_2,
    model_names=None,
    verbose=True,
    return_values=False,
    decimal_places=4,
):
    """
    Hanley & McNeil (1982) large-sample z-test for difference in correlated AUCs.
    Returns (auc1, auc2, p_value).

    Parameters
    ----------
    y_true : array-like
        True binary class labels.
    y_scores_1 : array-like
        Predicted probabilities or decision scores from the first model.
    y_scores_2 : array-like
        Predicted probabilities or decision scores from the second model.
    model_names : list or tuple of str, optional
        Optional names for the models, used for printed output.
        Defaults to ("Model 1", "Model 2") if not provided.
    verbose : bool, default=True
        If True, prints a formatted summary of the comparison, including AUCs
        and the computed p-value.
    return_values : bool, default=False
        If True, returns the tuple (auc1, auc2, p_value) instead of only
        printing the results. This is useful for programmatic access or when
        integrating into other functions such as `show_roc_curve()`.

    Returns
    -------
    tuple of floats, optional
        (auc1, auc2, p_value) — only returned if `return_values=True`.
    """
    auc1 = roc_auc_score(y_true, y_scores_1)
    auc2 = roc_auc_score(y_true, y_scores_2)
    n1 = np.sum(y_true)
    n2 = len(y_true) - n1
    q1 = auc1 / (2 - auc1)
    q2 = 2 * auc1**2 / (1 + auc1)
    se = np.sqrt(
        (auc1 * (1 - auc1) + (n1 - 1) * (q1 - auc1**2) + (n2 - 1) * (q2 - auc1**2))
        / (n1 * n2)
    )
    z = (auc1 - auc2) / se
    p = 2 * (1 - norm.cdf(abs(z)))

    if model_names is None:
        model_names = ("Model 1", "Model 2")

    if verbose:
        print(
            f"\nHanley & McNeil AUC Comparison (Approximation of DeLong's Test):\n"
            f"  {model_names[0]} AUC = {auc1:.{decimal_places}f}\n"
            f"  {model_names[1]} AUC = {auc2:.{decimal_places}f}\n"
            f"  p-value = {p:.{decimal_places}f}\n"
        )

    if return_values:
        return auc1, auc2, p


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
        - 'heteroskedastic': Boolean (True if heteroskedasticity detected at α=0.05)
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
                        f"Spearman: ρ={round(corr, decimals)}, "
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
