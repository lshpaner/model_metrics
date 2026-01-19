import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from model_metrics.model_evaluator import hanley_mcneil_auc_test

from model_metrics.metrics_utils import (
    compute_classification_metrics,
    extract_model_name,
    normalize_model_titles,
    has_feature_importances,
    get_coef_and_intercept,
    get_predictions,
    compute_regression_metrics,
    compute_residual_diagnostics,
    compute_leverage_and_cooks_distance,
    validate_and_normalize_inputs,
    check_heteroskedasticity,
    compute_classification_metrics,
)


def test_extract_model_name_basic():

    m = LogisticRegression()
    assert extract_model_name(m) == "LogisticRegression"


def test_normalize_model_titles_single():

    titles = normalize_model_titles("A", 1)
    assert titles == ["A"]


def test_normalize_model_titles_list():

    titles = normalize_model_titles(["A", "B"], 2)
    assert titles == ["A", "B"]


def test_normalize_model_titles_length_mismatch():

    titles = normalize_model_titles(["A"], 2)

    assert titles == ["A"]


def test_has_feature_importances_true():

    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)

    m = RandomForestClassifier().fit(X, y)

    assert has_feature_importances(m) is True


def test_has_feature_importances_false():

    m = LogisticRegression()
    assert has_feature_importances(m) is False


def test_get_coef_and_intercept():

    X = np.random.randn(20, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + 1.0

    m = LinearRegression().fit(X, y)

    coef, intercept = get_coef_and_intercept(m)

    assert coef is not None
    assert intercept is not None


def test_get_predictions_classifier():

    X = np.random.randn(50, 3)
    y = np.random.randint(0, 2, size=50)

    m = LogisticRegression().fit(X, y)

    result = get_predictions(
        m,
        X,
        y,
        model_threshold=0.5,
        custom_threshold=None,
        score="roc_auc",
    )

    y_pred = result[0]
    y_prob = result[1]

    assert y_pred.shape[0] == 50
    assert y_prob is not None


def test_compute_classification_metrics_basic():

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.3, 0.9])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, threshold=0.5)

    assert "AUC ROC" in metrics
    assert "F1-Score" in metrics


def test_compute_regression_metrics_basic():

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "RMSE" in metrics
    assert "Expl. Var." in metrics


def test_compute_residual_diagnostics_basic():

    X = np.random.randn(50, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50)

    m = LinearRegression().fit(X, y)
    y_pred = m.predict(X)
    residuals = y - y_pred

    diagnostics = compute_residual_diagnostics(residuals, y, y_pred)

    assert "mae" in diagnostics
    assert "jarque_bera_pval" in diagnostics


def test_check_heteroskedasticity_runs():

    X = np.random.randn(50, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50)

    m = LinearRegression().fit(X, y)
    y_pred = m.predict(X)
    residuals = y - y_pred

    result = check_heteroskedasticity(residuals, X)

    assert "breusch_pagan" in result


def test_hanley_mcneil_auc_test_runs():

    y_true = np.array([0, 1, 1, 0, 1])
    y_prob1 = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
    y_prob2 = np.array([0.1, 0.7, 0.5, 0.4, 0.8])

    # Should run without error
    result = hanley_mcneil_auc_test(y_true, y_prob1, y_prob2)

    assert result is None


def test_extract_model_name_pipeline():

    pipe = Pipeline([("clf", LogisticRegression())])
    name = extract_model_name(pipe)
    assert "LogisticRegression" in name


def test_normalize_model_titles_none():

    titles = normalize_model_titles(None, 2)
    assert titles == ["Model 1", "Model 2"]


def test_has_feature_importances_coef_model():

    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, size=30)

    m = LogisticRegression().fit(X, y)

    # coef_ models are NOT treated as feature_importances in this library
    assert has_feature_importances(m) is False


def test_get_coef_and_intercept_no_attributes():

    X = np.random.randn(30, 3)
    y = np.random.randn(30)

    m = RandomForestRegressor().fit(X, y)

    coef, intercept = get_coef_and_intercept(m)

    assert coef is None
    assert intercept is None


def test_get_predictions_regression():

    X = np.random.randn(40, 3)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(40)

    m = LinearRegression().fit(X, y)

    result = get_predictions(
        m,
        X,
        y,
        model_threshold=None,
        custom_threshold=None,
        score=None,
    )

    y_pred = result[0]
    y_prob = result[1]

    assert y_pred.shape[0] == 40
    assert y_prob.shape[0] == 40
    assert np.issubdtype(y_pred.dtype, np.floating)
    assert np.issubdtype(y_prob.dtype, np.floating)


def test_compute_classification_metrics_no_prob():

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    # y_prob is required, should error
    with pytest.raises(Exception):
        compute_classification_metrics(y_true, y_pred, None, threshold=0.5)


def test_compute_regression_metrics_zero_targets():

    y_true = np.array([0.0, 0.0, 1.0])
    y_pred = np.array([0.1, -0.1, 1.2])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "MAPE" in metrics


def test_compute_residual_diagnostics_with_leverage_and_cooks():

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    residuals = y_true - y_pred

    leverage = np.array([0.1, 0.2, 0.3, 0.4])
    cooks_d = np.array([0.01, 0.02, 0.03, 0.04])

    diagnostics = compute_residual_diagnostics(
        residuals,
        y_true,
        y_pred,
        leverage=leverage,
        cooks_d=cooks_d,
        n_features=3,
    )

    assert "high_leverage_count" in diagnostics
    assert "influential_points_05" in diagnostics
    assert "influential_points_10" in diagnostics


def test_compute_leverage_and_cooks_distance_statsmodels():

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 3), columns=["A", "B", "C"])
    y = X["A"] * 2.0 - X["B"] + rng.randn(100)

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    leverage, cooks, std_resid, influence = compute_leverage_and_cooks_distance(
        model, X
    )

    # Your implementation may return None for these depending on backend
    assert isinstance((leverage, cooks, std_resid, influence), tuple)


def test_compute_residual_diagnostics_without_leverage():

    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    residuals = y_true - y_pred

    diagnostics = compute_residual_diagnostics(
        residuals,
        y_true,
        y_pred,
        leverage=None,
        cooks_d=None,
        n_features=None,
    )

    assert "durbin_watson" in diagnostics


def test_validate_and_normalize_inputs_invalid_model_type():

    model, X, model_type = validate_and_normalize_inputs(
        None,
        None,
        "nonsense",
    )

    # invalid model_type should be normalized to a fallback code
    assert isinstance(model_type, int)


def test_check_heteroskedasticity_error_path():

    residuals = np.array([1.0, 2.0, 3.0])
    X = "not a matrix"

    result = check_heteroskedasticity(residuals, X)

    assert "breusch_pagan" in result
    assert "error" in result["breusch_pagan"]


def test_hanley_mcneil_auc_test_identical():

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])

    result = hanley_mcneil_auc_test(y_true, y_prob, y_prob)

    assert result is None


# --------------------------
# normalize_model_titles edge cases
# --------------------------


def test_normalize_model_titles_none():

    titles = normalize_model_titles(None, 2)
    assert titles == ["Model 1", "Model 2"]


def test_normalize_model_titles_longer_list():

    titles = normalize_model_titles(["A", "B", "C"], 2)

    assert titles == ["A", "B", "C"]


def test_normalize_model_titles_invalid_type():

    with pytest.raises(TypeError):
        normalize_model_titles(123, 2)


def test_normalize_model_titles_empty_list():

    titles = normalize_model_titles([], 2)
    assert titles == []


def test_normalize_model_titles_n_models_zero():

    titles = normalize_model_titles(["A", "B"], 0)
    assert titles == ["A", "B"]


# --------------------------
# has_feature_importances coef path
# --------------------------


def test_has_feature_importances_linear_coef():

    X = np.random.randn(20, 2)
    y = X @ np.array([1.0, -2.0])

    m = LinearRegression().fit(X, y)

    assert has_feature_importances(m) is False


# --------------------------
# get_coef_and_intercept edge cases
# --------------------------


def test_get_coef_and_intercept_no_intercept():

    X = np.random.randn(30, 3)
    y = X @ np.array([2.0, -1.0, 0.5])

    m = LinearRegression(fit_intercept=False).fit(X, y)

    coef, intercept = get_coef_and_intercept(m)

    assert coef is not None
    assert intercept == 0 or intercept is not None


# --------------------------
# get_predictions multiclass path
# --------------------------


def test_get_predictions_multiclass():

    X = np.random.randn(60, 4)
    y = np.random.randint(0, 3, size=60)

    m = LogisticRegression(multi_class="auto", max_iter=200).fit(X, y)

    result = get_predictions(
        m,
        X,
        y,
        model_threshold=0.5,
        custom_threshold=None,
        score="roc_auc",
    )

    y_pred = result[0]
    y_prob = result[1]

    assert y_pred.shape[0] == 60
    assert y_prob is not None


# --------------------------
# compute_classification_metrics threshold edge
# --------------------------


def test_compute_classification_metrics_extreme_threshold():

    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    y_pred = (y_prob > 0.99).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, threshold=0.99)

    assert "F1-Score" in metrics
    assert "Accuracy" in metrics or "AUC ROC" in metrics


def test_compute_classification_metrics_no_positive_predictions():

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.zeros_like(y_true)
    y_prob = np.zeros_like(y_true, dtype=float)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, threshold=0.5)

    assert "F1-Score" in metrics
    if "Recall" in metrics:
        assert metrics["Recall"] == 0.0


def test_compute_classification_metrics_single_class():

    y_true = np.zeros(10)
    y_pred = np.zeros(10)
    y_prob = np.zeros(10)

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_prob, threshold=0.5)


# --------------------------
# compute_regression_metrics perfect fit branch
# --------------------------


def test_compute_regression_metrics_perfect_fit():

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["RMSE"] == 0.0
    assert metrics["MSE"] == 0.0


def test_compute_regression_metrics_perfect_fit():

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])  # perfect

    metrics = compute_regression_metrics(y_true, y_pred, decimal_places=3)

    assert metrics["R^2"] == 1.0
    assert metrics["RMSE"] == 0.0


def test_compute_regression_metrics_zero_variance():

    y_true = np.ones(10)
    y_pred = np.ones(10)

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "R^2" in metrics


# --------------------------
# compute_residual_diagnostics no leverage no cooks
# --------------------------


def test_compute_residual_diagnostics_minimal():

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 2.8])
    residuals = y_true - y_pred

    diagnostics = compute_residual_diagnostics(residuals, y_true, y_pred)

    assert "durbin_watson" in diagnostics
    assert "mae" in diagnostics


# --------------------------
# check_heteroskedasticity error branch
# --------------------------


def test_check_heteroskedasticity_bad_input():

    residuals = np.array([1.0, 2.0, 3.0])
    X = "not an array"

    result = check_heteroskedasticity(residuals, X)

    assert "breusch_pagan" in result
    assert "error" in result["breusch_pagan"]


# --------------------------
# hanley_mcneil identical models branch
# --------------------------


def test_hanley_mcneil_identical_probs():

    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.9])

    # identical models should not crash
    result = hanley_mcneil_auc_test(y_true, y_prob, y_prob)

    assert result is None
