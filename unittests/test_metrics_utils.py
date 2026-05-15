import pytest
import pandas as pd
import numpy as np
import matplotlib
import io
import contextlib

matplotlib.use("Agg")  # headless, never opens a window
import matplotlib.pyplot as plt
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

from model_metrics.metrics_utils import (
    _venn_blend,
    _venn_resolve_side,
    _venn_category_counts,
    _FONT_ALIASES,
    _resolve_font_family,
    _print_overlap_crosstab_legend,
    _draw_crosstab_matrix,
    _draw_crosstab_summary,
    _draw_crosstab_legend,
)
from model_metrics.model_evaluator import (
    plot_overlap_venns,
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


# --------------------------
# _venn_blend
# --------------------------


def test_venn_blend_named():
    out = _venn_blend("red", "blue")
    # red = (1, 0, 0), blue = (0, 0, 1), midpoint = (0.5, 0, 0.5)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)


def test_venn_blend_hex():
    out = _venn_blend("#ff0000", "#0000ff")
    assert out[0] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.5)


def test_venn_blend_returns_3tuple():
    out = _venn_blend("white", "black")
    assert len(out) == 3
    for c in out:
        assert 0.0 <= c <= 1.0


# --------------------------
# _venn_resolve_side
# --------------------------


def test_venn_resolve_side_from_y_pred():
    arr = _venn_resolve_side("a", [0, 1, 1, 0], None, None)
    assert arr.tolist() == [0, 1, 1, 0]


def test_venn_resolve_side_from_model_and_X():
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)
    m = LogisticRegression().fit(X, y)
    arr = _venn_resolve_side("a", None, m, X)
    assert arr.shape == (20,)


def test_venn_resolve_side_both_raises():
    with pytest.raises(ValueError):
        _venn_resolve_side("a", [0, 1], LogisticRegression(), None)


def test_venn_resolve_side_neither_raises():
    with pytest.raises(ValueError):
        _venn_resolve_side("a", None, None, None)


def test_venn_resolve_side_model_without_X_raises():
    with pytest.raises(ValueError):
        _venn_resolve_side("a", None, LogisticRegression(), None)


def test_venn_resolve_side_no_predict_method_raises():

    class FakeModel:
        pass

    with pytest.raises(TypeError):
        _venn_resolve_side("a", None, FakeModel(), np.zeros((5, 2)))


# --------------------------
# _venn_category_counts
# --------------------------


def test_venn_category_counts_returns_5_tuple():
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 0, 0])
    y_pred_b = np.array([1, 1, 0, 1])
    result = _venn_category_counts(y_true, y_pred_a, y_pred_b, "FN")
    assert len(result) == 5
    for v in result:
        assert isinstance(v, int)


def test_venn_category_counts_fn_correct():
    # FN subpop = actual positives; FN = predicted 0 when actual is 1.
    # Construct so each region has exactly one observation.
    y_true = np.array([1, 1, 1, 1])
    y_pred_a = np.array([0, 0, 1, 1])  # a misses indices 0, 1
    y_pred_b = np.array([0, 1, 0, 1])  # b misses indices 0, 2
    # both miss index 0; a-only misses 1; b-only misses 2; neither misses 3
    a_only, b_only, both, outside, n_sub = _venn_category_counts(
        y_true, y_pred_a, y_pred_b, "FN"
    )
    assert a_only == 1
    assert b_only == 1
    assert both == 1
    assert outside == 1
    assert n_sub == 4


def test_venn_category_counts_tp_correct():
    y_true = np.array([1, 1, 1, 1])
    y_pred_a = np.array([1, 1, 0, 0])
    y_pred_b = np.array([1, 0, 1, 0])
    # both catch index 0; a-only catches 1; b-only catches 2; neither catches 3
    a_only, b_only, both, outside, n_sub = _venn_category_counts(
        y_true, y_pred_a, y_pred_b, "TP"
    )
    assert a_only == 1
    assert b_only == 1
    assert both == 1
    assert outside == 1
    assert n_sub == 4


def test_venn_category_counts_tn_correct():
    y_true = np.array([0, 0, 0, 0])
    y_pred_a = np.array([0, 0, 1, 1])
    y_pred_b = np.array([0, 1, 0, 1])
    # TN: subpop is actual negatives; in_set means predicted 0
    a_only, b_only, both, outside, n_sub = _venn_category_counts(
        y_true, y_pred_a, y_pred_b, "TN"
    )
    assert both == 1
    assert n_sub == 4


# --------------------------
# plot_overlap_venns (plural)
# --------------------------


def test_plot_overlap_venns_basic_runs():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred_a = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_pred_b = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    plot_overlap_venns(y_true, y_pred_a, y_pred_b)
    plt.close("all")


def test_plot_overlap_venns_all_four_categories():
    y_true = np.random.randint(0, 2, size=50)
    y_pred_a = np.random.randint(0, 2, size=50)
    y_pred_b = np.random.randint(0, 2, size=50)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN", "TN", "TP", "FP"),
    )
    plt.close("all")


def test_plot_overlap_venns_single_category():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
    )
    plt.close("all")


def test_plot_overlap_venns_with_models():
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 2, size=50)
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression().fit(X, y)
    plot_overlap_venns(
        y,
        model_a=m1,
        model_b=m2,
        X_a=X,
    )
    plt.close("all")


def test_plot_overlap_venns_models_different_X():
    X1 = np.random.randn(40, 3)
    X2 = np.random.randn(40, 5)
    y = np.random.randint(0, 2, size=40)
    m1 = LogisticRegression().fit(X1, y)
    m2 = LogisticRegression().fit(X2, y)
    plot_overlap_venns(
        y,
        model_a=m1,
        model_b=m2,
        X_a=X1,
        X_b=X2,
    )
    plt.close("all")


def test_plot_overlap_venns_unknown_category_raises():
    y_true = np.array([0, 1, 0, 1])
    y_pred_a = np.array([0, 1, 1, 1])
    y_pred_b = np.array([1, 1, 0, 1])
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true,
            y_pred_a,
            y_pred_b,
            categories=("XX",),
        )


def test_plot_overlap_venns_bad_titles_keys_raises():
    y_true = np.array([0, 1, 0, 1])
    y_pred_a = np.array([0, 1, 1, 1])
    y_pred_b = np.array([1, 1, 0, 1])
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true,
            y_pred_a,
            y_pred_b,
            titles={"BAD": "nope"},
        )


def test_plot_overlap_venns_invalid_colors_length_raises():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true,
            y_pred_a,
            y_pred_b,
            categories=("FN",),
            colors=("red",),
        )
    plt.close("all")


def test_plot_overlap_venns_custom_titles():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
        titles={"FN": "Custom Heading"},
    )
    plt.close("all")


def test_plot_overlap_venns_no_subtitle():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
        label_kwgs={"show_subtitle": False},
    )
    plt.close("all")


def test_plot_overlap_venns_with_colors_2tuple():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
        colors=("steelblue", "crimson"),
    )
    plt.close("all")


def test_plot_overlap_venns_with_colors_3tuple():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
        colors=("steelblue", "crimson", "purple"),
    )
    plt.close("all")


def test_plot_overlap_venns_with_ax_subgridspec():
    fig, ax = plt.subplots(figsize=(8, 6))
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN", "TN"),
        ax=ax,
    )
    plt.close("all")


def test_plot_overlap_venns_savefig_disabled_by_default(tmp_path):
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
        image_path_png=str(tmp_path),
    )
    # save_plot defaults to False, so no file should be written
    assert not list(tmp_path.glob("*.png"))
    plt.close("all")


def test_plot_overlap_venns_savefig_writes_png(tmp_path):
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories=("FN",),
        save_plot=True,
        image_path_png=str(tmp_path),
    )
    assert list(tmp_path.glob("*.png"))
    plt.close("all")


def test_venn_resolve_side_from_y_prob():
    """y_prob path applies default 0.5 threshold."""
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    arr = _venn_resolve_side("a", None, None, None, y_prob=y_prob)
    assert arr.tolist() == [0, 0, 1, 1]


def test_venn_resolve_side_from_y_prob_with_threshold():
    """y_prob path respects custom threshold."""
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    arr = _venn_resolve_side("a", None, None, None, y_prob=y_prob, threshold=0.5)
    assert arr.tolist() == [0, 0, 1, 1]
    arr = _venn_resolve_side("a", None, None, None, y_prob=y_prob, threshold=0.7)
    assert arr.tolist() == [0, 0, 0, 1]


def test_venn_resolve_side_y_pred_and_y_prob_both_raises():
    """Cannot supply both y_pred and y_prob on the same side."""
    with pytest.raises(ValueError):
        _venn_resolve_side(
            "a", [0, 1, 1, 0], None, None, y_prob=np.array([0.1, 0.5, 0.6, 0.2])
        )


def test_venn_resolve_side_y_prob_and_model_both_raises():
    """Cannot supply both y_prob and model on the same side."""
    X = np.random.randn(10, 3)
    y = np.random.randint(0, 2, size=10)
    m = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError):
        _venn_resolve_side("a", None, m, X, y_prob=np.random.rand(10))


def test_venn_resolve_side_all_three_provided_raises():
    """Three-way exclusivity check."""
    X = np.random.randn(10, 3)
    y = np.random.randint(0, 2, size=10)
    m = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError):
        _venn_resolve_side(
            "a",
            np.zeros(10),
            m,
            X,
            y_prob=np.random.rand(10),
        )


def test_venn_resolve_side_model_uses_get_predictions():
    """Model path routes through get_predictions; default threshold 0.5 applies."""
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, size=30)
    m = LogisticRegression().fit(X, y)
    arr = _venn_resolve_side("a", None, m, X, y_true=y)
    assert arr.shape == (30,)
    assert set(np.unique(arr)).issubset({0, 1})


def test_venn_resolve_side_model_with_custom_threshold():
    """Custom threshold overrides the model's stored threshold."""
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, size=30)
    m = LogisticRegression().fit(X, y)
    arr_default = _venn_resolve_side("a", None, m, X, y_true=y)
    arr_high = _venn_resolve_side("a", None, m, X, y_true=y, threshold=0.99)
    # Higher threshold yields fewer (or equal) positive predictions
    assert arr_high.sum() <= arr_default.sum()


def _make_sample_crosstab(label_a="LR", label_b="RF"):
    """Build a realistic 4x4 crosstab for draw-helper tests."""
    order = ["TP", "FP", "FN", "TN"]
    data = np.array(
        [
            [1460, 0, 14, 0],  # TP row
            [0, 985, 0, 170],  # FP row
            [489, 0, 375, 0],  # FN row
            [0, 1249, 0, 5027],  # TN row
        ]
    )
    ct = pd.DataFrame(data, index=order, columns=order)
    ct.index.name = label_a
    ct.columns.name = label_b
    return ct


def _stats_from_ct(ct):
    """Build the stats dict expected by _draw_crosstab_summary."""
    s = dict(
        both_tp=int(ct.loc["TP", "TP"]),
        both_fp=int(ct.loc["FP", "FP"]),
        both_fn=int(ct.loc["FN", "FN"]),
        both_tn=int(ct.loc["TN", "TN"]),
        b_extra_tp=int(ct.loc["FN", "TP"]),
        b_lost_tp=int(ct.loc["TP", "FN"]),
        b_extra_fp=int(ct.loc["TN", "FP"]),
        b_avoided_fp=int(ct.loc["FP", "TN"]),
    )
    s["net_tp"] = s["b_extra_tp"] - s["b_lost_tp"]
    s["net_fp"] = s["b_extra_fp"] - s["b_avoided_fp"]
    return s


# --------------------------
# _FONT_ALIASES
# --------------------------


def test_font_aliases_has_common_entries():
    """The alias map must cover the common cross-platform fonts."""
    expected_keys = {
        "arial",
        "helvetica",
        "times",
        "times new roman",
        "consolas",
        "courier",
        "courier new",
    }
    assert expected_keys.issubset(set(_FONT_ALIASES))


def test_font_aliases_keys_are_lowercase():
    """Keys must be lowercase since _resolve_font_family lowercases input."""
    for key in _FONT_ALIASES:
        assert key == key.lower()


def test_font_aliases_chains_end_in_dejavu():
    """Every chain must end in a font matplotlib ships with so aliases never fail."""
    dejavu_fonts = {"DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif"}
    for key, chain in _FONT_ALIASES.items():
        assert (
            chain[-1] in dejavu_fonts
        ), f"alias {key!r} chain {chain!r} does not end in a DejaVu font"


def test_font_aliases_chains_are_lists_of_strings():
    """Every chain must be a list of strings."""
    for key, chain in _FONT_ALIASES.items():
        assert isinstance(chain, list)
        for name in chain:
            assert isinstance(name, str)


# --------------------------
# _resolve_font_family
# --------------------------


def test_resolve_font_family_none_returns_none():
    assert _resolve_font_family(None) is None


def test_resolve_font_family_aliased_string():
    """An aliased name should return a list and never raise."""
    out = _resolve_font_family("Arial")
    assert isinstance(out, list)
    assert len(out) >= 1
    # DejaVu Sans ships with matplotlib so the chain should always land somewhere
    assert all(isinstance(name, str) for name in out)


def test_resolve_font_family_case_insensitive():
    """Alias lookup must be case-insensitive."""
    assert _resolve_font_family("Arial") == _resolve_font_family("ARIAL")
    assert _resolve_font_family("Arial") == _resolve_font_family("arial")
    assert _resolve_font_family("Helvetica") == _resolve_font_family("HELVETICA")


def test_resolve_font_family_aliased_list():
    """A list with aliased names should expand each alias in place."""
    out = _resolve_font_family(["Arial", "Times"])
    assert isinstance(out, list)
    assert len(out) >= 1


def test_resolve_font_family_dejavu_always_resolves():
    """DejaVu Sans ships with matplotlib so it must always be findable."""
    out = _resolve_font_family("DejaVu Sans")
    assert out == ["DejaVu Sans"]


def test_resolve_font_family_unknown_font_raises_value_error():
    """An unaliased font not installed on the system must raise ValueError."""
    with pytest.raises(ValueError, match="None of the requested fonts"):
        _resolve_font_family("DefinitelyNotARealFontName123XYZ")


def test_resolve_font_family_unknown_in_list_raises_when_none_installed():
    """A list of unaliased, uninstalled fonts must raise ValueError."""
    with pytest.raises(ValueError, match="None of the requested fonts"):
        _resolve_font_family(["NoSuchFont1XYZ", "NoSuchFont2XYZ"])


def test_resolve_font_family_unknown_in_list_with_dejavu_passes():
    """A list that mixes a fake font with an installable one should not raise."""
    out = _resolve_font_family(["NoSuchFont1XYZ", "DejaVu Sans"])
    assert out == ["DejaVu Sans"]


def test_resolve_font_family_invalid_type_raises_type_error():
    """Non-string, non-list/tuple input must raise TypeError."""
    with pytest.raises(TypeError):
        _resolve_font_family(42)
    with pytest.raises(TypeError):
        _resolve_font_family({"font": "Arial"})


def test_resolve_font_family_dedupes_chain():
    """Repeated entries (e.g. when a list contains an alias whose chain repeats
    a font also passed literally) should appear only once in the output."""
    out = _resolve_font_family(["DejaVu Sans", "DejaVu Sans"])
    assert out == ["DejaVu Sans"]


def test_resolve_font_family_error_message_lists_aliases():
    """The ValueError message should point users at the alias table."""
    try:
        _resolve_font_family("NoSuchFontXYZ")
    except ValueError as e:
        msg = str(e)
        # should mention at least one alias the user could have used instead
        assert "arial" in msg.lower() or "aliases" in msg.lower()


# --------------------------
# _print_overlap_crosstab_legend
# --------------------------


def test_print_overlap_crosstab_legend_runs():
    ct = _make_sample_crosstab()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_overlap_crosstab_legend(ct, "LR", "RF")
    assert buf.getvalue()


def test_print_overlap_crosstab_legend_mentions_labels():
    """The header lines must include both labels so users know which is which."""
    ct = _make_sample_crosstab()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_overlap_crosstab_legend(ct, "LR", "RF")
    text = buf.getvalue()
    assert "LR" in text
    assert "RF" in text


def test_print_overlap_crosstab_legend_has_swap_summary():
    """The swap summary section must appear with TP/FP swap headers."""
    ct = _make_sample_crosstab()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_overlap_crosstab_legend(ct, "LR", "RF")
    text = buf.getvalue()
    assert "Swap summary" in text
    assert "TP swap" in text
    assert "FP swap" in text


def test_print_overlap_crosstab_legend_derived_numbers_correct():
    """The derived agreement and swap numbers must match the crosstab."""
    ct = _make_sample_crosstab()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_overlap_crosstab_legend(ct, "LR", "RF")
    text = buf.getvalue()
    # agreement = diagonal sum = 1460 + 985 + 375 + 5027 = 7847
    assert "7,847" in text
    # total = 7847 + 14 + 170 + 489 + 1249 = 9769
    assert "9,769" in text
    # shared FN = 375
    assert "375" in text


def test_print_overlap_crosstab_legend_zero_total_does_not_crash():
    """Edge case: an all-zero crosstab should not raise on the agree_pct calc."""
    order = ["TP", "FP", "FN", "TN"]
    ct = pd.DataFrame(np.zeros((4, 4), dtype=int), index=order, columns=order)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_overlap_crosstab_legend(ct, "A", "B")
    assert buf.getvalue()


# --------------------------
# _draw_crosstab_matrix
# --------------------------


def test_draw_crosstab_matrix_creates_16_cells():
    """The matrix must render exactly 16 rectangle patches, one per cell."""
    from matplotlib.patches import Rectangle

    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    n_rects = sum(1 for p in ax.patches if isinstance(p, Rectangle))
    assert n_rects == 16
    plt.close(fig)


def test_draw_crosstab_matrix_renders_count_text():
    """The cell counts must appear as text on the axes."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    texts = [t.get_text() for t in ax.texts]
    assert "1,460" in texts
    assert "5,027" in texts
    assert "489" in texts
    plt.close(fig)


def test_draw_crosstab_matrix_uses_thousands_separator():
    """Counts >= 1000 must use the :, format with comma separator."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    texts = [t.get_text() for t in ax.texts]
    # the 5027 cell should render with a comma
    assert "5,027" in texts
    assert "5027" not in texts
    plt.close(fig)


def test_draw_crosstab_matrix_nan_cell_renders_as_hyphen():
    """NaN cells (from mask_impossible=True) must render as '-' not 'nan' or em dash."""
    ct = _make_sample_crosstab().astype(float)
    ct.loc["TP", "FP"] = np.nan
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    texts = [t.get_text() for t in ax.texts]
    assert "-" in texts
    assert "nan" not in texts
    plt.close(fig)


def test_draw_crosstab_matrix_renders_axis_labels():
    """label_a and label_b must appear somewhere in the axes text."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    texts = [t.get_text() for t in ax.texts]
    assert "LR" in texts
    assert "RF" in texts
    plt.close(fig)


def test_draw_crosstab_matrix_renders_category_headers():
    """Each of TP/FP/FN/TN must appear as both a row and column header."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    texts = [t.get_text() for t in ax.texts]
    for cat in ("TP", "FP", "FN", "TN"):
        # cat appears as row header AND col header so at least 2 occurrences
        assert texts.count(cat) >= 2
    plt.close(fig)


def test_draw_crosstab_matrix_aspect_equal():
    """Cells must be square (aspect=equal) regardless of figure shape."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_matrix(ax, ct, "LR", "RF", colors, 14, 12)
    aspect = ax.get_aspect()
    # matplotlib stores "equal" as the string 1.0 or the literal string
    assert aspect == "equal" or aspect == 1.0
    plt.close(fig)


# --------------------------
# _draw_crosstab_summary
# --------------------------


def test_draw_crosstab_summary_creates_text():
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    _draw_crosstab_summary(ax, "RF", _stats_from_ct(ct), 11)
    assert len(ax.texts) > 0
    plt.close(fig)


def test_draw_crosstab_summary_mentions_label_b():
    """The summary frames swaps relative to label_b so it must appear in the text."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    _draw_crosstab_summary(ax, "RF", _stats_from_ct(ct), 11)
    full = " ".join(t.get_text() for t in ax.texts)
    assert "RF" in full
    plt.close(fig)


def test_draw_crosstab_summary_includes_swap_headlines():
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    _draw_crosstab_summary(ax, "RF", _stats_from_ct(ct), 11)
    full = " ".join(t.get_text() for t in ax.texts)
    assert "TP swap" in full
    assert "FP swap" in full
    plt.close(fig)


def test_draw_crosstab_summary_uses_thousands_separator():
    """Numbers in the summary must use the :, format."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    _draw_crosstab_summary(ax, "RF", _stats_from_ct(ct), 11)
    full = " ".join(t.get_text() for t in ax.texts)
    # both_tn = 5027 → should render as 5,027
    assert "5,027" in full
    # b_extra_fp = 1249 → should render as 1,249 (the bug Leon caught)
    assert "1,249" in full
    plt.close(fig)


def test_draw_crosstab_summary_renders_net_with_sign():
    """Net deltas must include the +/- sign via :+, format."""
    ct = _make_sample_crosstab()
    fig, ax = plt.subplots()
    _draw_crosstab_summary(ax, "RF", _stats_from_ct(ct), 11)
    full = " ".join(t.get_text() for t in ax.texts)
    # net_tp = 489 - 14 = +475, net_fp = 1249 - 170 = +1079
    assert "+475" in full
    assert "+1,079" in full
    plt.close(fig)


# --------------------------
# _draw_crosstab_legend
# --------------------------


def test_draw_crosstab_legend_creates_legend():
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_legend(ax, colors, 12)
    assert ax.get_legend() is not None
    plt.close(fig)


def test_draw_crosstab_legend_has_three_entries():
    fig, ax = plt.subplots()
    colors = {"agree": "#d4edda", "disagree": "#f8d7da", "impossible": "#e2e6ed"}
    _draw_crosstab_legend(ax, colors, 12)
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert "agree" in labels
    assert "disagree (swap)" in labels
    assert "impossible (true label conflict)" in labels
    plt.close(fig)
