import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_metrics.model_evaluator import show_cumulative_feature_performance
from model_metrics.feature_selection_utils import _retained_fractions
from model_metrics.metrics_utils import _resolve_task, _validate_metrics_for_task


@pytest.fixture(autouse=True)
def _mpl_hygiene():
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.close("all")
        plt.show = orig_show


# -----------------------------------------------------------------------------------
# Lightweight NaN-tolerant stubs. These avoid fitting any real estimator, so the
# tests are fast, deterministic, and free of the OpenMP threading issues that make
# HistGradientBoosting hang in some WSL/conda environments. Predictions ignore NaN
# (via nanmean), which is exactly what the ablation needs.
# -----------------------------------------------------------------------------------
def _nan_score(X):
    Xv = np.asarray(X, dtype=float)
    return np.nan_to_num(np.nanmean(Xv, axis=1), nan=0.0)


class _StubEstimator:
    def __init__(self, n_features):
        self.feature_importances_ = np.linspace(1.0, 0.1, n_features)


class _Inner:
    def __init__(self, est):
        self.estimator = est


class _StubClfWrapper:
    def __init__(self, names):
        self.estimator = _Inner(_StubEstimator(len(names)))
        self._names = names

    def get_feature_names(self):
        return self._names

    def predict_proba(self, X):
        p1 = 1.0 / (1.0 + np.exp(-_nan_score(X)))
        return np.column_stack([1.0 - p1, p1])


class _StubRegressor:
    def __init__(self, names):
        self.feature_names_in_ = np.array(names)
        self.feature_importances_ = np.linspace(1.0, 0.1, len(names))

    def predict(self, X):
        return _nan_score(X)


@pytest.fixture
def clf_bundle():
    rng = np.random.RandomState(0)
    cols = [f"c{i}" for i in range(6)]
    X = pd.DataFrame(rng.randn(300, 6), columns=cols)
    y = (X["c0"] + X["c1"] + rng.randn(300) * 0.4 > 0).astype(int).values
    return _StubClfWrapper(cols), X, y


@pytest.fixture
def reg_bundle():
    rng = np.random.RandomState(0)
    cols = [f"r{i}" for i in range(6)]
    X = pd.DataFrame(rng.randn(300, 6), columns=cols)
    y = (X["r0"] * 2 + X["r1"] + rng.randn(300) * 0.4).values
    return _StubRegressor(cols), X, y


def test_requires_y(clf_bundle):
    model, X, _ = clf_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(model=model, X=X, y=None)


def test_both_model_and_yprobs_raises(clf_bundle):
    model, X, y = clf_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(model=model, X=X, y=y, y_probs={1: y})


def test_neither_model_nor_yprobs_raises(clf_bundle):
    _, _, y = clf_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(y=y)


def test_model_mode_returns_scores_df(clf_bundle):
    model, X, y = clf_bundle
    df = show_cumulative_feature_performance(model, X, y, return_df=True)
    assert "n_features" in df.columns and len(df) >= 1


def test_return_retention_df(clf_bundle):
    model, X, y = clf_bundle
    ret = show_cumulative_feature_performance(model, X, y, return_retention=True)
    assert ret.index.name == "retain_level"


def test_return_both_is_dict(clf_bundle):
    model, X, y = clf_bundle
    out = show_cumulative_feature_performance(
        model, X, y, return_df=True, return_retention=True
    )
    assert set(out.keys()) == {"scores", "retention"}


def test_return_features_is_list(clf_bundle):
    model, X, y = clf_bundle
    feats = show_cumulative_feature_performance(model, X, y, return_features=True)
    assert isinstance(feats, list) and all(isinstance(f, str) for f in feats)


def test_return_features_retain_subset(clf_bundle):
    model, X, y = clf_bundle
    full = show_cumulative_feature_performance(model, X, y, return_features=True)
    top = show_cumulative_feature_performance(
        model, X, y, retain_threshold=0.98, return_features=True
    )
    assert set(top) <= set(full) and len(top) >= 1


def test_return_features_with_other_flags_is_dict(clf_bundle):
    model, X, y = clf_bundle
    out = show_cumulative_feature_performance(
        model, X, y, return_df=True, return_retention=True, return_features=True
    )
    assert set(out.keys()) == {"scores", "retention", "features"}


def test_return_features_unavailable_in_yprobs_mode(clf_bundle):
    _, _, y = clf_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(
            y=y, y_probs={1: y, 2: y}, return_features=True
        )


def test_yprobs_mode_runs(clf_bundle):
    _, _, y = clf_bundle
    yprobs = {k: np.random.RandomState(k).rand(len(y)) for k in range(1, 5)}
    df = show_cumulative_feature_performance(y=y, y_probs=yprobs, return_df=True)
    assert len(df) == 4


def test_yprobs_must_be_dict(clf_bundle):
    _, _, y = clf_bundle
    with pytest.raises(TypeError):
        show_cumulative_feature_performance(y=y, y_probs=[0.1, 0.2, 0.3])


def test_task_classifier_is_classification(clf_bundle):
    model, _, _ = clf_bundle
    assert _resolve_task("auto", model, None) == "classification"


def test_task_regressor_is_regression(reg_bundle):
    model, _, _ = reg_bundle
    assert _resolve_task("auto", model, None) == "regression"


def test_family_reg_metric_on_classifier_raises(clf_bundle):
    model, X, y = clf_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(model, X, y, metrics=["roc_auc", "mae"])


def test_family_clf_metric_on_regressor_raises(reg_bundle):
    model, X, y = reg_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(model, X, y, metrics=["r2", "roc_auc"])


def test_yprobs_mixed_family_raises(clf_bundle):
    _, _, y = clf_bundle
    yprobs = {k: np.random.RandomState(k).rand(len(y)) for k in range(1, 4)}
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(
            y=y, y_probs=yprobs, metrics=["roc_auc", "mae"]
        )


def test_validate_metrics_direct():
    with pytest.raises(ValueError):
        _validate_metrics_for_task(["mae"], "classification")
    with pytest.raises(ValueError):
        _validate_metrics_for_task(["roc_auc"], "regression")
    _validate_metrics_for_task(["roc_auc", "precision"], "classification")
    _validate_metrics_for_task(["r2", "rmse"], "regression")


def test_label_metrics_run(clf_bundle):
    model, X, y = clf_bundle
    df = show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "precision", "recall", "f1", "accuracy"],
        return_df=True,
    )
    for col in ["ROC AUC", "Precision", "Recall", "F1", "Accuracy"]:
        assert col in df.columns


def test_regression_defaults_run(reg_bundle):
    model, X, y = reg_bundle
    df = show_cumulative_feature_performance(model, X, y, return_df=True)
    assert any("R" in c for c in df.columns)


def test_display_absolute_and_gains(clf_bundle):
    model, X, y = clf_bundle
    show_cumulative_feature_performance(model, X, y, display="absolute")
    show_cumulative_feature_performance(model, X, y, display="cumulative_gains")


def test_display_invalid_raises(clf_bundle):
    model, X, y = clf_bundle
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(model, X, y, display="bogus")


def test_x_as_percent_and_smooth(clf_bundle):
    model, X, y = clf_bundle
    show_cumulative_feature_performance(model, X, y, x_as_percent=True, smooth=True)


def test_retain_threshold_runs(clf_bundle):
    model, X, y = clf_bundle
    show_cumulative_feature_performance(model, X, y, retain_threshold=0.95)


def test_saves_via_arbitrary_extension(tmp_path, clf_bundle):
    model, X, y = clf_bundle
    show_cumulative_feature_performance(
        model, X, y, image_filename=str(tmp_path / "perf.pdf")
    )
    assert (tmp_path / "perf.pdf").exists()


def test_accepts_ax(clf_bundle):
    model, X, y = clf_bundle
    fig, ax = plt.subplots()
    show_cumulative_feature_performance(model, X, y, ax=ax)


def test_retained_fractions_reach_one_at_full():
    results = pd.DataFrame({"n_features": [1, 2, 3], "ROC AUC": [0.7, 0.85, 0.95]})
    resolved = [("ROC AUC", None, True, "score")]
    frac = _retained_fractions(results, resolved)
    assert frac["ROC AUC"].iloc[-1] == pytest.approx(1.0)


def test_retained_fractions_inverts_lower_is_better():
    results = pd.DataFrame({"n_features": [1, 2], "Brier": [0.2, 0.1]})
    resolved = [("Brier", None, False, "score")]
    frac = _retained_fractions(results, resolved)
    assert frac["Brier"].iloc[0] == pytest.approx(0.5)
    assert frac["Brier"].iloc[-1] == pytest.approx(1.0)
