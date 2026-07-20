import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_metrics.model_evaluator import show_cumulative_feature_performance
from model_metrics.feature_selection_utils import _retained_fractions
from model_metrics.metrics_utils import _resolve_task, _validate_metrics_for_task
from model_metrics.feature_selection_utils import _resolve_feature_groups
from sklearn.metrics import roc_auc_score


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


# --------------------------------------------------------------------------- #
# order= as a subset (regression coverage)
# --------------------------------------------------------------------------- #
def test_subset_order_holds_out_unlisted_features(flat_clf):
    """A partial `order` must blank everything it does not name.

    Before the fix, unlisted features stayed in play, so the final row was the
    full-model score wearing a top-k label.
    """
    model, X, y = flat_clf
    full = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], return_df=True
    )
    sub = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], order=["f0", "f1"], return_df=True
    )
    assert len(sub) == 2
    assert sub["ROC AUC"].iloc[-1] != pytest.approx(full["ROC AUC"].iloc[-1])


def test_subset_order_matches_manual_blanking(flat_clf):
    model, X, y = flat_clf
    keep = ["f0", "f1", "f2"]
    sub = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], order=keep, return_df=True
    )
    Xm = X.copy()
    for c in X.columns:
        if c not in keep:
            Xm[c] = np.nan
    manual = roc_auc_score(y, model.predict_proba(Xm)[:, 1])
    assert sub["ROC AUC"].iloc[-1] == pytest.approx(manual)


def test_subset_order_warns_about_held_out(flat_clf, capsys):
    model, X, y = flat_clf
    show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], order=["f0", "f1"]
    )
    assert "not in the ranking" in capsys.readouterr().out


def test_full_order_is_unchanged(flat_clf):
    """A complete `order` (just a reordering) must behave as before."""
    model, X, y = flat_clf
    default = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], return_df=True
    )
    full_order = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], order=list(X.columns), return_df=True
    )
    assert len(default) == len(full_order)
    assert default["ROC AUC"].iloc[-1] == pytest.approx(full_order["ROC AUC"].iloc[-1])


# --------------------------------------------------------------------------- #
# grouped sweeps
# --------------------------------------------------------------------------- #
def test_grouped_sweep_counts_variables_and_blanks_together(encoded_clf):
    model, Xt, y, ct = encoded_clf
    res = show_cumulative_feature_performance(
        model,
        Xt,
        y,
        metrics=["roc_auc"],
        feature_groups="auto",
        column_transformer=ct,
        return_df=True,
    )
    assert res["n_features"].max() == 3  # variables, not the 5 columns
    assert res.iloc[-1]["features"] == ["CKD Stage", "age", "egfr"] or set(
        res.iloc[-1]["features"]
    ) == {"age", "egfr", "CKD Stage"}

    # a grouped step must match manual blanking of all member columns
    mapping = _resolve_feature_groups("auto", list(Xt.columns), ct)
    row = res.iloc[1]
    keep_cols = {c for c in Xt.columns if mapping.get(c, c) in set(row["features"])}
    Xm = Xt.copy()
    for c in Xt.columns:
        if c not in keep_cols:
            Xm[c] = np.nan
    assert roc_auc_score(y, model.predict_proba(Xm)[:, 1]) == pytest.approx(
        row["ROC AUC"]
    )


# --------------------------------------------------------------------------- #
# include_retained_pct and the features column
# --------------------------------------------------------------------------- #
def test_features_column_is_ranked_prefix_and_last(flat_clf):
    model, X, y = flat_clf
    res = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], return_df=True
    )
    assert list(res.columns)[-1] == "features"
    order = res.iloc[-1]["features"]
    for _, row in res.iterrows():
        assert row["features"] == order[: row["n_features"]]


def test_features_column_still_last_with_pct(flat_clf):
    model, X, y = flat_clf
    res = show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "brier"],
        return_df=True,
        include_retained_pct=True,
    )
    assert list(res.columns)[-1] == "features"


def test_retained_pct_default_off(flat_clf):
    model, X, y = flat_clf
    res = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], return_df=True
    )
    assert not any("% of full" in c for c in res.columns)


def test_retained_pct_higher_is_better_ratio(flat_clf):
    model, X, y = flat_clf
    res = show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], return_df=True, include_retained_pct=True
    )
    full = res["ROC AUC"].iloc[-1]
    assert np.allclose(res["ROC AUC (% of full)"], res["ROC AUC"] / full * 100)
    assert res["ROC AUC (% of full)"].iloc[-1] == pytest.approx(100.0)


def test_retained_pct_lower_is_better_is_inverted(flat_clf):
    model, X, y = flat_clf
    res = show_cumulative_feature_performance(
        model, X, y, metrics=["brier"], return_df=True, include_retained_pct=True
    )
    b = "Brier Score (lower is better)"
    full = res[b].iloc[-1]
    assert np.allclose(res[f"{b} (% of full)"], full / res[b] * 100)
    assert res[f"{b} (% of full)"].iloc[-1] == pytest.approx(100.0)


def test_no_features_column_in_yprobs_mode(flat_clf):
    _, _, y = flat_clf
    yp = {k: np.random.RandomState(k).rand(len(y)) for k in range(1, 4)}
    res = show_cumulative_feature_performance(
        y=y, y_probs=yp, metrics=["roc_auc"], return_df=True, include_retained_pct=True
    )
    assert "features" not in res.columns
    assert "ROC AUC (% of full)" in res.columns


# --------------------------------------------------------------------------- #
# retain dict
# --------------------------------------------------------------------------- #
def test_retain_dict_equals_separate_args(flat_clf):
    model, X, y = flat_clf
    a = show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "average_precision"],
        retain={"average_precision": 0.95},
        return_features=True,
    )
    b = show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "average_precision"],
        retain_metric="average_precision",
        retain_threshold=0.95,
        return_features=True,
    )
    assert a == b


def test_retain_with_separate_args_raises(flat_clf):
    model, X, y = flat_clf
    with pytest.raises(ValueError, match="not both"):
        show_cumulative_feature_performance(
            model,
            X,
            y,
            metrics=["roc_auc"],
            retain={"roc_auc": 0.95},
            retain_threshold=0.9,
        )
    with pytest.raises(ValueError, match="not both"):
        show_cumulative_feature_performance(
            model,
            X,
            y,
            metrics=["roc_auc"],
            retain={"roc_auc": 0.95},
            retain_metric="roc_auc",
        )


def test_retain_must_be_nonempty_dict(flat_clf):
    model, X, y = flat_clf
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(
            model, X, y, metrics=["roc_auc"], retain=0.95
        )
    with pytest.raises(ValueError):
        show_cumulative_feature_performance(model, X, y, metrics=["roc_auc"], retain={})


def test_retain_single_metric_only(flat_clf):
    model, X, y = flat_clf
    with pytest.raises(ValueError, match="single metric"):
        show_cumulative_feature_performance(
            model,
            X,
            y,
            metrics=["roc_auc", "average_precision"],
            retain={"roc_auc": 0.9, "average_precision": 0.9},
        )


def test_unknown_retain_metric_raises_with_input_keys(flat_clf):
    model, X, y = flat_clf
    with pytest.raises(ValueError) as exc:
        show_cumulative_feature_performance(
            model, X, y, metrics=["roc_auc", "brier"], retain={"average_precision": 0.9}
        )
    msg = str(exc.value)
    assert "'roc_auc'" in msg and "'brier'" in msg
    assert "lower is better" not in msg  # display labels must not leak


# --------------------------------------------------------------------------- #
# per-metric styling
# --------------------------------------------------------------------------- #
def _markers(fig):
    return [
        ln.get_marker()
        for ax in fig.axes
        for ln in ax.get_lines()
        if ln.get_marker() not in (None, "None")
    ]


def test_default_markers_are_circles(flat_clf):
    model, X, y = flat_clf
    show_cumulative_feature_performance(model, X, y, metrics=["roc_auc"])
    assert set(_markers(plt.gcf())) == {"o"}


def test_flat_marker_kwgs_applies_to_all(flat_clf):
    model, X, y = flat_clf
    show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "average_precision"],
        marker_kwgs={"marker": "^", "markersize": 9},
    )
    assert set(_markers(plt.gcf())) == {"^"}


def test_per_metric_marker_kwgs(flat_clf):
    model, X, y = flat_clf
    show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "average_precision"],
        marker_kwgs={"roc_auc": {"marker": "^"}, "average_precision": {"marker": "s"}},
    )
    assert set(_markers(plt.gcf())) == {"^", "s"}


def test_per_metric_curve_colors(flat_clf):
    import matplotlib.colors as mc

    model, X, y = flat_clf
    show_cumulative_feature_performance(
        model,
        X,
        y,
        metrics=["roc_auc", "average_precision"],
        curve_kwgs={
            "roc_auc": {"color": "navy"},
            "average_precision": {"color": "orange"},
        },
    )
    colors = {
        mc.to_hex(ln.get_color()) for ax in plt.gcf().axes for ln in ax.get_lines()
    }
    assert mc.to_hex("navy") in colors and mc.to_hex("orange") in colors


def test_markers_honoured_in_smooth_path(flat_clf):
    """The smoothed branch used to hardcode 'o', ignoring marker_kwgs."""
    model, X, y = flat_clf
    show_cumulative_feature_performance(
        model, X, y, metrics=["roc_auc"], smooth=True, marker_kwgs={"marker": "D"}
    )
    assert set(_markers(plt.gcf())) == {"D"}


def test_unknown_per_metric_style_key_raises(flat_clf):
    model, X, y = flat_clf
    with pytest.raises(ValueError, match="unknown metric keys"):
        show_cumulative_feature_performance(
            model, X, y, metrics=["roc_auc"], marker_kwgs={"nope": {"marker": "^"}}
        )
