import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_metrics.metrics_utils import get_predictions
from model_metrics.model_evaluator import (
    show_confusion_matrix,
    show_roc_curve,
    show_pr_curve,
    show_calibration_curve,
    show_lift_chart,
    show_gain_chart,
    plot_threshold_metrics,
    overlap_summary,
    show_cumulative_feature_performance,
)

LABELS = np.array(["azoo", "normal", "oligo"])


# Move these fixtures into conftest.py
@pytest.fixture
def mc_data():
    X, y = make_classification(
        n_samples=600,
        n_classes=3,
        n_informative=6,
        weights=[0.2, 0.5, 0.3],
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X, pd.Series(LABELS[y])


@pytest.fixture
def mc_model(mc_data):
    X, y = mc_data
    return LogisticRegression(max_iter=1000).fit(X, y)


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


# --- binary-only guards -------------------------------------------------------


def test_get_predictions_rejects_multiclass(mc_data, mc_model):
    X, y = mc_data
    with pytest.raises(ValueError, match="binary classification only"):
        get_predictions(mc_model, X, y, None, None, None)


@pytest.mark.parametrize(
    "func, kwargs",
    [
        (show_lift_chart, {}),
        (show_gain_chart, {}),
    ],
)
def test_lift_gain_reject_multiclass(mc_data, mc_model, func, kwargs):
    X, y = mc_data
    with pytest.raises(ValueError, match="binary classification only"):
        func(y_prob=mc_model.predict_proba(X), y=y, **kwargs)
    with pytest.raises(ValueError, match="binary classification only"):
        func(model=mc_model, X=X, y=y, **kwargs)


def test_threshold_metrics_rejects_multiclass(mc_data, mc_model):
    X, y = mc_data
    with pytest.raises(ValueError, match="binary classification only"):
        plot_threshold_metrics(model=mc_model, X_test=X, y_test=y)


def test_overlap_rejects_multiclass(mc_data, mc_model):
    X, y = mc_data
    preds = mc_model.predict(X)
    with pytest.raises(ValueError, match="binary classification only"):
        overlap_summary(y, y_pred_a=preds, y_pred_b=preds)


def test_cumulative_feature_performance_rejects_multiclass(mc_data, mc_model):
    X, y = mc_data
    with pytest.raises(ValueError, match="binary classification only"):
        show_cumulative_feature_performance(model=mc_model, X=X, y=y, order=list(X.columns))


# --- show_confusion_matrix ----------------------------------------------------


def test_confusion_matrix_multiclass_labels(mc_data, mc_model):
    X, y = mc_data
    _, ax = plt.subplots()
    show_confusion_matrix(model=mc_model, X=X, y=y, ax=ax)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(LABELS)
    cell_texts = [t.get_text() for t in ax.texts]
    assert not {"TN", "FP", "FN", "TP"} & set(cell_texts)
    assert sum(int(t.replace(",", "")) for t in cell_texts) == len(y)


def test_confusion_matrix_custom_class_labels(mc_data, mc_model):
    X, y = mc_data
    _, ax = plt.subplots()
    names = ["Azoospermia", "Normal", "Oligospermia"]
    show_confusion_matrix(
        y_prob=mc_model.predict_proba(X), y=y, class_labels=names, ax=ax
    )
    assert [t.get_text() for t in ax.get_yticklabels()] == names


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"custom_threshold": 0.4}, "Thresholds do not apply"),
        ({"class_labels": ["a", "b"]}, "class_labels has 2 entries"),
    ],
)
def test_confusion_matrix_multiclass_raises(mc_data, mc_model, kwargs, match):
    X, y = mc_data
    with pytest.raises(ValueError, match=match):
        show_confusion_matrix(y_prob=mc_model.predict_proba(X), y=y, **kwargs)


# --- show_roc_curve / show_pr_curve -------------------------------------------


def test_roc_one_curve_per_class(mc_data, mc_model):
    X, y = mc_data
    p = mc_model.predict_proba(X)
    _, ax = plt.subplots()
    show_roc_curve(y_prob=p, y=y, ax=ax, decimal_places=3)
    labels = _legend(ax)
    for j, c in enumerate(LABELS):
        expected = round(roc_auc_score(y == c, p[:, j]), 3)
        assert f"{c} vs rest (AUC = {expected:.3f})" in labels
    assert "Random Guess" in labels


def test_roc_overlay_macro_auc_matches_sklearn(mc_data, mc_model):
    X, y = mc_data
    p = mc_model.predict_proba(X)
    _, ax = plt.subplots()
    show_roc_curve(y_prob=[p, p], y=y, overlay=True, model_title=["A", "B"], ax=ax, decimal_places=3)
    macro = roc_auc_score(y, p, multi_class="ovr", average="macro")
    assert f"A (macro AUC = {macro:.3f})" in _legend(ax)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"show_operating_point": True}, "binary models only"),
        ({"delong": (np.zeros(3), np.zeros(3))}, "binary models only"),
        ({"group_category": pd.Series(["a"] * 600)}, "not supported for multiclass"),
    ],
)
def test_roc_binary_only_options_raise(mc_data, mc_model, kwargs, match):
    X, y = mc_data
    with pytest.raises(ValueError, match=match):
        show_roc_curve(y_prob=mc_model.predict_proba(X), y=y, **kwargs)


def test_pr_one_curve_per_class(mc_data, mc_model):
    X, y = mc_data
    _, ax = plt.subplots()
    show_pr_curve(model=mc_model, X=X, y=y, ax=ax)
    assert len(ax.get_lines()) == len(LABELS)
    assert all("vs rest (AP = " in lbl for lbl in _legend(ax))


def test_pr_overlay_macro_label(mc_data, mc_model):
    X, y = mc_data
    p = mc_model.predict_proba(X)
    _, ax = plt.subplots()
    show_pr_curve(y_prob=[p, p], y=y, overlay=True, legend_metric="aucpr", ax=ax)
    assert all("macro AUCPR" in lbl for lbl in _legend(ax))


# --- show_calibration_curve ---------------------------------------------------


def test_calibration_one_curve_per_class(mc_data, mc_model):
    X, y = mc_data
    _, ax = plt.subplots()
    show_calibration_curve(model=mc_model, X=X, y=y, ax=ax)
    labels = _legend(ax)
    assert sum("vs rest (Brier:" in lbl for lbl in labels) == len(LABELS)
    assert "Perfectly Calibrated" in labels


def test_calibration_overlay_rejects_multiclass(mc_data, mc_model):
    X, y = mc_data
    p = mc_model.predict_proba(X)
    with pytest.raises(ValueError, match="not supported for multiclass"):
        show_calibration_curve(y_prob=[p, p], y=y, overlay=True)


# --- return_df and model_tuner-style wrappers ---------------------------------


class _Wrapper:
    """Mimics a model_tuner Model: classes_ lives on the inner estimator."""

    def __init__(self, est):
        self.estimator = est

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def test_confusion_matrix_return_df(mc_data, mc_model):
    from sklearn.metrics import confusion_matrix

    X, y = mc_data
    dfs = show_confusion_matrix(
        model=[mc_model, mc_model],
        X=X,
        y=y,
        model_title=["A", "B"],
        subplots=True,
        return_df=True,
    )
    assert list(dfs) == ["A", "B"]
    expected = confusion_matrix(y, mc_model.predict(X), labels=mc_model.classes_)
    assert (dfs["A"].values == expected).all()
    assert dfs["A"].index.tolist() == [f"Actual {c}" for c in LABELS]


def test_confusion_matrix_default_returns_none(mc_data, mc_model):
    X, y = mc_data
    assert show_confusion_matrix(model=mc_model, X=X, y=y) is None


def test_wrapper_classes_used_when_test_set_missing_a_class(mc_data, mc_model):
    X, y = mc_data
    keep = (y != "azoo").values
    dfs = show_confusion_matrix(
        model=_Wrapper(mc_model), X=X[keep], y=y[keep], return_df=True
    )
    assert dfs["Model 1"].shape == (3, 3)


def test_missing_class_without_classes_attr_raises(mc_data, mc_model):
    X, y = mc_data
    keep = (y != "azoo").values
    with pytest.raises(ValueError, match="every class must appear"):
        show_confusion_matrix(y_prob=mc_model.predict_proba(X)[keep], y=y[keep])


def test_encoded_label_model_matches_raw_label_model(mc_data, mc_model):
    from sklearn.preprocessing import LabelEncoder
    from model_metrics.model_evaluator import summarize_model_performance

    X, y = mc_data
    enc = LabelEncoder().fit(y)
    encoded = _Wrapper(LogisticRegression(max_iter=1000).fit(X, enc.transform(y)))
    with pytest.warns(UserWarning, match="label-encoded targets"):
        got = summarize_model_performance(
            model=encoded, X=X, y=y, per_class=True, return_df=True
        )
    ref = summarize_model_performance(
        model=mc_model, X=X, y=y, per_class=True, return_df=True
    )
    assert got.equals(ref)


def test_class_labels_with_encoded_model_no_warning(mc_data, mc_model):
    import warnings
    from sklearn.preprocessing import LabelEncoder
    from model_metrics.model_evaluator import summarize_model_performance

    X, y = mc_data
    enc = LabelEncoder().fit(y)
    encoded = _Wrapper(LogisticRegression(max_iter=1000).fit(X, enc.transform(y)))
    ref = summarize_model_performance(
        model=mc_model, X=X, y=y, per_class=True, return_df=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for y_in in (y, enc.transform(y)):  # names or codes
            got = summarize_model_performance(
                model=encoded,
                X=X,
                y=y_in,
                class_labels=list(enc.classes_),
                per_class=True,
                return_df=True,
            )
            assert got.equals(ref)


def test_class_labels_wrong_length_raises(mc_data, mc_model):
    from model_metrics.model_evaluator import summarize_model_performance

    X, y = mc_data
    with pytest.raises(ValueError, match="class_labels has 2 entries"):
        summarize_model_performance(
            model=mc_model, X=X, y=y, class_labels=["a", "b"], per_class=True
        )


# --- normalize / highlight_diagonal -------------------------------------------


@pytest.mark.parametrize("norm", ["true", "pred", "all"])
def test_confusion_matrix_normalize_matches_sklearn(mc_data, mc_model, norm):
    from sklearn.metrics import confusion_matrix

    X, y = mc_data
    dfs = show_confusion_matrix(model=mc_model, X=X, y=y, normalize=norm, return_df=True)
    expected = confusion_matrix(
        y, mc_model.predict(X), labels=mc_model.classes_, normalize=norm
    )
    assert np.allclose(dfs["Model 1"].values, expected)


def test_confusion_matrix_normalized_cell_text(mc_data, mc_model):
    X, y = mc_data
    _, ax = plt.subplots()
    show_confusion_matrix(model=mc_model, X=X, y=y, normalize="true", ax=ax)
    texts = [t.get_text() for t in ax.texts]
    assert all("\n(" in t for t in texts)  # proportion with count beneath


def test_confusion_matrix_highlight_diagonal(mc_data, mc_model):
    from matplotlib.patches import Rectangle

    X, y = mc_data
    _, ax = plt.subplots()
    show_confusion_matrix(model=mc_model, X=X, y=y, highlight_diagonal=True, ax=ax)
    borders = [p for p in ax.patches if isinstance(p, Rectangle) and not p.get_fill()]
    assert len(borders) == len(LABELS)


def test_confusion_matrix_invalid_normalize(mc_data, mc_model):
    X, y = mc_data
    with pytest.raises(ValueError, match="normalize must be"):
        show_confusion_matrix(model=mc_model, X=X, y=y, normalize="rows")


@pytest.mark.parametrize(
    "cell_values, check",
    [
        ("both", lambda t: "\n(" in t),
        ("proportion", lambda t: "\n" not in t and "." in t),
        ("count", lambda t: "." not in t),
    ],
)
def test_confusion_matrix_cell_values(mc_data, mc_model, cell_values, check):
    X, y = mc_data
    _, ax = plt.subplots()
    show_confusion_matrix(
        model=mc_model, X=X, y=y, normalize="true", cell_values=cell_values, ax=ax
    )
    assert all(check(t.get_text()) for t in ax.texts)


def test_cell_values_proportion_requires_normalize(mc_data, mc_model):
    X, y = mc_data
    with pytest.raises(ValueError, match="requires `normalize`"):
        show_confusion_matrix(model=mc_model, X=X, y=y, cell_values="proportion")
