import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from model_metrics.model_evaluator import (
    show_roc_curve,
    show_pr_curve,
    show_confusion_matrix,
    show_calibration_curve,
    show_lift_chart,
    show_gain_chart,
    plot_threshold_metrics,
    combine_plots,
    plot_overlap_venns,
    overlap_table,
    overlap_summary,
    overlap_crosstab,
    plot_overlap_crosstab,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 3), columns=["A", "B", "C"])
    y = np.random.randint(0, 2, size=100)
    return X, y


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]
    return X, y, model, y_prob


@pytest.fixture
def two_model_clf_data():
    """Two trained classifiers on the same data for overlap function tests."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    model_a = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    model_b = LogisticRegression(max_iter=1000, C=0.1, random_state=0).fit(X, y)
    return X, y, model_a, model_b


# ==============================================================================
# combine_plots — new parameters (hspace, wspace, height_ratios)
# ==============================================================================


@patch("matplotlib.pyplot.show")
def test_combine_plots_hspace(mock_show, clf_data):
    X, y, model, y_prob = clf_data
    combine_plots(
        plot_calls=[
            (show_roc_curve, {"y_prob": y_prob, "y": y}),
            (show_pr_curve, {"y_prob": y_prob, "y": y}),
        ],
        n_cols=2,
        hspace=0.5,
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_combine_plots_wspace(mock_show, clf_data):
    X, y, model, y_prob = clf_data
    combine_plots(
        plot_calls=[
            (show_roc_curve, {"y_prob": y_prob, "y": y}),
            (show_pr_curve, {"y_prob": y_prob, "y": y}),
        ],
        n_cols=2,
        wspace=0.3,
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_combine_plots_hspace_and_wspace(mock_show, clf_data):
    X, y, model, y_prob = clf_data
    combine_plots(
        plot_calls=[
            (show_roc_curve, {"y_prob": y_prob, "y": y}),
            (show_pr_curve, {"y_prob": y_prob, "y": y}),
        ],
        n_cols=2,
        hspace=0.4,
        wspace=0.3,
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_combine_plots_height_ratios(mock_show, clf_data):
    X, y, model, y_prob = clf_data
    combine_plots(
        plot_calls=[
            (show_roc_curve, {"y_prob": y_prob, "y": y}),
            (show_pr_curve, {"y_prob": y_prob, "y": y}),
            (show_confusion_matrix, {"y_prob": y_prob, "y": y}),
            (show_calibration_curve, {"y_prob": y_prob, "y": y}),
        ],
        n_cols=2,
        height_ratios=[1, 0.7],
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_combine_plots_hspace_skips_tight_layout(mock_show, clf_data):
    """When hspace is set, constrained_layout is used instead of tight_layout."""
    X, y, model, y_prob = clf_data
    with patch("matplotlib.pyplot.tight_layout") as mock_tl:
        combine_plots(
            plot_calls=[(show_roc_curve, {"y_prob": y_prob, "y": y})],
            hspace=0.4,
            tight_layout=True,
        )
        mock_tl.assert_not_called()


@patch("matplotlib.pyplot.show")
def test_combine_plots_font_injection(mock_show, clf_data):
    """label_fontsize and tick_fontsize are injected into panel functions."""
    X, y, model, y_prob = clf_data
    called_kwargs = {}

    def spy_roc(y_prob, y, label_fontsize=12, tick_fontsize=10, **kwargs):
        called_kwargs["label_fontsize"] = label_fontsize
        called_kwargs["tick_fontsize"] = tick_fontsize
        show_roc_curve(
            y_prob=y_prob,
            y=y,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

    combine_plots(
        plot_calls=[(spy_roc, {"y_prob": y_prob, "y": y})],
        label_fontsize=16,
        tick_fontsize=14,
    )
    assert called_kwargs.get("label_fontsize") == 16
    assert called_kwargs.get("tick_fontsize") == 14


@patch("matplotlib.pyplot.show")
def test_combine_plots_per_panel_font_override(mock_show, clf_data):
    """Per-panel font size overrides combine_plots defaults."""
    X, y, model, y_prob = clf_data
    called_kwargs = {}

    def spy_roc(y_prob, y, label_fontsize=12, tick_fontsize=10, **kwargs):
        called_kwargs["label_fontsize"] = label_fontsize
        show_roc_curve(
            y_prob=y_prob,
            y=y,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            **kwargs,
        )

    combine_plots(
        plot_calls=[(spy_roc, {"y_prob": y_prob, "y": y, "label_fontsize": 20})],
        label_fontsize=12,
    )
    assert called_kwargs.get("label_fontsize") == 20


# ==============================================================================
# show_confusion_matrix — model_threshold as list (X provided)
# ==============================================================================


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_model_threshold_list_with_X(mock_show, sample_data):
    """model_threshold as list should index per model when X is provided."""
    X, y = sample_data
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression(C=0.5).fit(X, y)

    show_confusion_matrix(
        model=[m1, m2],
        X=X,
        y=y,
        model_threshold=[0.3, 0.4],
        subplots=True,
        save_plot=False,
    )
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_model_threshold_scalar_with_X(mock_show, sample_data):
    """model_threshold as scalar should broadcast to all models when X is provided."""
    X, y = sample_data
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression(C=0.5).fit(X, y)

    show_confusion_matrix(
        model=[m1, m2],
        X=X,
        y=y,
        model_threshold=0.35,
        subplots=True,
        save_plot=False,
    )
    assert mock_show.called


# ==============================================================================
# show_confusion_matrix — model_threshold branches when X is None
# ==============================================================================


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_model_threshold_scalar_no_X(mock_show, sample_data):
    """model_threshold as scalar applied when X is None (y_prob path)."""
    X, y = sample_data
    y_prob = np.random.rand(len(y))

    show_confusion_matrix(
        y_prob=y_prob,
        y=y,
        model_threshold=0.3,
        save_plot=False,
    )
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_model_threshold_list_no_X(mock_show, sample_data):
    """model_threshold as list indexed per model when X is None."""
    X, y = sample_data
    y_prob1 = np.random.rand(len(y))
    y_prob2 = np.random.rand(len(y))

    show_confusion_matrix(
        y_prob=[y_prob1, y_prob2],
        y=y,
        model_threshold=[0.3, 0.4],
        subplots=True,
        save_plot=False,
    )
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_show_confusion_matrix_model_threshold_dict_no_X(mock_show, sample_data):
    """model_threshold as dict keyed by model title when X is None."""
    X, y = sample_data
    y_prob = np.random.rand(len(y))

    show_confusion_matrix(
        y_prob=y_prob,
        y=y,
        model_title="MyModel",
        model_threshold={"MyModel": 0.35},
        save_plot=False,
    )
    assert mock_show.called


# ==============================================================================
# Overlay ax= — all 6 functions that had the plt.figure() bug
# ==============================================================================


def test_show_roc_curve_overlay_external_ax(clf_data):
    """overlay=True draws onto supplied ax, not a new figure."""
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show") as mock_show:
        show_roc_curve(
            y_prob=[y_prob, y_prob2],
            y=y,
            model_title=["M1", "M2"],
            overlay=True,
            ax=ax,
            save_plot=False,
        )
        mock_show.assert_not_called()
    assert len(ax.lines) > 0


def test_show_pr_curve_overlay_external_ax(clf_data):
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show") as mock_show:
        show_pr_curve(
            y_prob=[y_prob, y_prob2],
            y=y,
            model_title=["M1", "M2"],
            overlay=True,
            ax=ax,
            save_plot=False,
        )
        mock_show.assert_not_called()
    assert len(ax.lines) > 0


def test_show_lift_chart_overlay_external_ax(clf_data):
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show") as mock_show:
        show_lift_chart(
            y_prob=[y_prob, y_prob2],
            y=y,
            model_title=["M1", "M2"],
            overlay=True,
            ax=ax,
            save_plot=False,
        )
        mock_show.assert_not_called()
    assert len(ax.lines) > 0


def test_show_gain_chart_overlay_external_ax(clf_data):
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show") as mock_show:
        show_gain_chart(
            y_prob=[y_prob, y_prob2],
            y=y,
            model_title=["M1", "M2"],
            overlay=True,
            ax=ax,
            save_plot=False,
        )
        mock_show.assert_not_called()
    assert len(ax.lines) > 0


def test_show_calibration_curve_overlay_external_ax(clf_data):
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show") as mock_show:
        show_calibration_curve(
            y_prob=[y_prob, y_prob2],
            y=y,
            model_title=["M1", "M2"],
            overlay=True,
            ax=ax,
            save_plot=False,
        )
        mock_show.assert_not_called()
    assert len(ax.lines) > 0


def test_plot_threshold_metrics_overlay_external_ax(clf_data):
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_threshold_metrics(
            y_prob=[y_prob, y_prob2],
            y_test=y,
            model_title=["M1", "M2"],
            overlay=True,
            ax=ax,
            save_plot=False,
        )
        mock_show.assert_not_called()
    assert len(ax.lines) > 0


def test_overlay_external_ax_does_not_save(clf_data, tmp_path):
    """overlay + ax= must not save even when save_plot=True."""
    X, y, model, y_prob = clf_data
    y_prob2 = np.random.rand(len(y))
    fig, ax = plt.subplots()
    show_roc_curve(
        y_prob=[y_prob, y_prob2],
        y=y,
        model_title=["M1", "M2"],
        overlay=True,
        ax=ax,
        save_plot=True,
        image_path_png=str(tmp_path),
    )
    assert list(tmp_path.iterdir()) == []


# ==============================================================================
# plot_overlap_venns
# ==============================================================================


@patch("matplotlib.pyplot.show")
def test_plot_overlap_venns_basic(mock_show, clf_data):
    """Standalone call triggers plt.show()."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN", "TN"),
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_overlap_venns_all_four_categories(mock_show, clf_data):
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN", "TN", "TP", "FP"),
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_overlap_venns_with_models_shared_X(mock_show, clf_data):
    """model_a / model_b path with a single shared feature matrix."""
    X, y, model, _ = clf_data
    m2 = LogisticRegression(C=0.5, max_iter=1000).fit(X, y)
    plot_overlap_venns(
        y_true=y,
        model_a=model,
        model_b=m2,
        X_a=X,
    )
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_overlap_venns_with_models_different_X(mock_show, clf_data):
    """model_a / model_b path with separate feature matrices."""
    X, y, model, _ = clf_data
    X2 = X.copy()
    X2["extra"] = np.random.rand(len(X2))
    m2 = LogisticRegression(C=0.5, max_iter=1000).fit(X2, y)
    plot_overlap_venns(
        y_true=y,
        model_a=model,
        model_b=m2,
        X_a=X,
        X_b=X2,
    )
    mock_show.assert_called_once()


def test_plot_overlap_venns_unknown_category_raises(clf_data):
    X, y, model, _ = clf_data
    y_pred = model.predict(X)
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true=y,
            y_pred_a=y_pred,
            y_pred_b=y_pred,
            categories=("XX",),
        )


def test_plot_overlap_venns_bad_titles_keys_raises(clf_data):
    X, y, model, _ = clf_data
    y_pred = model.predict(X)
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true=y,
            y_pred_a=y_pred,
            y_pred_b=y_pred,
            titles={"BAD": "nope"},
        )


def test_plot_overlap_venns_invalid_colors_length_raises(clf_data):
    X, y, model, _ = clf_data
    y_pred = model.predict(X)
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true=y,
            y_pred_a=y_pred,
            y_pred_b=y_pred,
            categories=("FN",),
            colors=("red",),
        )


def test_plot_overlap_venns_external_ax_no_show(clf_data):
    """When ax= is supplied, the function must not call plt.show()."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_overlap_venns(
            y_true=y,
            y_pred_a=y_pred_a,
            y_pred_b=y_pred_b,
            categories=("FN", "TN"),
            ax=ax,
        )
        mock_show.assert_not_called()


def test_plot_overlap_venns_external_ax_does_not_save(clf_data, tmp_path):
    """ax + save_plot=True must not save (caller owns saving)."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN",),
        ax=ax,
        save_plot=True,
        image_path_png=str(tmp_path),
    )
    assert list(tmp_path.iterdir()) == []


@patch("matplotlib.pyplot.show")
def test_plot_overlap_venns_savefig_writes_png(mock_show, clf_data, tmp_path):
    """Standalone with save_plot=True writes a PNG."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN",),
        save_plot=True,
        image_path_png=str(tmp_path),
    )
    assert list(tmp_path.glob("*.png"))


@patch("matplotlib.pyplot.show")
def test_plot_overlap_venns_savefig_disabled_by_default(mock_show, clf_data, tmp_path):
    """Standalone without save_plot=True does not write a file."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN",),
        image_path_png=str(tmp_path),
    )
    assert list(tmp_path.iterdir()) == []


@patch("matplotlib.pyplot.show")
def test_combine_plots_with_overlap_venns(mock_show, clf_data):
    """plot_overlap_venns slots into combine_plots via the ax subgridspec hook."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    venn_kwargs = {
        "y_true": y,
        "y_pred_a": y_pred_a,
        "y_pred_b": y_pred_b,
    }
    combine_plots(
        plot_calls=[
            (show_roc_curve, {"y_prob": y_prob, "y": y}),
            (plot_overlap_venns, {**venn_kwargs, "categories": ("FN",)}),
            (plot_overlap_venns, {**venn_kwargs, "categories": ("TN",)}),
            (plot_overlap_venns, {**venn_kwargs, "categories": ("TP",)}),
        ],
        n_cols=2,
    )
    mock_show.assert_called_once()


def test_plot_overlap_venns_accepts_bare_string_category():
    y_true = np.random.randint(0, 2, size=30)
    y_pred_a = np.random.randint(0, 2, size=30)
    y_pred_b = np.random.randint(0, 2, size=30)
    plot_overlap_venns(
        y_true,
        y_pred_a,
        y_pred_b,
        categories="FN",
    )
    plt.close("all")


@patch("model_metrics.model_evaluator.plt.show")
def test_plot_overlap_venns_label_kwgs_all_off(mock_show, clf_data):
    """Every toggle off renders a bare venn without error."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN",),
        label_kwgs={
            "show_title": False,
            "show_subtitle": False,
            "show_set_labels": False,
            "show_set_totals": False,
            "show_inner_count": False,
            "show_inner_role": False,
        },
    )
    plt.close("all")


@patch("model_metrics.model_evaluator.plt.show")
def test_plot_overlap_venns_label_kwgs_partial(mock_show, clf_data):
    """Partial label_kwgs leaves unspecified toggles at their defaults."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN", "TP"),
        label_kwgs={"show_inner_role": False},
    )
    plt.close("all")


@patch("model_metrics.model_evaluator.plt.show")
def test_plot_overlap_venns_label_kwgs_none(mock_show, clf_data):
    """label_kwgs=None is equivalent to passing no label_kwgs at all."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN",),
        label_kwgs=None,
    )
    plt.close("all")


@patch("model_metrics.model_evaluator.plt.show")
def test_plot_overlap_venns_label_kwgs_unknown_key(mock_show, clf_data):
    """Unknown keys in label_kwgs are silently ignored, not raised."""
    X, y, model, y_prob = clf_data
    y_pred_a = model.predict(X)
    y_pred_b = (np.random.rand(len(y)) > 0.5).astype(int)
    plot_overlap_venns(
        y_true=y,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        categories=("FN",),
        label_kwgs={"show_inner_count": False, "bogus_key": True},
    )
    plt.close("all")


def test_overlap_table_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 0, 0])
    y_pred_b = np.array([1, 1, 1, 0])
    df = overlap_table(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    # _pred suffix was added in the column rename
    assert list(df.columns) == [
        "y_true",
        "A_pred",
        "B_pred",
        "A_category",
        "B_category",
        "agree",
    ]


def test_overlap_table_with_index():
    y_true = np.array([1, 0, 1])
    df = overlap_table(
        y_true,
        [1, 0, 0],
        [1, 1, 1],
        label_a="A",
        label_b="B",
        index=["p1", "p2", "p3"],
    )
    assert df.index.tolist() == ["p1", "p2", "p3"]


def test_overlap_summary_shape_and_index():
    y_true = np.random.randint(0, 2, size=50)
    y_pred_a = np.random.randint(0, 2, size=50)
    y_pred_b = np.random.randint(0, 2, size=50)
    s = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    assert list(s.index) == ["TP", "FP", "FN", "TN"]
    assert {"n_A", "n_B", "both", "A_only", "B_only", "outside", "subpop"} == set(
        s.columns
    )


def test_overlap_summary_consistency_with_venn():
    # The summary counts must match _venn_category_counts exactly
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred_a = np.array([1, 0, 1, 0, 1, 0])
    y_pred_b = np.array([1, 1, 0, 1, 1, 0])
    s = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    # FN subpop = actual positives = 3
    assert s.loc["FN", "subpop"] == 3
    # TP subpop is also actual positives = 3
    assert s.loc["TP", "subpop"] == 3
    # TN subpop = actual negatives = 3
    assert s.loc["TN", "subpop"] == 3


def test_overlap_table_y_prob_path():
    """y_prob_a / y_prob_b path produces same df shape as y_pred_a / y_pred_b."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob_a = np.array([0.6, 0.4, 0.3, 0.7, 0.8, 0.2])
    y_prob_b = np.array([0.9, 0.55, 0.1, 0.6, 0.3, 0.45])
    df = overlap_table(
        y_true,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        label_a="A",
        label_b="B",
    )
    assert list(df.columns) == [
        "y_true",
        "A_pred",
        "B_pred",
        "A_category",
        "B_category",
        "agree",
    ]
    assert len(df) == 6


def test_overlap_table_y_prob_with_threshold():
    """Threshold parameter is honored on the y_prob path."""
    y_true = np.array([1, 1, 0, 0])
    y_prob_a = np.array([0.6, 0.4, 0.3, 0.7])
    y_prob_b = np.array([0.9, 0.55, 0.1, 0.6])
    df_low = overlap_table(
        y_true,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        threshold_a=0.3,
        threshold_b=0.3,
        label_a="A",
        label_b="B",
    )
    df_high = overlap_table(
        y_true,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        threshold_a=0.7,
        threshold_b=0.7,
        label_a="A",
        label_b="B",
    )
    # Lower threshold predicts more positives
    assert df_low["A_pred"].sum() >= df_high["A_pred"].sum()
    assert df_low["B_pred"].sum() >= df_high["B_pred"].sum()


def test_overlap_table_model_path():
    """model_a + X_a path works end-to-end."""
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, size=30)
    m1 = LogisticRegression().fit(X, y)
    m2 = LogisticRegression(C=0.5).fit(X, y)
    df = overlap_table(
        y,
        model_a=m1,
        model_b=m2,
        X_a=X,
        label_a="A",
        label_b="B",
    )
    assert len(df) == 30
    assert df["A_pred"].isin([0, 1]).all()


def test_overlap_table_three_way_exclusivity_raises():
    """Cannot supply y_pred and y_prob on the same side."""
    y_true = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError):
        overlap_table(
            y_true,
            y_pred_a=[1, 0, 0, 1],
            y_prob_a=[0.5, 0.5, 0.5, 0.5],
            y_pred_b=[0, 1, 0, 1],
        )


def test_overlap_table_with_index():
    """Custom index is applied to the returned df."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 0, 0])
    y_pred_b = np.array([1, 1, 1, 0])
    patient_ids = ["P1", "P2", "P3", "P4"]
    df = overlap_table(
        y_true,
        y_pred_a,
        y_pred_b,
        label_a="A",
        label_b="B",
        index=patient_ids,
    )
    assert list(df.index) == patient_ids


def test_overlap_summary_column_names():
    """Summary returns the 7 expected columns in the right order."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred_a = np.array([1, 0, 0, 1, 1, 0])
    y_pred_b = np.array([1, 1, 0, 0, 0, 1])
    df = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    assert list(df.columns) == [
        "n_A",
        "n_B",
        "both",
        "A_only",
        "B_only",
        "outside",
        "subpop",
    ]


def test_overlap_summary_partition_identity():
    """The four partition columns sum to subpop on every row."""
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=200)
    y_pred_a = np.random.randint(0, 2, size=200)
    y_pred_b = np.random.randint(0, 2, size=200)
    df = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    partition_sum = df["both"] + df["A_only"] + df["B_only"] + df["outside"]
    assert (partition_sum == df["subpop"]).all()


def test_overlap_summary_per_model_decomposition():
    """n_A = both + A_only and n_B = both + B_only on every row."""
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=200)
    y_pred_a = np.random.randint(0, 2, size=200)
    y_pred_b = np.random.randint(0, 2, size=200)
    df = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    assert (df["n_A"] == df["both"] + df["A_only"]).all()
    assert (df["n_B"] == df["both"] + df["B_only"]).all()


def test_overlap_summary_row_index():
    """Rows are indexed in confusion-matrix order: TP, FP, FN, TN."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 0, 0])
    y_pred_b = np.array([1, 1, 1, 0])
    df = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    assert list(df.index) == ["TP", "FP", "FN", "TN"]


def test_overlap_summary_subpop_correctness():
    """subpop is actual positives for TP/FN, actual negatives for FP/TN."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0])  # 4 positives, 3 negatives
    y_pred_a = np.zeros(7, dtype=int)
    y_pred_b = np.zeros(7, dtype=int)
    df = overlap_summary(y_true, y_pred_a, y_pred_b, label_a="A", label_b="B")
    assert df.loc["TP", "subpop"] == 4
    assert df.loc["FN", "subpop"] == 4
    assert df.loc["FP", "subpop"] == 3
    assert df.loc["TN", "subpop"] == 3


def test_overlap_summary_y_prob_path():
    """y_prob path produces the same output shape."""
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=50)
    y_prob_a = np.random.rand(50)
    y_prob_b = np.random.rand(50)
    df = overlap_summary(
        y_true,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        label_a="LR",
        label_b="DT",
    )
    assert df.shape == (4, 7)
    assert (
        df["both"] + df["LR_only"] + df["DT_only"] + df["outside"] == df["subpop"]
    ).all()


def test_overlap_summary_y_prob_threshold_changes_output():
    """Different thresholds produce different partition counts."""
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=100)
    y_prob_a = np.random.rand(100)
    y_prob_b = np.random.rand(100)
    df_low = overlap_summary(
        y_true,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        threshold_a=0.3,
        threshold_b=0.3,
    )
    df_high = overlap_summary(
        y_true,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        threshold_a=0.7,
        threshold_b=0.7,
    )
    # At least one cell differs
    assert not df_low.equals(df_high)


def test_overlap_summary_verbose_prints_legend(capsys):
    """verbose=True prints the column legend."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 0, 1])
    y_pred_b = np.array([1, 1, 0, 0])
    overlap_summary(
        y_true,
        y_pred_a,
        y_pred_b,
        label_a="LR",
        label_b="DT",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "overlap_summary columns" in captured.out
    assert "LR_only" in captured.out
    assert "DT_only" in captured.out


def test_overlap_summary_three_way_exclusivity_raises():
    """Cannot supply y_pred and y_prob on the same side."""
    y_true = np.array([1, 1, 0, 0])
    with pytest.raises(ValueError):
        overlap_summary(
            y_true,
            y_pred_a=[1, 0, 0, 1],
            y_prob_a=[0.5, 0.5, 0.5, 0.5],
            y_pred_b=[0, 1, 0, 1],
        )


@patch("model_metrics.model_evaluator.plt.show")
def test_plot_overlap_venns_y_prob_path(mock_show, clf_data):
    """y_prob_a / y_prob_b path renders without error."""
    X, y, model, y_prob = clf_data
    y_prob_b = np.random.rand(len(y))
    plot_overlap_venns(
        y_true=y,
        y_prob_a=y_prob,
        y_prob_b=y_prob_b,
        categories=("FN",),
        label_a="LR",
        label_b="DT",
    )
    plt.close("all")


@patch("model_metrics.model_evaluator.plt.show")
def test_plot_overlap_venns_y_prob_with_threshold(mock_show, clf_data):
    """y_prob path honors threshold_a / threshold_b."""
    X, y, model, y_prob = clf_data
    y_prob_b = np.random.rand(len(y))
    plot_overlap_venns(
        y_true=y,
        y_prob_a=y_prob,
        y_prob_b=y_prob_b,
        threshold_a=0.3,
        threshold_b=0.7,
        categories=("FN", "TP"),
        label_a="LR",
        label_b="DT",
    )
    plt.close("all")


def test_plot_overlap_venns_y_pred_and_y_prob_raises(clf_data):
    """Cannot mix y_pred and y_prob on the same side."""
    X, y, model, y_prob = clf_data
    y_pred = model.predict(X)
    with pytest.raises(ValueError):
        plot_overlap_venns(
            y_true=y,
            y_pred_a=y_pred,
            y_prob_a=y_prob,
            y_pred_b=y_pred,
        )
    plt.close("all")


# ==============================================================================
# overlap_crosstab tests
# ==============================================================================


def test_overlap_crosstab_returns_4x4_dataframe(two_model_clf_data):
    """Result must be a 4x4 DataFrame indexed by TP/FP/FN/TN."""
    X, y, model_a, model_b = two_model_clf_data
    ct = overlap_crosstab(y_true=y, model_a=model_a, model_b=model_b, X_a=X)
    assert isinstance(ct, pd.DataFrame)
    assert ct.shape == (4, 4)
    assert list(ct.index) == ["TP", "FP", "FN", "TN"]
    assert list(ct.columns) == ["TP", "FP", "FN", "TN"]


def test_overlap_crosstab_total_equals_observations(two_model_clf_data):
    """Sum of all cells must equal the observation count."""
    X, y, model_a, model_b = two_model_clf_data
    ct = overlap_crosstab(y_true=y, model_a=model_a, model_b=model_b, X_a=X)
    assert ct.values.sum() == len(y)


def test_overlap_crosstab_index_and_columns_named(two_model_clf_data):
    """Index name must be label_a, columns name must be label_b."""
    X, y, model_a, model_b = two_model_clf_data
    ct = overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        label_a="LR_A",
        label_b="LR_B",
    )
    assert ct.index.name == "LR_A"
    assert ct.columns.name == "LR_B"


def test_overlap_crosstab_impossible_cells_are_zero(two_model_clf_data):
    """The 8 structurally impossible cells must always be 0."""
    X, y, model_a, model_b = two_model_clf_data
    ct = overlap_crosstab(y_true=y, model_a=model_a, model_b=model_b, X_a=X)
    pos = {"TP", "FN"}
    for ra in ct.index:
        for cb in ct.columns:
            if (ra in pos) != (cb in pos):
                assert (
                    ct.loc[ra, cb] == 0
                ), f"impossible cell ({ra}, {cb}) = {ct.loc[ra, cb]}, expected 0"


def test_overlap_crosstab_y_pred_path(two_model_clf_data):
    """Binary predictions path produces same shape and total."""
    X, y, model_a, model_b = two_model_clf_data
    y_pred_a = model_a.predict(X)
    y_pred_b = model_b.predict(X)
    ct = overlap_crosstab(y_true=y, y_pred_a=y_pred_a, y_pred_b=y_pred_b)
    assert ct.shape == (4, 4)
    assert ct.values.sum() == len(y)


def test_overlap_crosstab_y_prob_path(two_model_clf_data):
    """Probability path applies the default 0.5 threshold."""
    X, y, model_a, model_b = two_model_clf_data
    y_prob_a = model_a.predict_proba(X)[:, 1]
    y_prob_b = model_b.predict_proba(X)[:, 1]
    ct = overlap_crosstab(y_true=y, y_prob_a=y_prob_a, y_prob_b=y_prob_b)
    assert ct.shape == (4, 4)
    assert ct.values.sum() == len(y)


def test_overlap_crosstab_threshold_changes_result(two_model_clf_data):
    """Different thresholds must produce different crosstabs."""
    X, y, model_a, model_b = two_model_clf_data
    y_prob_a = model_a.predict_proba(X)[:, 1]
    y_prob_b = model_b.predict_proba(X)[:, 1]
    ct_low = overlap_crosstab(
        y_true=y,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        threshold_a=0.3,
        threshold_b=0.3,
    )
    ct_high = overlap_crosstab(
        y_true=y,
        y_prob_a=y_prob_a,
        y_prob_b=y_prob_b,
        threshold_a=0.7,
        threshold_b=0.7,
    )
    assert not ct_low.equals(ct_high)


def test_overlap_crosstab_three_way_exclusivity_raises(two_model_clf_data):
    """Supplying more than one prediction source per side raises ValueError."""
    X, y, model_a, model_b = two_model_clf_data
    with pytest.raises(ValueError):
        overlap_crosstab(
            y_true=y,
            y_pred_a=model_a.predict(X),
            y_prob_a=model_a.predict_proba(X)[:, 1],
            y_pred_b=model_b.predict(X),
        )


def test_overlap_crosstab_normalize_sums_to_one(two_model_clf_data):
    """With normalize=True, all cells sum to 1.0."""
    X, y, model_a, model_b = two_model_clf_data
    ct = overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        normalize=True,
    )
    assert ct.values.sum() == pytest.approx(1.0)


def test_overlap_crosstab_mask_impossible_nans_impossible_cells(two_model_clf_data):
    """With mask_impossible=True, the 8 structurally impossible cells become NaN."""
    X, y, model_a, model_b = two_model_clf_data
    ct = overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        mask_impossible=True,
    )
    pos = {"TP", "FN"}
    for ra in ct.index:
        for cb in ct.columns:
            if (ra in pos) != (cb in pos):
                assert np.isnan(ct.loc[ra, cb])
            else:
                assert not np.isnan(ct.loc[ra, cb])


def test_overlap_crosstab_verbose_prints_legend(two_model_clf_data, capsys):
    """verbose=True should print the legend before returning."""
    X, y, model_a, model_b = two_model_clf_data
    overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        label_a="LR",
        label_b="RF",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "overlap_crosstab cells" in captured.out
    assert "Swap summary" in captured.out
    assert "LR" in captured.out
    assert "RF" in captured.out


def test_overlap_crosstab_diagonal_equals_agreement(two_model_clf_data):
    """Diagonal sum must equal the per-observation agreement count."""
    from model_metrics.metrics_utils import _overlap_table_categorize

    X, y, model_a, model_b = two_model_clf_data
    y_pred_a = model_a.predict(X)
    y_pred_b = model_b.predict(X)
    ct = overlap_crosstab(y_true=y, y_pred_a=y_pred_a, y_pred_b=y_pred_b)
    cat_a = _overlap_table_categorize(np.asarray(y), y_pred_a)
    cat_b = _overlap_table_categorize(np.asarray(y), y_pred_b)
    expected_agreement = int(np.sum(cat_a == cat_b))
    assert int(np.trace(ct.values)) == expected_agreement


def test_overlap_crosstab_x_b_defaults_to_x_a(two_model_clf_data):
    """X_b=None should default to X_a when both models use the same features."""
    X, y, model_a, model_b = two_model_clf_data
    ct_explicit = overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        X_b=X,
    )
    ct_default = overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
    )
    assert ct_explicit.equals(ct_default)


def test_overlap_crosstab_model_without_x_raises(two_model_clf_data):
    """Supplying a model without a feature matrix raises ValueError."""
    X, y, model_a, model_b = two_model_clf_data
    with pytest.raises(ValueError):
        overlap_crosstab(y_true=y, model_a=model_a, model_b=model_b)


@patch("matplotlib.pyplot.show")
def test_plot_overlap_crosstab_summary_body_fontsize_accepted(
    mock_show, two_model_clf_data
):
    """summary_body_fontsize is accepted as a keyword and the call completes."""
    X, y, model_a, model_b = two_model_clf_data
    plot_overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        summary_body_fontsize=18,
    )
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_overlap_crosstab_summary_body_fontsize_applied(
    mock_show, two_model_clf_data
):
    """summary_body_fontsize reaches the italic body lines in the summary panel."""
    plt.close("all")
    X, y, model_a, model_b = two_model_clf_data
    plot_overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        summary_body_fontsize=30,
    )
    # Walk every axes in the figure; any italic-style text artist is a body line
    italic_sizes = [
        t.get_fontsize()
        for ax in plt.gcf().axes
        for t in ax.texts
        if t.get_style() == "italic"
    ]
    assert any(s == 30 for s in italic_sizes), f"got {italic_sizes}"


@patch("matplotlib.pyplot.show")
def test_plot_overlap_crosstab_label_fontsize_controls_summary_headlines(
    mock_show, two_model_clf_data
):
    """label_fontsize flows into the summary panel headlines as label_fontsize + 5."""
    plt.close("all")
    X, y, model_a, model_b = two_model_clf_data
    plot_overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        label_fontsize=22,
    )
    bold_sizes = [
        t.get_fontsize()
        for ax in plt.gcf().axes
        for t in ax.texts
        if t.get_weight() == "bold"
    ]
    # The four summary headlines should render at 22 + 5 = 27pt
    assert 27 in bold_sizes, f"expected 27pt headlines, got {bold_sizes}"


@patch("matplotlib.pyplot.show")
def test_plot_overlap_crosstab_table_only_matrix_anchored_north(
    mock_show, two_model_clf_data
):
    """In table_only mode, the matrix axes is anchored north (regression for top-whitespace bug)."""
    plt.close("all")
    X, y, model_a, model_b = two_model_clf_data
    plot_overlap_crosstab(
        y_true=y,
        model_a=model_a,
        model_b=model_b,
        X_a=X,
        table_only=True,
    )
    fig = plt.gcf()
    # table_only renders the matrix as the only axes
    ax = fig.axes[0]
    assert ax.get_anchor() == "N"


def test_overlap_crosstab_decimal_places_rounds_normalized_proportions():
    """decimal_places=2 rounds normalized proportions to 2 decimals."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0])
    y_pred_a = np.array([1, 1, 0, 0, 0, 0, 1])
    y_pred_b = np.array([1, 0, 1, 0, 1, 0, 0])
    ct = overlap_crosstab(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        normalize=True,
        decimal_places=2,
    )
    for v in ct.values.flatten():
        if not np.isnan(v):
            assert v == round(v, 2), f"got {v}, expected {round(v, 2)}"


def test_overlap_crosstab_decimal_places_none_preserves_full_precision():
    """Default decimal_places=None does not round; 1/3 stays at 0.3333..."""
    y_true = np.array([1, 1, 1])
    y_pred_a = np.array([1, 1, 0])
    y_pred_b = np.array([1, 1, 0])
    ct = overlap_crosstab(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        normalize=True,
    )
    # (FN, FN) is 1/3, which should NOT equal round(1/3, 2) = 0.33
    val = ct.loc["FN", "FN"]
    assert val != round(val, 2)
    assert abs(val - 1 / 3) < 1e-10


def test_overlap_crosstab_decimal_places_with_mask_impossible_preserves_nan():
    """decimal_places does not touch NaN cells from mask_impossible."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 1, 0])
    y_pred_b = np.array([1, 0, 1, 0])
    ct = overlap_crosstab(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        mask_impossible=True,
        decimal_places=2,
    )
    pos = {"TP", "FN"}
    for ra in ct.index:
        for cb in ct.columns:
            if (ra in pos) != (cb in pos):
                assert np.isnan(ct.loc[ra, cb])


def test_overlap_crosstab_decimal_places_zero_rounds_to_whole_numbers():
    """decimal_places=0 rounds proportions to 0.0 or 1.0."""
    y_true = np.array([1, 1, 1])
    y_pred_a = np.array([1, 1, 0])
    y_pred_b = np.array([1, 1, 0])
    ct = overlap_crosstab(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        normalize=True,
        decimal_places=0,
    )
    # 2/3 rounds to 1.0, 1/3 rounds to 0.0
    for v in ct.values.flatten():
        if not np.isnan(v):
            assert v in (0.0, 1.0), f"got {v}"


def test_overlap_crosstab_decimal_places_no_op_on_integer_counts():
    """Without normalize, decimal_places leaves integer counts unchanged."""
    y_true = np.array([1, 1, 0, 0])
    y_pred_a = np.array([1, 0, 1, 0])
    y_pred_b = np.array([1, 1, 0, 0])
    ct_no_round = overlap_crosstab(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
    )
    ct_rounded = overlap_crosstab(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        decimal_places=2,
    )
    assert ct_no_round.equals(ct_rounded)
