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
