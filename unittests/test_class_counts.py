import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from model_metrics.metrics_utils import _class_support
from model_metrics.model_evaluator import (
    show_calibration_curve,
    show_pr_curve,
    show_roc_curve,
)

CLASS_LABELS = ["a", "b", "c", "d"]
COUNTS = {"a": 500, "b": 300, "c": 150, "d": 50}  # n = 1,000
SUPPORT = {
    "a": "n = 500, 50.0%",
    "b": "n = 300, 30.0%",
    "c": "n = 150, 15.0%",
    "d": "n = 50, 5.0%",
}


################################################################################
# Fixtures
################################################################################


@pytest.fixture
def mc_data():
    """Encoded 4-class labels and a matching (n, 4) probability matrix."""
    rng = np.random.default_rng(222)
    y = np.repeat(np.arange(4), list(COUNTS.values()))
    rng.shuffle(y)
    y_prob = rng.dirichlet(np.ones(4), size=len(y))
    y_prob[np.arange(len(y)), y] += 0.5  # give the curves some signal
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    return pd.Series(y, name="target"), y_prob


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(222)
    y = rng.integers(0, 2, 400)
    y_prob = np.clip(y * 0.3 + rng.uniform(0, 0.7, 400), 0, 1)
    return pd.Series(y), y_prob


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _class_labels_on(ax):
    """Legend labels for the per-class curves only."""
    return [l for l in ax.get_legend_handles_labels()[1] if " vs rest" in l]


################################################################################
# _class_support
################################################################################


class TestClassSupport:
    def test_count_and_prevalence(self):
        assert _class_support([0, 1, 1, 2], 1) == "n = 2, 50.0%"

    def test_thousands_separator(self):
        y = np.r_[np.ones(1234), np.zeros(8766)]
        assert _class_support(y, 1) == "n = 1,234, 12.3%"

    def test_string_labels(self):
        y = ["used_died", "used_alive", "used_alive", "used_alive"]
        assert _class_support(y, "used_died") == "n = 1, 25.0%"

    def test_series_input(self):
        assert _class_support(pd.Series([3, 3, 4]), 3) == "n = 2, 66.7%"

    def test_column_vector_is_flattened(self):
        y = np.array([[0], [1], [1], [1]])
        assert _class_support(y, 1) == "n = 3, 75.0%"

    def test_absent_class(self):
        assert _class_support([0, 0, 1], 9) == "n = 0, 0.0%"


################################################################################
# show_roc_curve
################################################################################


class TestRocClassCounts:
    def test_counts_in_legend_by_default(self, mc_data, ax):
        y, y_prob = mc_data
        show_roc_curve(y_prob=y_prob, y=y, class_labels=CLASS_LABELS, ax=ax)
        labels = _class_labels_on(ax)
        assert len(labels) == 4
        for c in CLASS_LABELS:
            (label,) = [l for l in labels if l.startswith(f"{c} vs rest")]
            assert re.fullmatch(
                rf"{c} vs rest \(AUC = \d\.\d{{2}}; {re.escape(SUPPORT[c])}\)",
                label,
            )

    def test_counts_disabled(self, mc_data, ax):
        y, y_prob = mc_data
        show_roc_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            show_class_counts=False,
            ax=ax,
        )
        for label in _class_labels_on(ax):
            assert re.fullmatch(r"\w vs rest \(AUC = \d\.\d{2}\)", label)

    def test_counts_follow_the_split_passed(self, mc_data, ax):
        y, y_prob = mc_data
        keep = np.asarray(y != 0)  # drop class "a" entirely
        keep[np.flatnonzero(y == 0)[:100]] = True  # keep 100 of class "a"
        show_roc_curve(
            y_prob=y_prob[keep], y=y[keep], class_labels=CLASS_LABELS, ax=ax
        )
        (label,) = [l for l in _class_labels_on(ax) if l.startswith("a vs")]
        assert "n = 100, 16.7%" in label  # 100 of 600

    def test_overlay_labels_unchanged(self, mc_data, ax):
        y, y_prob = mc_data
        show_roc_curve(
            y_prob=[y_prob, y_prob],
            y=y,
            class_labels=CLASS_LABELS,
            model_title=["m1", "m2"],
            overlay=True,
            ax=ax,
        )
        labels = ax.get_legend_handles_labels()[1]
        assert not any("n = " in l for l in labels)
        assert any(l.startswith("m1 (macro AUC = ") for l in labels)

    def test_binary_labels_unchanged(self, binary_data, ax):
        y, y_prob = binary_data
        show_roc_curve(y_prob=y_prob, y=y, ax=ax)
        labels = ax.get_legend_handles_labels()[1]
        assert any(re.fullmatch(r"AUC = \d\.\d{2}", l) for l in labels)
        assert not any("n = " in l for l in labels)


################################################################################
# show_pr_curve
################################################################################


def _hlines(ax):
    """Horizontal lines drawn with axhline (constant y spanning the axes)."""
    out = []
    for line in ax.lines:
        ys = np.asarray(line.get_ydata(), dtype=float)
        xs = np.asarray(line.get_xdata(), dtype=float)
        if len(ys) == 2 and ys[0] == ys[1] and tuple(xs) == (0.0, 1.0):
            out.append(line)
    return out


class TestPrClassCounts:
    def test_counts_in_legend_by_default(self, mc_data, ax):
        y, y_prob = mc_data
        show_pr_curve(y_prob=y_prob, y=y, class_labels=CLASS_LABELS, ax=ax)
        for c in CLASS_LABELS:
            (label,) = [l for l in _class_labels_on(ax) if l.startswith(f"{c} vs")]
            assert re.fullmatch(
                rf"{c} vs rest \(AP = \d\.\d{{2}}; {re.escape(SUPPORT[c])}\)",
                label,
            )

    def test_aucpr_metric_with_counts(self, mc_data, ax):
        y, y_prob = mc_data
        show_pr_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            legend_metric="aucpr",
            ax=ax,
        )
        (label,) = [l for l in _class_labels_on(ax) if l.startswith("d vs")]
        assert label.startswith("d vs rest (AUCPR = ")
        assert label.endswith(f"; {SUPPORT['d']})")

    def test_counts_disabled(self, mc_data, ax):
        y, y_prob = mc_data
        show_pr_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            show_class_counts=False,
            ax=ax,
        )
        for label in _class_labels_on(ax):
            assert re.fullmatch(r"\w vs rest \(AP = \d\.\d{2}\)", label)

    def test_prevalence_baselines_drawn(self, mc_data, ax):
        y, y_prob = mc_data
        show_pr_curve(y_prob=y_prob, y=y, class_labels=CLASS_LABELS, ax=ax)
        baselines = _hlines(ax)
        assert len(baselines) == 4
        assert sorted(l.get_ydata()[0] for l in baselines) == pytest.approx(
            [0.05, 0.15, 0.30, 0.50]
        )
        for l in baselines:
            assert l.get_linestyle() == ":"
            assert l.get_label().startswith("_")  # kept out of the legend

    def test_baseline_color_matches_class_curve(self, mc_data, ax):
        y, y_prob = mc_data
        show_pr_curve(y_prob=y_prob, y=y, class_labels=CLASS_LABELS, ax=ax)
        prev = {c: n / 1000 for c, n in COUNTS.items()}
        curves = {
            l.get_label().split(" vs")[0]: l
            for l in ax.lines
            if " vs rest" in l.get_label()
        }
        for line in _hlines(ax):
            (c,) = [k for k, p in prev.items() if p == pytest.approx(line.get_ydata()[0])]
            assert line.get_color() == curves[c].get_color()

    def test_prevalence_baselines_disabled(self, mc_data, ax):
        y, y_prob = mc_data
        show_pr_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            show_prevalence_baseline=False,
            ax=ax,
        )
        assert _hlines(ax) == []

    def test_binary_has_no_baseline_or_counts(self, binary_data, ax):
        y, y_prob = binary_data
        show_pr_curve(y_prob=y_prob, y=y, ax=ax)
        assert _hlines(ax) == []
        assert not any("n = " in l for l in ax.get_legend_handles_labels()[1])


################################################################################
# show_calibration_curve
################################################################################


class TestCalibrationClassCounts:
    def test_brier_and_counts_by_default(self, mc_data, ax):
        y, y_prob = mc_data
        show_calibration_curve(
            y_prob=y_prob, y=y, class_labels=CLASS_LABELS, bins=5, ax=ax
        )
        for c in CLASS_LABELS:
            (label,) = [l for l in _class_labels_on(ax) if l.startswith(f"{c} vs")]
            assert re.fullmatch(
                rf"{c} vs rest \(Brier: \d\.\d{{3}}; {re.escape(SUPPORT[c])}\)",
                label,
            )

    def test_counts_only(self, mc_data, ax):
        y, y_prob = mc_data
        show_calibration_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            bins=5,
            show_brier_score=False,
            ax=ax,
        )
        assert sorted(_class_labels_on(ax)) == [
            f"{c} vs rest ({SUPPORT[c]})" for c in CLASS_LABELS
        ]

    def test_brier_only(self, mc_data, ax):
        y, y_prob = mc_data
        show_calibration_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            bins=5,
            show_class_counts=False,
            ax=ax,
        )
        for label in _class_labels_on(ax):
            assert re.fullmatch(r"\w vs rest \(Brier: \d\.\d{3}\)", label)

    def test_neither(self, mc_data, ax):
        y, y_prob = mc_data
        show_calibration_curve(
            y_prob=y_prob,
            y=y,
            class_labels=CLASS_LABELS,
            bins=5,
            show_brier_score=False,
            show_class_counts=False,
            ax=ax,
        )
        assert sorted(_class_labels_on(ax)) == [f"{c} vs rest" for c in CLASS_LABELS]

    def test_binary_labels_unchanged(self, binary_data, ax):
        y, y_prob = binary_data
        show_calibration_curve(
            y_prob=y_prob, y=y, model_title="m", bins=5, ax=ax
        )
        labels = ax.get_legend_handles_labels()[1]
        assert any(re.fullmatch(r"m \(Brier: \d\.\d{3}\)", l) for l in labels)
        assert not any("n = " in l for l in labels)