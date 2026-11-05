"""Tests for grouped ROC / PR curves and the group_category helpers.

Covers integer legend counts, input types, index alignment, missing groups,
and single-class groups.
"""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from model_metrics.metrics_utils import _prepare_group_category
from model_metrics.model_evaluator import show_pr_curve, show_roc_curve


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture
def grouped():
    """Float target, shuffled non-default index, three groups."""
    rng = np.random.default_rng(0)
    n = 600
    idx = rng.permutation(n) + 1000
    y = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    p = np.clip(y.values * 0.3 + rng.random(n) * 0.7, 0, 1)
    g = pd.Series(rng.choice(["A", "B", "C"], n), index=idx, name="stage")
    return y, p, g


def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


CURVES = [show_roc_curve, show_pr_curve]


# --------------------------------------------------------------------------- #
# Legend counts                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func", CURVES)
def test_counts_are_integers_with_float_target(func, grouped):
    y, p, g = grouped
    _, ax = plt.subplots()
    func(y_prob=p, y=y, group_category=g, ax=ax)
    labels = [l for l in _legend(ax) if "Count:" in l]
    assert len(labels) == 3
    for l in labels:
        assert ".0" not in l.split("Count:")[1]
        assert re.search(r"Count: [\d,]+, Pos: [\d,]+, Neg: [\d,]+$", l)


@pytest.mark.parametrize("func", CURVES)
def test_counts_match_data(func, grouped):
    y, p, g = grouped
    _, ax = plt.subplots()
    func(y_prob=p, y=y, group_category=g, ax=ax)
    for gr in ["A", "B", "C"]:
        m = (g == gr).values
        n, pos = int(m.sum()), int(y.values[m].sum())
        (label,) = [l for l in _legend(ax) if f" for {gr} = " in l]
        assert label.endswith(f"Count: {n:,}, Pos: {pos:,}, Neg: {n - pos:,}")


# --------------------------------------------------------------------------- #
# Input types and alignment                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func", CURVES)
def test_numpy_inputs(func, grouped):
    y, p, g = grouped
    _, ax = plt.subplots()
    func(y_prob=p, y=y.values, group_category=g.values, ax=ax)
    assert sum("Count:" in l for l in _legend(ax)) == 3


@pytest.mark.parametrize("func", CURVES)
def test_list_group_category(func, grouped):
    y, p, g = grouped
    _, ax = plt.subplots()
    func(y_prob=p, y=y, group_category=g.tolist(), ax=ax)
    assert sum("Count:" in l for l in _legend(ax)) == 3


def test_same_labels_different_order_is_reindexed(grouped):
    y, _, g = grouped
    shuffled = g.sample(frac=1, random_state=1)
    _, masks_shuffled = _prepare_group_category(shuffled, y, "m")
    _, masks_aligned = _prepare_group_category(g, y, "m")
    for (g1, m1), (g2, m2) in zip(
        sorted(masks_shuffled, key=lambda t: t[0]),
        sorted(masks_aligned, key=lambda t: t[0]),
    ):
        assert g1 == g2
        np.testing.assert_array_equal(m1, m2)


def test_mismatched_index_warns_and_aligns_by_position(grouped):
    y, _, g = grouped
    with pytest.warns(UserWarning, match="aligning by position"):
        _, masks = _prepare_group_category(g, y.reset_index(drop=True), "m")
    (_, mask_a) = [t for t in masks if t[0] == "A"][0]
    np.testing.assert_array_equal(mask_a, (g == "A").values)


@pytest.mark.parametrize("func", CURVES)
def test_length_mismatch_raises(func, grouped):
    y, p, g = grouped
    with pytest.raises(ValueError, match="must be the same length"):
        func(y_prob=p, y=y, group_category=g.iloc[:-1])


# --------------------------------------------------------------------------- #
# Missing and degenerate groups                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func", CURVES)
def test_missing_groups_excluded_with_warning(func, grouped):
    y, p, g = grouped
    g = g.copy()
    g.iloc[:5] = None
    _, ax = plt.subplots()
    with pytest.warns(UserWarning, match="5 rows with missing"):
        func(y_prob=p, y=y, group_category=g, ax=ax)
    labels = [l for l in _legend(ax) if "Count:" in l]
    assert len(labels) == 3
    assert not any("nan" in l.lower() or "None" in l for l in labels)
    counts = [
        int(re.search(r"Count: ([\d,]+)", l).group(1).replace(",", ""))
        for l in labels
    ]
    assert sum(counts) == len(y) - 5


@pytest.mark.parametrize("func", CURVES)
def test_single_class_group_skipped_with_warning(func, grouped):
    y, p, g = grouped
    y = y.copy()
    y[g == "C"] = 0.0
    _, ax = plt.subplots()
    with pytest.warns(UserWarning, match="Skipping group 'C'"):
        func(y_prob=p, y=y, group_category=g, ax=ax)
    labels = [l for l in _legend(ax) if "Count:" in l]
    assert len(labels) == 2
    assert not any(" for C = " in l for l in labels)


@pytest.mark.parametrize("func", CURVES)
def test_all_groups_single_class_raises(func, grouped):
    y, p, g = grouped
    y = pd.Series(np.where(g == "A", 1.0, 0.0), index=y.index)
    g2 = g.where(g == "A", "B")  # A is all positive, B all negative
    with pytest.warns(UserWarning, match="Skipping group"):
        with pytest.raises(ValueError, match="both classes present"):
            func(y_prob=p, y=y, group_category=g2)


# --------------------------------------------------------------------------- #
# Saved filenames                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "func, suffix", [(show_roc_curve, "roc_auc"), (show_pr_curve, "precision_recall")]
)
def test_unnamed_group_category_filename(func, suffix, grouped, tmp_path):
    y, p, g = grouped
    func(
        y_prob=p,
        y=y.values,
        group_category=g.values,
        model_title="m",
        save_plot=True,
        image_path_png=str(tmp_path),
    )
    assert (tmp_path / f"m_group_{suffix}.png").exists()


@pytest.mark.parametrize(
    "func, suffix", [(show_roc_curve, "roc_auc"), (show_pr_curve, "precision_recall")]
)
def test_named_group_category_filename(func, suffix, grouped, tmp_path):
    y, p, g = grouped
    func(
        y_prob=p,
        y=y,
        group_category=g,
        model_title="m",
        save_plot=True,
        image_path_png=str(tmp_path),
    )
    assert (tmp_path / f"m_stage_{suffix}.png").exists()
