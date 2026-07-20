"""Tests for show_cumulative_feature_importance.

The previous version of this file was an accidental copy of the performance
tests, leaving the importance function with no coverage at all.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from model_metrics.model_evaluator import show_cumulative_feature_importance
from model_metrics.feature_selection_utils import (
    _rank_features_by_importance,
    _clean_feature_labels,
)


# --------------------------------------------------------------------------- #
# Basic contract                                                              #
# --------------------------------------------------------------------------- #
def test_requires_model():
    with pytest.raises(ValueError):
        show_cumulative_feature_importance(None)


def test_returns_none_by_default(rf_reg):
    assert show_cumulative_feature_importance(rf_reg) is None


def test_return_df_columns(rf_reg):
    df = show_cumulative_feature_importance(rf_reg, return_df=True)
    assert list(df.columns) == [
        "feature",
        "importance",
        "importance_pct",
        "cumulative_pct",
    ]


def test_importance_pct_sums_to_one(rf_reg):
    df = show_cumulative_feature_importance(rf_reg, return_df=True)
    assert df["importance_pct"].sum() == pytest.approx(1.0)


def test_cumulative_is_monotonic_and_ends_at_one(rf_reg):
    df = show_cumulative_feature_importance(rf_reg, return_df=True)
    assert df["cumulative_pct"].is_monotonic_increasing
    assert df["cumulative_pct"].iloc[-1] == pytest.approx(1.0)


def test_sorted_descending_by_importance(rf_reg):
    df = show_cumulative_feature_importance(rf_reg, return_df=True)
    assert df["importance"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Threshold / shortlist                                                        #
# --------------------------------------------------------------------------- #
def test_return_features_is_list(rf_reg):
    feats = show_cumulative_feature_importance(rf_reg, return_features=True)
    assert isinstance(feats, list) and len(feats) >= 1


def test_return_both_is_dict(rf_reg):
    out = show_cumulative_feature_importance(
        rf_reg, return_df=True, return_features=True
    )
    assert set(out.keys()) == {"features", "df"}


def test_shortlist_is_ranked_prefix(rf_reg):
    out = show_cumulative_feature_importance(
        rf_reg, return_df=True, return_features=True
    )
    ranked = out["df"]["feature"].tolist()
    assert out["features"] == ranked[: len(out["features"])]


def test_shortlist_crosses_threshold(rf_reg):
    out = show_cumulative_feature_importance(
        rf_reg, threshold=0.80, return_df=True, return_features=True
    )
    k = len(out["features"])
    cum = out["df"]["cumulative_pct"].tolist()
    assert cum[k - 1] >= 0.80  # the last selected feature crosses it
    if k > 1:
        assert cum[k - 2] < 0.80  # the one before it does not


def test_higher_threshold_selects_at_least_as_many(rf_reg):
    lo = show_cumulative_feature_importance(
        rf_reg, threshold=0.50, return_features=True
    )
    hi = show_cumulative_feature_importance(
        rf_reg, threshold=0.95, return_features=True
    )
    assert len(hi) >= len(lo)
    assert set(lo) <= set(hi)


def test_console_summary_printed(rf_reg, capsys):
    show_cumulative_feature_importance(rf_reg, threshold=0.80)
    out = capsys.readouterr().out
    assert "of total importance" in out


# --------------------------------------------------------------------------- #
# Display options                                                              #
# --------------------------------------------------------------------------- #
def test_clean_names_does_not_change_returned_names(rf_reg):
    feats = show_cumulative_feature_importance(
        rf_reg, clean_names={"bmi": "Body Mass Index"}, return_features=True
    )
    assert "bmi" in feats and "Body Mass Index" not in feats


def test_clean_names_accepts_callable(rf_reg):
    show_cumulative_feature_importance(rf_reg, clean_names=lambda n: n.upper())


def test_show_annotation_off(rf_reg):
    show_cumulative_feature_importance(rf_reg, show_annotation=False)


def test_bar_kwgs_applied(rf_reg):
    fig, ax = plt.subplots()
    show_cumulative_feature_importance(
        rf_reg, bar_kwgs={"edgecolor": "black", "linewidth": 2}, ax=ax
    )
    widths = [p.get_linewidth() for p in ax.patches]
    assert widths and all(w == 2 for w in widths)


def test_accepts_ax_and_suppresses_save(tmp_path, rf_reg):
    fig, ax = plt.subplots()
    show_cumulative_feature_importance(
        rf_reg, ax=ax, image_filename=str(tmp_path / "nope.png")
    )
    assert not (tmp_path / "nope.png").exists()


def test_saves_via_arbitrary_extension(tmp_path, rf_reg):
    show_cumulative_feature_importance(rf_reg, image_filename=str(tmp_path / "imp.pdf"))
    assert (tmp_path / "imp.pdf").exists()


def test_title_and_fontsize_options(rf_reg):
    show_cumulative_feature_importance(
        rf_reg, title="Custom", label_fontsize=14, tick_fontsize=8
    )


# --------------------------------------------------------------------------- #
# Estimator variety                                                            #
# --------------------------------------------------------------------------- #
def test_works_with_linear_coef_model():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 4), columns=list("abcd"))
    y = (X["a"] * 2 + X["b"] > 0).astype(int)
    lr = LogisticRegression(max_iter=500).fit(X, y)
    df = show_cumulative_feature_importance(lr, return_df=True)
    assert len(df) == 4 and (df["importance"] >= 0).all()


def test_explicit_feature_names_override():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 3), columns=["x0", "x1", "x2"])
    y = X["x0"] * 3 + rng.randn(100) * 0.1
    rf3 = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    df = show_cumulative_feature_importance(
        rf3, feature_names=["alpha", "beta", "gamma"], return_df=True
    )
    assert set(df["feature"]) == {"alpha", "beta", "gamma"}


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #
def test_rank_helper_matches_function_output(rf_reg):
    df = show_cumulative_feature_importance(rf_reg, return_df=True)
    ranked = _rank_features_by_importance(rf_reg)
    assert df["feature"].tolist() == ranked["feature"].tolist()


def test_clean_feature_labels_dict_and_callable():
    assert _clean_feature_labels(["a", "b"], {"a": "Alpha"}) == ["Alpha", "b"]
    assert _clean_feature_labels(["a", "b"], str.upper) == ["A", "B"]
    assert _clean_feature_labels(["a", "b"], None) == ["a", "b"]


# --------------------------------------------------------------------------- #
# feature_groups                                                               #
# --------------------------------------------------------------------------- #
def test_grouping_sums_importance_and_preserves_total(encoded_clf):
    model, Xt, _, ct = encoded_clf
    ung = show_cumulative_feature_importance(model, return_df=True)
    grp = show_cumulative_feature_importance(
        model, feature_groups="auto", column_transformer=ct, return_df=True
    )
    assert len(grp) < len(ung)
    assert grp["importance"].sum() == pytest.approx(ung["importance"].sum())
    stage_split = ung[ung["feature"].str.contains("CKD Stage")]["importance"].sum()
    stage_grouped = grp.loc[grp["feature"] == "CKD Stage", "importance"].iloc[0]
    assert stage_grouped == pytest.approx(stage_split)


def test_grouping_accepts_dict_and_callable(encoded_clf):
    model, Xt, _, _ = encoded_clf
    as_dict = {c: ("CKD Stage" if "CKD Stage" in c else c) for c in Xt.columns}
    d = show_cumulative_feature_importance(
        model, feature_groups=as_dict, return_df=True
    )
    assert "CKD Stage" in d["feature"].tolist()

    c = show_cumulative_feature_importance(
        model, feature_groups=lambda n: n.split("__")[-1].split("_")[0], return_df=True
    )
    assert len(c) <= len(as_dict)


def test_grouping_auto_without_transformer_raises(encoded_clf):
    model, _, _, _ = encoded_clf
    with pytest.raises(ValueError, match="column_transformer"):
        show_cumulative_feature_importance(model, feature_groups="auto")


def test_grouping_invalid_spec_raises(encoded_clf):
    model, _, _, _ = encoded_clf
    with pytest.raises(ValueError):
        show_cumulative_feature_importance(model, feature_groups="not-auto")
