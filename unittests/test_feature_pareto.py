import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_metrics.model_evaluator import show_feature_pareto


def test_pareto_requires_model():
    with pytest.raises(ValueError):
        show_feature_pareto(None)


def test_pareto_bars_runs(rf_reg):
    assert show_feature_pareto(rf_reg, display="bars") is None


def test_pareto_curve_runs(rf_reg):
    assert show_feature_pareto(rf_reg, display="curve") is None


def test_pareto_invalid_display_raises(rf_reg):
    with pytest.raises(ValueError):
        show_feature_pareto(rf_reg, display="nope")


def test_pareto_return_df(rf_reg):
    df = show_feature_pareto(rf_reg, return_df=True)
    assert list(df.columns) == [
        "feature",
        "importance",
        "importance_pct",
        "cumulative_pct",
    ]


def test_pareto_return_features_is_list(rf_reg):
    feats = show_feature_pareto(rf_reg, return_features=True)
    assert isinstance(feats, list) and len(feats) >= 1


def test_pareto_return_both_is_dict(rf_reg):
    out = show_feature_pareto(rf_reg, return_features=True, return_df=True)
    assert set(out.keys()) == {"features", "df"}


def test_pareto_top_n_limits_bars(rf_reg):
    # top_n truncates the drawn set but the returned df is still full
    df = show_feature_pareto(rf_reg, top_n=3, return_df=True)
    assert len(df) == 8


def test_pareto_clean_names(rf_reg):
    show_feature_pareto(rf_reg, clean_names={"bmi": "Body Mass Index"})


def test_pareto_smooth_off(rf_reg):
    show_feature_pareto(rf_reg, smooth=False)


def test_pareto_saves_pdf(tmp_path, rf_reg):
    show_feature_pareto(rf_reg, image_filename=str(tmp_path / "pareto.pdf"))
    assert (tmp_path / "pareto.pdf").exists()


def test_pareto_accepts_ax(rf_reg):
    fig, ax = plt.subplots()
    show_feature_pareto(rf_reg, ax=ax)


# --------------------------------------------------------------------------- #
# feature_groups
# --------------------------------------------------------------------------- #
def test_grouping_applies_to_pareto(encoded_clf):
    model, _, _, ct = encoded_clf
    df = show_feature_pareto(
        model, feature_groups="auto", column_transformer=ct, return_df=True
    )
    assert set(df["feature"]) == {"age", "egfr", "CKD Stage"}


def test_pareto_curve_and_threshold_kwgs(rf_reg):
    """Line styling dicts merge over the hardcoded defaults."""
    fig, ax = plt.subplots()
    show_feature_pareto(
        rf_reg,
        curve_kwgs={"linewidth": 4},
        threshold_kwgs={"linewidth": 2.5},
        bar_kwgs={"edgecolor": "black"},
        ax=ax,
    )
    widths = [ln.get_linewidth() for a in fig.axes for ln in a.get_lines()]
    assert 4 in widths and 2.5 in widths


def test_pareto_threshold_count_printed(rf_reg, capsys):
    show_feature_pareto(rf_reg, threshold=0.80)
    assert "of cumulative importance" in capsys.readouterr().out
