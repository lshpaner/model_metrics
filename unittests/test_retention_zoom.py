import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_metrics.model_evaluator import show_retention_zoom
from model_metrics.feature_selection_utils import _retention_cutoff


@pytest.fixture
def dip_scores():
    """AP crosses 80% of full at k=232, dips below it over 238-270, recovers at 271."""
    n = np.arange(1, 401)
    full = 0.59
    ap = np.minimum(full, full * n / 290)
    ap[237:270] = 0.41
    return pd.DataFrame(
        {
            "n_features": n,
            "ROC AUC": np.full(len(n), 0.98),
            "Average Precision": ap,
            "features": [["x"]] * len(n),
        }
    )


# -----------------------------------------------------------------------------------
# _retention_cutoff
# -----------------------------------------------------------------------------------
def test_cutoff_first_vs_sustained():
    n = [1, 2, 3, 4]
    frac = [0.9, 0.5, 0.95, 1.0]
    assert _retention_cutoff(n, frac, 0.8, "first") == 1
    assert _retention_cutoff(n, frac, 0.8, "sustained") == 3


def test_cutoff_unreached_returns_none():
    assert _retention_cutoff([1, 2], [0.5, 0.6], 0.9) is None


def test_cutoff_bad_crossing():
    with pytest.raises(ValueError, match="crossing"):
        _retention_cutoff([1], [1.0], 0.9, "last")


# -----------------------------------------------------------------------------------
# show_retention_zoom
# -----------------------------------------------------------------------------------
def test_first_crossing_matches_first_hit(dip_scores):
    out = show_retention_zoom(
        dip_scores,
        retain_metric="average_precision",
        retention_levels=(0.8,),
        return_df=True,
    )
    assert out.loc["80%", "n_features"] == 232


def test_sustained_skips_the_dip(dip_scores):
    out = show_retention_zoom(
        dip_scores,
        retain_metric="Average Precision",
        retention_levels=(0.8,),
        crossing="sustained",
        return_df=True,
    )
    assert out.loc["80%", "n_features"] == 271


def test_table_values_match_scores(dip_scores):
    out = show_retention_zoom(
        dip_scores, retain_metric="average_precision", return_df=True
    )
    for _, row in out.iterrows():
        src = dip_scores.loc[dip_scores["n_features"] == row["n_features"]]
        assert row["Average Precision"] == pytest.approx(
            src["Average Precision"].iloc[0]
        )


def test_panel_count_follows_levels(dip_scores):
    show_retention_zoom(dip_scores, retention_levels=(0.8, 0.9, 0.95), n_cols=2)
    fig = plt.gcf()
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3


def test_default_metrics_ignore_extra_columns(dip_scores):
    df = dip_scores.assign(**{"ROC AUC (% of full)": 100.0})
    out = show_retention_zoom(df, return_df=True)
    assert list(out.columns) == ["n_features", "ROC AUC", "Average Precision"]


def test_lower_is_better_metric():
    n = np.arange(1, 201)
    df = pd.DataFrame(
        {"n_features": n, "Brier Score (lower is better)": 0.05 + 0.1 * np.exp(-n / 20)}
    )
    out = show_retention_zoom(
        df, metrics=["brier"], retention_levels=(0.9,), return_df=True
    )
    k = out.loc["90%", "n_features"]
    full = df["Brier Score (lower is better)"].iloc[-1]
    hit = df.loc[df["n_features"] == k, "Brier Score (lower is better)"].iloc[0]
    assert full / hit >= 0.9


def test_unsorted_input_is_sorted(dip_scores):
    shuffled = dip_scores.sample(frac=1, random_state=0)
    out = show_retention_zoom(
        shuffled,
        retain_metric="average_precision",
        retention_levels=(0.8,),
        return_df=True,
    )
    assert out.loc["80%", "n_features"] == 232


def test_saves_image(dip_scores, tmp_path):
    path = tmp_path / "zoom.png"
    show_retention_zoom(dip_scores, image_filename=str(path))
    assert path.exists()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"retention_levels": (1.2,)}, "retention_levels"),
        ({"metrics": ["f1"]}, "not found"),
        ({"retain_metric": "brier"}, "not found"),
        ({"window": 0}, "window"),
        ({"crossing": "last"}, "crossing"),
        ({"curve_kwgs": {"nope": {"color": "k"}}}, "unknown metric keys"),
    ],
)
def test_validation(dip_scores, kwargs, match):
    with pytest.raises(ValueError, match=match):
        show_retention_zoom(dip_scores, **kwargs)


def test_requires_n_features():
    with pytest.raises(ValueError, match="n_features"):
        show_retention_zoom(pd.DataFrame({"ROC AUC": [0.9]}))


def test_xlim_clipped_to_feature_range():
    n = np.arange(1, 7)
    df = pd.DataFrame(
        {"n_features": n, "Average Precision": [0.41, 0.59, 0.68, 0.71, 0.735, 0.737]}
    )
    show_retention_zoom(df, retention_levels=(0.8, 0.99), window=50)
    for ax in [a for a in plt.gcf().axes if a.get_visible()]:
        assert ax.get_xlim() == (1, 6)


def test_default_legend_is_figure_level(dip_scores):
    show_retention_zoom(dip_scores)
    fig = plt.gcf()
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_legend_loc_inside_first_panel(dip_scores):
    show_retention_zoom(dip_scores, legend_loc="best")
    fig = plt.gcf()
    assert not fig.legends
    assert fig.axes[0].get_legend() is not None


def test_text_wrap_applies_to_panel_titles(dip_scores):
    show_retention_zoom(dip_scores, text_wrap=30)
    fig = plt.gcf()
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert titles and all("\n" in t for t in titles)
    assert all(len(line) <= 30 for t in titles for line in t.split("\n"))


def test_text_wrap_leaves_suptitle_alone(dip_scores):
    show_retention_zoom(dip_scores, text_wrap=20)
    assert "\n" not in plt.gcf()._suptitle.get_text()


def test_suptitle_wrap(dip_scores):
    show_retention_zoom(dip_scores, suptitle_wrap=20)
    fig = plt.gcf()
    assert "\n" in fig._suptitle.get_text()
    assert all("\n" not in ax.get_title() for ax in fig.axes if ax.get_visible())


def test_scalar_retention_level(dip_scores):
    out = show_retention_zoom(
        dip_scores, retain_metric="average_precision", retention_levels=0.8, return_df=True
    )
    assert list(out.index) == ["80%"]
    fig = plt.gcf()
    assert len(fig.axes) == 1


def test_suptitle_y(dip_scores):
    show_retention_zoom(dip_scores, suptitle_y=1.05)
    assert plt.gcf()._suptitle.get_position()[1] == pytest.approx(1.05)