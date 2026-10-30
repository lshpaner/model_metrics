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
    assert list(out.columns) == [
        "n_features",
        "achieved",
        "prev_n_features",
        "prev_achieved",
        "ROC AUC",
        "Average Precision",
    ]


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
    # classic titles are one line, so any wrap there would come from suptitle_wrap
    show_retention_zoom(dip_scores, suptitle_wrap=20, title_style="classic")
    fig = plt.gcf()
    assert "\n" in fig._suptitle.get_text()
    assert all("\n" not in ax.get_title() for ax in fig.axes if ax.get_visible())


def test_suptitle_wrap_leaves_target_titles_two_lines(dip_scores):
    show_retention_zoom(dip_scores, suptitle_wrap=20)
    fig = plt.gcf()
    assert all(ax.get_title().count("\n") == 1 for ax in fig.axes if ax.get_visible())


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

# -----------------------------------------------------------------------------------
# achieved retention, jumps, merge_shared, title_style
# -----------------------------------------------------------------------------------
@pytest.fixture
def jump_scores():
    """AP sits at 59.5% of full through k=25, jumps to 96.4% at k=26, reaches
    full at k=100. So the 80/90/95% levels all resolve to 26 and 99% to 100."""
    n = np.arange(1, 101)
    full = 0.5
    ap = np.where(n <= 25, full * 0.595, full * 0.964)
    ap[-1] = full
    return pd.DataFrame(
        {"n_features": n, "ROC AUC": np.full(len(n), 0.9), "Average Precision": ap}
    )


_JUMP = dict(retain_metric="average_precision", retention_levels=(0.8, 0.9, 0.95, 0.99))


def _visible(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def _level_lines(ax):
    return [l for l in ax.get_lines() if l.get_linestyle() == ":"]


def test_achieved_and_prev_columns(jump_scores):
    out = show_retention_zoom(jump_scores, return_df=True, **_JUMP)
    row = out.loc["80%"]
    assert row["n_features"] == 26
    assert row["achieved"] == pytest.approx(0.964)
    assert row["prev_n_features"] == 25
    assert row["prev_achieved"] == pytest.approx(0.595)
    # achieved always meets the target, the step before never does (first crossing)
    for lvl, r in out.iterrows():
        L = float(lvl.rstrip("%")) / 100
        assert r["achieved"] >= L - 1e-12
        assert r["prev_achieved"] < L


def test_prev_is_nan_when_cutoff_is_first_row():
    df = pd.DataFrame({"n_features": [1, 2, 3], "Average Precision": [0.9, 0.95, 1.0]})
    out = show_retention_zoom(df, retention_levels=(0.8,), return_df=True)
    assert out.loc["80%", "n_features"] == 1
    assert pd.isna(out.loc["80%", "prev_n_features"])
    assert pd.isna(out.loc["80%", "prev_achieved"])
    assert "; " not in plt.gcf().axes[0].get_title()  # no jump without a prior step


def test_no_merge_one_panel_per_level(jump_scores):
    show_retention_zoom(jump_scores, **_JUMP)
    axes = _visible(plt.gcf())
    assert len(axes) == 4
    assert all(len(_level_lines(ax)) == 1 for ax in axes)


def test_merge_shared_collapses_same_cutoff(jump_scores):
    show_retention_zoom(jump_scores, merge_shared=True, **_JUMP)
    axes = _visible(plt.gcf())
    assert len(axes) == 2
    first, last = axes
    assert len(_level_lines(first)) == 3  # one line per target sharing k=26
    assert first.get_title().startswith("80%\u201395% target: minimum 26 features")
    assert last.get_title().startswith("99% target: minimum 100 features")


def test_merge_shared_keeps_one_table_row_per_level(jump_scores):
    a = show_retention_zoom(jump_scores, return_df=True, **_JUMP)
    b = show_retention_zoom(jump_scores, merge_shared=True, return_df=True, **_JUMP)
    pd.testing.assert_frame_equal(a, b)


def test_merge_shared_no_effect_when_cutoffs_distinct(dip_scores):
    show_retention_zoom(
        dip_scores,
        retain_metric="average_precision",
        retention_levels=(0.8, 0.9, 0.95),
        merge_shared=True,
    )
    assert len(_visible(plt.gcf())) == 3


def test_jump_title_shows_preceding_step(jump_scores):
    show_retention_zoom(jump_scores, **_JUMP)
    title = _visible(plt.gcf())[0].get_title()
    head, body = title.split("\n")
    assert head == (
        "80% target: minimum 26 features (96.4% of full Average Precision; 59.5% at 25)"
    )
    assert body.startswith("At 26: ROC AUC 0.900, Average Precision 0.482")


def test_within_tolerance_title_has_no_preceding_step(dip_scores):
    show_retention_zoom(dip_scores, retain_metric="average_precision", retention_levels=0.8)
    head = plt.gcf().axes[0].get_title().split("\n")[0]
    assert head == "80% target: minimum 232 features (80.0% of full Average Precision)"


def test_overshoot_tol_controls_jump_note(jump_scores):
    # 96.4% vs 80% target overshoots by 0.164: shown at the default tol, hidden at 0.2
    show_retention_zoom(jump_scores, overshoot_tol=0.2, **_JUMP)
    head = _visible(plt.gcf())[0].get_title().split("\n")[0]
    assert "at 25" not in head
    # merged panel compares against the highest target (95%): 0.014 overshoot
    show_retention_zoom(jump_scores, merge_shared=True, overshoot_tol=0.02, **_JUMP)
    assert "at 25" not in _visible(plt.gcf())[0].get_title()
    show_retention_zoom(jump_scores, merge_shared=True, overshoot_tol=0.01, **_JUMP)
    assert "59.5% at 25" in _visible(plt.gcf())[0].get_title()


def test_classic_title_style(jump_scores):
    show_retention_zoom(jump_scores, title_style="classic", **_JUMP)
    title = _visible(plt.gcf())[0].get_title()
    assert title == "80% of full Average Precision: 26 features (AUC 0.900, AP 0.482)"


def test_classic_title_merged_range(jump_scores):
    show_retention_zoom(jump_scores, title_style="classic", merge_shared=True, **_JUMP)
    title = _visible(plt.gcf())[0].get_title()
    assert title.startswith("80%\u201395% of full Average Precision: 26 features")


def test_bad_title_style_raises(dip_scores):
    with pytest.raises(ValueError, match="title_style"):
        show_retention_zoom(dip_scores, title_style="fancy")