"""Tests for the risk-adjusted funnel plot.

Covers the aggregation and Poisson-limit math, the separation of the drawing
threshold from the flagging threshold, the overdispersion path, the style-dict
contract, and the save / axis plumbing.

The fixtures are local to this module: no other test file needs a case-level
predictions frame.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_metrics.funnel_plot import (
    funnel_plot,
    group_oe,
    poisson_limits,
    estimate_phi,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cohort():
    """Case-level frame with one badly-performing group planted.

    Group "g_03" runs well below its expected rate at high volume, so it is
    flagged under any reasonable alpha. The rest are drawn from their own
    predicted risk, so they sit near O/E = 1.
    """
    rng = np.random.RandomState(0)
    frames = []
    for i in range(20):
        n = int(rng.randint(40, 600))
        p = np.clip(rng.normal(0.35, 0.10, n), 0.02, 0.98)
        shift = -0.22 if i == 3 else 0.0
        y = rng.binomial(1, np.clip(p + shift, 0.01, 0.99))
        frames.append(
            pd.DataFrame(
                {"unit": f"g_{i:02d}", "y_true": y, "y_pred_proba": p}
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def tiny():
    """Four cases over two groups, small enough to verify by hand."""
    return pd.DataFrame(
        {
            "unit": ["a", "a", "b", "b", "b"],
            "y_true": [1, 0, 1, 1, 0],
            "y_pred_proba": [0.5, 0.5, 0.4, 0.4, 0.2],
        }
    )


# --------------------------------------------------------------------------- #
# group_oe                                                                     #
# --------------------------------------------------------------------------- #
def test_group_oe_computes_counts_by_hand(tiny):
    g = group_oe(tiny, "unit", "y_true", "y_pred_proba")
    assert g.loc["a", "observed"] == 1
    assert g.loc["a", "expected"] == pytest.approx(1.0)
    assert g.loc["a", "n"] == 2
    assert g.loc["a", "oe"] == pytest.approx(1.0)
    assert g.loc["b", "observed"] == 2
    assert g.loc["b", "expected"] == pytest.approx(1.0)
    assert g.loc["b", "oe"] == pytest.approx(2.0)


def test_group_oe_indexed_by_group_col(tiny):
    g = group_oe(tiny, "unit", "y_true", "y_pred_proba")
    assert g.index.name == "unit"
    assert list(g.columns) == ["observed", "expected", "n", "oe"]


def test_group_oe_min_volume_drops_small_groups(tiny):
    g = group_oe(tiny, "unit", "y_true", "y_pred_proba", min_volume=3)
    assert list(g.index) == ["b"]


def test_group_oe_does_not_mutate_input(tiny):
    before = tiny.copy()
    group_oe(tiny, "unit", "y_true", "y_pred_proba")
    pd.testing.assert_frame_equal(tiny, before)


# --------------------------------------------------------------------------- #
# poisson_limits                                                               #
# --------------------------------------------------------------------------- #
def test_limits_bracket_the_target():
    lo, hi = poisson_limits([10.0, 50.0, 200.0], 0.05)
    assert np.all(lo < 1.0) and np.all(hi > 1.0)


def test_limits_tighten_as_expected_count_grows():
    lo, hi = poisson_limits([10.0, 100.0, 1000.0], 0.05)
    width = hi - lo
    assert width[0] > width[1] > width[2]


def test_tighter_alpha_gives_wider_limits():
    lo95, hi95 = poisson_limits([100.0], 0.05)
    lo998, hi998 = poisson_limits([100.0], 0.002)
    assert lo998[0] < lo95[0] and hi998[0] > hi95[0]


def test_phi_of_one_leaves_limits_unchanged():
    a = poisson_limits([50.0, 200.0], 0.05, phi=1.0)
    b = poisson_limits([50.0, 200.0], 0.05)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_phi_widens_by_exactly_sqrt_phi():
    phi = 4.0
    lo1, hi1 = poisson_limits([100.0], 0.05, phi=1.0)
    lo4, hi4 = poisson_limits([100.0], 0.05, phi=phi)
    assert (hi4[0] - 1.0) == pytest.approx((hi1[0] - 1.0) * np.sqrt(phi))
    assert (1.0 - lo4[0]) == pytest.approx((1.0 - lo1[0]) * np.sqrt(phi))


def test_limits_preserve_input_shape():
    lo, hi = poisson_limits(np.arange(5.0) + 10, 0.05)
    assert lo.shape == (5,) and hi.shape == (5,)


# --------------------------------------------------------------------------- #
# estimate_phi                                                                 #
# --------------------------------------------------------------------------- #
def test_phi_is_floored_at_one():
    g = pd.DataFrame({"observed": [10, 10], "expected": [10.0, 10.0]})
    assert estimate_phi(g) == 1.0


def test_phi_exceeds_one_under_dispersion():
    g = pd.DataFrame({"observed": [30, 2, 25, 4], "expected": [10.0] * 4})
    assert estimate_phi(g) > 1.0


def test_phi_matches_mean_squared_z():
    g = pd.DataFrame({"observed": [16, 4], "expected": [9.0, 9.0]})
    z = (g["observed"] - g["expected"]) / np.sqrt(g["expected"])
    assert estimate_phi(g) == pytest.approx(np.sum(z**2) / 2)


# --------------------------------------------------------------------------- #
# Basic contract                                                               #
# --------------------------------------------------------------------------- #
def test_group_col_is_required(cohort):
    with pytest.raises(TypeError):
        funnel_plot(cohort)


def test_returns_frame_with_expected_columns(cohort):
    g = funnel_plot(cohort, "unit")
    assert list(g.columns) == ["observed", "expected", "n", "oe", "outlier"]
    assert g.index.name == "unit"


def test_returned_frame_covers_every_group(cohort):
    g = funnel_plot(cohort, "unit")
    assert len(g) == cohort["unit"].nunique()


def test_empty_after_min_volume_raises(cohort):
    with pytest.raises(ValueError, match="min_volume"):
        funnel_plot(cohort, "unit", min_volume=10**6)


def test_outlier_column_is_boolean(cohort):
    g = funnel_plot(cohort, "unit")
    assert g["outlier"].dtype == bool


# --------------------------------------------------------------------------- #
# Flagging                                                                     #
# --------------------------------------------------------------------------- #
def test_planted_group_is_flagged(cohort):
    g = funnel_plot(cohort, "unit")
    assert bool(g.loc["g_03", "outlier"])
    assert g.loc["g_03", "oe"] < 1.0


def test_most_groups_are_not_flagged(cohort):
    g = funnel_plot(cohort, "unit")
    assert g["outlier"].sum() < len(g) / 2


def test_looser_flag_alpha_flags_at_least_as_many(cohort):
    strict = funnel_plot(cohort, "unit", flag_alpha=0.002)
    loose = funnel_plot(cohort, "unit", flag_alpha=0.05)
    assert loose["outlier"].sum() >= strict["outlier"].sum()
    assert set(strict.index[strict["outlier"]]) <= set(loose.index[loose["outlier"]])


def test_drawn_bands_do_not_affect_flagging(cohort):
    """limit_alphas is presentation only; flag_alpha alone decides outliers."""
    a = funnel_plot(cohort, "unit", limit_alphas=(0.05, 0.002), flag_alpha=0.002)
    b = funnel_plot(cohort, "unit", limit_alphas=(0.10,), flag_alpha=0.002)
    pd.testing.assert_series_equal(a["outlier"], b["outlier"])


def test_min_volume_filters_before_flagging(cohort):
    g = funnel_plot(cohort, "unit", min_volume=200)
    assert (g["n"] >= 200).all()


# --------------------------------------------------------------------------- #
# Overdispersion                                                               #
# --------------------------------------------------------------------------- #
def test_overdispersion_flags_are_a_subset_of_plain(cohort):
    plain = funnel_plot(cohort, "unit")
    od = funnel_plot(cohort, "unit", overdispersion=True)
    assert set(od.index[od["outlier"]]) <= set(plain.index[plain["outlier"]])


def test_phi_stored_on_returned_frame(cohort):
    g = funnel_plot(cohort, "unit", overdispersion=True)
    assert g.attrs["phi"] > 1.0


def test_phi_is_one_when_overdispersion_off(cohort):
    g = funnel_plot(cohort, "unit")
    assert g.attrs["phi"] == 1.0


def test_phi_appended_to_default_title(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", overdispersion=True, ax=ax)
    assert "phi=" in ax.get_title()


def test_phi_appended_to_custom_title(cohort):
    """Regression: phi used to be baked into the default title only, so a
    custom title silently hid the fact that the limits had been inflated."""
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", overdispersion=True, title="My Title", ax=ax)
    title = ax.get_title()
    assert title.startswith("My Title") and "phi=" in title


def test_empty_title_suppresses_phi_too(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", overdispersion=True, title="", ax=ax)
    assert ax.get_title() == ""


def test_no_phi_in_title_when_overdispersion_off(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", title="My Title", ax=ax)
    assert ax.get_title() == "My Title"


# --------------------------------------------------------------------------- #
# Style dicts                                                                  #
# --------------------------------------------------------------------------- #
def test_unknown_curve_kwgs_key_raises(cohort):
    with pytest.raises(ValueError, match="linecolor"):
        funnel_plot(cohort, "unit", curve_kwgs={"linecolor": {"color": "red"}})


def test_unknown_point_kwgs_key_raises(cohort):
    with pytest.raises(ValueError, match="outliers"):
        funnel_plot(cohort, "unit", point_kwgs={"outliers": {"color": "red"}})


def test_short_limits_list_raises(cohort):
    with pytest.raises(ValueError, match="limit_alphas"):
        funnel_plot(
            cohort,
            "unit",
            limit_alphas=(0.05, 0.002),
            curve_kwgs={"limits": [{"ls": "-"}]},
        )


def test_partial_point_kwgs_preserves_defaults(cohort):
    """Overriding one property must not reset the others."""
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", point_kwgs={"outlier": {"color": "#000000"}}, ax=ax)
    outlier_coll = ax.collections[1]
    assert outlier_coll.get_sizes()[0] == 45  # default size survives


def test_point_kwgs_passes_scatter_only_kwargs(cohort):
    fig, ax = plt.subplots()
    funnel_plot(
        cohort,
        "unit",
        point_kwgs={"inlier": {"marker": "^", "alpha": 0.5, "s": 99}},
        ax=ax,
    )
    assert ax.collections[0].get_sizes()[0] == 99


def test_limits_list_styles_bands_positionally(cohort):
    fig, ax = plt.subplots()
    funnel_plot(
        cohort,
        "unit",
        limit_alphas=(0.05, 0.002),
        curve_kwgs={
            "limits": [{"lw": 3.0, "color": "red"}, {"lw": 1.5, "color": "blue"}]
        },
        ax=ax,
    )
    widths = {ln.get_linewidth() for ln in ax.get_lines()}
    assert 3.0 in widths and 1.5 in widths


def test_limits_dict_applies_to_every_band(cohort):
    fig, ax = plt.subplots()
    funnel_plot(
        cohort,
        "unit",
        limit_alphas=(0.05, 0.002),
        curve_kwgs={"limits": {"lw": 2.5}},
        ax=ax,
    )
    band_widths = [ln.get_linewidth() for ln in ax.get_lines()]
    assert band_widths.count(2.5) == 4  # two bands, upper and lower each


def test_target_kwgs_styles_reference_line(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", curve_kwgs={"target": {"lw": 4.0}}, ax=ax)
    assert any(ln.get_linewidth() == 4.0 for ln in ax.get_lines())


def test_more_bands_than_default_linestyles(cohort):
    """The linestyle cycle must not run out with four bands."""
    fig, ax = plt.subplots()
    funnel_plot(
        cohort, "unit", limit_alphas=(0.20, 0.05, 0.01, 0.002), ax=ax
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert sum(lab.endswith("% limit") for lab in labels) == 4


# --------------------------------------------------------------------------- #
# Labels and axes                                                              #
# --------------------------------------------------------------------------- #
def test_default_xlabel_is_domain_neutral(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", unit_label="branch", ax=ax)
    assert ax.get_xlabel() == "Branch volume (number of cases)"
    assert "procedure" not in ax.get_xlabel().lower()


def test_custom_labels_applied(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", xlabel="Cases", ylabel="O over E", ax=ax)
    assert ax.get_xlabel() == "Cases" and ax.get_ylabel() == "O over E"


def test_axis_limits_applied(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", xlim=(0, 700), ylim=(0.5, 1.5), ax=ax)
    assert ax.get_xlim() == (0, 700) and ax.get_ylim() == (0.5, 1.5)


def test_legend_can_be_suppressed(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", show_legend=False, ax=ax)
    assert ax.get_legend() is None


def test_lower_bands_stay_out_of_the_legend(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", limit_alphas=(0.05, 0.002), ax=ax)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels.count("95% limit") == 1


def test_annotate_outliers_labels_flagged_points(cohort):
    fig, ax = plt.subplots()
    g = funnel_plot(cohort, "unit", annotate_outliers=True, ax=ax)
    assert len(ax.texts) == int(g["outlier"].sum())


def test_no_annotations_by_default(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", ax=ax)
    assert len(ax.texts) == 0


def test_gridlines_toggle(cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", gridlines=True, ax=ax)
    assert any(line.get_visible() for line in ax.get_xgridlines())


# --------------------------------------------------------------------------- #
# Saving and axes plumbing                                                     #
# --------------------------------------------------------------------------- #
def test_saves_via_arbitrary_extension(tmp_path, cohort):
    funnel_plot(cohort, "unit", image_filename=str(tmp_path / "funnel.pdf"))
    assert (tmp_path / "funnel.pdf").exists()


def test_saves_auto_stem_into_dirs(tmp_path, cohort):
    funnel_plot(
        cohort,
        "unit",
        unit_label="branch",
        outcome_label="churn",
        save_plot=True,
        image_path_png=str(tmp_path / "png"),
        image_path_svg=str(tmp_path / "svg"),
    )
    assert (tmp_path / "png" / "funnel_churn_branch.png").exists()
    assert (tmp_path / "svg" / "funnel_churn_branch.svg").exists()


def test_save_plot_without_path_raises(cohort):
    with pytest.raises(ValueError):
        funnel_plot(cohort, "unit", save_plot=True)


def test_accepts_ax_and_suppresses_save(tmp_path, cohort):
    fig, ax = plt.subplots()
    funnel_plot(cohort, "unit", ax=ax, image_filename=str(tmp_path / "nope.png"))
    assert not (tmp_path / "nope.png").exists()


def test_ax_draws_onto_supplied_axes(cohort):
    fig, ax = plt.subplots()
    before = len(plt.get_fignums())
    funnel_plot(cohort, "unit", ax=ax)
    assert len(plt.get_fignums()) == before  # no new figure created
    assert ax.collections  # points landed on the supplied axes


def test_combine_plots_call_shape(cohort):
    """combine_plots dispatches as func(**merged, ax=ax)."""
    fig, ax = plt.subplots()
    merged = {"df": cohort, "group_col": "unit", "unit_label": "branch"}
    g = funnel_plot(**merged, ax=ax)
    assert isinstance(g, pd.DataFrame)
