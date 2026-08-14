"""
Risk-adjusted funnel plot, generalized over grouping unit and outcome.

Works for any provider level (center, surgeon, region) and any binary
outcome, given one row per case with:
    - a grouping id column
    - a 0/1 outcome column
    - a model risk column (predicted probability)

The observed-over-expected (O/E) ratio per group is plotted against the
group's case volume, with exact Poisson control limits that widen for
low-volume groups. Points outside the chosen limit are risk-adjusted
outliers.

Example
-------
    g = funnel_plot(
        df_pred_death,
        id_col="physicianid",
        y_col="y_true",
        p_col="y_pred_proba",
        unit_label="surgeon",
        outcome_label="death_in_6mon",
    )
    outliers = g[g["outlier"]].sort_values("oe")
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from .plot_utils import (
    save_plot_images,
    apply_axis_limits,
    apply_plot_title,
    apply_legend,
)


def group_oe(df, id_col, y_col, p_col, min_volume=1):
    """
    Aggregate to one row per group: observed, expected, volume, O/E.
    Groups below min_volume are dropped (small-n instability control).
    """
    g = df.groupby(id_col).agg(
        observed=(y_col, "sum"),
        expected=(p_col, "sum"),
        n=(y_col, "size"),
    )
    g = g[g["n"] >= min_volume]
    g["oe"] = g["observed"] / g["expected"]
    return g


def poisson_limits(expected_grid, alpha, phi=1.0):
    """
    Exact Poisson control limits for an O/E ratio across a grid of expected
    counts, as multiples of the target O/E = 1.

    phi is an overdispersion inflation factor (Spiegelhalter). phi = 1 is the
    standard Poisson funnel. phi > 1 widens the limits when there is genuine
    between-group variation beyond chance; estimate it with estimate_phi().
    """
    lam = np.asarray(expected_grid, dtype=float)
    lower_obs = stats.chi2.ppf(alpha / 2, 2 * lam) / 2
    upper_obs = stats.chi2.ppf(1 - alpha / 2, 2 * (lam + 1)) / 2
    lo = lower_obs / lam
    hi = upper_obs / lam
    if phi > 1.0:
        lo = 1.0 + (lo - 1.0) * np.sqrt(phi)
        hi = 1.0 + (hi - 1.0) * np.sqrt(phi)
    return lo, hi


def estimate_phi(g):
    """
    Overdispersion factor via the dispersion of standardized O/E residuals
    (Spiegelhalter 2005). phi ~ 1 means no overdispersion; phi > 1 means real
    between-group variation and the plain funnel over-flags.
    """
    z = (g["observed"] - g["expected"]) / np.sqrt(g["expected"])
    phi = np.sum(z**2) / len(g)
    return max(phi, 1.0)


def funnel_plot(
    df,
    id_col="centerid",
    y_col="y_true",
    p_col="y_pred_proba",
    unit_label="center",
    outcome_label="outcome",
    min_volume=1,
    flag_alpha=0.002,
    limit_alphas=(0.05, 0.002),
    overdispersion=False,
    annotate_outliers=False,
    n_grid=300,
    title=None,
    xlabel=None,
    ylabel="Observed / Expected",
    xlim=None,
    ylim=None,
    figsize=(9, 6),
    label_fontsize=12,
    tick_fontsize=10,
    title_fontsize=14,
    text_wrap=None,
    gridlines=False,
    grid_kwgs=None,
    inlier_color="steelblue",
    outlier_color="crimson",
    limit_color="grey",
    limit_linestyles=("-", ":"),
    target_color="black",
    target_linestyle="--",
    marker_size=25,
    outlier_marker_size=45,
    show_legend=True,
    legend_loc="upper right",
    legend_fontsize=8,
    save_plot=False,
    image_path_png=None,
    image_path_svg=None,
    image_filename=None,
    dpi=None,
    show_plot=True,
    ax=None,
):
    """
    Plot a risk-adjusted funnel of O/E against group volume.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per case, containing id_col, y_col and p_col.
    id_col : str
        Grouping unit column (center, surgeon, region, and so on).
    y_col : str
        Binary observed outcome column (0/1).
    p_col : str
        Predicted risk column used to build expected counts.
    unit_label, outcome_label : str
        Used in axis labels, the title and the default save filename.
    min_volume : int
        Groups with fewer cases than this are dropped before plotting.
    flag_alpha : float
        Two-sided alpha used to flag outliers against each group's own
        expected count. 0.002 is the 99.8% (three-sigma) limit.
    limit_alphas : sequence of float
        Alphas for the drawn control bands, paired with limit_linestyles.
    overdispersion : bool
        Inflate the limits by sqrt(phi) using estimate_phi().
    annotate_outliers : bool
        Label flagged points with their group id. Off by default; the
        returned frame is usually easier to read.
    n_grid : int
        Number of volume points used to draw the smooth control bands.
    xlim, ylim : tuple, optional
        Axis limits as (min, max). ylim is the useful one when a very
        low-expected group throws the O/E axis out to a large value and
        compresses everything else.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis (used by combine_plots). When supplied,
        the function skips figure sizing, saving and showing.

    Returns
    -------
    g : pandas.DataFrame
        Per-group observed, expected, volume (n), O/E and outlier flag,
        indexed by id_col. Flagged groups are g[g["outlier"]].
    """
    if len(limit_alphas) > len(limit_linestyles):
        raise ValueError(
            "limit_linestyles must be at least as long as limit_alphas."
        )

    g = group_oe(df, id_col, y_col, p_col, min_volume=min_volume)
    if g.empty:
        raise ValueError(
            f"No groups remain after min_volume={min_volume} filtering."
        )

    phi = estimate_phi(g) if overdispersion else 1.0

    rate = g["observed"].sum() / g["n"].sum()  # overall event rate
    vol_grid = np.linspace(g["n"].min(), g["n"].max(), n_grid)
    exp_grid = vol_grid * rate

    lo_c, hi_c = poisson_limits(g["expected"].values, flag_alpha, phi)
    g["outlier"] = (g["oe"].values < lo_c) | (g["oe"].values > hi_c)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    ax.axhline(
        1.0,
        color=target_color,
        lw=1,
        ls=target_linestyle,
        label="Expected (O/E = 1)",
    )

    for alpha, ls in zip(limit_alphas, limit_linestyles):
        lo, hi = poisson_limits(exp_grid, alpha, phi)
        band_label = f"{(1 - alpha) * 100:g}% limit"
        ax.plot(vol_grid, hi, color=limit_color, lw=1, ls=ls, label=band_label)
        ax.plot(vol_grid, lo, color=limit_color, lw=1, ls=ls)

    inl = g[~g["outlier"]]
    out = g[g["outlier"]]
    ax.scatter(
        inl["n"],
        inl["oe"],
        s=marker_size,
        color=inlier_color,
        label="Within limits",
    )
    ax.scatter(
        out["n"],
        out["oe"],
        s=outlier_marker_size,
        color=outlier_color,
        zorder=5,
        label="Outlier",
    )

    if annotate_outliers:
        for gid, row in out.iterrows():
            ax.annotate(
                str(gid),
                (row["n"], row["oe"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=tick_fontsize - 2,
            )

    disp = f"  (phi={phi:.2f})" if overdispersion else ""
    x_label = (
        xlabel
        if xlabel is not None
        else f"{unit_label.capitalize()} volume (number of procedures)"
    )

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    apply_plot_title(
        title,
        f"{outcome_label}: risk-adjusted {unit_label} funnel{disp}",
        text_wrap=text_wrap,
        fontsize=title_fontsize,
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    apply_axis_limits(ax, xlim=xlim, ylim=ylim)

    if gridlines:
        ax.grid(**(grid_kwgs or {"visible": True, "alpha": 0.3}))

    if show_legend:
        apply_legend(legend_loc=legend_loc, fontsize=legend_fontsize, ax=ax)

    if owns_fig:
        fig.tight_layout()

    if owns_fig:
        save_plot_images(
            filename=f"funnel_{outcome_label}_{unit_label}",
            save_plot=save_plot,
            image_path_png=image_path_png,
            image_path_svg=image_path_svg,
            image_filename=image_filename,
            fig=fig,
            dpi=dpi,
        )

    if show_plot and owns_fig:
        plt.show()
    elif owns_fig:
        plt.close(fig)

    return g