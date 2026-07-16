import matplotlib.pyplot as plt
import textwrap
import math
import numpy as np
import os


def save_plot_images(
    filename,
    save_plot,
    image_path_png,
    image_path_svg,
    image_filename=None,
    fig=None,
    dpi=None,
    bbox_inches="tight",
):
    """
    Save a matplotlib figure.

    Saving is triggered when ``save_plot=True`` or ``image_filename`` is given.

    Two ways to name the output, which can be combined:

    * ``image_filename`` WITH an extension (e.g. ``"figure.pdf"`` or
      ``"out/figure.tiff"``) is saved verbatim in that format. No output
      directory is required, and any extension matplotlib supports works.
    * ``image_path_png`` / ``image_path_svg`` save ``<stem>.png`` / ``<stem>.svg``
      into those directories, where the stem is taken from ``image_filename``
      (without its extension) or, if none was given, from ``filename``.

    Parameters
    ----------
    filename : str
        Auto-generated base stem used when ``image_filename`` is None.
    save_plot : bool
        Trigger saving with the auto stem into the PNG/SVG directories.
    image_path_png, image_path_svg : str or None
        Output directories for the legacy PNG / SVG outputs.
    image_filename : str or None
        Custom name. With an extension it is saved verbatim (any format);
        without one it is used as the stem for the directory outputs. Providing
        it triggers saving regardless of ``save_plot``.
    fig : matplotlib.figure.Figure or None
        Figure to save. Defaults to the current figure.
    dpi : int or None
        DPI for raster formats (ignored by vector formats).
    bbox_inches : str, default="tight"
        Passed through to ``savefig``.

    Raises
    ------
    ValueError
        If saving was requested but nothing is saveable (a bare stem with no
        extension and no output directory).
    """
    should_save = save_plot or (image_filename is not None)
    if not should_save:
        return

    fig = fig or plt.gcf()

    # Determine the stem for directory-based (png/svg) outputs.
    name_for_stem = image_filename if image_filename is not None else filename
    stem, ext = (
        os.path.splitext(os.path.basename(name_for_stem)) if name_for_stem else ("", "")
    )

    targets = []

    # 1) Verbatim save when image_filename carries an extension (any format).
    if image_filename is not None and ext:
        targets.append(image_filename)

    # 2) Legacy directory + stem outputs.
    if image_path_png:
        targets.append(os.path.join(image_path_png, f"{stem}.png"))
    if image_path_svg:
        targets.append(os.path.join(image_path_svg, f"{stem}.svg"))

    if not targets:
        raise ValueError(
            "Cannot save figure: saving was requested but nothing is saveable. "
            f"`image_filename={image_filename!r}` has no file extension and no "
            "`image_path_png` / `image_path_svg` was given. Add an extension "
            "(e.g. '.png', '.jpg', '.pdf', '.svg') or pass an output directory."
        )

    for path in targets:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)


def apply_axis_limits(ax, xlim=None, ylim=None):
    """
    Apply axis limits to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to apply limits to.
    xlim : tuple, optional
        X-axis limits as (min, max).
    ylim : tuple, optional
        Y-axis limits as (min, max).

    Examples
    --------
    apply_axis_limits(ax, xlim=(0, 100), ylim=(-10, 10))
    apply_axis_limits(ax, ylim=(0, 1))  # Only set y-limits
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def apply_plot_title(
    title,
    default_title,
    text_wrap=None,
    fontsize=12,
    ax=None,
    fig=None,
    suptitle_y=None,
):
    """
    Apply title to plot with optional text wrapping.

    Parameters
    ----------
    title : str or None
        User-provided title. If None, use default_title. If "", no title shown.
    default_title : str
        The default title to use when title is None (function-specific).
    text_wrap : int, optional
        Maximum width before wrapping text.
    fontsize : int, default=12
        Font size for title.
    ax : matplotlib.axes.Axes, optional
        Axes object for subplot title. If None, uses plt.title().
    fig : matplotlib.figure.Figure, optional
        Figure object for suptitle. Takes precedence over ax if provided.
    suptitle_y : float, optional
        Y-position for suptitle (0-1 range). Only used when fig is provided.

    Examples
    --------
    # Subplot title
    apply_plot_title(None, "Default Title", ax=ax, fontsize=12)

    # Figure suptitle
    apply_plot_title(title, "Default Title", fig=fig, fontsize=14, suptitle_y=0.995)

    # No title
    apply_plot_title("", "Default Title", ax=ax)
    """
    if title is None:
        final_title = default_title  # Use the function-specific default
    elif title == "":
        return  # Explicitly no title
    else:
        final_title = title  # Use custom title

    if final_title and text_wrap:
        final_title = "\n".join(textwrap.wrap(final_title, width=text_wrap))

    if final_title:
        if fig:
            # Use suptitle for figure-level title
            if suptitle_y is not None:
                fig.suptitle(final_title, fontsize=fontsize, y=suptitle_y)
            else:
                fig.suptitle(final_title, fontsize=fontsize)
        elif ax:
            # Use set_title for subplot-level title
            ax.set_title(final_title, fontsize=fontsize)
        else:
            # Fallback to plt.title()
            plt.title(final_title, fontsize=fontsize)


def apply_legend(
    legend_loc="best",
    fontsize=10,
    ax=None,
    handles=None,
    labels=None,
    **legend_kwargs,
):
    """Apply legend with standardized positioning."""
    if legend_loc == "bottom":
        # Resize figure ONCE (check if already resized to avoid doing it multiple times)
        if ax is not None:
            fig = ax.get_figure()

            # Only resize if not already done (check a flag we set)
            if not hasattr(fig, "_resized_for_bottom_legend"):
                current_height = fig.get_figheight()
                fig.set_figheight(current_height + 2)
                fig._resized_for_bottom_legend = True  # Set flag to prevent re-resizing

        kwargs = {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.2),
            "fontsize": fontsize,
            **legend_kwargs,
        }
    else:
        kwargs = {"loc": legend_loc, "fontsize": fontsize, **legend_kwargs}

    if handles and labels:
        kwargs["handles"] = handles
        kwargs["labels"] = labels

    if ax:
        ax.legend(**kwargs)
    else:
        plt.legend(**kwargs)


def _should_show_in_resid_legend(legend_kwgs, item_type):
    """
    Determine if a legend item should be shown.

    Parameters
    ----------
    legend_kwgs : dict, bool, or None
        Legend configuration dictionary
    item_type : str
        Type of legend item: 'groups', 'clusters', 'centroids', 'het_tests', 'cooks'

    Returns
    -------
    bool
        Whether to show this item type in the legend
    """
    # Handle boolean/None cases first
    if legend_kwgs is None or legend_kwgs is True:
        return True  # Default: show everything
    if legend_kwgs is False:
        return False  # Hide everything

    # Handle dict case
    if not isinstance(legend_kwgs, dict):
        return True  # Fallback

    key_map = {
        "groups": "show_groups",
        "clusters": "show_clusters",
        "centroids": "show_centroids",
        "het_tests": "show_het_tests",
        "cooks": "show_cooks",
    }

    return legend_kwgs.get(key_map.get(item_type), True)


def _get_resid_legend_formatting_kwgs(legend_kwgs, tick_fontsize):
    """
    Extract matplotlib legend formatting kwargs.

    Parameters
    ----------
    legend_kwgs : dict or None
        Legend configuration dictionary
    tick_fontsize : int
        Default font size for ticks

    Returns
    -------
    dict
        Kwargs to pass to ax.legend()
    """
    if legend_kwgs is None:
        return {"fontsize": tick_fontsize - 2}

    # Extract formatting parameters
    formatting_keys = [
        "fontsize",
        "frameon",
        "framealpha",
        "shadow",
        "title",
        "title_fontsize",
        "ncol",
        "columnspacing",
        "labelspacing",
    ]

    fmt_kwgs = {
        k: v for k, v in legend_kwgs.items() if k in formatting_keys and v is not None
    }

    # Set default fontsize if not specified
    if "fontsize" not in fmt_kwgs:
        fmt_kwgs["fontsize"] = tick_fontsize - 2

    return fmt_kwgs


def setup_subplots(num_models, n_cols=2, n_rows=None, figsize=None):
    """Set up subplot grid for multiple models."""
    if n_rows is None:
        n_rows = math.ceil(num_models / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
    )

    # Always return a flattened array for consistent indexing
    if isinstance(axes, np.ndarray):
        return fig, axes.flatten()
    else:
        return fig, np.array([axes])


def normalize_curve_styles(curve_kwgs, model_title, num_models):
    """Normalize curve styling kwargs to list format."""
    if isinstance(curve_kwgs, dict):
        return [curve_kwgs.get(name, {}) for name in model_title]
    elif isinstance(curve_kwgs, list):
        return curve_kwgs
    else:
        return [{}] * num_models
