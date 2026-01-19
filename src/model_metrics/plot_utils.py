import matplotlib.pyplot as plt
import textwrap
import math
import numpy as np


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
        kwargs = {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.15 if ax is None else -0.25),
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
