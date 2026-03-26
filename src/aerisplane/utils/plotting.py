"""Plotting utilities and style setup for AerisPlane.

Provides consistent, publication-quality plot styling inspired by AeroSandbox's
pretty_plots. All plots use seaborn theming with a clean, readable palette.

Usage:
    from aerisplane.utils.plotting import set_style, show_plot

Call ``set_style()`` once at the top of a notebook or script.
Call ``show_plot()`` after building a figure to apply final formatting.
"""

from __future__ import annotations

# AerisPlane color palette — clear, distinguishable, colorblind-friendly
PALETTE = [
    "#4285F4",  # blue
    "#EA4335",  # red
    "#34A853",  # green
    "#ECB22E",  # gold
    "#9467BD",  # purple
    "#8C564B",  # brown
    "#E377C2",  # pink
    "#7F7F7F",  # gray
]


def set_style() -> None:
    """Apply AerisPlane default plot style.

    Sets seaborn theme, high DPI, and clean formatting defaults.
    Call once at the top of a notebook or script.
    """
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        font_scale=1.0,
    )
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["axes.formatter.useoffset"] = False
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["grid.linewidth"] = 0.8


def show_plot(
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool | None = None,
    tight_layout: bool = True,
) -> None:
    """Apply final formatting to the current figure.

    Parameters
    ----------
    title : str or None
        Figure title.
    xlabel, ylabel : str or None
        Axis labels for all axes.
    legend : bool or None
        If True, show legend. If None, auto-detect.
    tight_layout : bool
        Whether to call tight_layout with minimal padding.
    """
    import matplotlib.pyplot as plt

    fig = plt.gcf()

    for ax in fig.get_axes():
        if title is not None:
            if len(fig.get_axes()) > 1:
                fig.suptitle(title, fontsize=13, fontweight="bold")
            else:
                ax.set_title(title, fontsize=12, fontweight="bold")

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Smart grid
        ax.grid(True, which="major", linewidth=0.8, alpha=0.3)
        ax.grid(True, which="minor", linewidth=0.4, alpha=0.15)
        ax.minorticks_on()

        # Auto-legend
        if legend is True:
            ax.legend(framealpha=0.9, edgecolor="none")
        elif legend is None:
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) >= 2:
                ax.legend(framealpha=0.9, edgecolor="none")

    if tight_layout:
        fig.tight_layout(pad=0.5)
