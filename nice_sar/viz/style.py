"""Shared matplotlib style for nice-sar figures.

One style for every notebook and documentation figure:

- all text at least 12 pt
- only essential text on the figure (explanation belongs in the surrounding text)
- tight margins and simple layouts
- one fixed colour per quantity and one colormap per kind of data

The series colours were checked for lightness, chroma, colour-vision-deficiency
separation and contrast on a white background.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "CMAPS",
    "COLORS",
    "add_colorbar",
    "add_scalebar",
    "image_axes",
    "use_nice_style",
]

#: One colour per quantity, used identically in every figure.
COLORS: dict[str, str] = {
    "HH": "#2a78d6",  # blue
    "HV": "#eb6834",  # orange
    "VV": "#1b9e77",  # teal
    "coh80": "#008300",  # green
    "coh20": "#4a3aa7",  # violet
    "forest": "#52514e",  # intact-forest reference (dark gray)
    "pasture": "#9a9892",  # non-forest reference (mid gray)
    "floor": "#b4b2ab",  # coherence estimator floor
    "event": "#f3dcc9",  # event bracket band
    "detection": "#c1121f",  # detection marker
    "outline": "#ffd400",  # outlines on imagery
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
}

#: Colormap and default display range per kind of data.
CMAPS: dict[str, dict[str, Any]] = {
    "backscatter": {"cmap": "gray", "vmin": -25.0, "vmax": -3.0},  # dB
    "hv": {"cmap": "gray", "vmin": -20.0, "vmax": -6.0},  # dB, forest HV
    "coherence": {"cmap": "magma", "vmin": 0.0, "vmax": 1.0},
    "phase": {"cmap": "twilight", "vmin": -np.pi, "vmax": np.pi},  # wrapped, radians
    "difference": {"cmap": "RdBu_r"},  # centre on zero with symmetric limits
    "index": {"cmap": "viridis"},  # sequential indices (RFDI, entropy, texture)
    "angle": {"cmap": "cividis", "vmin": 0.0, "vmax": 90.0},  # e.g. alpha angle
}

BASE_PT = 13


def use_nice_style(dpi: int = 110) -> None:
    """Apply the nice-sar matplotlib defaults.

    Args:
        dpi: Figure DPI for inline display and saving. 110 keeps notebook images
            sharp while keeping the files small.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": BASE_PT,
            "axes.titlesize": BASE_PT,
            "axes.labelsize": BASE_PT,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
            "axes.edgecolor": "#c9c8c3",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink2"],
            "ytick.color": COLORS["ink2"],
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "image.interpolation": "nearest",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
        }
    )


def image_axes(ax: Any, title: str | None = None) -> None:
    """Strip ticks, grid and spines from an image panel.

    Args:
        ax: Matplotlib axes.
        title: Optional panel title.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)


def add_colorbar(im: Any, ax: Any, label: str = "", **kwargs: Any) -> Any:
    """Add a slim colorbar with a label.

    Args:
        im: The mappable returned by ``imshow``.
        ax: Axes (or list of axes) the colorbar belongs to.
        label: Colorbar label, including units.
        **kwargs: Passed to ``Figure.colorbar``.

    Returns:
        The colorbar.
    """
    fig = np.atleast_1d(ax).flat[0].figure
    kwargs.setdefault("shrink", 0.85)
    kwargs.setdefault("aspect", 25)
    kwargs.setdefault("pad", 0.02)
    cb = fig.colorbar(im, ax=ax, **kwargs)
    cb.set_label(label)
    cb.outline.set_visible(False)
    return cb


def add_scalebar(
    ax: Any,
    pixel_m: float,
    length_m: float | None = None,
    loc: str = "lower right",
    color: str = "white",
) -> None:
    """Draw a scale bar on an image panel.

    Args:
        ax: Axes showing an image with one pixel per data unit.
        pixel_m: Pixel size in metres.
        length_m: Bar length in metres; defaults to a round number near a fifth
            of the image width.
        loc: Matplotlib location string.
        color: Bar and text colour.
    """
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    width_px = abs(np.diff(ax.get_xlim())[0])
    if length_m is None:
        target = width_px * pixel_m / 5
        exp = 10 ** np.floor(np.log10(target))
        length_m = float(min((1, 2, 5, 10), key=lambda m: abs(m * exp - target)) * exp)
    label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    bar = AnchoredSizeBar(
        ax.transData,
        length_m / pixel_m,
        label,
        loc,
        pad=0.4,
        color=color,
        frameon=False,
        size_vertical=max(width_px / 150, 1),
        fontproperties=FontProperties(size=12),
    )
    ax.add_artist(bar)
