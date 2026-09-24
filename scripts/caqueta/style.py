"""Figure style: large readable text, tight margins, and a fixed validated palette.

Rules applied to every figure (see scripts/caqueta/README.md):

- all text at least 12 pt
- only essential text on the figure; explanation lives in each folder's README
- tight bounding box, minimal whitespace, simple layouts

Colours are fixed per quantity, never per rank. Both series pairs were checked with
the dataviz palette validator (lightness, chroma, colour-vision-deficiency
separation, and contrast on the light surface all pass).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Series colours (one hue per quantity, used identically in every figure)
HH = "#2a78d6"  # blue
HV = "#eb6834"  # orange
COH80 = "#008300"  # green
COH20 = "#4a3aa7"  # violet

# Neutral references and marks
FOREST_REF = "#52514e"  # stable-forest reference (dark gray)
PASTURE_REF = "#9a9892"  # cleared-before-series reference (mid gray)
FLOOR = "#b4b2ab"  # zero-coherence estimator floor
EVENT_BAND = "#f3dcc9"  # HV-dated event bracket (light warm band)
OUTLINE = "#ffd400"  # patch outline on imagery (visible on RGB and grayscale)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

BASE_PT = 13


def apply() -> None:
    """Set matplotlib defaults for all Caquetá figures."""
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
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 200,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02,
        }
    )


def image_axes(ax: plt.Axes) -> None:
    """Strip ticks, grid, and spines from an image panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig: plt.Figure, path: Path) -> Path:
    """Save with a tight bounding box and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path
