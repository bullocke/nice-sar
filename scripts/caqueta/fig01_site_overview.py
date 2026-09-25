#!/usr/bin/env python
"""Site overview: Sentinel-2 at the start and end of the series, and RADD alerts.

Writes ``local_examples/caqueta/01_site/overview.png`` with three panels:

1. clearest Sentinel-2 true-colour image near the start of the NISAR series
2. clearest image near the end
3. RADD alert date (blue: during the NISAR series, light to dark by date;
   gray: alerts before the series), with numbered case-study locations

Usage:
    python scripts/caqueta/fig01_site_overview.py
"""

# ruff: noqa: E501  (long lines are Markdown in README templates)
from __future__ import annotations

import logging

import config
import data
import matplotlib.pyplot as plt
import numpy as np
import select_cases
import style
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = config.OUT_DIR / "01_site"
# Sequential blue ramp (light = early, dark = late)
DATE_CMAP = LinearSegmentedColormap.from_list("date", ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"])


def _clearest(scenes: list[data.S2Scene], lo: int, hi: int) -> data.S2Scene:
    cands = [s for s in scenes if lo <= s.day <= hi]
    usable = [s for s in cands if data.s2_usable(s)] or cands
    return max(usable, key=lambda s: data.s2_clear_fraction(s))


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = data.load()
    scenes = data.s2_scenes()
    early = _clearest(scenes, ds.first_day - 15, ds.first_day + 60)
    late = _clearest(scenes, ds.last_day - 60, ds.last_day + 15)
    cases = select_cases.read()

    alert = ds.radd.alert_date
    during = np.where((alert >= ds.first_day) & (alert <= ds.last_day), alert, np.nan)
    before = np.where((alert > -9999) & (alert < ds.first_day), 1.0, np.nan)

    h, w = ds.grid.shape
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for ax, scene in ((axes[0], early), (axes[1], late)):
        ax.imshow(data.s2_rgb(scene), interpolation="nearest", extent=(0, w, h, 0))
        ax.set_title(config.to_date(scene.day).strftime("%d %b %Y"))
    ax = axes[2]
    ax.set_facecolor("white")
    ax.imshow(before, cmap=ListedColormap(["#d9d8d4"]), interpolation="nearest")
    im = ax.imshow(
        during, cmap=DATE_CMAP, vmin=ds.first_day, vmax=ds.last_day, interpolation="nearest"
    )
    ax.set_title("RADD alert date")
    for a in axes:
        style.image_axes(a)
    # 5 km scale bar on the first panel
    bar = 5000 / config.PIXEL_M
    axes[0].plot([20, 20 + bar], [h - 25, h - 25], color="white", lw=4)
    axes[0].text(20 + bar / 2, h - 40, "5 km", color="white", ha="center", fontsize=12)

    ticks = [config.to_day(d) for d in ("2025-12-01", "2026-03-01", "2026-06-01", "2026-09-01")]
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02, ticks=ticks)
    cb.ax.set_yticklabels([config.to_date(t).strftime("%b %Y") for t in ticks])
    style.save(fig, OUT / "overview.png")

    # Case locations on a larger map so the numbers stay readable
    fig, ax = plt.subplots(figsize=(14, 11.5))
    ax.set_facecolor("white")
    ax.imshow(before, cmap=ListedColormap(["#d9d8d4"]), interpolation="nearest")
    im = ax.imshow(
        during, cmap=DATE_CMAP, vmin=ds.first_day, vmax=ds.last_day, interpolation="nearest"
    )
    for c in cases:
        ax.text(
            c.cols.mean(),
            c.rows.mean(),
            c.number,
            fontsize=12,
            ha="center",
            va="center",
            color=style.INK,
            bbox={"boxstyle": "circle,pad=0.15", "fc": "white", "ec": style.INK_2, "lw": 1},
        )
    style.image_axes(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, ticks=ticks)
    cb.ax.set_yticklabels([config.to_date(t).strftime("%b %Y") for t in ticks])
    style.save(fig, OUT / "case_locations.png")

    lines = [
        "# Site overview",
        "",
        f"AOI {config.AOI} (west, south, east, north), about 22 x 18 km in NISAR frame "
        f"{config.TRACK:03d} {config.DIRECTION} {config.FRAME:03d} (Caquetá, Colombia). "
        "It is the densest cluster of RADD alerts in the frame during the NISAR series.",
        "",
        f"- **Left**: Sentinel-2 {config.to_date(early.day)} (clearest image near the start). "
        "5 km scale bar.",
        f"- **Middle**: Sentinel-2 {config.to_date(late.day)} (clearest image near the end).",
        "- **Right**: RADD alert date. Blue = alerts during the NISAR series "
        f"({config.to_date(ds.first_day)} to {config.to_date(ds.last_day)}), light to dark by date; "
        "gray = alerts before the series (already cleared land).",
        "",
        "`case_locations.png` repeats the RADD map at a larger size with the case "
        "studies in `../04_cases/` numbered:",
        "",
        "| # | Case |",
        "|--:|---|",
    ]
    lines += [f"| {c.number} | {c.case_id} |" for c in cases]
    lines += ["", "Regenerate: `python scripts/caqueta/fig01_site_overview.py`."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
