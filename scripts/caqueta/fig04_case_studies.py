#!/usr/bin/env python
"""Case-study figures: optical chips, backscatter, and coherence for each case.

For every case in ``cases.csv`` (see select_cases.py) this writes one PNG to
``local_examples/caqueta/04_cases/<category>/<case_id>.png`` with:

- top row: four Sentinel-2 true-colour chips (two before and two after the
  HV-dated event; evenly spaced dates for controls), case outline in yellow
- middle: HH and HV backscatter (dB). Thick = case mean, thin = one
  representative pixel, dashed gray = stable-forest HV median
- bottom: HH coherence. Each pair is a horizontal segment from its first to its
  second date. Thick = case mean, thin = representative pixel, gray = stable-forest
  median for the same pair (80 m)
- shaded band: HV-dated event bracket; dotted vertical line: RADD alert date

A README per category explains the selection rule and what to look for.

Usage:
    python scripts/caqueta/fig04_case_studies.py
"""

from __future__ import annotations

import logging

import analysis
import config
import data
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import select_cases
import style
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = select_cases.CASES_DIR

CATEGORY_TEXT = {
    "dip_detected": (
        "Dip detected",
        "Clearings with an abrupt HV drop (forest-normalized patch mean, step >= "
        f"{config.PATCH_STEP_DB} dB, >= {config.ABRUPT_FRAC:.0%} of it between two "
        "consecutive dual-pol dates) **and** 80 m coherence in the lowest pair inside "
        f"the HV bracket at least {-config.DIP_Z} pre-event standard deviations below "
        "the patch's own pre-event mean (after removing each pair's stable-forest "
        "median). Ranked by the size of the coherence drop.\n\n"
        "What to look for: the coherence segment spanning the shaded band sits below "
        "the preceding segments (and near the gray forest reference or lower), then "
        "coherence jumps up in the following pairs as the cleared surface stays stable.",
    ),
    "no_dip": (
        "No dip",
        "Clearings with the same abrupt HV drop but **no** coherence decrease in the "
        "pair(s) inside the HV bracket (z >= 0). Ranked by HV step.\n\n"
        "What to look for: coherence rises straight from forest level to the "
        "post-clearing level inside the bracket. Possible reasons: coherence dropped "
        "one cycle *before* the HV step (felling before the slash is burned; check the "
        "pair just left of the shaded band), the bracket holds two 12-day pairs and "
        "the event fell at the boundary, the forest was already partly cleared "
        "(low pre-event coherence), or the drop is hidden by pair-to-pair noise.",
    ),
    "gradual_decline": (
        "Gradual decline",
        f"Clearings whose HV decline (>= {config.PATCH_STEP_DB} dB overall) is spread over "
        f"several dates: no single interval carries >= {config.GRADUAL_FRAC:.0%} of it. "
        "Ranked by total step.\n\n"
        "What to look for: HV declining over 1-3 months, consistent with understory "
        "clearing before felling or progressive clearing within the patch. The HV "
        "bracket is less meaningful here, so the coherence test is weaker.",
    ),
    "radd_only": (
        "RADD alert without NISAR response",
        "Patches with a RADD high-confidence alert during the series but a "
        f"forest-normalized HV step < {config.PATCH_NO_STEP_DB} dB.\n\n"
        "What to look for: whether the optical chips show a clearing. Reasons for no "
        "HV step include clearing before the first usable dual-pol date (9 Dec 2025), "
        "so HV is already at pasture level from the start (e.g. radd_only_1, brown in "
        "the first chip), small or partial disturbances, or RADD commission errors.",
    ),
    "stable_forest": (
        "Stable forest control",
        "Random 1 ha squares of RADD forest baseline at least 100 m from any alert.\n\n"
        "What to look for: the natural pair-to-pair spread of coherence and "
        "backscatter in intact forest; this is the noise a disturbance signal must "
        "exceed. The CSV metrics for controls come from a best-fit split with no "
        "event, so they show what the metrics give by chance.",
    ),
    "pre_series_pasture": (
        "Pasture cleared before the series",
        "Random 1 ha squares alerted by RADD at least 90 days before the first NISAR "
        "date (already cleared land: pasture, crops, or regrowth).\n\n"
        "What to look for: the post-disturbance reference, with lower HV and higher "
        "coherence than forest.",
    ),
}


def _dates(days) -> list:
    return [config.to_date(d) for d in np.atleast_1d(days)]


def _choose_chips(case, scenes: list[data.S2Scene], window) -> list[data.S2Scene]:
    """Two clear chips before and two after the event (or four spread out)."""
    clear = [s for s in scenes if data.s2_usable(s, window)]
    if not clear:
        return []
    if np.isfinite(case.t0):
        before = [s for s in clear if s.day <= case.t0]
        after = [s for s in clear if s.day >= case.t1]
        chosen = before[-2:] + after[:1] + after[-1:] if after else before[-4:]
        # drop duplicate if only one "after" scene
        seen, out = set(), []
        for s in chosen:
            if s.day not in seen:
                seen.add(s.day)
                out.append(s)
        return out
    idx = np.linspace(0, len(clear) - 1, 4).round().astype(int)
    return [clear[i] for i in sorted(set(idx))]


def _segments(ax, pairs: data.PairStack, values, color, lw, alpha=1.0, ls="-"):
    for r, s, v in zip(pairs.ref, pairs.sec, values, strict=True):
        if np.isfinite(v):
            ax.plot(
                _dates([r, s]),
                [v, v],
                color=color,
                lw=lw,
                alpha=alpha,
                ls=ls,
                solid_capstyle="butt",
            )


def plot_case(ds: data.Dataset, case, scenes: list[data.S2Scene]) -> None:
    window = case.window(ds.grid.shape)
    series = analysis.patch_series(ds, case.rows, case.cols)
    pr, pc = case.rep_pixel
    pixel = analysis.patch_series(ds, np.array([pr]), np.array([pc]))
    forest = analysis.forest_reference(ds)
    chips = _choose_chips(case, scenes, window)

    fig = plt.figure(figsize=(12, 9.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 1])
    chip_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :], sharex=ax_b)

    # --- optical chips with case outline ------------------------------------------
    mask = np.zeros((window[0].stop - window[0].start, window[1].stop - window[1].start))
    inside = (
        (case.rows >= window[0].start)
        & (case.rows < window[0].stop)
        & (case.cols >= window[1].start)
        & (case.cols < window[1].stop)
    )
    mask[case.rows[inside] - window[0].start, case.cols[inside] - window[1].start] = 1
    mask = np.kron(mask, np.ones((2, 2)))  # 20 m -> 10 m S2 grid
    for ax, scene in zip(chip_axes, chips + [None] * (4 - len(chips)), strict=True):
        style.image_axes(ax)
        if scene is None:
            ax.set_visible(False)
            continue
        ax.imshow(data.s2_rgb(scene, window), interpolation="nearest")
        ax.contour(mask, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
        ax.set_title(config.to_date(scene.day).strftime("%d %b %Y"), pad=3)

    # --- backscatter ---------------------------------------------------------------
    for pol, color in (("HH", style.HH), ("HV", style.HV)):
        days = getattr(ds, pol.lower()).days
        ax_b.plot(_dates(days), series[pol], color=color, lw=2.5, marker="o", ms=4)
        ax_b.plot(_dates(days), pixel[pol], color=color, lw=1, alpha=0.6)
    ax_b.plot(_dates(ds.hv.days), forest["HV"], color=style.FOREST_REF, lw=1.5, ls="--")
    ax_b.set_ylabel("Backscatter (dB)")

    # --- coherence -----------------------------------------------------------------
    _segments(ax_c, ds.coh80, forest["coh80"], style.FOREST_REF, 2.5, alpha=0.5)
    _segments(ax_c, ds.coh20, series["coh20"], style.COH20, 3)
    _segments(ax_c, ds.coh80, series["coh80"], style.COH80, 3)
    _segments(ax_c, ds.coh80, pixel["coh80"], style.COH80, 1, alpha=0.7)
    ax_c.set_ylabel("Coherence")
    ax_c.set_ylim(0, 1)

    # --- event marks ---------------------------------------------------------------
    for ax in (ax_b, ax_c):
        if np.isfinite(case.t0):
            ax.axvspan(*_dates([case.t0, case.t1]), color=style.EVENT_BAND, lw=0, zorder=0)
        if np.isfinite(case.radd_day) and case.radd_day >= ds.first_day:
            ax.axvline(_dates(case.radd_day)[0], color=style.INK_2, lw=1.2, ls=":")
    ax_c.xaxis.set_major_locator(mdates.MonthLocator())
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.setp(ax_b.get_xticklabels(), visible=False)

    ax_b.legend(
        handles=[
            Line2D([], [], color=style.HH, lw=2.5, label="HH"),
            Line2D([], [], color=style.HV, lw=2.5, label="HV"),
            Line2D([], [], color=style.FOREST_REF, lw=1.5, ls="--", label="Forest HV"),
        ],
        ncol=3,
        loc="best",
    )
    ax_c.legend(
        handles=[
            Line2D([], [], color=style.COH80, lw=3, label="80 m"),
            Line2D([], [], color=style.COH20, lw=3, label="20 m"),
            Line2D([], [], color=style.FOREST_REF, lw=2.5, alpha=0.5, label="Forest 80 m"),
        ],
        ncol=3,
        loc="best",
    )
    style.save(fig, OUT / case.category / f"{case.case_id}.png")


def write_readmes(cases) -> None:
    by_cat: dict[str, list] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    for cat, items in by_cat.items():
        title, text = CATEGORY_TEXT[cat]
        rows = [
            "| Case | Area (ha) | Lat, lon | HV bracket | RADD alert | HV step (dB) | "
            "80 m coherence change (z) |",
            "|---|--:|---|---|---|--:|--:|",
        ]
        for c in items:
            bracket = f"{c.hv_bracket_start} to {c.hv_bracket_end}" if c.hv_bracket_start else "-"
            rows.append(
                f"| {c.case_id} | {c.area_ha} | {c.lat}, {c.lon} | {bracket} | "
                f"{c.radd_alert or '-'} | {c.hv_step_db} | {c.coh80_diff} ({c.coh80_z}) |"
            )
        readme = f"""# {title}

{text}

## Cases

{chr(10).join(rows)}

## Reading the figures

- **Chips**: Sentinel-2 true colour (B4/B3/B2, reflectance 0-0.15), Cloud Score+
  masked (clouds shown light gray); yellow outline = case pixels. Only chips at
  least 80% clear and haze-free (median blue reflectance <= 0.06) are used. A faint
  horizontal seam can appear where two Sentinel-2 granules meet (each granule is
  atmospherically corrected separately). Two chips before
  and two after the HV-dated event (controls: spread over the series).
- **Backscatter**: thick = case mean (averaged in linear power), thin = one
  representative pixel (HV step closest to the case median), dashed gray =
  stable-forest HV median for that date.
- **Coherence**: each pair is a horizontal segment from its first to its second
  acquisition. Green = 80 m, violet = 20 m (case mean); thin green = the
  representative pixel at 80 m; gray = stable-forest 80 m median for the same pair.
  The expected value for a fully decorrelated pair is about 0.08 at 80 m and 0.21
  at 20 m (estimator bias with 112 and 18 looks).
- **Shaded band**: HV-dated event bracket (last dual-pol date before and first after
  the HV drop). **Dotted line**: median RADD alert date for the case.

Regenerate with `python scripts/caqueta/select_cases.py` then
`python scripts/caqueta/fig04_case_studies.py`.
"""
        (OUT / cat / "README.md").write_text(readme)


def write_index_readme(cases) -> None:
    lines = [
        "# Case studies",
        "",
        "One folder per category, each with its own README describing the selection "
        "rule and what to look for. `cases.csv` lists every case with its metrics; "
        "`cases.npz` holds the pixel indices so all figures use identical pixels.",
        "",
        "Case numbers match the labels on `../01_site/overview.png`.",
        "",
        "| # | Case | Category |",
        "|--:|---|---|",
    ]
    lines += [f"| {c.number} | {c.case_id} | {c.category} |" for c in cases]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    style.apply()
    ds = data.load()
    scenes = data.s2_scenes()
    cases = select_cases.read()
    for case in cases:
        plot_case(ds, case, scenes)
        logger.info("Wrote %s", case.case_id)
    write_readmes(cases)
    write_index_readme(cases)


if __name__ == "__main__":
    main()
