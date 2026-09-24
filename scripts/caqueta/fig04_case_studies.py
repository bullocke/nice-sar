#!/usr/bin/env python
"""Case-study figures: image chips and time series for each clearing and control.

For every case in ``cases.csv`` (see select_cases.py) this writes one PNG to
``local_examples/caqueta/04_cases/<category>/<case_id>.png``:

- three chip rows on the same window and zoom: Sentinel-2 true colour, GCOV HV,
  and HH coherence, two dates before and two after the event
- Sentinel-2 NBR: case mean against the stable-forest median (optical timing)
- HH and HV backscatter (case mean, one pixel, stable-forest HV)
- coherence relative to stable forest in the same pair (80 m and 20 m), with the
  +/-2 sigma band of forest noise for an area of the case's size
- shaded bands: HV-dated drop and coherence dip; dotted line: RADD alert date

The first case of each clearing category also gets ``<case_id>_coh20.png`` with
20 m coherence in the chip row. A README per category explains the selection
rule and what to look for.

Usage:
    python scripts/caqueta/fig04_case_studies.py
"""

# ruff: noqa: E501  (long lines are Markdown in README templates)
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
from matplotlib.patches import Patch as LegendPatch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = select_cases.CASES_DIR
# Cases that also get a variant with 20 m coherence chips (<case_id>_coh20.png):
# the first case of each clearing category.
COH20_VARIANT_CATEGORIES = ("dip_detected", "no_dip", "gradual_decline")
HV_RANGE_DB = (-16.0, -6.0)  # forest about -10 dB, pasture about -12.5 dB
COH_RANGE = (0.0, 0.8)
DIP_HATCH = style.COH80  # coherence-dip period: green hatching over the HV band
MIN_CHIP_GAP_D = 20  # the two "before" chips are at least this far apart
REUSE_TOLERANCE_D = 12  # take an unused date/pair only if at most this much farther

CATEGORY_TEXT = {
    "dip_detected": (
        "Coherence dip detected",
        f"Clearings with a forest-normalized HV drop of at least {config.PATCH_STEP_DB} dB "
        "(case mean) **and** at least one 80 m coherence pair more than "
        f"{config.DIP_SIGMA:g} sigma below stable forest in the same pair, searched from "
        f"{config.DIP_SEARCH_BEFORE_D} days before the HV drop to the onset of the "
        "post-clearing rise. Sigma is the pair-to-pair spread of intact-forest areas "
        "of the same size. Ranked by the depth of the dip in sigma units.\n\n"
        "What to look for: the green band (coherence dip) relative to the orange band "
        "(HV drop) and to the NBR drop. A dip that precedes the HV drop fits felling "
        "(coherence lost) before the slash is burned (HV lost).",
    ),
    "no_dip": (
        "No coherence dip",
        f"Clearings with an HV drop of at least {config.PATCH_STEP_DB} dB but **no** 80 m pair "
        "below -1 sigma of forest noise near the event. Ranked by HV step.\n\n"
        "What to look for: coherence moving straight from forest level to the "
        "post-clearing level. Possible reasons: the clearing happened within a single "
        "pair that also contains stable ground, the outline mixes cleared and intact "
        "pixels, or the dip is hidden in forest noise.",
    ),
    "gradual_decline": (
        "Gradual decline",
        f"Clearings whose HV decline (>= {config.PATCH_STEP_DB} dB overall) is spread over "
        f"several dates: no single interval carries >= {config.GRADUAL_FRAC:.0%} of it. "
        "Ranked by total step.\n\n"
        "What to look for: HV declining over one to three months, consistent with "
        "progressive clearing or understory clearing before felling; compare with "
        "the NBR panel to see when the canopy was actually removed.",
    ),
    "radd_only": (
        "RADD alert without NISAR HV response",
        "Clearings with a RADD high-confidence alert during the series but a "
        f"forest-normalized HV step < {config.PATCH_NO_STEP_DB} dB.\n\n"
        "What to look for: whether NBR and the chips show a clearing. Reasons for no "
        "HV step include clearing before the first usable dual-pol date (9 Dec 2025), "
        "re-clearing of young regrowth that had little HV to lose, small or partial "
        "disturbances, or RADD commission errors.",
    ),
    "stable_forest": (
        "Stable forest control",
        "Random 1 ha squares of RADD forest baseline at least 100 m from any alert.\n\n"
        "What to look for: the natural pair-to-pair spread of coherence and "
        "backscatter in intact forest; the gray band in the coherence panel is this "
        "spread (+/-2 sigma) for a 1 ha area.",
    ),
    "pre_series_pasture": (
        "Pasture cleared before the series",
        "Random 1 ha squares alerted by RADD at least 90 days before the first NISAR "
        "date (already cleared land: pasture, crops, or regrowth).\n\n"
        "What to look for: the post-disturbance reference, with low NBR, lower HV, and "
        "coherence above forest.",
    ),
}


def _dates(days) -> list:
    return [config.to_date(d) for d in np.atleast_1d(days)]


def _usable_days(s2: data.S2Stack, window) -> list[int]:
    return [t for t in range(len(s2.days)) if s2.usable(t, window)]


def _choose_chips(case, s2: data.S2Stack, window) -> list[int]:
    """Indices of two usable dates before and two after the event (or four spread)."""
    usable = _usable_days(s2, window)
    if not usable:
        return []
    days = s2.days
    if np.isfinite(case.event_start):
        before_all = [t for t in usable if days[t] <= case.event_start]
        before = before_all[-1:]
        if before:
            # second "before" chip at least MIN_CHIP_GAP_D earlier, so the two differ
            earlier = [t for t in before_all if days[t] <= days[before[0]] - MIN_CHIP_GAP_D]
            before = earlier[-1:] + before
        after = [t for t in usable if days[t] >= case.event_end]
        picked = before + (after[:1] + after[-1:] if after else [])
        return sorted(set(picked), key=picked.index)[:4]
    idx = np.linspace(0, len(usable) - 1, 4).round().astype(int)
    return [usable[i] for i in sorted(set(idx))]


def _segments(ax, pairs: data.PairStack, values, color, lw, alpha=1.0):
    for r, s, v in zip(pairs.ref, pairs.sec, values, strict=True):
        if np.isfinite(v):
            ax.plot(_dates([r, s]), [v, v], color=color, lw=lw, alpha=alpha, solid_capstyle="butt")


def _side(case, day: float) -> str | None:
    """Whether a chip date is before or after the case's event."""
    if not np.isfinite(case.event_start):
        return None
    return "before" if day <= case.event_start else "after"


def _match_hv(ds: data.Dataset, case, chip_days: list[int]) -> list[int]:
    """Index of the HV date nearest each chip date, on the same side of the event.

    A date not yet shown is preferred when it is at most ``REUSE_TOLERANCE_D`` days
    farther than the nearest one; otherwise the nearest date is repeated.
    """
    days = ds.hv.days
    used: set[int] = set()
    out = []
    for day in chip_days:
        side = _side(case, day)
        ok = np.ones(len(days), bool)
        if side == "before":
            ok &= days <= case.event_start
        elif side == "after":
            ok &= days >= case.event_end
        cand = np.flatnonzero(ok) if ok.any() else np.arange(len(days))
        dist = np.abs(days[cand] - day)
        order = cand[np.argsort(dist)]
        limit = dist.min() + REUSE_TOLERANCE_D
        pick = next((i for i in order if i not in used and abs(days[i] - day) <= limit), order[0])
        used.add(int(pick))
        out.append(int(pick))
    return out


def _match_pair(pairs: data.PairStack, case, chip_days: list[int]) -> list[int]:
    """Index of the coherence pair closest to each chip date, same side of the event.

    Distance is zero when the chip date falls inside the pair; ties go to the pair
    whose midpoint is nearest; an unused pair is preferred within
    ``REUSE_TOLERANCE_D`` days of the best. "Before" chips only get pairs ending by the event
    start and "after" chips only pairs starting at or after the event end, so no
    chip shows a pair spanning the event.
    """
    used: set[int] = set()
    out = []
    for day in chip_days:
        side = _side(case, day)
        ok = np.ones(len(pairs.ref), bool)
        if side == "before":
            ok &= pairs.sec <= case.event_start
        elif side == "after":
            ok &= pairs.ref >= case.event_end
        cand = np.flatnonzero(ok) if ok.any() else np.arange(len(pairs.ref))
        gap = np.maximum(pairs.ref[cand] - day, 0) + np.maximum(day - pairs.sec[cand], 0)
        mid = np.abs((pairs.ref[cand] + pairs.sec[cand]) / 2 - day)
        order = cand[np.lexsort((mid, gap))]
        best = gap.min()
        dist = dict(zip(cand, gap, strict=True))
        pick = next(
            (i for i in order if i not in used and dist[i] <= best + REUSE_TOLERANCE_D),
            order[0],
        )
        used.add(int(pick))
        out.append(int(pick))
    return out


def _pair_label(ref: int, sec: int) -> str:
    a, b = config.to_date(ref), config.to_date(sec)
    if a.month == b.month:
        return f"{a:%d}–{b:%d %b}"
    return f"{a:%d %b}–{b:%d %b}"


def _case_mask(case, window) -> np.ndarray:
    """Case pixels as a 0/1 array on the 20 m window."""
    mask = np.zeros((window[0].stop - window[0].start, window[1].stop - window[1].start))
    inside = (
        (case.rows >= window[0].start)
        & (case.rows < window[0].stop)
        & (case.cols >= window[1].start)
        & (case.cols < window[1].stop)
    )
    mask[case.rows[inside] - window[0].start, case.cols[inside] - window[1].start] = 1
    return mask


def nbr_series(ds: data.Dataset, s2: data.S2Stack, case, window) -> tuple:
    """Case-mean and stable-forest NBR on dates usable for this case.

    A date counts if the chip window is usable (clear, haze-free) and at least 80%
    of the case pixels are clear.
    """
    days, case_v, forest_v = [], [], []
    sf = ds.masks["stable_forest"][window]
    for t in _usable_days(s2, window):
        v = s2.nbr[t][case.rows, case.cols]
        if np.isfinite(v).mean() < config.S2_MIN_CLEAR:
            continue
        days.append(s2.days[t])
        case_v.append(np.nanmean(v))
        forest_v.append(np.nanmedian(s2.nbr[t][window][sf]) if sf.any() else np.nan)
    return np.array(days), np.array(case_v), np.array(forest_v)


def optical_drop(ds: data.Dataset, s2: data.S2Stack, case) -> tuple[int, int] | None:
    """Dates bracketing the optical clearing: last NBR >= forest level, first after <= cleared.

    Uses the case-mean NBR on dates clear over the case (``nbr_series``). Returns
    ``None`` if the case never looks forested, or never looks cleared afterwards.
    """
    d, v, _ = nbr_series(ds, s2, case, case.window(ds.grid.shape))
    forested = np.flatnonzero(v >= config.NBR_FOREST_MIN)
    if not forested.size:
        return None
    later = [i for i in range(forested[-1] + 1, len(d)) if v[i] <= config.NBR_CLEARED_MAX]
    if not later:
        return None
    return int(d[forested[-1]]), int(d[later[0]])


def spanning_pairs_text(ds: data.Dataset, case, drop: tuple[int, int] | None) -> str:
    """80 m pairs overlapping the optical clearing: forest level and case - forest."""
    if drop is None:
        return "-"
    pairs = ds.coh80
    forest = analysis.forest_reference(ds)["coh80"]
    series = analysis.patch_series(ds, case.rows, case.cols)["coh80"]
    parts = [
        f"{_pair_label(pairs.ref[p], pairs.sec[p])}: {forest[p]:.2f} / {series[p] - forest[p]:+.2f}"
        for p in range(len(pairs.ref))
        if pairs.ref[p] < drop[1] and pairs.sec[p] > drop[0]
    ]
    return "; ".join(parts) or "-"


def _bands(ax, case) -> None:
    if np.isfinite(case.t0):
        ax.axvspan(*_dates([case.t0, case.t1]), color=style.EVENT_BAND, lw=0, zorder=0)
    if np.isfinite(case.dip_start):
        ax.axvspan(
            *_dates([case.dip_start, case.dip_end]),
            facecolor="none",
            edgecolor=DIP_HATCH,
            hatch="///",
            lw=0,
            zorder=0.5,
        )
    if np.isfinite(case.radd_day):
        ax.axvline(_dates(case.radd_day)[0], color=style.INK_2, lw=1.2, ls=":")


def plot_case(ds: data.Dataset, s2: data.S2Stack, case, coh_kind: str = "coh80") -> None:
    """One case figure: chip rows (S2, HV, coherence) above three time-series panels."""
    window = case.window(ds.grid.shape)
    series = analysis.patch_series(ds, case.rows, case.cols)
    pr, pc = case.rep_pixel
    pixel = analysis.patch_series(ds, np.array([pr]), np.array([pc]))
    forest = analysis.forest_reference(ds)
    chips = _choose_chips(case, s2, window)
    chip_days = [int(s2.days[t]) for t in chips]
    coh_pairs: data.PairStack = getattr(ds, coh_kind)
    hv_idx = _match_hv(ds, case, chip_days)
    pair_idx = _match_pair(coh_pairs, case, chip_days)
    scenes = {s.day: s for s in data.s2_scenes()}

    fig = plt.figure(figsize=(12, 17.5))
    gs = fig.add_gridspec(
        6, 5, height_ratios=[1, 1, 1, 0.75, 1, 1], width_ratios=[1, 1, 1, 1, 0.06]
    )
    rgb_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    hv_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    coh_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]
    ax_n = fig.add_subplot(gs[3, :4])
    ax_b = fig.add_subplot(gs[4, :4], sharex=ax_n)
    ax_c = fig.add_subplot(gs[5, :4], sharex=ax_n)

    # --- chip rows: same window and zoom in all three rows ----------------------------
    mask20 = _case_mask(case, window)
    mask10 = np.kron(mask20, np.ones((2, 2)))  # 20 m -> 10 m S2 grid
    hv_im = coh_im = None
    for i in range(4):
        for ax in (rgb_axes[i], hv_axes[i], coh_axes[i]):
            style.image_axes(ax)
        if i >= len(chips):
            for ax in (rgb_axes[i], hv_axes[i], coh_axes[i]):
                ax.set_visible(False)
            continue
        rgb_axes[i].imshow(data.s2_rgb(scenes[chip_days[i]], window), interpolation="nearest")
        rgb_axes[i].contour(mask10, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
        rgb_axes[i].set_title(config.to_date(chip_days[i]).strftime("%d %b %Y"), pad=3)

        h = hv_idx[i]
        hv_im = hv_axes[i].imshow(
            ds.hv.values[h][window],
            cmap="gray",
            vmin=HV_RANGE_DB[0],
            vmax=HV_RANGE_DB[1],
            interpolation="nearest",
        )
        hv_axes[i].contour(mask20, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
        hv_axes[i].set_title(config.to_date(ds.hv.days[h]).strftime("%d %b %Y"), pad=3)

        p = pair_idx[i]
        coh_im = coh_axes[i].imshow(
            coh_pairs.values[p][window],
            cmap="gray",
            vmin=COH_RANGE[0],
            vmax=COH_RANGE[1],
            interpolation="nearest",
        )
        coh_axes[i].contour(mask20, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
        coh_axes[i].set_title(_pair_label(coh_pairs.ref[p], coh_pairs.sec[p]), pad=3)

    rgb_axes[0].set_ylabel("Sentinel-2")
    hv_axes[0].set_ylabel("HV")
    coh_axes[0].set_ylabel(f"Coherence {coh_kind[3:]} m")
    fig.add_subplot(gs[0, 4]).set_visible(False)
    if hv_im is not None:
        fig.colorbar(hv_im, cax=fig.add_subplot(gs[1, 4]), label="dB")
        fig.colorbar(coh_im, cax=fig.add_subplot(gs[2, 4]))

    # --- optical timing ---------------------------------------------------------------
    nd, nc, nf = nbr_series(ds, s2, case, window)
    ax_n.plot(_dates(nd), nf, color=style.FOREST_REF, lw=1.5, ls="--")
    ax_n.plot(_dates(nd), nc, color=style.INK, lw=2.5, marker="o", ms=5)
    ax_n.set_ylabel("NBR")
    ax_n.set_ylim(-0.4, 0.8)

    # --- backscatter ------------------------------------------------------------------
    for pol, color in (("HH", style.HH), ("HV", style.HV)):
        days = getattr(ds, pol.lower()).days
        ax_b.plot(_dates(days), series[pol], color=color, lw=2.5, marker="o", ms=4)
        ax_b.plot(_dates(days), pixel[pol], color=color, lw=1, alpha=0.6)
    ax_b.plot(_dates(ds.hv.days), forest["HV"], color=style.FOREST_REF, lw=1.5, ls="--")
    ax_b.set_ylabel("Backscatter (dB)")

    # --- coherence relative to forest ---------------------------------------------------
    sigma = float(case.noise_sd)
    ax_c.axhspan(-2 * sigma, 2 * sigma, color=style.GRID, lw=0, zorder=0)
    ax_c.axhline(0, color=style.FOREST_REF, lw=1.5)
    _segments(ax_c, ds.coh20, series["coh20"] - forest["coh20"], style.COH20, 3)
    _segments(ax_c, ds.coh80, series["coh80"] - forest["coh80"], style.COH80, 3)
    _segments(ax_c, ds.coh80, pixel["coh80"] - forest["coh80"], style.COH80, 1, alpha=0.7)
    ax_c.set_ylabel("Coherence − forest")
    ax_c.set_ylim(-0.5, 0.7)

    for ax in (ax_n, ax_b, ax_c):
        _bands(ax, case)
    ax_c.xaxis.set_major_locator(mdates.MonthLocator())
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_c.set_xlim(*_dates([ds.first_day - 5, ds.last_day + 5]))
    for ax in (ax_n, ax_b):
        plt.setp(ax.get_xticklabels(), visible=False)

    band_handles = []
    if np.isfinite(case.t0):
        band_handles.append(LegendPatch(color=style.EVENT_BAND, label="HV drop"))
    if np.isfinite(case.dip_start):
        band_handles.append(
            LegendPatch(facecolor="none", edgecolor=DIP_HATCH, hatch="///", label="Coherence dip")
        )
    ax_n.legend(
        handles=[
            Line2D([], [], color=style.INK, lw=2.5, marker="o", label="Case"),
            Line2D([], [], color=style.FOREST_REF, lw=1.5, ls="--", label="Forest"),
            *band_handles,
        ],
        ncol=4,
        loc="lower left",
    )
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
            LegendPatch(color=style.GRID, label="±2σ forest"),
        ],
        ncol=3,
        loc="upper left",
    )
    suffix = "" if coh_kind == "coh80" else "_coh20"
    style.save(fig, OUT / case.category / f"{case.case_id}{suffix}.png")


def write_readmes(ds: data.Dataset, s2: data.S2Stack, cases) -> None:
    by_cat: dict[str, list] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    for cat, items in by_cat.items():
        title, text = CATEGORY_TEXT[cat]
        rows = [
            "| Case | Area (ha) | Outline | Optical clearing (NBR) | Coherence dip | HV drop | "
            "RADD alert | HV step (dB) | Lowest 80 m coherence − forest (sigma) |",
            "|---|--:|---|---|---|---|---|--:|--:|",
        ]
        timing = [
            "| Case | 80 m pairs spanning the optical clearing: forest coherence / case − forest |",
            "|---|---|",
        ]
        for c in items:
            hv = f"{c.hv_bracket_start} to {c.hv_bracket_end}" if c.hv_bracket_start else "-"
            dip = f"{c.coh_dip_start} to {c.coh_dip_end}" if c.coh_dip_start else "-"
            drop = optical_drop(ds, s2, c) if c.hv_bracket_start else None
            opt = f"{config.to_date(drop[0])} to {config.to_date(drop[1])}" if drop else "-"
            rows.append(
                f"| {c.case_id} | {c.area_ha} | {c.delineation} | {opt} | {dip} | {hv} | "
                f"{c.radd_alert or '-'} | {c.hv_step_db} | {c.coh80_min_delta} "
                f"({c.coh80_min_sigma}) |"
            )
            timing.append(f"| {c.case_id} | {spanning_pairs_text(ds, c, drop)} |")
        readme = f"""# {title}

{text}

## Cases

{chr(10).join(rows)}

"Optical clearing" is the interval between the last date the case-mean NBR is
forest-like (>= {config.NBR_FOREST_MIN}) and the first later date it is cleared (<= {config.NBR_CLEARED_MAX}); "-" means the
case never looks forested or never looks cleared on the usable dates (often a
"RADD seed" outline, or clearing before the first usable image).

## Does the clearing show up in the pair that spans it?

{chr(10).join(timing)}

A clearing lowers coherence only if the forest was coherent in that pair to begin
with. Where the spanning pair has forest coherence around 0.45-0.5 the case drops
0.2-0.35 below forest; where it is low (e.g. 0.17 in the 21 Dec - 2 Jan pair, close
to the 80 m estimator floor of about 0.08, most likely rain) there is little room
left and the drop stays within forest noise.

## Reading the figures

- **Outline** (yellow): the whole clearing, taken from Sentinel-2. Pixels whose NBR
  was forest-like (>= {config.NBR_FOREST_MIN}) in the last two usable images at least {config.PRE_GAP_D} days
  before the HV drop, and cleared (<= {config.NBR_CLEARED_MAX}, a drop of at least {config.NBR_DROP_MIN}) in the
  first two usable images after it, connected to the RADD seed. "RADD seed" in the
  table means the optical outline failed (clouds) and the smaller RADD-based seed
  is used instead.
- **Sentinel-2 chips**: true colour, Cloud Score+ masked (clouds light gray), only
  dates at least 80% clear and haze-free (median blue <= 0.06). Two before the
  event (the earlier of the HV drop and the coherence dip; at least 20 days apart)
  and two after. A faint
  seam can appear where two Sentinel-2 granules meet.
- **HV chips**: GCOV HV, -16 (black) to -6 dB (white), the dual-pol date nearest
  each Sentinel-2 date on the same side of the event. Single-date 20 m HV is
  speckled; forest is about -10 dB and cleared land about -12.5 dB.
- **Coherence chips**: HH coherence 0 (black) to 0.8 (white) for the pair closest to
  each Sentinel-2 date, on the same side of the event (never the pair spanning
  it). 80 m in the main figures (4 x 4 blocks of 20 m pixels); `<case>_coh20.png`
  repeats the figure with 20 m coherence for the first case of each clearing
  category.
- **NBR**: Sentinel-2 normalized burn ratio, (B8 - B12) / (B8 + B12); case mean (black)
  and stable-forest median in the window (dashed), on dates clear over the case.
  Forest is about 0.6; felled or burned ground drops below 0.
- **Backscatter**: thick = case mean (linear power), thin = one representative
  pixel, dashed = stable-forest HV median.
- **Coherence − forest**: each pair is a segment from its first to its second date,
  showing the case's coherence minus the stable-forest median for the **same
  pair**. This removes changes that affect the whole scene: the 21 Dec - 2 Jan pair,
  for example, is low everywhere (forest 0.17 against 0.25-0.52 in neighbouring
  pairs), most likely weather, and does not indicate disturbance. Green = 80 m,
  violet = 20 m (case mean), thin green = one pixel at 80 m. The gray band is
  +/-2 sigma of the same quantity for intact-forest areas of the case's size.
  Sigma is about 0.06-0.075 and barely shrinks with area, so forest coherence
  varies coherently in space rather than only as estimation noise.
- **Bands**: orange fill = HV drop (last dual-pol date before and first after the
  fitted HV step); green hatching = coherence dip (pairs more than {config.DIP_SIGMA:g} sigma below
  forest). **Dotted line**: median in-series RADD alert date for the case (RADD
  lags the NISAR HV drop by about two weeks on average).

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
    s2 = data.load_s2_stack()
    cases = select_cases.read()
    variants = {
        next(c.case_id for c in cases if c.category == cat)
        for cat in COH20_VARIANT_CATEGORIES
        if any(c.category == cat for c in cases)
    }
    for case in cases:
        plot_case(ds, s2, case, "coh80")
        if case.case_id in variants:
            plot_case(ds, s2, case, "coh20")
        logger.info("Wrote %s", case.case_id)
    write_readmes(ds, s2, cases)
    write_index_readme(cases)


if __name__ == "__main__":
    main()
