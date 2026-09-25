#!/usr/bin/env python
"""Case-study figures: image chips and time series for each clearing and control.

For every case in ``cases.csv`` (see select_cases.py) this writes one PNG to
``local_examples/caqueta/04_cases/<category>/forest_<reference>/<case_id>.png``:

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

Figures go to ``<category>/forest_<reference>/``: ``scene`` removes weather with the
stable-forest median over the study area, ``ring`` with intact forest around each
case.

Usage:
    python scripts/caqueta/fig04_case_studies.py                 # both references
    python scripts/caqueta/fig04_case_studies.py --reference ring
"""

# ruff: noqa: E501  (long lines are Markdown in README templates)
from __future__ import annotations

import argparse
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
COH20_VARIANT_CATEGORIES = (
    "forest_clearing",
    "cleared_land_disturbance",
    "clearing_in_low_coherence_pair",
)
HV_RANGE_DB = (-16.0, -6.0)  # forest about -10 dB, pasture about -12.5 dB
COH_RANGE = (0.0, 0.8)
DIP_HATCH = style.COH80  # coherence-dip period: green hatching over the HV band
MIN_CHIP_GAP_D = 20  # the two "before" chips are at least this far apart
REUSE_TOLERANCE_D = 12
CHIP_S2_TOLERANCE_D = 12  # S2 image may lie this far outside a chip's pair  # take an unused date/pair only if at most this much farther

CATEGORY_TEXT = {
    "forest_clearing": (
        "Forest clearing with a coherence dip",
        "Forest before the event (case-mean Sentinel-2 NBR >= "
        f"{config.NBR_FOREST_MIN}), at least one 80 m coherence pair more than "
        "DIP_SIGMA_X sigma below the case's own recent level, and evidence of "
        f"clearing (NBR drop to <= {config.NBR_CLEARED_MAX}, or an HV step >= "
        f"{config.PATCH_STEP_DB} dB). Ranked by the depth of the dip.\n\n"
        "What to look for: a large dip in the pair(s) spanning the felling (hatched), "
        "sometimes preceded by a smaller dip (early degradation or understory "
        "clearing), then a large, sustained rise above forest once the surface is "
        "non-forest. Compare the dip timing with the NBR drop and the later HV drop.",
    ),
    "cleared_land_disturbance": (
        "Disturbance of already-cleared land (e.g. fire)",
        "Non-forest before the event (NBR below forest level, or coherence well "
        "above forest), with a flagged coherence dip. These are typically burns of "
        "felled vegetation or pasture. Ranked by the depth of the dip.\n\n"
        "What to look for: coherence already above forest (a stable, non-forest "
        "surface) that collapses for one pair, often to about forest level, then "
        "recovers. Measured against intact forest this looks like no change, which "
        "is why dips are measured against each case's own history.",
    ),
    "degradation": (
        "Dip in forest without clearing",
        "Forest before the event, a flagged coherence dip, but no optical clearing "
        f"(NBR never falls to <= {config.NBR_CLEARED_MAX} in the usable images) and an HV "
        f"step < {config.PATCH_STEP_DB} dB. Candidates are degradation (selective or "
        "understory clearing) or clearings hidden by cloud in the optical record. "
        "Ranked by the depth of the dip.\n\n"
        "What to look for: whether the chips show thinning or partial clearing, and "
        "whether coherence rises afterwards (non-forest) or returns to forest level.",
    ),
    "clearing_in_low_coherence_pair": (
        "Clearing inside a low-coherence pair",
        "Forest cleared (NBR drop or HV step) with no flagged dip, where the pairs "
        f"spanning the clearing had stable-forest coherence below {config.LOW_FOREST_COH} "
        "(e.g. 0.17 in the rainy 21 Dec - 2 Jan pair, close to the 80 m estimator "
        "floor of about 0.08). Ranked by HV step.\n\n"
        "What to look for: coherence is low in the spanning pair, but so is intact "
        "forest, so the clearing cannot pull it much lower. The event is dated by "
        "the sharp rise in the following pair instead.",
    ),
    "clearing_without_dip": (
        "Clearing without a coherence dip",
        "Forest cleared (NBR drop or HV step) with no flagged dip even though the "
        f"forest was coherent (>= {config.LOW_FOREST_COH}) in the spanning pairs. Ranked by "
        "HV step.\n\n"
        "What to look for: whether the clearing was gradual (spread over several "
        "pairs, each below the threshold), whether the outline mixes cleared and "
        "intact pixels, or whether the optical and HV dates disagree.",
    ),
    "radd_only": (
        "RADD alert without a NISAR or optical response",
        "Areas with a RADD high-confidence alert during the series but no flagged "
        f"coherence dip, no optical clearing, and an HV step < {config.PATCH_NO_STEP_DB} dB.\n\n"
        "What to look for: whether the chips show any change. Candidates are "
        "clearing before the series, small or partial disturbances, or RADD "
        "commission errors.",
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
        "What to look for: the post-disturbance reference, with low NBR, lower HV, "
        "and coherence above forest, plus any burns during the series.",
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
    """Case-mean NBR (``analysis.case_nbr``) and the stable-forest median in the window."""
    days, vals = analysis.case_nbr(s2, case.rows, case.cols, ds.grid.shape)
    sf = ds.masks["stable_forest"][window]
    idx = {d: t for t, d in enumerate(s2.days)}
    forest = [np.nanmedian(s2.nbr[idx[d]][window][sf]) if sf.any() else np.nan for d in days]
    return days, vals, np.array(forest)


def spanning_pairs_text(ds: data.Dataset, case, reference: str) -> str:
    """80 m pairs overlapping the optical clearing: forest level and case - forest."""
    lo, hi = case._day("optical_start"), case._day("optical_end")
    if not np.isfinite(lo):
        return "-"
    pairs = ds.coh80
    forest = analysis.reference_series(ds, case.rows, case.cols, reference)["coh80"]
    series = analysis.patch_series(ds, case.rows, case.cols)["coh80"]
    parts = [
        f"{_pair_label(pairs.ref[p], pairs.sec[p])}: {forest[p]:.2f} / {series[p] - forest[p]:+.2f}"
        for p in range(len(pairs.ref))
        if pairs.ref[p] < hi and pairs.sec[p] > lo
    ]
    return "; ".join(parts) or "-"


def _bands(ax, case) -> None:
    if np.isfinite(case.t0):
        ax.axvspan(*_dates([case.t0, case.t1]), color=style.EVENT_BAND, lw=0, zorder=0)
    for ref, sec, _ in case.dip_pairs:
        ax.axvspan(
            *_dates([ref, sec]),
            facecolor="none",
            edgecolor=DIP_HATCH,
            hatch="///",
            lw=0,
            zorder=0.5,
        )
    if np.isfinite(case.radd_day):
        ax.axvline(_dates(case.radd_day)[0], color=style.INK_2, lw=1.2, ls=":")


def _nearest_hv(ds: data.Dataset, day: float) -> int:
    return int(np.argmin(np.abs(ds.hv.days - day)))


def _columns(ds: data.Dataset, s2: data.S2Stack, case, window, pairs: data.PairStack) -> list:
    """Chip columns as dicts with ``s2`` (index or None), ``hv``, ``pair``, ``dip``.

    With a flagged coherence dip, the columns follow the deepest dip: the pair
    before it, the dip pair, the pair after it, and the last pair (post-
    disturbance). Each column's Sentinel-2 image is the usable date inside that
    pair nearest its midpoint (or within ``CHIP_S2_TOLERANCE_D`` of the pair),
    and its HV date is the dual-pol date nearest that image (or the pair
    midpoint). Without a dip, columns are chosen from Sentinel-2 dates around the
    event as before.
    """
    if not case.dip_pairs:
        chips = _choose_chips(case, s2, window)
        days = [int(s2.days[t]) for t in chips]
        hv = _match_hv(ds, case, days)
        pr = _match_pair(pairs, case, days)
        return [
            {"s2": t, "hv": h, "pair": q, "dip": False}
            for t, h, q in zip(chips, hv, pr, strict=True)
        ]
    ref, sec, _ = min(case.dip_pairs, key=lambda d: d[2])
    d = int(np.flatnonzero((pairs.ref == ref) & (pairs.sec == sec))[0])
    n = len(pairs.ref)
    order = []
    for q in (d - 1, d, d + 1, n - 1):
        q = min(max(q, 0), n - 1)
        if q not in order:
            order.append(q)
    usable = _usable_days(s2, window)
    cols = []
    for q in order:
        r, e = pairs.ref[q], pairs.sec[q]
        mid = (r + e) / 2
        near = [
            t for t in usable if r - CHIP_S2_TOLERANCE_D <= s2.days[t] <= e + CHIP_S2_TOLERANCE_D
        ]
        t = min(near, key=lambda t: abs(s2.days[t] - mid)) if near else None
        hv = _nearest_hv(ds, s2.days[t] if t is not None else mid)
        cols.append({"s2": t, "hv": hv, "pair": q, "dip": q == d})
    return cols


def plot_case(
    ds: data.Dataset,
    s2: data.S2Stack,
    case,
    coh_kind: str = "coh80",
    reference: str = "ring",
) -> None:
    """One case figure: chip rows (S2, HV, coherence) above three time-series panels."""
    window = case.window(ds.grid.shape)
    series = analysis.patch_series(ds, case.rows, case.cols)
    pr, pc = case.rep_pixel
    pixel = analysis.patch_series(ds, np.array([pr]), np.array([pc]))
    forest = analysis.forest_reference(ds)
    coh_pairs: data.PairStack = getattr(ds, coh_kind)
    columns = _columns(ds, s2, case, window, coh_pairs)
    ref_coh = analysis.reference_series(ds, case.rows, case.cols, reference)
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
        if i >= len(columns):
            for ax in (rgb_axes[i], hv_axes[i], coh_axes[i]):
                ax.set_visible(False)
            continue
        col = columns[i]
        if col["s2"] is not None:
            day = int(s2.days[col["s2"]])
            rgb_axes[i].imshow(data.s2_rgb(scenes[day], window), interpolation="nearest")
            rgb_axes[i].contour(mask10, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
            rgb_axes[i].set_title(config.to_date(day).strftime("%d %b %Y"), pad=3)
        else:
            rgb_axes[i].set_title("no clear image", pad=3, color=style.INK_2)

        h = col["hv"]
        hv_im = hv_axes[i].imshow(
            ds.hv.values[h][window],
            cmap="gray",
            vmin=HV_RANGE_DB[0],
            vmax=HV_RANGE_DB[1],
            interpolation="nearest",
        )
        hv_axes[i].contour(mask20, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
        hv_axes[i].set_title(config.to_date(ds.hv.days[h]).strftime("%d %b %Y"), pad=3)

        q = col["pair"]
        coh_im = coh_axes[i].imshow(
            coh_pairs.values[q][window],
            cmap="gray",
            vmin=COH_RANGE[0],
            vmax=COH_RANGE[1],
            interpolation="nearest",
        )
        coh_axes[i].contour(mask20, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
        coh_axes[i].set_title(_pair_label(coh_pairs.ref[q], coh_pairs.sec[q]), pad=3)
        if col["dip"]:
            # frame the dip pair in the colour of the hatched dip band
            for spine in coh_axes[i].spines.values():
                spine.set_visible(True)
                spine.set_color(DIP_HATCH)
                spine.set_linewidth(4)

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
    # Dip threshold per pair: the case's own recent level (median of the previous
    # pairs) minus DIP_SIGMA x the forest change noise for this area.
    delta80 = series["coh80"] - ref_coh["coh80"]
    threshold = (
        delta80
        - analysis.own_history_change(delta80)
        - config.DIP_SIGMA[reference] * float(case.change_sd)
    )
    _segments(ax_c, ds.coh80, threshold, style.INK_2, 1.2, ls="--")
    _segments(ax_c, ds.coh20, series["coh20"] - ref_coh["coh20"], style.COH20, 3)
    _segments(ax_c, ds.coh80, delta80, style.COH80, 3)
    _segments(ax_c, ds.coh80, pixel["coh80"] - ref_coh["coh80"], style.COH80, 1, alpha=0.7)
    ax_c.set_ylabel("Coherence − forest" if reference == "scene" else "Coherence − nearby forest")
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
    if case.dip_pairs:
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
            LegendPatch(color=style.GRID, label="±2σ forest level"),
            Line2D([], [], color=style.INK_2, lw=1.2, ls="--", label="Dip threshold"),
        ],
        ncol=4,
        loc="upper left",
    )
    suffix = "" if coh_kind == "coh80" else "_coh20"
    style.save(fig, OUT / case.category / f"forest_{reference}" / f"{case.case_id}{suffix}.png")


REFERENCE_TEXT = {
    "scene": (
        "the stable-forest median over the whole study area for the same pair "
        "(the same reference for every case)"
    ),
    "ring": (
        f"the mean of intact forest {config.RING_INNER_PX * config.PIXEL_M:.0f}-"
        f"{config.RING_OUTER_PX * config.PIXEL_M:.0f} m around the case for the same pair. Rain is "
        "patchy, so nearby forest shares the case's weather much more closely than the "
        "scene-wide median: the noise of the reference-corrected coherence is 2-3 times "
        "smaller"
    ),
}


def write_readmes(ds: data.Dataset, cases, reference: str) -> None:
    by_cat: dict[str, list] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    sigma = config.DIP_SIGMA[reference]
    false_rate = config.DIP_FALSE_RATE[reference]
    for cat, items in by_cat.items():
        title, text = CATEGORY_TEXT[cat]
        text = text.replace("DIP_SIGMA_X", f"{sigma:g}")
        rows = [
            "| Case | Area (ha) | Outline | Before | Coherence dips (sigma) | "
            "Optical clearing (NBR) | HV drop | RADD alert | HV step (dB) |",
            "|---|--:|---|---|---|---|---|---|--:|",
        ]
        timing = [
            "| Case | 80 m pairs spanning the optical clearing: reference coherence / "
            "case − reference |",
            "|---|---|",
        ]
        context = [
            "| Case | Before: pairs ending by (n) | 80 m before: case / ring | "
            "20 m before: case / ring | 80 m minimum: case / ring (pair) | "
            "20 m minimum: case / ring (pair) |",
            "|---|---|--:|--:|---|---|",
        ]
        for c in items:
            hv = f"{c.hv_bracket_start} to {c.hv_bracket_end}" if c.hv_bracket_start else "-"
            dips = (
                "; ".join(f"{_pair_label(r, s_)} ({sig:+.1f})" for r, s_, sig in c.dip_pairs) or "-"
            )
            opt = f"{c.optical_start} to {c.optical_end}" if c.optical_start else "-"
            rows.append(
                f"| {c.case_id} | {c.area_ha} | {c.delineation} | {c.state_before} | {dips} | "
                f"{opt} | {hv} | {c.radd_alert or '-'} | {c.hv_step_db} |"
            )
            timing.append(f"| {c.case_id} | {spanning_pairs_text(ds, c, reference)} |")
            before = c._day("optical_start") if c.optical_start else c.event_start
            if np.isfinite(before):
                ctx = analysis.coherence_context(ds, c.rows, c.cols, before)
                a, b = ctx["coh80"], ctx["coh20"]
                context.append(
                    f"| {c.case_id} | {config.to_date(before)} ({a['n_pre']}) | "
                    f"{a['pre_inside']:.2f} / {a['pre_ring']:.2f} | "
                    f"{b['pre_inside']:.2f} / {b['pre_ring']:.2f} | "
                    f"{a['min_inside']:.2f} / {a['min_ring']:.2f} "
                    f"({_pair_label(a['min_ref'], a['min_sec'])}) | "
                    f"{b['min_inside']:.2f} / {b['min_ring']:.2f} "
                    f"({_pair_label(b['min_ref'], b['min_sec'])}) |"
                )
        readme = f"""# {title}

**Weather reference: `{reference}`.** Coherence is compared with {REFERENCE_TEXT[reference]}.
Figures using the other reference are in `../forest_{"ring" if reference == "scene" else "scene"}/`.

{text}

## Cases

{chr(10).join(rows)}

"Optical clearing" is the interval between the last date the case-mean NBR is
forest-like (>= {config.NBR_FOREST_MIN}) and the first later date it is cleared (<= {config.NBR_CLEARED_MAX}); "-" means the
case never looks forested or never looks cleared on the usable dates.

## Coherence before and at its lowest, against surrounding forest

{chr(10).join(context) if len(context) > 2 else "No event dates for this category."}

"Case" is the mean coherence over the outline; "ring" is intact forest from
{config.RING_INNER_PX * config.PIXEL_M:.0f} to {config.RING_OUTER_PX * config.PIXEL_M:.0f} m outside it (no RADD alerts; the {config.RING_INNER_PX * config.PIXEL_M:.0f} m gap keeps 80 m cells that
straddle the outline out of the ring). "Before" averages the pairs ending on or
before the last forest-like Sentinel-2 date (the start of the optical clearing),
or the event start where there is no optical interval. "Minimum" is the pair with
the lowest case coherence.

## Does the clearing show up in the pair that spans it?

{chr(10).join(timing)}

A clearing lowers coherence only if the forest was coherent in that pair to begin
with; where the reference forest is itself close to the 80 m estimator floor (about
0.08), there is little room left for a dip.

## Reading the figures

- **Outline** (yellow): the whole disturbed area from Sentinel-2: pixels whose NBR
  dropped by at least {config.NBR_DROP_MIN} and ended <= {config.NBR_CLEARED_MAX} between the last two usable images at
  least {config.PRE_GAP_D} days before the HV drop and the first two after it, connected to the
  RADD seed ("RADD seed" = optical outline failed, the seed is used). "Before" is
  the land state before the event from the case-mean NBR (forest if >= {config.NBR_FOREST_MIN}).
- **Chip columns**: when a coherence dip is flagged, the four columns follow the
  deepest dip: the pair before it, **the dip pair (framed in green, matching the
  hatched band)**, the pair after it, and the last pair of the series. Each
  column's Sentinel-2 image is the clear date inside that pair (or within
  {CHIP_S2_TOLERANCE_D} days of it; "no clear image" otherwise) and its HV image the nearest
  dual-pol date. Without a dip, two columns come before the event and two after
  it, matched to clear Sentinel-2 dates.
- **Sentinel-2 chips**: true colour, Cloud Score+ masked (clouds light gray), at
  least 80% clear and haze-free (median blue <= 0.06). A faint seam can appear
  where two Sentinel-2 granules meet.
- **HV chips**: GCOV HV, -16 (black) to -6 dB (white). Single-date 20 m HV is
  speckled; forest is about -10 dB and cleared land about -12.5 dB.
- **Coherence chips**: HH coherence 0 (black) to 0.8 (white). 80 m in the main
  figures (4 x 4 blocks of 20 m pixels); `<case>_coh20.png` repeats the figure with
  20 m coherence for the first case of the main clearing categories.
- **NBR**: Sentinel-2 normalized burn ratio, (B8 - B12) / (B8 + B12); case mean
  (black) and stable-forest median in the window (dashed). Forest is about 0.6;
  felled or burned ground drops below 0.
- **Backscatter**: thick = case mean (linear power), thin = one representative
  pixel, dashed = stable-forest HV median.
- **Coherence panel**: each pair is a segment from its first to its second date,
  showing the case's coherence minus the reference for the same pair. Green = 80 m,
  violet = 20 m (case mean), thin green = one pixel at 80 m. **Dashed segments**
  are the dip threshold per pair: the case's own recent level (median of its
  previous {config.CHANGE_BASELINE_PAIRS} pairs) minus {sigma:g} sigma of the change noise for intact-forest
  areas of the case's size; a green segment below its dashed segment is a flagged
  dip. The **gray band** is +/-2 sigma of the *level* (case minus reference) for
  intact forest, i.e. how far an intact area normally sits from the reference;
  values well above it indicate non-forest.
- **Bands**: orange fill = HV drop (last dual-pol date before and first after the
  fitted HV step); green hatching = each flagged coherence dip. At {sigma:g} sigma
  with the `{reference}` reference, {false_rate} of intact-forest areas show any flagged
  dip over the whole series. **Dotted line**: median in-series RADD alert date
  (RADD lags the NISAR HV drop by about two weeks on average).

Regenerate with `python scripts/caqueta/select_cases.py --reference {reference}` then
`python scripts/caqueta/fig04_case_studies.py --reference {reference}`.
"""
        folder = OUT / cat / f"forest_{reference}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "README.md").write_text(readme)


def write_index_readme() -> None:
    lines = [
        "# Case studies",
        "",
        "One folder per category. Inside each, `forest_scene/` and `forest_ring/` hold "
        "the figures for the two weather references used to detect coherence dips:",
        "",
        "- `forest_scene`: coherence minus the stable-forest median over the whole study area",
        "- `forest_ring`: coherence minus intact forest 80-300 m around each case "
        "(lower noise, because rain is patchy)",
        "",
        "Each subfolder has a README describing the selection rule and what to look "
        "for. `cases_forest_<reference>.csv` lists every case with its metrics; the "
        "matching `.npz` holds the pixel indices. The two references select different "
        "cases, so `forest_clearing_1` in one is not the same clearing as in the other.",
        "",
        "Case numbers match the labels on `../01_site/case_locations_forest_<reference>.png`.",
    ]
    for reference in config.REFERENCES:
        if not select_cases.csv_path(reference).exists():
            continue
        lines += ["", f"## `{reference}` reference", "", "| # | Case |", "|--:|---|"]
        lines += [f"| {c.number} | {c.case_id} |" for c in select_cases.read(reference)]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reference",
        choices=config.REFERENCES,
        nargs="+",
        default=list(config.REFERENCES),
        help="Weather reference(s) to plot (reads cases_forest_<reference>.csv)",
    )
    args = parser.parse_args()
    style.apply()
    ds = data.load()
    s2 = data.load_s2_stack()
    for reference in args.reference:
        cases = select_cases.read(reference)
        variants = {
            next(c.case_id for c in cases if c.category == cat)
            for cat in COH20_VARIANT_CATEGORIES
            if any(c.category == cat for c in cases)
        }
        for case in cases:
            plot_case(ds, s2, case, "coh80", reference)
            if case.case_id in variants:
                plot_case(ds, s2, case, "coh20", reference)
            logger.info("Wrote %s (%s)", case.case_id, reference)
        write_readmes(ds, cases, reference)
    write_index_readme()


if __name__ == "__main__":
    main()
