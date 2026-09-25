#!/usr/bin/env python
"""Simplified forest-clearing figures (20 m coherence, forest-ring reference).

A lighter version of the fig04 case-study figure for presenting to a general
audience. For each ``forest_clearing`` case it writes one PNG to
``local_examples/caqueta/05_lab_examples/forest_clearing/<case_id>.png``:

- three chip rows (Sentinel-2, HV, 20 m HH coherence) with the image in which each
  sensor first detects the disturbance framed in green
- NBR, HH/HV backscatter, and 20 m coherence time series, with one green line per
  panel at that sensor's first detection
- one shared legend between the chips and the time series

First detection per sensor:

- NBR: the first date the case-mean NBR is <= NBR_CLEARED_MAX after having been
  forest-like (``analysis.optical_drop``)
- HV: the first dual-pol date after the fitted HV step (``analysis.hv_step``)
- coherence: the second date of the first 20 m pair whose own-history change
  (ring-referenced) is below -DIP_SIGMA["ring"] x the 20 m forest change noise

Cases come from ``analysis.select_cases`` with the ring reference and up to
``N_CASES`` forest clearings (the first 8 are the fig04 cases); the list is cached
as ``04_cases/cases_forest_ring_lab.csv/.npz``.

Usage:
    python scripts/caqueta/fig05_lab_examples.py
    python scripts/caqueta/fig05_lab_examples.py --reselect   # redo case selection
"""

# ruff: noqa: E501  (long lines are Markdown in the README template)
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import analysis
import config
import data
import fig04_case_studies as fig04
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import select_cases
import style
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REFERENCE = "ring"
CATEGORY = "forest_clearing"
N_CASES = 16
SUFFIX = "_lab"
OUT = config.OUT_DIR / "05_lab_examples" / CATEGORY
CLUSTER_D = 15  # detections this close share a chip column
DETECT = style.COH80  # green: detection line and chip frame
RAW_COH = "#e87ba4"  # magenta: unadjusted coherence (validated against violet and green)
ROWS = ("s2", "hv", "coh")


@dataclass
class Detections:
    """Day of first detection per sensor (NaN if none) and the 20 m dip."""

    s2: float
    hv: float
    coh: float
    coh_pair: int | None  # index into ds.coh20
    coh_sigma: float
    last_before: float  # last observation of any sensor before its detected change

    def get(self, row: str) -> float:
        return getattr(self, row)


def load_cases(ds: data.Dataset, s2: data.S2Stack, reselect: bool) -> list:
    """Forest-clearing cases (ring reference), selecting and caching them if needed."""
    if reselect or not select_cases.csv_path(REFERENCE, SUFFIX).exists():
        patches = analysis.select_cases(
            ds,
            s2,
            REFERENCE,
            per_category={CATEGORY: N_CASES},
            categories=(CATEGORY,),
        )
        select_cases.write(patches, ds.grid, REFERENCE, SUFFIX)
    return [c for c in select_cases.read(REFERENCE, SUFFIX) if c.category == CATEGORY]


def check_original(cases: list) -> set[str]:
    """Log whether the first cases match the fig04 ring cases pixel for pixel."""
    original = {c.case_id: c for c in select_cases.read(REFERENCE) if c.category == CATEGORY}
    same = set()
    for c in cases:
        o = original.get(c.case_id)
        if o is None:
            continue
        if np.array_equal(o.rows, c.rows) and np.array_equal(o.cols, c.cols):
            same.add(c.case_id)
        else:
            logger.warning("%s differs from the fig04 case of the same name", c.case_id)
    logger.info("%d of %d original cases reproduced", len(same), len(original))
    return same


def detect(ds: data.Dataset, case) -> tuple[Detections, np.ndarray, np.ndarray]:
    """First detection per sensor, plus the case 20 m coherence and its ring delta."""
    coh = analysis.patch_series(ds, case.rows, case.cols)["coh20"]
    delta = coh - analysis.reference_series(ds, case.rows, case.cols, REFERENCE)["coh20"]
    change = analysis.own_history_change(delta)
    _, change_sd = analysis.forest_noise(ds, len(case.rows), kind="coh20", reference=REFERENCE)
    flagged = np.flatnonzero(change < -config.DIP_SIGMA[REFERENCE] * change_sd)
    p = int(flagged[0]) if flagged.size else None
    det = Detections(
        s2=case._day("optical_end"),
        hv=case.t1,
        coh=float(ds.coh20.sec[p]) if p is not None else np.nan,
        coh_pair=p,
        coh_sigma=float(change[p] / change_sd) if p is not None else np.nan,
        last_before=float(
            np.nanmin(
                [
                    case._day("optical_start"),
                    case.t0,
                    float(ds.coh20.ref[p]) if p is not None else np.nan,
                ]
            )
        ),
    )
    return det, coh, delta


def _clusters(days: list[float]) -> list[float]:
    """Group sorted days closer than CLUSTER_D; return each group's mean."""
    groups: list[list[float]] = []
    for d in sorted(days):
        if groups and d - groups[-1][-1] <= CLUSTER_D:
            groups[-1].append(d)
        else:
            groups.append([d])
    return [float(np.mean(g)) for g in groups]


def chip_columns(obs: dict[str, np.ndarray], det: Detections) -> dict[str, list]:
    """Observation index per row for the four chip columns.

    ``obs[row]`` holds each row's observation days (coherence: the pair's second
    date). Columns are anchored in time: the last undisturbed observation (at or
    before ``det.last_before``, else the row's first date), the first two groups
    of detection dates, and the last date. Each row's own
    detection goes in the column nearest its date; other cells take the row's
    observation nearest the anchor, without repeats and in date order (a cell
    with no such observation is left empty).
    """
    found = [det.get(r) for r in ROWS if np.isfinite(det.get(r))]
    mid = _clusters(found)[:2]
    out = {}
    for row in ROWS:
        days = obs[row]
        own = det.get(row)
        own_i = int(np.flatnonzero(days == own)[0]) if np.isfinite(own) else None
        cells: list[int | None] = [None] * 4
        if own_i is not None:
            col = 1 + int(np.argmin([abs(own - a) for a in mid]))
            cells[col] = own_i
        before = np.flatnonzero(days <= det.last_before)
        cells[0] = int(before[-1]) if before.size else 0
        if cells[0] == own_i:
            cells[0] = None
        if cells[3] is None and len(days) - 1 not in cells:
            cells[3] = len(days) - 1
        for col in (1, 2):
            if cells[col] is not None:
                continue
            # keep the row in date order: strictly between its filled neighbours
            lo = max((days[c] for c in cells[:col] if c is not None), default=-np.inf)
            hi = min((days[c] for c in cells[col + 1 :] if c is not None), default=np.inf)
            free = [i for i in range(len(days)) if i not in cells and lo < days[i] < hi]
            if col < len(mid):
                anchor = mid[col - 1]
            else:  # one detection group: show the observation after it
                prev = cells[1]
                anchor = days[prev] + 1 if prev is not None else mid[0] + CLUSTER_D
                free = [i for i in free if days[i] >= anchor] or free
            if free:
                cells[col] = min(free, key=lambda i: abs(days[i] - anchor))
        out[row] = cells
    return out


def _frame(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DETECT)
        spine.set_linewidth(4)


def _pair_steps(ax, pairs: data.PairStack, values: np.ndarray, color: str) -> None:
    """One segment per pair, joined end-to-start by thin dashed connectors."""
    fig04._segments(ax, pairs, values, color, 3)
    ok = np.flatnonzero(np.isfinite(values))
    for a, b in zip(ok[:-1], ok[1:], strict=True):
        ax.plot(
            fig04._dates([pairs.sec[a], pairs.ref[b]]),
            [values[a], values[b]],
            color=color,
            lw=1.2,
            ls="--",
        )


def _mark(ax, day: float) -> None:
    if np.isfinite(day):
        ax.axvline(fig04._dates(day)[0], color=DETECT, lw=2.5, zorder=3)


def plot_case(ds: data.Dataset, s2: data.S2Stack, case, out_dir=OUT) -> Detections:
    window = case.window(ds.grid.shape)
    series = analysis.patch_series(ds, case.rows, case.cols)
    forest = analysis.forest_reference(ds)
    det, coh_raw, coh_delta = detect(ds, case)
    scenes = {s.day: s for s in data.s2_scenes()}

    # Sentinel-2 dates: clear over the chip window, plus the NBR detection date
    s2_idx = fig04._usable_days(s2, window)
    if np.isfinite(det.s2):
        s2_idx = sorted(set(s2_idx) | set(np.flatnonzero(s2.days == det.s2)))
    obs = {
        "s2": s2.days[s2_idx].astype(float),
        "hv": ds.hv.days.astype(float),
        "coh": ds.coh20.sec.astype(float),
    }
    cols = chip_columns(obs, det)

    fig = plt.figure(figsize=(12, 16.5))
    gs = fig.add_gridspec(
        7, 5, height_ratios=[1, 1, 1, 0.22, 0.75, 0.9, 0.9], width_ratios=[1, 1, 1, 1, 0.06]
    )
    chip_axes = {row: [fig.add_subplot(gs[k, i]) for i in range(4)] for k, row in enumerate(ROWS)}
    ax_leg = fig.add_subplot(gs[3, :4])
    ax_n = fig.add_subplot(gs[4, :4])
    ax_b = fig.add_subplot(gs[5, :4], sharex=ax_n)
    ax_c = fig.add_subplot(gs[6, :4], sharex=ax_n)

    # --- chips ---------------------------------------------------------------------
    mask20 = fig04._case_mask(case, window)
    mask10 = np.kron(mask20, np.ones((2, 2)))
    hv_im = coh_im = None
    for row in ROWS:
        for i, ax in enumerate(chip_axes[row]):
            style.image_axes(ax)
            j = cols[row][i]
            if j is None:
                ax.set_visible(False)
                continue
            if row == "s2":
                day = int(obs["s2"][j])
                ax.imshow(data.s2_rgb(scenes[day], window), interpolation="nearest")
                ax.contour(mask10, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
                ax.set_title(config.to_date(day).strftime("%d %b %Y"), pad=3)
            elif row == "hv":
                hv_im = ax.imshow(
                    ds.hv.values[j][window],
                    cmap="gray",
                    vmin=fig04.HV_RANGE_DB[0],
                    vmax=fig04.HV_RANGE_DB[1],
                    interpolation="nearest",
                )
                ax.contour(mask20, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
                ax.set_title(config.to_date(ds.hv.days[j]).strftime("%d %b %Y"), pad=3)
            else:
                coh_im = ax.imshow(
                    ds.coh20.values[j][window],
                    cmap="gray",
                    vmin=fig04.COH_RANGE[0],
                    vmax=fig04.COH_RANGE[1],
                    interpolation="nearest",
                )
                ax.contour(mask20, levels=[0.5], colors=style.OUTLINE, linewidths=1.5)
                ax.set_title(fig04._pair_label(ds.coh20.ref[j], ds.coh20.sec[j]), pad=3)
            if obs[row][j] == det.get(row):
                _frame(ax)
    chip_axes["s2"][0].set_ylabel("Sentinel-2")
    chip_axes["hv"][0].set_ylabel("HV")
    chip_axes["coh"][0].set_ylabel("Coherence 20 m")
    fig.add_subplot(gs[0, 4]).set_visible(False)
    if hv_im is not None:
        fig.colorbar(hv_im, cax=fig.add_subplot(gs[1, 4]), label="dB")
    if coh_im is not None:
        fig.colorbar(coh_im, cax=fig.add_subplot(gs[2, 4]))

    # --- NBR -------------------------------------------------------------------------
    nd, nc, nf = fig04.nbr_series(ds, s2, case, window)
    ax_n.plot(fig04._dates(nd), nf, color=style.FOREST_REF, lw=1.5, ls="--")
    ax_n.plot(fig04._dates(nd), nc, color=style.INK, lw=2.5, marker="o", ms=5)
    ax_n.set_ylabel("NBR")
    ax_n.set_ylim(min(-0.4, np.nanmin(nc) - 0.05), 0.8)
    _mark(ax_n, det.s2)

    # --- backscatter -----------------------------------------------------------------
    for pol, color in (("HH", style.HH), ("HV", style.HV)):
        days = getattr(ds, pol.lower()).days
        ax_b.plot(fig04._dates(days), series[pol], color=color, lw=2.5, marker="o", ms=4)
    ax_b.plot(fig04._dates(ds.hv.days), forest["HV"], color=style.FOREST_REF, lw=1.5, ls="--")
    ax_b.set_ylabel("Backscatter (dB)")
    _mark(ax_b, det.hv)

    # --- coherence -------------------------------------------------------------------
    ax_c.axhline(0, color=style.FOREST_REF, lw=1)
    _pair_steps(ax_c, ds.coh20, coh_raw, RAW_COH)
    _pair_steps(ax_c, ds.coh20, coh_delta, style.COH20)
    ax_c.set_ylabel("Coherence (20 m)")
    ax_c.set_ylim(-0.4, 0.8)
    _mark(ax_c, det.coh)

    ax_c.xaxis.set_major_locator(mdates.MonthLocator())
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_c.set_xlim(*fig04._dates([ds.first_day - 5, ds.last_day + 5]))
    for ax in (ax_n, ax_b):
        plt.setp(ax.get_xticklabels(), visible=False)

    # --- shared legend ---------------------------------------------------------------
    ax_leg.axis("off")
    handles = [
        Line2D([], [], color=style.INK, lw=2.5, marker="o", label="Disturbance polygon"),
        Line2D([], [], color=style.FOREST_REF, lw=1.5, ls="--", label="Stable Forest"),
        Line2D([], [], color=style.HH, lw=2.5, marker="o", ms=4, label="HH"),
        Line2D([], [], color=style.HV, lw=2.5, marker="o", ms=4, label="HV"),
        Line2D([], [], color=style.COH20, lw=3, label="Coherence − nearby forest"),
        Line2D([], [], color=RAW_COH, lw=3, label="Coherence"),
        Line2D(
            [],
            [],
            color=DETECT,
            lw=0,
            marker="|",
            ms=18,
            mew=2.5,
            label="First detection (framed image)",
        ),
    ]
    ax_leg.legend(handles=handles, ncol=4, loc="center", columnspacing=1.5, handlelength=2.2)

    style.save(fig, out_dir / f"{case.case_id}.png")
    return det


def _day_str(day: float) -> str:
    return config.to_date(day).isoformat() if np.isfinite(day) else "-"


def write_readme(ds: data.Dataset, rows: list[tuple], original: set[str]) -> None:
    table = [
        "| Case | Area (ha) | NBR detection | HV detection | 20 m coherence dip (sigma) | In `04_cases` |",
        "|---|--:|---|---|---|---|",
    ]
    for case, det in rows:
        dip = (
            f"{fig04._pair_label(ds.coh20.ref[det.coh_pair], ds.coh20.sec[det.coh_pair])} "
            f"({det.coh_sigma:+.1f})"
            if det.coh_pair is not None
            else "none"
        )
        table.append(
            f"| {case.case_id} | {case.area_ha} | {_day_str(det.s2)} | {_day_str(det.hv)} | "
            f"{dip} | {'yes' if case.case_id in original else 'new'} |"
        )
    text = f"""# Forest clearing examples (simplified)

Simplified versions of the forest-clearing case figures in `../../04_cases/forest_clearing/forest_ring/`, with 20 m coherence throughout. Cases marked "yes" are the same polygons as there; "new" cases were chosen by the same rules (forest before, a flagged 80 m coherence dip, and optical or HV evidence of clearing), continuing the same ranking.

## Reading the figures

- **Chips**: Sentinel-2 true colour, GCOV HV (-16 to -6 dB), and 20 m HH coherence (0 to 0.8), with the disturbance polygon in yellow. Columns are aligned in time as closely as each sensor's dates allow: before the disturbance, around the first detections, and the last date. Each chip is titled with its own date.
- **Green frame and green line**: the first observation in which each sensor detects the disturbance. The frame in each chip row and the line in the matching time-series panel mark the same date.
- **NBR**: Sentinel-2 normalized burn ratio for the disturbance polygon and the Stable Forest median. Detection is the first date NBR falls to <= {config.NBR_CLEARED_MAX} after being forest-like (>= {config.NBR_FOREST_MIN}).
- **Backscatter**: HH and HV means over the polygon; dashed is Stable Forest HV. Detection is the first HV date after the best single step down in the series. This is the best-fitting step, not a significance test.
- **Coherence**: each 12-day pair is a horizontal segment from its first to its second date, and dashed connectors join consecutive pairs. Magenta is the polygon's 20 m coherence. Violet is the same value minus intact forest {config.RING_INNER_PX * config.PIXEL_M:.0f}-{config.RING_OUTER_PX * config.PIXEL_M:.0f} m around the polygon in the same pair, which removes weather effects shared with nearby forest. Detection is the second date of the first pair whose violet value falls more than {config.DIP_SIGMA[REFERENCE]:g} sigma below the median of the previous {config.CHANGE_BASELINE_PAIRS} pairs. Sigma is the pair-to-pair spread of the same quantity for intact-forest areas of the polygon's size.

## Cases

{chr(10).join(table)}

Coherence dips here are detected on 20 m coherence, so their dates can differ from the 80 m dips listed in `04_cases`.

Regenerate with `python scripts/caqueta/fig05_lab_examples.py` (add `--reselect` to redo the case selection).
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reselect", action="store_true", help="Redo the case selection")
    args = parser.parse_args()
    style.apply()
    ds = data.load()
    s2 = data.load_s2_stack()
    cases = load_cases(ds, s2, args.reselect)
    original = check_original(cases)
    rows = []
    for case in cases:
        det = plot_case(ds, s2, case)
        rows.append((case, det))
        logger.info(
            "Wrote %s (NBR %s, HV %s, coherence %s)",
            case.case_id,
            _day_str(det.s2),
            _day_str(det.hv),
            _day_str(det.coh),
        )
    write_readme(ds, rows, original)


if __name__ == "__main__":
    main()
