#!/usr/bin/env python
"""Forest clearings whose Sentinel-2 response after the HV drop is weak.

Screens the full ring-reference candidate pool (``analysis.candidate_pool``) for
``forest_clearing`` cases whose case-mean NBR never falls to <= NBR_CLEARED_MAX
after the HV bracket end. These met the optical rule only because NBR crossed the
threshold at some other point in the series, so they are candidates for
disturbances NISAR responds to but Sentinel-2 shows only weakly.

Each case is drawn in the simplified fig05 format and written, with a README, to
``local_examples/caqueta/06_weak_optical/``. The case list is cached as
``04_cases/cases_forest_ring_weakopt.csv/.npz``.

Usage:
    python scripts/caqueta/fig06_weak_optical.py
    python scripts/caqueta/fig06_weak_optical.py --reselect   # rebuild the list
"""

# ruff: noqa: E501  (long lines are Markdown in the README template)
from __future__ import annotations

import argparse
import logging

import analysis
import config
import data
import fig05_lab_examples as fig05
import numpy as np
import select_cases
import style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REFERENCE = "ring"
SUFFIX = "_weakopt"
CASE_PREFIX = "weak_optical"
OUT = config.OUT_DIR / "06_weak_optical"


def _nbr_after(s2: data.S2Stack, rows, cols, shape, day: float) -> float:
    days, vals = analysis.case_nbr(s2, rows, cols, shape)
    after = vals[days > day]
    return float(after.min()) if after.size else np.nan


def screen(ds: data.Dataset, s2: data.S2Stack) -> list[analysis.Patch]:
    """forest_clearing candidates whose NBR stays above NBR_CLEARED_MAX after the HV drop."""
    pool = analysis.candidate_pool(ds, s2, REFERENCE)
    clearing = [c for c in pool if analysis.categorize(c[3]) == "forest_clearing"]
    weak = [
        c
        for c in clearing
        if _nbr_after(s2, c[0], c[1], ds.grid.shape, c[3]["hv"]["t1"]) > config.NBR_CLEARED_MAX
    ]
    logger.info(
        "%d forest_clearing candidates, %d with NBR > %.2f after the HV drop",
        len(clearing),
        len(weak),
        config.NBR_CLEARED_MAX,
    )
    # two RADD seeds can grow into the same Sentinel-2 outline: keep one
    unique: dict[tuple, tuple] = {}
    for c in weak:
        unique.setdefault((c[0].tobytes(), c[1].tobytes()), c)
    weak = sorted(unique.values(), key=lambda c: c[3]["deepest_sigma"])
    return [
        analysis._make_patch(ds, f"{CASE_PREFIX}_{k}", "forest_clearing", rows, cols, how, m)
        for k, (rows, cols, how, m) in enumerate(weak, start=1)
    ]


def load_cases(ds: data.Dataset, s2: data.S2Stack, reselect: bool) -> list:
    if reselect or not select_cases.csv_path(REFERENCE, SUFFIX).exists():
        select_cases.write(screen(ds, s2), ds.grid, REFERENCE, SUFFIX)
    return select_cases.read(REFERENCE, SUFFIX)


def write_readme(ds: data.Dataset, s2: data.S2Stack, rows: list[tuple]) -> None:
    table = [
        "| Case | Area (ha) | Outline | NBR before (median) | NBR min (date) | NBR min after HV drop | HV step (dB) | HV detection | 20 m coherence dip (sigma) |",
        "|---|--:|---|--:|---|--:|--:|---|---|",
    ]
    for case, det in rows:
        days, vals = analysis.case_nbr(s2, case.rows, case.cols, ds.grid.shape)
        pre = vals[days < case.event_start]
        i = int(np.argmin(vals))
        after = _nbr_after(s2, case.rows, case.cols, ds.grid.shape, case.t1)
        dip = (
            f"{fig05.fig04._pair_label(ds.coh20.ref[det.coh_pair], ds.coh20.sec[det.coh_pair])} "
            f"({det.coh_sigma:+.1f})"
            if det.coh_pair is not None
            else "none"
        )
        table.append(
            f"| {case.case_id} | {case.area_ha} | {case.delineation} | "
            f"{np.median(pre) if pre.size else np.nan:.2f} | "
            f"{vals[i]:.2f} ({config.to_date(days[i])}) | {after:.2f} | {case.hv_step_db} | "
            f"{fig05._day_str(det.hv)} | {dip} |"
        )
    text = f"""# Forest clearings with a weak Sentinel-2 response

Screened from the full `ring` candidate pool: every `forest_clearing` candidate (forest before, a flagged 80 m coherence dip, and an NBR drop to <= {config.NBR_CLEARED_MAX} or an HV step >= {config.PATCH_STEP_DB} dB) whose case-mean NBR stays above {config.NBR_CLEARED_MAX} on every usable Sentinel-2 date after the HV drop. The screen was meant to find disturbance that NISAR responds to but Sentinel-2 shows weakly. Compare "NBR min (date)" with "HV detection": when the NBR minimum comes first, the clearing is plain in Sentinel-2 and only the HV step is late. NBR is then back above the threshold simply because the next clear image comes months later, after regrowth. Ordered by the depth of the 80 m coherence dip.

The figures use the format of `../05_lab_examples/forest_clearing/` (see its README): green frames and lines mark each sensor's first detection. The NBR "detection" is still the first date NBR falls to <= {config.NBR_CLEARED_MAX}, which for these cases can come before the HV drop.

## Cases

{chr(10).join(table)}

"NBR before" is the median case-mean NBR before the first 80 m coherence dip; "NBR min after HV drop" is the lowest value on usable dates after the HV bracket end.

Regenerate with `python scripts/caqueta/fig06_weak_optical.py` (add `--reselect` to rebuild the case list).
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reselect", action="store_true", help="Rebuild the case list")
    args = parser.parse_args()
    style.apply()
    ds = data.load()
    s2 = data.load_s2_stack()
    rows = []
    for case in load_cases(ds, s2, args.reselect):
        det = fig05.plot_case(ds, s2, case, out_dir=OUT)
        rows.append((case, det))
        logger.info("Wrote %s", case.case_id)
    write_readme(ds, s2, rows)


if __name__ == "__main__":
    main()
