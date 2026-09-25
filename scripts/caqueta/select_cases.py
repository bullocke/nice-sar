#!/usr/bin/env python
"""Select the case-study clearings and controls (step 1 of the figure pipeline).

Runs ``analysis.select_cases`` and writes:

- ``local_examples/caqueta/04_cases/cases_forest_<reference>.csv``: one row per
  case with its category, location, timing, and the metrics used to categorize it
- ``local_examples/caqueta/04_cases/cases_forest_<reference>.npz``: the pixel
  rows/cols of each case, so every figure uses exactly the same pixels

``<reference>`` is the weather reference used for coherence dips: ``scene``
(stable-forest median over the AOI) or ``ring`` (intact forest around each case).

Usage:
    python scripts/caqueta/select_cases.py                 # both references
    python scripts/caqueta/select_cases.py --reference ring
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import analysis
import config
import data
import numpy as np
import rasterio
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CASES_DIR = config.OUT_DIR / "04_cases"


def csv_path(reference: str) -> Path:
    """Case list for one weather reference ("scene" or "ring")."""
    return CASES_DIR / f"cases_forest_{reference}.csv"


def npz_path(reference: str) -> Path:
    return CASES_DIR / f"cases_forest_{reference}.npz"


FIELDS = [
    "number",
    "case_id",
    "category",
    "area_ha",
    "lat",
    "lon",
    "delineation",
    "state_before",
    "event_start",
    "event_end",
    "coh_dips",
    "deepest_sigma",
    "optical_start",
    "optical_end",
    "hv_bracket_start",
    "hv_bracket_end",
    "hv_step_db",
    "radd_alert",
    "spanning_forest_coh",
    "noise_sd",
    "change_sd",
    "rep_row",
    "rep_col",
]


@dataclass
class CaseRecord:
    """A saved case: its CSV row plus pixel indices."""

    row: dict
    rows: np.ndarray
    cols: np.ndarray

    def __getattr__(self, name: str):
        return self.row[name]

    @property
    def t0(self) -> float:
        v = self.row["hv_bracket_start"]
        return config.to_day(v) if v else np.nan

    @property
    def t1(self) -> float:
        v = self.row["hv_bracket_end"]
        return config.to_day(v) if v else np.nan

    def _day(self, field: str) -> float:
        v = self.row[field]
        return config.to_day(v) if v else np.nan

    @property
    def event_start(self) -> float:
        """Earliest flagged coherence dip, or the HV bracket start if none."""
        return self._day("event_start")

    @property
    def event_end(self) -> float:
        """End of the deepest flagged dip, or the HV bracket end if none."""
        return self._day("event_end")

    @property
    def dip_pairs(self) -> list[tuple[int, int, float]]:
        """Flagged coherence dips as (ref day, sec day, change in sigma)."""
        out = []
        for item in filter(None, self.row["coh_dips"].split(";")):
            span, sigma = item.split(":")
            ref, sec = span.split("/")
            out.append((config.to_day(ref), config.to_day(sec), float(sigma)))
        return out

    @property
    def radd_day(self) -> float:
        v = self.row["radd_alert"]
        return config.to_day(v) if v else np.nan

    @property
    def rep_pixel(self) -> tuple[int, int]:
        return int(self.row["rep_row"]), int(self.row["rep_col"])

    def window(
        self, shape: tuple[int, int], min_size: int = 60, pad: int = 12
    ) -> tuple[slice, slice]:
        """Square NISAR-grid window centred on the case, shifted to stay in ``shape``.

        At least ``min_size`` px (60 px = 1.2 km) so the surroundings are visible.
        """
        size = max(min_size, max(np.ptp(self.rows), np.ptp(self.cols)) + 2 * pad)
        centre = (int(round(self.rows.mean())), int(round(self.cols.mean())))
        out = []
        for c, n in zip(centre, shape, strict=True):
            start = min(max(c - size // 2, 0), n - size)
            out.append(slice(start, start + size))
        return out[0], out[1]


def _fmt_day(day: float) -> str:
    return "" if not np.isfinite(day) else config.to_date(day).isoformat()


def write(cases: list[analysis.Patch], grid: data.Grid, reference: str) -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    arrays = {}
    with csv_path(reference).open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for n, p in enumerate(cases, start=1):
            r, c = p.center
            x, y = rasterio.transform.xy(grid.transform, r, c)
            lon, lat = to_ll.transform(x, y)
            w.writerow(
                {
                    "number": n,
                    "case_id": p.case_id,
                    "category": p.category,
                    "area_ha": round(p.area_ha, 2),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "delineation": p.delineation,
                    "state_before": p.state_before,
                    "event_start": _fmt_day(p.event_start),
                    "event_end": _fmt_day(p.event_end),
                    "coh_dips": p.dips,
                    "deepest_sigma": round(p.deepest_sigma, 2),
                    "optical_start": _fmt_day(p.optical_start),
                    "optical_end": _fmt_day(p.optical_end),
                    "hv_bracket_start": _fmt_day(p.t0),
                    "hv_bracket_end": _fmt_day(p.t1),
                    "hv_step_db": round(p.hv_step_db, 2),
                    "radd_alert": _fmt_day(p.radd_day),
                    "spanning_forest_coh": round(p.spanning_forest_coh, 2),
                    "noise_sd": round(p.noise_sd, 3),
                    "change_sd": round(p.change_sd, 3),
                    "rep_row": p.rep_pixel[0],
                    "rep_col": p.rep_pixel[1],
                }
            )
            arrays[f"{p.case_id}_rows"] = p.rows
            arrays[f"{p.case_id}_cols"] = p.cols
    np.savez_compressed(npz_path(reference), **arrays)
    logger.info("Wrote %s (%d cases)", csv_path(reference), len(cases))


def read(reference: str = "ring") -> list[CaseRecord]:
    """Load the saved cases for one weather reference (run this script first)."""
    arrays = np.load(npz_path(reference))
    with csv_path(reference).open() as f:
        return [
            CaseRecord(row, arrays[f"{row['case_id']}_rows"], arrays[f"{row['case_id']}_cols"])
            for row in csv.DictReader(f)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reference",
        choices=config.REFERENCES,
        nargs="+",
        default=list(config.REFERENCES),
        help="Weather reference(s): scene-wide forest median and/or local forest ring",
    )
    args = parser.parse_args()
    ds = data.load()
    s2 = data.load_s2_stack()
    for reference in args.reference:
        write(analysis.select_cases(ds, s2, reference), ds.grid, reference)


if __name__ == "__main__":
    main()
