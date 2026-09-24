#!/usr/bin/env python
"""Select the case-study clearings and controls (step 1 of the figure pipeline).

Runs ``analysis.select_cases`` and writes:

- ``local_examples/caqueta/04_cases/cases.csv``: one row per case with its
  category, location, HV-dated bracket, RADD date, and the metrics used to
  categorize it
- ``local_examples/caqueta/04_cases/cases.npz``: the pixel rows/cols of each case,
  so every figure uses exactly the same pixels

Usage:
    python scripts/caqueta/select_cases.py
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass

import analysis
import config
import data
import numpy as np
import rasterio
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CASES_DIR = config.OUT_DIR / "04_cases"
CSV_PATH = CASES_DIR / "cases.csv"
NPZ_PATH = CASES_DIR / "cases.npz"
FIELDS = [
    "number",
    "case_id",
    "category",
    "area_ha",
    "lat",
    "lon",
    "hv_bracket_start",
    "hv_bracket_end",
    "radd_alert",
    "delineation",
    "hv_step_db",
    "hv_abrupt_frac",
    "coh_dip_start",
    "coh_dip_end",
    "coh80_min_delta",
    "coh80_min_sigma",
    "noise_sd",
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

    @property
    def dip_start(self) -> float:
        v = self.row["coh_dip_start"]
        return config.to_day(v) if v else np.nan

    @property
    def dip_end(self) -> float:
        v = self.row["coh_dip_end"]
        return config.to_day(v) if v else np.nan

    @property
    def event_start(self) -> float:
        """Earliest of the HV bracket start and the coherence-dip start."""
        return float(np.nanmin([self.t0, self.dip_start])) if np.isfinite(self.t0) else np.nan

    @property
    def event_end(self) -> float:
        """Latest of the HV bracket end and the coherence-dip end."""
        return float(np.nanmax([self.t1, self.dip_end])) if np.isfinite(self.t1) else np.nan

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


def write(cases: list[analysis.Patch], grid: data.Grid) -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    arrays = {}
    with CSV_PATH.open("w", newline="") as f:
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
                    "hv_bracket_start": _fmt_day(p.t0),
                    "hv_bracket_end": _fmt_day(p.t1),
                    "radd_alert": _fmt_day(p.radd_day),
                    "delineation": p.delineation,
                    "hv_step_db": round(p.hv_step_db, 2),
                    "hv_abrupt_frac": round(p.hv_abrupt_frac, 2),
                    "coh_dip_start": _fmt_day(p.coh_dip_start),
                    "coh_dip_end": _fmt_day(p.coh_dip_end),
                    "coh80_min_delta": round(p.coh80_min_delta, 3),
                    "coh80_min_sigma": round(p.coh80_min_sigma, 2),
                    "noise_sd": round(p.noise_sd, 3),
                    "rep_row": p.rep_pixel[0],
                    "rep_col": p.rep_pixel[1],
                }
            )
            arrays[f"{p.case_id}_rows"] = p.rows
            arrays[f"{p.case_id}_cols"] = p.cols
    np.savez_compressed(NPZ_PATH, **arrays)
    logger.info("Wrote %s and %s (%d cases)", CSV_PATH, NPZ_PATH, len(cases))


def read() -> list[CaseRecord]:
    """Load the saved cases (run this script first)."""
    arrays = np.load(NPZ_PATH)
    with CSV_PATH.open() as f:
        return [
            CaseRecord(row, arrays[f"{row['case_id']}_rows"], arrays[f"{row['case_id']}_cols"])
            for row in csv.DictReader(f)
        ]


def main() -> None:
    ds = data.load()
    s2 = data.load_s2_stack()
    write(analysis.select_cases(ds, s2), ds.grid)


if __name__ == "__main__":
    main()
