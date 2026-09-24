#!/usr/bin/env python
"""Run the Caquetá analysis and figure pipeline in order.

Data-fetch steps are not run by default because they need network access and
credentials and take a long time. Run them once first (see README.md):

    python scripts/caqueta/fetch_nisar_subsets.py   # NASA Earthdata, ~45 min
    python scripts/caqueta/fetch_radd.py --template NISAR_Data/caqueta/GCOV/GCOV_20251103_HH.tif
    python scripts/caqueta/fetch_sentinel2.py       # Earth Engine, ~10 min

Then:

    python scripts/caqueta/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "select_cases.py",
    "fig01_site_overview.py",
    "fig02_event_aligned.py",
    "fig03_spanning_pair.py",
    "fig04_case_studies.py",
]


def main() -> None:
    for step in STEPS:
        print(f"==> {step}", flush=True)
        subprocess.run([sys.executable, str(HERE / step)], check=True)


if __name__ == "__main__":
    main()
