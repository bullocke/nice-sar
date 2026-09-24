#!/usr/bin/env python
"""Download PROVISIONAL GCOV and GUNW granules over the Rondônia deforestation frontier.

Example of maturity-aware search and download with nice-sar. The AOI covers the
"fishbone" clearing pattern east of Ariquemes, Rondônia, Brazil. Frame 068 D 093
covers ~99% of the AOI. The script downloads:

- GCOV: an early (2026-06-18) and a late (2026-08-29) dry-season acquisition,
  to compare HH/HV backscatter before and after the main clearing/burning season.
  This frame alternates dual-pol (DHDH) and single-pol (SHSH) acquisitions every
  12 days; both dates are dual-pol so HV is available.
- GUNW: an early (2026-06-30/07-12) and a late (2026-08-29/09-10) 12-day pair,
  to compare coherence. Pairs spanning the 2026-07-27 to 2026-08-10 instrument
  data gap are avoided.

Each file is ~2 GB. Requires NASA Earthdata credentials (~/.netrc or
EARTHDATA_USERNAME/EARTHDATA_PASSWORD).

Usage:
    python scripts/examples/amazon_frontier_download.py
    python scripts/examples/amazon_frontier_download.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py

from nice_sar.io.download import download_granules
from nice_sar.io.products import read_identification
from nice_sar.search import search_nisar, summarize_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("asf_search").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

AOI = (-63.5, -10.0, -62.5, -9.0)  # west, south, east, north
TRACK, DIRECTION, FRAME = 68, "D", 93

# Reference acquisition dates to download for each product.
WANTED = {
    "GCOV": ["2026-06-18", "2026-08-29"],
    "GUNW": ["2026-06-30", "2026-08-29"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("NISAR_Data/amazon_rondonia"))
    parser.add_argument("--dry-run", action="store_true", help="Search only")
    args = parser.parse_args()

    manifest = []
    for product, dates in WANTED.items():
        results = search_nisar(
            product,
            bbox=AOI,
            maturity="provisional",
            track=TRACK,
            frame=FRAME,
            direction=DIRECTION,
            start="2026-06-17",
        )
        selected = []
        for date in dates:
            matches = [
                (r, s)
                for r, s in zip(results, summarize_results(results), strict=True)
                if s.start and s.start.startswith(date)
            ]
            if not matches:
                raise SystemExit(f"No {product} granule found for {date}")
            # Prefer full-frame products and the highest processing counter.
            matches.sort(key=lambda m: (bool(m[1].full_frame), m[1].granule_id))
            selected.append(matches[-1])

        for _, s in selected:
            logger.info(
                "%s %s %s %s %.2f GB %s",
                product,
                s.maturity,
                s.crid,
                s.start,
                s.size_gb or 0,
                s.granule_id,
            )
        if args.dry_run:
            continue

        paths = download_granules([r for r, _ in selected], args.output_dir / product)
        for (_, s), path in zip(selected, paths, strict=True):
            with h5py.File(path, "r") as h5:
                ident = read_identification(h5)
            if (ident["track"], ident["frame"]) != (TRACK, FRAME):
                raise RuntimeError(f"Unexpected track/frame in {path}: {ident}")
            manifest.append({**s.to_dict(), "path": str(path), "identification": ident})

    if manifest:
        out = args.output_dir / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2, default=str))
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
