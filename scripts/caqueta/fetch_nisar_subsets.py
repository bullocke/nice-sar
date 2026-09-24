#!/usr/bin/env python
"""Stream a GCOV/GUNW time series over a Caquetá (Colombia) deforestation hotspot.

The AOI (74.36-74.16°W, 0.76-0.92°N, ~22 x 18 km) lies in NISAR frame 083 D 088,
a supplemental PROVISIONAL frame with acquisitions from November 2025 onward.
It was chosen as the densest cluster of RADD disturbance alerts within the frame,
with alert dates spread across most of the NISAR acquisition intervals.

Only the pixels inside the AOI are streamed from each remote HDF5 file (via
:func:`nice_sar.io.subset.subset_product`); full scenes are not downloaded.

Outputs (under ``--output-dir``):

- ``GCOV/``: HH (all dates) and HV (dual-pol dates) backscatter, 20 m
- ``GUNW/``: HH coherence magnitude for each pair at 20 m and 80 m
- ``manifest.json``: one entry per GeoTIFF with granule ID, dates, and layer

Requires NASA Earthdata credentials.

Usage:
    python scripts/caqueta/fetch_nisar_subsets.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nice_sar.auth.earthdata import get_https_filesystem, login
from nice_sar.io.subset import subset_product
from nice_sar.search import search_nisar, summarize_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
for name in ("asf_search", "nice_sar.io.subset", "nice_sar.io.geotiff"):
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

AOI = (-74.36, 0.76, -74.16, 0.92)  # west, south, east, north
TRACK, DIRECTION, FRAME = 83, "D", 88


def _latest_per_date(summaries: list) -> list:
    """Keep one granule per acquisition (pair): full frame, highest counter."""
    best: dict[tuple, object] = {}
    for s in sorted(summaries, key=lambda s: (bool(s.full_frame), s.granule_id)):
        best[(s.start[:10], (s.secondary_start or "")[:10])] = s
    return sorted(best.values(), key=lambda s: s.start)


def _rename(path: Path, stem: str) -> Path:
    target = path.with_name(stem + ".tif")
    path.replace(target)
    sidecar = path.with_name(path.name + ".json")
    if sidecar.exists():
        sidecar.replace(target.with_name(target.name + ".json"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("NISAR_Data/caqueta"))
    args = parser.parse_args()

    login()
    fs = get_https_filesystem()
    manifest: list[dict] = []

    for product in ("GCOV", "GUNW"):
        results = search_nisar(
            product, track=TRACK, frame=FRAME, direction=DIRECTION, max_results=500
        )
        granules = _latest_per_date(summarize_results(results))
        logger.info("%s: %d acquisitions", product, len(granules))
        out_dir = args.output_dir / product
        out_dir.mkdir(parents=True, exist_ok=True)

        for s in granules:
            date = s.start[:10].replace("-", "")
            jobs: list[tuple[str, dict]] = []
            if product == "GCOV":
                pol_mode = s.granule_id.split("_")[9]  # DHDH (HH+HV) or SHSH (HH)
                pols = ["HH", "HV"] if pol_mode.startswith("D") else ["HH"]
                stem = f"GCOV_{date}"
                jobs.append((stem, {"polarizations": pols}))
            else:
                sec = (s.secondary_start or "")[:10].replace("-", "")
                for posting in (20, 80):
                    stem = f"GUNW_{date}_{sec}_coh{posting}m"
                    jobs.append(
                        (
                            stem,
                            {
                                "polarizations": ["HH"],
                                "layers": ["coherenceMagnitude"],
                                "posting": posting,
                            },
                        )
                    )

            for stem, kwargs in jobs:
                if list(out_dir.glob(stem + "*.tif")):
                    logger.info("Exists, skipping: %s", stem)
                    paths = sorted(out_dir.glob(stem + "*.tif"))
                else:
                    written = subset_product(
                        source=s.url,
                        product=product,
                        bbox=AOI,
                        output_dir=out_dir,
                        filesystem=fs,
                        confirm=False,
                        **kwargs,
                    )
                    paths = []
                    for p in written:
                        # e.g. GCOV_freqA_HV_HVHV_2026-06-18_subset.tif -> ..._HV
                        layer = p.name.split("_")[2]
                        name = stem if product == "GUNW" else f"{stem}_{layer}"
                        paths.append(_rename(p, name))
                    logger.info("Wrote %s", ", ".join(p.name for p in paths))
                for p in paths:
                    manifest.append(
                        {
                            "file": str(p.relative_to(args.output_dir)),
                            "product": product,
                            "granule_id": s.granule_id,
                            "maturity": s.maturity,
                            "crid": s.crid,
                            "start": s.start,
                            "secondary_start": s.secondary_start,
                        }
                    )

    out = args.output_dir / "manifest.json"
    out.write_text(json.dumps({"aoi": AOI, "files": manifest}, indent=2))
    logger.info("Wrote %s (%d files)", out, len(manifest))


if __name__ == "__main__":
    main()
