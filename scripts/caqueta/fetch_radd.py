#!/usr/bin/env python
"""Export RADD disturbance alerts onto the NISAR grid of the Caquetá subsets.

RADD (Reiche et al., 2021) alerts are Sentinel-1 based forest disturbance
alerts for the humid tropics, distributed in Google Earth Engine. This script
samples the latest South America release onto the exact grid (CRS, 20 m
transform, shape) of a NISAR subset GeoTIFF, so alerts and NISAR pixels align
one-to-one.

Output ``RADD_reference.tif`` bands:

1. ``alert_date``: days since 2025-01-01 (day 0) of the first alert; nodata -9999.
   RADD alert dates mark the first Sentinel-1 observation flagged as disturbed,
   so they can lag the actual event by up to one or two S1 revisits.
2. ``alert_conf``: 2 = low, 3 = high confidence; 0 = no alert.
3. ``forest_baseline``: 1 where RADD's primary humid forest baseline applies.

Requires an initialized Earth Engine account (``earthengine authenticate``).

Usage:
    python scripts/caqueta/fetch_radd.py \\
        --template NISAR_Data/caqueta/GCOV/GCOV_20251103_HH.tif
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import ee
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RADD = "projects/radar-wur/raddalert/v1"


def _to_numpy(image: ee.Image, profile: dict) -> np.ndarray:
    t = profile["transform"]
    grid = {
        "dimensions": {"width": profile["width"], "height": profile["height"]},
        "affineTransform": {
            "scaleX": t.a,
            "shearX": t.b,
            "translateX": t.c,
            "shearY": t.d,
            "scaleY": t.e,
            "translateY": t.f,
        },
        "crsCode": profile["crs"].to_string(),
    }
    arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
    return np.stack([arr[name] for name in arr.dtype.names])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NISAR_Data/caqueta/RADD_reference.tif"),
    )
    args = parser.parse_args()

    ee.Initialize()
    alerts = (
        ee.ImageCollection(RADD)
        .filterMetadata("layer", "contains", "alert")
        .filterMetadata("geography", "equals", "sa")
        .sort("system:time_end", False)
        .first()
    )
    release = alerts.get("system:index").getInfo()
    # Date is encoded YYDOY (e.g. 26123); convert to days since 2025-01-01.
    code = alerts.select("Date")
    year = code.divide(1000).floor()
    doy = code.mod(1000)
    jan1 = ee.Image(0).where(year.eq(24), -366).where(year.eq(26), 365)
    jan1 = jan1.where(year.lte(23), year.subtract(25).multiply(365.25).floor())
    days = jan1.add(doy).subtract(1)
    forest = (
        ee.ImageCollection(RADD)
        .filterMetadata("layer", "contains", "forest_baseline")
        .filterMetadata("geography", "equals", "sa")
        .first()
    )
    stack = ee.Image.cat(
        days.unmask(-9999).toInt32().rename("alert_date"),
        alerts.select("Alert").unmask(0).toInt32().rename("alert_conf"),
        forest.unmask(0).gt(0).toInt32().rename("forest_baseline"),
    )

    with rasterio.open(args.template) as src:
        profile = src.profile.copy()
    data = _to_numpy(stack, profile)

    profile.update(count=3, dtype="int32", nodata=-9999, compress="deflate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(data.astype("int32"))
        dst.descriptions = ("alert_date", "alert_conf", "forest_baseline")
        dst.update_tags(
            radd_release=release,
            alert_date_units="days since 2025-01-01",
        )
    logger.info(
        "Wrote %s from RADD %s: %.1f%% alerted, %.1f%% forest baseline",
        args.output,
        release,
        100 * (data[1] > 0).mean(),
        100 * data[2].mean(),
    )


if __name__ == "__main__":
    main()
