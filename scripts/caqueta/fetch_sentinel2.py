#!/usr/bin/env python
"""Fetch mostly clear Sentinel-2 L2A images over the Caquetá AOI from Earth Engine.

Used for the optical context chips in the figures. For each date between the
first and last NISAR acquisition:

1. Mosaic all Sentinel-2 L2A (``COPERNICUS/S2_SR_HARMONIZED``) granules
   acquired that day.
2. Mask clouds and shadows with Cloud Score+ (``cs_cdf >= 0.6`` is clear).
   Cloud Score+ is designed for exactly this; it outperforms the L2A scene
   classification band in persistently cloudy tropics.
3. Keep the date if at least ``--min-clear`` of the AOI is clear.
4. Sample bands B2, B3, B4 (blue, green, red), B8 (NIR), B12 (SWIR2) and the
   clear mask at 10 m on the NISAR grid's origin and CRS (EPSG:32618), so each
   20 m NISAR pixel covers exactly 2 x 2 Sentinel-2 pixels.

Output: ``NISAR_Data/caqueta/S2/S2_<YYYY-MM-DD>.tif`` (uint16 reflectance x 10000,
band descriptions B2, B3, B4, B8, B12, clear). Existing files are skipped.

Requires an initialized Earth Engine account.

Usage:
    python scripts/caqueta/fetch_sentinel2.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta

import config
import ee
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BANDS = ["B2", "B3", "B4", "B8", "B12"]
CLEAR_THRESHOLD = 0.6  # Cloud Score+ cs_cdf; Google's recommended default
ROW_TILE = 600  # rows per computePixels request (keeps each request < 48 MB)


def _nisar_grid() -> tuple[rasterio.crs.CRS, rasterio.Affine, tuple[int, int]]:
    with rasterio.open(config.RADD_TIF) as src:
        return src.crs, src.transform, src.shape


def _clear_dates(aoi: ee.Geometry, start: str, end: str, min_clear: float) -> list[str]:
    """Dates whose daily mosaic is at least ``min_clear`` clear over the AOI."""
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
    s2 = s2.filterDate(start, end)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    linked = s2.linkCollection(cs, ["cs_cdf"])

    def score(img: ee.Image) -> ee.Feature:
        frac = (
            img.select("cs_cdf")
            .gte(CLEAR_THRESHOLD)
            .reduceRegion(ee.Reducer.mean(), aoi, 200, maxPixels=1e9)
            .get("cs_cdf")
        )
        return ee.Feature(
            None, {"d": img.date().format("YYYY-MM-dd"), "clear": ee.Algorithms.If(frac, frac, 0)}
        )

    rows = ee.FeatureCollection(linked.map(score)).getInfo()["features"]
    best: dict[str, float] = {}
    for r in rows:
        d, c = r["properties"]["d"], r["properties"]["clear"]
        best[d] = max(best.get(d, 0.0), c)  # overlapping granules: take the best
    dates = sorted(d for d, c in best.items() if c >= min_clear)
    logger.info("%d of %d dates are >= %.0f%% clear", len(dates), len(best), 100 * min_clear)
    return dates


def _daily_image(aoi: ee.Geometry, day: str) -> ee.Image:
    start = ee.Date(day)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, start.advance(1, "day"))
        .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"])
    )
    mosaic = s2.mosaic()
    clear = mosaic.select("cs_cdf").gte(CLEAR_THRESHOLD).rename("clear")
    return mosaic.select(BANDS).toUint16().addBands(clear.toUint16())


def _compute(
    image: ee.Image, crs: str, transform: rasterio.Affine, width: int, height: int
) -> np.ndarray:
    """Sample ``image`` on a 10 m grid in row tiles; returns (bands, H, W)."""
    tiles = []
    for r0 in range(0, height, ROW_TILE):
        h = min(ROW_TILE, height - r0)
        arr = ee.data.computePixels(
            {
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": width, "height": h},
                    "affineTransform": {
                        "scaleX": transform.a,
                        "shearX": 0,
                        "translateX": transform.c,
                        "shearY": 0,
                        "scaleY": transform.e,
                        "translateY": transform.f + r0 * transform.e,
                    },
                    "crsCode": crs,
                },
            }
        )
        tiles.append(np.stack([arr[b] for b in arr.dtype.names]))
    return np.concatenate(tiles, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-clear", type=float, default=0.5)
    args = parser.parse_args()

    ee.Initialize()
    crs, t20, (h20, w20) = _nisar_grid()
    t10 = rasterio.Affine(t20.a / 2, 0, t20.c, 0, t20.e / 2, t20.f)
    width, height = w20 * 2, h20 * 2
    aoi = ee.Geometry.Rectangle(list(config.AOI))

    files = json.loads(config.MANIFEST.read_text())["files"]
    days = [config.to_day(f["start"]) for f in files]
    days += [config.to_day(f["secondary_start"]) for f in files if f["secondary_start"]]
    start = config.to_date(min(days) - 15).isoformat()
    end = (config.to_date(max(days)) + timedelta(days=15)).isoformat()

    config.S2_DIR.mkdir(parents=True, exist_ok=True)
    for day in _clear_dates(aoi, start, end, args.min_clear):
        out = config.S2_DIR / f"S2_{day}.tif"
        if out.exists():
            continue
        data = _compute(_daily_image(aoi, day), crs.to_string(), t10, width, height)
        profile = {
            "driver": "GTiff",
            "dtype": "uint16",
            "count": data.shape[0],
            "height": height,
            "width": width,
            "crs": crs,
            "transform": t10,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
        }
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data.astype("uint16"))
            dst.descriptions = (*BANDS, "clear")
        logger.info("Wrote %s (%.0f%% clear)", out.name, 100 * data[-1].mean())


if __name__ == "__main__":
    main()
