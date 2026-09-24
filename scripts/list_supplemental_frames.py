#!/usr/bin/env python
"""Inventory the NISAR supplemental PROVISIONAL frames.

PROVISIONAL forward processing covers acquisitions from 2026-06-17 onward. The
NISAR science team also back-processed a limited set of frames to build longer
time series for validation ("Supplemental Provisional Products"). This script
finds them by listing every PROVISIONAL granule acquired before the forward
processing start date and grouping them by track, direction, and frame.

Only CMR search metadata is queried, so no Earthdata login is required.

Usage:
    python scripts/list_supplemental_frames.py
    python scripts/list_supplemental_frames.py --products GCOV GUNW RSLC \\
        --output docs/data/supplemental_provisional_frames.csv
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import Polygon

from nice_sar.search.maturity import nisar_short_names, parse_granule_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
FORWARD_START = "2026-06-17T00:00:00Z"
COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"


@dataclass
class FrameRecord:
    """Summary of one supplemental frame for one product."""

    product: str
    track: int
    direction: str
    frame: int
    n_granules: int
    first_date: str
    last_date: str
    n_full_frame: int
    lat: float
    lon: float
    west: float
    south: float
    east: float
    north: float
    example_granule: str


def fetch_granules(short_name: str, end: str) -> list[dict]:
    """Page through all CMR granules in a collection acquired before ``end``."""
    params = {
        "short_name": short_name,
        "temporal[]": f"2025-01-01T00:00:00Z,{end}",
        "page_size": 2000,
    }
    entries: list[dict] = []
    search_after = None
    while True:
        headers = {"CMR-Search-After": search_after} if search_after else {}
        resp = requests.get(CMR_GRANULES, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        page = resp.json()["feed"]["entry"]
        entries.extend(page)
        search_after = resp.headers.get("CMR-Search-After")
        if not page or not search_after:
            break
    logger.info("%s: %d granules before %s", short_name, len(entries), end)
    return entries


def _footprint(entry: dict) -> Polygon | None:
    rings = entry.get("polygons")
    if rings:
        vals = [float(v) for v in rings[0][0].split()]
        return Polygon(zip(vals[1::2], vals[0::2], strict=True))
    boxes = entry.get("boxes")
    if boxes:
        s, w, n, e = (float(v) for v in boxes[0].split())
        return Polygon([(w, s), (e, s), (e, n), (w, n)])
    return None


def summarize(product: str, entries: list[dict], min_dates: int) -> list[FrameRecord]:
    """Group granules by (track, direction, frame)."""
    groups: dict[tuple[int, str, int], list] = defaultdict(list)
    for entry in entries:
        parsed = parse_granule_name(entry["title"])
        groups[(parsed.track, parsed.direction, parsed.frame)].append((parsed, entry))

    records = []
    for (track, direction, frame), items in sorted(groups.items()):
        dates = sorted(p.start.date().isoformat() for p, _ in items)
        if len(set(dates)) < min_dates:
            continue
        shapes = [s for s in (_footprint(e) for _, e in items) if s is not None]
        west, south, east, north = shapes[0].bounds if shapes else (float("nan"),) * 4
        centroid = shapes[0].centroid if shapes else None
        records.append(
            FrameRecord(
                product=product,
                track=track,
                direction=direction,
                frame=frame,
                n_granules=len(items),
                first_date=dates[0],
                last_date=dates[-1],
                n_full_frame=sum(p.full_frame for p, _ in items),
                lat=round(centroid.y, 3) if centroid else float("nan"),
                lon=round(centroid.x, 3) if centroid else float("nan"),
                west=round(west, 3),
                south=round(south, 3),
                east=round(east, 3),
                north=round(north, 3),
                example_granule=items[0][0].granule_id,
            )
        )
    return records


def add_country(df: pd.DataFrame) -> pd.DataFrame:
    """Label frames with the Natural Earth country nearest the frame centroid."""
    try:
        import geopandas as gpd

        countries = gpd.read_file(COUNTRIES_URL)[["NAME", "CONTINENT", "geometry"]]
    except Exception as exc:  # noqa: BLE001 - labels are optional
        logger.warning("Country labels skipped: %s", exc)
        return df
    points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin_nearest(
        points.to_crs("EPSG:3857"), countries.to_crs("EPSG:3857"), how="left"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    out = df.copy()
    out["country"] = joined["NAME"].values
    out["continent"] = joined["CONTINENT"].values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--products", nargs="+", default=["GCOV", "GUNW"])
    parser.add_argument("--end", default=FORWARD_START, help="Exclusive end time")
    parser.add_argument(
        "--min-dates",
        type=int,
        default=2,
        help="Drop frames with fewer distinct dates (removes forward-processing "
        "edge cases on 2026-06-16)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data/supplemental_provisional_frames.csv"),
    )
    args = parser.parse_args()

    records: list[FrameRecord] = []
    for product in args.products:
        (short_name,) = nisar_short_names(product, "provisional")
        entries = fetch_granules(short_name, args.end)
        records.extend(summarize(product, entries, args.min_dates))

    df = add_country(pd.DataFrame([asdict(r) for r in records]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    for product, sub in df.groupby("product"):
        logger.info(
            "%s: %d frames, %d granules, %s to %s",
            product,
            len(sub),
            sub["n_granules"].sum(),
            sub["first_date"].min(),
            sub["last_date"].max(),
        )
    logger.info("Wrote %s (generated %s)", args.output, date.today().isoformat())


if __name__ == "__main__":
    main()
