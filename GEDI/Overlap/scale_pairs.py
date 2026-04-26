"""Tiled GEDI L4A pre/post overlap discovery for large AOIs.

Scales the feasibility-notebook workflow to AOIs that exceed the GEE per-request
memory cap. The AOI is split into ~0.5 deg tiles and each tile is processed as
an independent GEE request. Results are written to per-tile Parquet files for
resumability, then concatenated into a master ``pairs.parquet``.

Design notes
------------
- Replaces the notebook's ``ee.Join.saveAll`` + ``aggregate_array("matches")``
  pattern with ``ee.Join.inner`` + a server-side ``map`` that flattens the
  nested pre/post properties into a single flat schema. Pair attributes are
  then pulled with one ``aggregate_array`` per column. This avoids materializing
  the nested match arrays that triggered the ``User memory limit exceeded`` error.
- Tiles whose Hansen 2020-2022 ``lossyear`` is empty are dropped before any GEDI
  work happens (cheap ``reduceRegion`` with ``anyNonZero``).
- Per-tile failures with ``ee.ee_exception.EEException`` are caught and recorded
  in ``run_log.csv``. ``--retry-failed`` re-runs those tiles split into 0.25 deg
  sub-tiles. Non-EE exceptions are surfaced loudly.

Run::

    python scale_pairs.py --aoi path/to/aoi.geojson --out-dir GEDI/Overlap/

    python GEDI/Overlap/scale_pairs.py --aoi <path.geojson> --out-dir GEDI/Overlap \
        [--pre-window 2019-04-18,2021-01-01] [--post-window 2022-01-01,2023-03-17] \
        [--tile-deg 0.5] [--workers 8] [--distance-m 25] [--strict-distance-m 10] \
        [--ee-project dyce-biomass] [--retry-failed]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scale_pairs")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIST_MAX_ERROR_M = 1.0  # GEE projection tolerance for withinDistance
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
GEDI04A_INDEX_ID = "LARSE/GEDI/GEDI04_A_002_INDEX"

# Fields to extract per shot. Matches the notebook's sample_pairs columns.
# ``delta_time`` (seconds since 2018-01-01 UTC) is required for SAR temporal
# matching downstream in ``sample_sar.py``.
SHOT_FIELDS = ("agbd", "agbd_se", "sensitivity", "beam", "shot_number", "delta_time")

RUN_LOG_COLUMNS = (
    "tile_id",
    "west",
    "south",
    "east",
    "north",
    "n_pre",
    "n_post",
    "n_pairs_25m",
    "n_pairs_10m",
    "latency_s",
    "status",
    "error",
)

# ---------------------------------------------------------------------------
# Run-log helpers (thread-safe append)
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


def append_run_log(log_path: Path, row: dict) -> None:
    """Append a single row to the run log. Thread-safe."""
    write_header = not log_path.exists()
    with _LOG_LOCK:
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in RUN_LOG_COLUMNS})


# ---------------------------------------------------------------------------
# AOI / tiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    tile_id: str
    west: float
    south: float
    east: float
    north: float

    @property
    def shapely(self) -> BaseGeometry:
        return box(self.west, self.south, self.east, self.north)

    @property
    def ee_geom(self) -> ee.Geometry:
        return ee.Geometry.Rectangle([self.west, self.south, self.east, self.north])


def read_aoi(path: Path) -> BaseGeometry:
    """Read AOI from a GeoJSON file and return its union geometry.

    Assumes EPSG:4326 (the GEE-native CRS). Shapefile inputs are not supported
    in this lightweight reader to avoid a hard pyproj dependency. Convert to
    GeoJSON first if needed.
    """
    if path.suffix.lower() not in {".geojson", ".json"}:
        raise ValueError(
            f"AOI must be GeoJSON (.geojson/.json), got {path.suffix!r}. "
            "Convert shapefiles to GeoJSON first."
        )
    with path.open() as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    geom = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    if geom.is_empty:
        raise ValueError(f"AOI in {path} is empty")
    return geom


def tile_aoi(aoi: BaseGeometry, tile_deg: float) -> list[Tile]:
    """Split the AOI bbox into a regular grid of size ``tile_deg`` and keep tiles
    that intersect the AOI. tile_id encodes integer grid index for stable naming.
    """
    minx, miny, maxx, maxy = aoi.bounds
    # Snap to grid origin so tile_ids are stable across AOI changes.
    x0 = (minx // tile_deg) * tile_deg
    y0 = (miny // tile_deg) * tile_deg
    tiles: list[Tile] = []
    nx = int(((maxx - x0) // tile_deg) + 1)
    ny = int(((maxy - y0) // tile_deg) + 1)
    for ix in range(nx):
        for iy in range(ny):
            west = x0 + ix * tile_deg
            south = y0 + iy * tile_deg
            east = west + tile_deg
            north = south + tile_deg
            tile_box = box(west, south, east, north)
            if not aoi.intersects(tile_box):
                continue
            # Skip tiles that only touch the AOI on an edge or corner.
            if aoi.intersection(tile_box).area <= 0:
                continue
            # tile_id uses signed-minute coordinates of the SW corner so it is
            # stable, sortable, and human-readable.
            tile_id = f"x{west:+09.4f}_y{south:+09.4f}".replace("+", "p").replace("-", "m")
            tiles.append(Tile(tile_id, west, south, east, north))
    return tiles


# ---------------------------------------------------------------------------
# GEE workflow (per-tile)
# ---------------------------------------------------------------------------


def _gedi_table_ids(tile: ee.Geometry, start_iso: str, end_iso: str) -> list[str]:
    sub = (
        ee.FeatureCollection(GEDI04A_INDEX_ID)
        .filterBounds(tile)
        .filter(ee.Filter.gte("time_start", start_iso))
        .filter(ee.Filter.lt("time_start", end_iso))
    )
    return sub.aggregate_array("table_id").getInfo()


def _quality_filter(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    return fc.filter(
        ee.Filter.And(
            ee.Filter.eq("l4_quality_flag", 1),
            ee.Filter.eq("degrade_flag", 0),
        )
    )


def _disturbance_filter(
    fc: ee.FeatureCollection, disturbed_mask: ee.Image
) -> ee.FeatureCollection:
    """Keep only shots whose location has Hansen lossyear in 2020-2022."""
    return disturbed_mask.rename("dist").sampleRegions(
        collection=fc, scale=30, geometries=True, tileScale=4
    ).filter(ee.Filter.eq("dist", 1))


def _build_window_fc(
    tile: ee.Geometry, start_iso: str, end_iso: str
) -> tuple[ee.FeatureCollection, int]:
    """Per-granule merge for one tile + temporal window. Returns (FC, n_granules)."""
    ids = _gedi_table_ids(tile, start_iso, end_iso)
    if not ids:
        return ee.FeatureCollection([]), 0
    parts = [
        _quality_filter(ee.FeatureCollection(tid).filterBounds(tile))
        for tid in ids
    ]
    return ee.FeatureCollection(parts).flatten(), len(ids)


def _saveall_join_within(
    pre_fc: ee.FeatureCollection,
    post_fc: ee.FeatureCollection,
    distance_m: float,
) -> ee.FeatureCollection:  # pragma: no cover - unused, kept for reference
    """Server-side spatial join (kept for reference; not used).

    Earlier drafts ran the pre/post pairing as a GEE ``Join``
    (``saveAll`` or ``inner``) followed by ``getInfo``, but for AOIs with
    thousands of disturbance-filtered shots this materializes the join
    against the live ``sampleRegions`` chain and exceeds the GEE per-request
    memory cap. The production path pulls both disturbed shot sets to
    DataFrames and pairs them with ``scipy.spatial.cKDTree`` in process.
    """
    distance_filter = ee.Filter.withinDistance(
        distance=distance_m,
        leftField=".geo",
        rightField=".geo",
        maxError=DIST_MAX_ERROR_M,
    )
    join = ee.Join.saveAll(
        matchesKey="matches", measureKey="distance_m", outer=False
    )
    return join.apply(pre_fc, post_fc, distance_filter)


def _pull_shots(fc: ee.FeatureCollection) -> pd.DataFrame:
    """Pull lon/lat + ``SHOT_FIELDS`` for every feature via ``getInfo``.

    Used on a disturbance-filtered FC of typically a few thousand shots per
    tile. The response is small (a few hundred KB) because each feature only
    carries scalar properties + a single point geometry.
    """
    raw = fc.getInfo()
    rows: list[dict] = []
    for feat in raw.get("features", []):
        props = feat.get("properties", {}) or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        row = {"lon": coords[0], "lat": coords[1]}
        for fld in SHOT_FIELDS:
            row[fld] = props.get(fld)
        rows.append(row)
    return pd.DataFrame(rows)


def _pair_shots_local(
    pre_df: pd.DataFrame, post_df: pd.DataFrame, distance_m: float
) -> pd.DataFrame:
    """Find all (pre, post) shot pairs within ``distance_m`` using a KD-tree.

    Inputs carry lon/lat in degrees. Coordinates are projected to a local
    equirectangular metric plane centred on the AOI, which is accurate to
    well under 1 m at the 25 m matching threshold. Returns one row per pair
    with the pre/post columns from ``SHOT_FIELDS`` and a ``distance_m`` column.
    """
    if pre_df.empty or post_df.empty:
        return pd.DataFrame()

    lat0 = float(np.deg2rad(pd.concat([pre_df["lat"], post_df["lat"]]).mean()))
    mx = 111_320.0 * float(np.cos(lat0))
    my = 110_540.0

    pre_xy = np.column_stack([pre_df["lon"].to_numpy() * mx,
                              pre_df["lat"].to_numpy() * my])
    post_xy = np.column_stack([post_df["lon"].to_numpy() * mx,
                               post_df["lat"].to_numpy() * my])

    tree = cKDTree(post_xy)
    matches = tree.query_ball_point(pre_xy, r=distance_m)

    rows: list[dict] = []
    for i, post_idxs in enumerate(matches):
        if not post_idxs:
            continue
        pre_row = pre_df.iloc[i]
        for j in post_idxs:
            post_row = post_df.iloc[j]
            dist = float(np.hypot(pre_xy[i, 0] - post_xy[j, 0],
                                  pre_xy[i, 1] - post_xy[j, 1]))
            row = {
                "pre_lon": float(pre_row["lon"]),
                "pre_lat": float(pre_row["lat"]),
                "post_lon": float(post_row["lon"]),
                "post_lat": float(post_row["lat"]),
                "distance_m": dist,
            }
            for fld in SHOT_FIELDS:
                row[f"pre_{fld}"] = pre_row.get(fld)
                row[f"post_{fld}"] = post_row.get(fld)
            rows.append(row)
    return pd.DataFrame(rows)


def _tile_has_disturbance(
    tile_geom: ee.Geometry, disturbed_mask: ee.Image
) -> bool:
    """Cheap Hansen prescreen: true if any 30 m pixel in the tile has loss in 2020-2022."""
    val = (
        disturbed_mask.reduceRegion(
            reducer=ee.Reducer.anyNonZero(),
            geometry=tile_geom,
            scale=30,
            maxPixels=1e10,
            tileScale=4,
            bestEffort=True,
        )
        .values()
        .get(0)
        .getInfo()
    )
    return bool(val)


@dataclass
class TileResult:
    tile_id: str
    n_pre: int
    n_post: int
    n_pairs_25m: int
    n_pairs_10m: int
    latency_s: float
    status: str
    error: str
    df: pd.DataFrame | None


def process_tile(
    tile: Tile,
    pre_window: tuple[str, str],
    post_window: tuple[str, str],
    distance_m: float,
    strict_distance_m: float,
    disturbed_mask: ee.Image,
) -> TileResult:
    """Run the full GEDI overlap workflow for one tile."""
    t0 = time.perf_counter()
    geom = tile.ee_geom

    # Hansen prescreen (cheap, single reduceRegion call).
    log.info("[%s] hansen prescreen", tile.tile_id)
    if not _tile_has_disturbance(geom, disturbed_mask):
        return TileResult(
            tile.tile_id, 0, 0, 0, 0,
            time.perf_counter() - t0, "skipped_no_disturbance", "", None,
        )

    log.info("[%s] building pre/post FCs", tile.tile_id)
    pre_full, n_pre_g = _build_window_fc(geom, pre_window[0], pre_window[1])
    post_full, n_post_g = _build_window_fc(geom, post_window[0], post_window[1])
    log.info("[%s] %d pre granules, %d post granules", tile.tile_id, n_pre_g, n_post_g)
    pre = _disturbance_filter(pre_full, disturbed_mask)
    post = _disturbance_filter(post_full, disturbed_mask)

    # Skip the redundant size().getInfo() calls — they trigger the expensive
    # sampleRegions chain twice. Pull once; n_pre / n_post are just len(df).
    log.info("[%s] pulling disturbed pre/post shots", tile.tile_id)
    pre_df = _pull_shots(pre)
    post_df = _pull_shots(post)
    n_pre = len(pre_df)
    n_post = len(post_df)
    log.info("[%s] disturbed pre=%d post=%d", tile.tile_id, n_pre, n_post)
    if n_pre == 0 or n_post == 0:
        return TileResult(
            tile.tile_id, n_pre, n_post, 0, 0,
            time.perf_counter() - t0, "ok", "", pd.DataFrame(),
        )

    log.info("[%s] pairing locally @ %.0fm", tile.tile_id, distance_m)
    df = _pair_shots_local(pre_df, post_df, distance_m)
    if not df.empty:
        df["tile_id"] = tile.tile_id
        df["within_strict"] = df["distance_m"] <= strict_distance_m

    n_pairs_25m = len(df)
    n_pairs_10m = int(df["within_strict"].sum()) if not df.empty else 0

    return TileResult(
        tile.tile_id, n_pre, n_post, n_pairs_25m, n_pairs_10m,
        time.perf_counter() - t0, "ok", "", df,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _row_for_tile(tile: Tile, res: TileResult) -> dict:
    return {
        "tile_id": tile.tile_id,
        "west": tile.west,
        "south": tile.south,
        "east": tile.east,
        "north": tile.north,
        "n_pre": res.n_pre,
        "n_post": res.n_post,
        "n_pairs_25m": res.n_pairs_25m,
        "n_pairs_10m": res.n_pairs_10m,
        "latency_s": round(res.latency_s, 2),
        "status": res.status,
        "error": res.error,
    }


def _worker(
    tile: Tile,
    tiles_dir: Path,
    log_path: Path,
    pre_window: tuple[str, str],
    post_window: tuple[str, str],
    distance_m: float,
    strict_distance_m: float,
    disturbed_mask: ee.Image,
) -> None:
    out_path = tiles_dir / f"{tile.tile_id}.parquet"
    if out_path.exists():
        log.info("[%s] skip (exists)", tile.tile_id)
        return

    try:
        res = process_tile(
            tile, pre_window, post_window, distance_m, strict_distance_m, disturbed_mask
        )
    except ee.ee_exception.EEException as exc:
        msg = str(exc).splitlines()[0][:200]
        log.warning("[%s] EE error: %s", tile.tile_id, msg)
        status = "memory_error" if "memory" in msg.lower() else "ee_error"
        append_run_log(
            log_path,
            {
                "tile_id": tile.tile_id, "west": tile.west, "south": tile.south,
                "east": tile.east, "north": tile.north,
                "n_pre": "", "n_post": "", "n_pairs_25m": "", "n_pairs_10m": "",
                "latency_s": "", "status": status, "error": msg,
            },
        )
        return

    if res.df is not None:
        # Empty df is fine — write a marker so we can skip on resume.
        res.df.to_parquet(out_path, index=False)

    append_run_log(log_path, _row_for_tile(tile, res))
    log.info(
        "[%s] %s n_pre=%d n_post=%d pairs25=%d pairs10=%d (%.1fs)",
        tile.tile_id, res.status, res.n_pre, res.n_post,
        res.n_pairs_25m, res.n_pairs_10m, res.latency_s,
    )


def concat_tiles(tiles_dir: Path, master_path: Path) -> int:
    """Concatenate every per-tile Parquet into the master file. Returns row count."""
    parts = sorted(tiles_dir.glob("*.parquet"))
    if not parts:
        log.warning("no per-tile Parquets in %s — nothing to concat", tiles_dir)
        return 0
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_parquet(master_path, index=False)
    log.info("wrote %s with %d rows from %d tiles", master_path, len(df), len(parts))
    return len(df)


def _split_tile(tile: Tile, factor: int = 2) -> list[Tile]:
    """Split a tile into ``factor x factor`` sub-tiles."""
    dx = (tile.east - tile.west) / factor
    dy = (tile.north - tile.south) / factor
    out: list[Tile] = []
    for i in range(factor):
        for j in range(factor):
            west = tile.west + i * dx
            south = tile.south + j * dy
            east = west + dx
            north = south + dy
            sub_id = f"{tile.tile_id}_s{i}{j}"
            out.append(Tile(sub_id, west, south, east, north))
    return out


def _failed_tiles_from_log(log_path: Path) -> list[Tile]:
    if not log_path.exists():
        return []
    df = pd.read_csv(log_path)
    failed = df[df["status"] == "memory_error"]
    return [
        Tile(r.tile_id, float(r.west), float(r.south), float(r.east), float(r.north))
        for r in failed.itertuples(index=False)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aoi", required=True, type=Path, help="GeoJSON or shapefile path")
    p.add_argument("--out-dir", type=Path, default=Path("GEDI/Overlap/outputs"))
    p.add_argument("--pre-window", default="2019-04-18,2021-01-01")
    p.add_argument("--post-window", default="2022-01-01,2023-03-17")
    p.add_argument("--tile-deg", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--distance-m", type=float, default=25.0)
    p.add_argument("--strict-distance-m", type=float, default=10.0)
    p.add_argument("--ee-project", default="dyce-biomass")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run only tiles in run_log.csv with status=memory_error, split 2x2",
    )
    return p.parse_args(argv)


def _parse_window(s: str) -> tuple[str, str]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"window must be 'YYYY-MM-DD,YYYY-MM-DD', got {s!r}")
    return parts[0] + "T00:00:00Z", parts[1] + "T00:00:00Z"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pre_window = _parse_window(args.pre_window)
    post_window = _parse_window(args.post_window)

    out_dir: Path = args.out_dir
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.csv"
    master_path = out_dir / "pairs.parquet"

    ee.Initialize(
        project=args.ee_project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    log.info("Earth Engine initialized (project=%s)", args.ee_project)

    hansen = ee.Image(HANSEN_ASSET).select("lossyear")
    disturbed_mask = hansen.gte(20).And(hansen.lte(22))

    if args.retry_failed:
        failed = _failed_tiles_from_log(log_path)
        if not failed:
            log.info("no memory_error tiles to retry")
            tiles: list[Tile] = []
        else:
            log.info("retrying %d failed tiles split 2x2 -> %d sub-tiles",
                     len(failed), len(failed) * 4)
            tiles = [sub for t in failed for sub in _split_tile(t, 2)]
    else:
        aoi = read_aoi(args.aoi)
        log.info("AOI bounds=%s area_deg2=%.4f", aoi.bounds, aoi.area)
        tiles = tile_aoi(aoi, args.tile_deg)

    log.info("tiles to process: %d", len(tiles))
    if not tiles:
        concat_tiles(tiles_dir, master_path)
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                _worker, t, tiles_dir, log_path,
                pre_window, post_window,
                args.distance_m, args.strict_distance_m, disturbed_mask,
            )
            for t in tiles
        ]
        try:
            for fut in as_completed(futures):
                fut.result()  # surface non-EE exceptions loudly
        except KeyboardInterrupt:
            log.warning("interrupted; cancelling pending tiles")
            for f in futures:
                f.cancel()
            raise

    concat_tiles(tiles_dir, master_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
