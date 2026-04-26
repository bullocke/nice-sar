"""PALSAR-2 ScanSAR per-pair HH/HV sampling for GEDI overlap pairs.

For each pre/post GEDI shot pair in ``pairs.parquet``, find the nearest-in-time
PALSAR-2 ScanSAR L2.2 scene within ``--window-days`` of the GEDI shot time and
sample HH and HV at the GEDI footprint (12.5 m circular buffer, 25 m mean).

Design
------
- GEDI ``delta_time`` is seconds since 2018-01-01T00:00:00Z.
- For each pair, build a server-side ``ee.Feature`` carrying ``pre_target_ms``,
  ``post_target_ms``, and a ``pre_lon``/``pre_lat`` point (used for both
  pre and post sampling — we treat the pair as a single ground location).
- Map ``_sample_one_pair`` over an ``ee.FeatureCollection`` of pair-points.
  Inside the map, filter ScanSAR by date window + bounds, sort by absolute
  time difference to target, take ``.first()``, and sample HH + HV.
- Convert DN to sigma-naught dB client-side: ``sigma0_dB = 10*log10(DN^2) - 83.0``
  (PALSAR-2 L2.2 standard calibration).
- Chunked at ~500 pairs/batch via ``getInfo()``; chunks parallelized with a
  ThreadPoolExecutor.
- Per-chunk Parquets in ``outputs/sar_chunks/`` for resumability; ``sar_log.csv``
  is append-only.
- Row count is preserved: if no scene matches, HH/HV are NaN and ``sar_status``
  records the failure mode.

Run::

    python GEDI/Overlap/sample_sar.py \
        --pairs GEDI/Overlap/outputs/pairs.parquet \
        --ee-project dyce-biomass
"""

from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sample_sar")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCANSAR_ASSET = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
GEDI_EPOCH_MS = 1_514_764_800_000  # 2018-01-01T00:00:00Z in ms
SAR_FIELDS = (
    "pre_HH_dn", "pre_HV_dn", "pre_BeamID", "pre_AntennaPointing",
    "pre_inc_near", "pre_inc_far", "pre_sar_time_ms",
    "post_HH_dn", "post_HV_dn", "post_BeamID", "post_AntennaPointing",
    "post_inc_near", "post_inc_far", "post_sar_time_ms",
)

SAR_LOG_COLUMNS = (
    "chunk_id", "n_in", "n_pre_matched", "n_post_matched",
    "latency_s", "status", "error",
)

_LOG_LOCK = threading.Lock()


def append_sar_log(log_path: Path, row: dict) -> None:
    write_header = not log_path.exists()
    with _LOG_LOCK:
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SAR_LOG_COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in SAR_LOG_COLUMNS})


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------


def _delta_time_to_ms(delta_time_s) -> float | None:
    """Convert GEDI ``delta_time`` (seconds since 2018-01-01) to epoch ms."""
    if delta_time_s is None or pd.isna(delta_time_s):
        return None
    return float(delta_time_s) * 1000.0 + GEDI_EPOCH_MS


def _build_pair_fc(chunk_df: pd.DataFrame) -> ee.FeatureCollection:
    """Build a server-side FC with one feature per pair carrying target times.

    Geometry is the pre-shot point (pairs are <=25 m apart so this is ~the same
    location for pre and post for SAR sampling purposes).
    """
    features = []
    for idx, row in chunk_df.iterrows():
        pre_ms = _delta_time_to_ms(row["pre_delta_time"])
        post_ms = _delta_time_to_ms(row["post_delta_time"])
        if pre_ms is None or post_ms is None:
            # Carry through with sentinel times; sampling will yield no scene.
            pre_ms = pre_ms or 0.0
            post_ms = post_ms or 0.0
        pt = ee.Geometry.Point([float(row["pre_lon"]), float(row["pre_lat"])])
        feat = ee.Feature(
            pt,
            {
                "row_idx": int(idx),
                "pre_target_ms": pre_ms,
                "post_target_ms": post_ms,
            },
        )
        features.append(feat)
    return ee.FeatureCollection(features)


def _sample_at_point(image: ee.Image, pt: ee.Geometry) -> ee.Dictionary:
    """Reduce HH and HV at a 12.5 m buffer with mean reducer at 25 m scale."""
    return image.select(["HH", "HV"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=pt.buffer(12.5),
        scale=25,
        bestEffort=True,
        maxPixels=1e6,
    )


def _nearest_scene_props(target_ms, pt: ee.Geometry, window_days: int) -> ee.Dictionary:
    """Return a dict of HH, HV, BeamID, AntennaPointing, inc near/far, time_ms.

    Filters ScanSAR by ``[target - window, target + window]`` and ``filterBounds``,
    sorts by absolute time difference, takes ``.first()``. If no scene, returns
    a dict with all values null.
    """
    target = ee.Number(target_ms)
    window_ms = ee.Number(window_days * 86400 * 1000)
    start = ee.Date(target.subtract(window_ms))
    end = ee.Date(target.add(window_ms))
    coll = (
        ee.ImageCollection(SCANSAR_ASSET)
        .filterDate(start, end)
        .filterBounds(pt)
    )
    # Drop single-pol (HH-only) scenes — they lack HV. The collection mixes
    # both modes; filter server-side.
    coll = coll.map(
        lambda img: img.set("_has_hv", img.bandNames().contains("HV"))
    ).filter(ee.Filter.eq("_has_hv", True))
    coll_with_dt = coll.map(
        lambda img: img.set(
            "abs_dt_ms",
            ee.Number(img.get("system:time_start")).subtract(target).abs(),
        )
    )
    n = coll_with_dt.size()
    img = ee.Image(coll_with_dt.sort("abs_dt_ms").first())

    sample = ee.Algorithms.If(
        n.gt(0),
        _sample_at_point(img, pt),
        ee.Dictionary({"HH": None, "HV": None}),
    )
    sample = ee.Dictionary(sample)

    # Metadata pulled defensively (image may be null when n=0).
    def _meta(img_):
        return ee.Dictionary(
            {
                "BeamID": img_.get("BeamID"),
                "AntennaPointing": img_.get("AntennaPointing"),
                "inc_near": img_.get("IncAngleNearRange"),
                "inc_far": img_.get("IncAngleFarRange"),
                "time_ms": img_.get("system:time_start"),
            }
        )

    meta = ee.Algorithms.If(
        n.gt(0),
        _meta(img),
        ee.Dictionary(
            {"BeamID": None, "AntennaPointing": None, "inc_near": None,
             "inc_far": None, "time_ms": None}
        ),
    )
    meta = ee.Dictionary(meta)

    return ee.Dictionary(
        {
            "HH_dn": sample.get("HH"),
            "HV_dn": sample.get("HV"),
            "BeamID": meta.get("BeamID"),
            "AntennaPointing": meta.get("AntennaPointing"),
            "inc_near": meta.get("inc_near"),
            "inc_far": meta.get("inc_far"),
            "time_ms": meta.get("time_ms"),
        }
    )


def _make_sample_one(window_days: int):
    """Build a server-side mapper that samples pre and post for one feature."""

    def _sample_one(feat):
        feat = ee.Feature(feat)
        pt = feat.geometry()
        pre = _nearest_scene_props(feat.get("pre_target_ms"), pt, window_days)
        post = _nearest_scene_props(feat.get("post_target_ms"), pt, window_days)
        return feat.set(
            {
                "pre_HH_dn": pre.get("HH_dn"),
                "pre_HV_dn": pre.get("HV_dn"),
                "pre_BeamID": pre.get("BeamID"),
                "pre_AntennaPointing": pre.get("AntennaPointing"),
                "pre_inc_near": pre.get("inc_near"),
                "pre_inc_far": pre.get("inc_far"),
                "pre_sar_time_ms": pre.get("time_ms"),
                "post_HH_dn": post.get("HH_dn"),
                "post_HV_dn": post.get("HV_dn"),
                "post_BeamID": post.get("BeamID"),
                "post_AntennaPointing": post.get("AntennaPointing"),
                "post_inc_near": post.get("inc_near"),
                "post_inc_far": post.get("inc_far"),
                "post_sar_time_ms": post.get("time_ms"),
            }
        )

    return _sample_one


# ---------------------------------------------------------------------------
# DN -> dB conversion
# ---------------------------------------------------------------------------


def _dn_to_db(dn) -> float:
    """PALSAR-2 L2.2 calibration: sigma0_dB = 10*log10(DN^2) - 83.0.

    Returns NaN for invalid DN (None, 0, NaN).
    """
    if dn is None or pd.isna(dn) or dn <= 0:
        return float("nan")
    return 10.0 * np.log10(float(dn) ** 2) - 83.0


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


def process_chunk(
    chunk_id: str,
    chunk_df: pd.DataFrame,
    window_days: int,
    out_path: Path,
    log_path: Path,
) -> None:
    """Process one chunk: build FC, server-side sample, getInfo, dB-convert, write."""
    if out_path.exists():
        log.info("[%s] skip (exists)", chunk_id)
        return

    t0 = time.time()
    try:
        fc = _build_pair_fc(chunk_df)
        sampled = fc.map(_make_sample_one(window_days))
        # Pull as flat list of dicts.
        raw = sampled.getInfo()
    except ee.ee_exception.EEException as e:
        msg = str(e)
        log.warning("[%s] EE error: %s", chunk_id, msg)
        append_sar_log(
            log_path,
            {
                "chunk_id": chunk_id, "n_in": len(chunk_df), "n_pre_matched": "",
                "n_post_matched": "", "latency_s": f"{time.time() - t0:.1f}",
                "status": "ee_error", "error": msg,
            },
        )
        return

    # Build a DataFrame indexed by row_idx so we can left-join back.
    by_idx: dict[int, dict] = {}
    for feat in raw.get("features", []):
        props = feat.get("properties", {}) or {}
        ridx = props.get("row_idx")
        if ridx is None:
            continue
        by_idx[int(ridx)] = props

    rows = []
    for idx, row in chunk_df.iterrows():
        p = by_idx.get(int(idx), {})
        out_row = row.to_dict()
        for f in SAR_FIELDS:
            out_row[f] = p.get(f)
        # Compute dB and timestamps.
        out_row["pre_HH_db"] = _dn_to_db(out_row.get("pre_HH_dn"))
        out_row["pre_HV_db"] = _dn_to_db(out_row.get("pre_HV_dn"))
        out_row["post_HH_db"] = _dn_to_db(out_row.get("post_HH_dn"))
        out_row["post_HV_db"] = _dn_to_db(out_row.get("post_HV_dn"))
        out_row["pre_sar_date"] = (
            pd.Timestamp(out_row["pre_sar_time_ms"], unit="ms", tz="UTC")
            if out_row.get("pre_sar_time_ms") is not None else pd.NaT
        )
        out_row["post_sar_date"] = (
            pd.Timestamp(out_row["post_sar_time_ms"], unit="ms", tz="UTC")
            if out_row.get("post_sar_time_ms") is not None else pd.NaT
        )
        # Status flag.
        has_pre = not pd.isna(out_row["pre_HV_db"])
        has_post = not pd.isna(out_row["post_HV_db"])
        if has_pre and has_post:
            out_row["sar_status"] = "ok"
        elif has_pre:
            out_row["sar_status"] = "no_post"
        elif has_post:
            out_row["sar_status"] = "no_pre"
        else:
            out_row["sar_status"] = "no_match"
        rows.append(out_row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    n_pre = int(df["pre_HV_db"].notna().sum())
    n_post = int(df["post_HV_db"].notna().sum())
    latency = time.time() - t0
    append_sar_log(
        log_path,
        {
            "chunk_id": chunk_id, "n_in": len(chunk_df), "n_pre_matched": n_pre,
            "n_post_matched": n_post, "latency_s": f"{latency:.1f}",
            "status": "ok", "error": "",
        },
    )
    log.info(
        "[%s] ok n_in=%d pre=%d post=%d (%.1fs)",
        chunk_id, len(chunk_df), n_pre, n_post, latency,
    )


def concat_chunks(chunks_dir: Path, master_path: Path) -> int:
    parts = sorted(chunks_dir.glob("*.parquet"))
    if not parts:
        log.warning("no chunk Parquets in %s", chunks_dir)
        return 0
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(master_path, index=False)
    log.info("wrote %s with %d rows from %d chunks", master_path, len(df), len(parts))
    return len(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pairs", type=Path, default=Path("GEDI/Overlap/outputs/pairs.parquet"))
    p.add_argument("--out", type=Path, default=Path("GEDI/Overlap/outputs/pairs_with_sar.parquet"))
    p.add_argument("--chunks-dir", type=Path, default=Path("GEDI/Overlap/outputs/sar_chunks"))
    p.add_argument("--log", type=Path, default=Path("GEDI/Overlap/outputs/sar_log.csv"))
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ee-project", default="dyce-biomass")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pairs.exists():
        log.error("pairs file not found: %s", args.pairs)
        return 2

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ee.Initialize(
        project=args.ee_project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    log.info("Earth Engine initialized (project=%s)", args.ee_project)

    pairs = pd.read_parquet(args.pairs).reset_index(drop=True)
    log.info("loaded %d pairs from %s", len(pairs), args.pairs)
    if "pre_delta_time" not in pairs.columns:
        log.error(
            "pairs.parquet missing 'pre_delta_time' column. Re-run scale_pairs.py "
            "with the patched SHOT_FIELDS that includes delta_time."
        )
        return 3

    n = len(pairs)
    n_chunks = (n + args.batch_size - 1) // args.batch_size

    def _run(i: int) -> None:
        chunk_id = f"chunk_{i:05d}"
        out_path = args.chunks_dir / f"{chunk_id}.parquet"
        chunk = pairs.iloc[i * args.batch_size : (i + 1) * args.batch_size]
        process_chunk(chunk_id, chunk, args.window_days, out_path, args.log)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, i): i for i in range(n_chunks)}
        try:
            for fut in as_completed(futs):
                fut.result()
        except KeyboardInterrupt:
            log.warning("interrupted; cancelling pending chunks")
            for f in futs:
                f.cancel()
            raise

    concat_chunks(args.chunks_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
