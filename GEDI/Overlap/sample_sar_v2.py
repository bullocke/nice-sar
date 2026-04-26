"""PALSAR-2 ScanSAR per-pair sampling with same-pass + same-path matching (v2).

This is a fork of ``sample_sar.py``. The only behavioural difference is that
each pair's pre and post scenes must share the same ``PassDirection``
(Asc/Desc) and ``RSP_Path_Number``. Mixing those across pre/post adds large
per-pixel ΔHV / ΔHH variance with no biomass signal, so this version
constrains the match to reduce that noise floor.

Algorithm (coordinated symmetric matching)
------------------------------------------
For each pair, server-side:

1. Build the pre and post candidate ScanSAR collections (date window +
   bounds + dual-pol filter, as in v1).
2. Compute the set of ``(PassDirection, RSP_Path_Number)`` keys present in
   each; intersect.
3. For each common key, take the nearest-in-time scene from pre and from
   post and compute ``total_dt = |pre_dt| + |post_dt|``.
4. Pick the key with the minimum ``total_dt``; sample HH/HV from those two
   scenes.
5. If the intersection is empty, both HH/HV are NaN and ``sar_status`` is
   ``no_pass_path_match``.

Because pre and post are chosen jointly, every matched row satisfies
``pre_PassDirection == post_PassDirection`` and
``pre_RSP_Path_Number == post_RSP_Path_Number`` by construction.

Run::

    python GEDI/Overlap/sample_sar_v2.py \
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
log = logging.getLogger("sample_sar_v2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCANSAR_ASSET = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
GEDI_EPOCH_MS = 1_514_764_800_000  # 2018-01-01T00:00:00Z in ms
SAR_FIELDS = (
    "pre_HH_dn", "pre_HV_dn", "pre_BeamID", "pre_AntennaPointing",
    "pre_PassDirection", "pre_RSP_Path_Number",
    "pre_inc_near", "pre_inc_far", "pre_sar_time_ms",
    "post_HH_dn", "post_HV_dn", "post_BeamID", "post_AntennaPointing",
    "post_PassDirection", "post_RSP_Path_Number",
    "post_inc_near", "post_inc_far", "post_sar_time_ms",
    "match_total_dt_s", "match_n_candidates",
)

SAR_LOG_COLUMNS = (
    "chunk_id", "n_in", "n_pre_matched", "n_post_matched",
    "n_pass_path_matched", "latency_s", "status", "error",
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
    if delta_time_s is None or pd.isna(delta_time_s):
        return None
    return float(delta_time_s) * 1000.0 + GEDI_EPOCH_MS


def _build_pair_fc(chunk_df: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for idx, row in chunk_df.iterrows():
        pre_ms = _delta_time_to_ms(row["pre_delta_time"])
        post_ms = _delta_time_to_ms(row["post_delta_time"])
        if pre_ms is None or post_ms is None:
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
    return image.select(["HH", "HV"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=pt.buffer(12.5),
        scale=25,
        bestEffort=True,
        maxPixels=1e6,
    )


def _candidate_collection(
    target_ms, pt: ee.Geometry, window_days: int,
    require_dual_pol: bool,
) -> ee.ImageCollection:
    """ScanSAR scenes within ±window_days of target_ms over pt, with abs_dt_ms."""
    target = ee.Number(target_ms)
    window_ms = ee.Number(window_days * 86400 * 1000)
    start = ee.Date(target.subtract(window_ms))
    end = ee.Date(target.add(window_ms))
    coll = (
        ee.ImageCollection(SCANSAR_ASSET)
        .filterDate(start, end)
        .filterBounds(pt)
    )
    if require_dual_pol:
        coll = coll.map(
            lambda img: img.set("_has_hv", img.bandNames().contains("HV"))
        ).filter(ee.Filter.eq("_has_hv", True))
    coll = coll.map(
        lambda img: img.set(
            "abs_dt_ms",
            ee.Number(img.get("system:time_start")).subtract(target).abs(),
        )
    )
    return coll


def _key_str(img: ee.Image) -> ee.String:
    """Build a 'PassDirection|RSP_Path_Number' key string for an image."""
    pd_ = ee.String(ee.Algorithms.If(
        img.get("PassDirection"), ee.String(img.get("PassDirection")), ee.String("UNK")
    ))
    rsp = ee.Algorithms.If(
        img.get("RSP_Path_Number"),
        ee.Number(img.get("RSP_Path_Number")).format("%d"),
        ee.String("UNK"),
    )
    return pd_.cat(ee.String("|")).cat(ee.String(rsp))


def _meta_dict(img: ee.Image) -> ee.Dictionary:
    return ee.Dictionary({
        "BeamID": img.get("BeamID"),
        "AntennaPointing": img.get("AntennaPointing"),
        "PassDirection": img.get("PassDirection"),
        "RSP_Path_Number": img.get("RSP_Path_Number"),
        "inc_near": img.get("IncAngleNearRange"),
        "inc_far": img.get("IncAngleFarRange"),
        "time_ms": img.get("system:time_start"),
    })


def _null_pair_dict() -> ee.Dictionary:
    keys = [
        "pre_HH_dn", "pre_HV_dn", "pre_BeamID", "pre_AntennaPointing",
        "pre_PassDirection", "pre_RSP_Path_Number",
        "pre_inc_near", "pre_inc_far", "pre_sar_time_ms",
        "post_HH_dn", "post_HV_dn", "post_BeamID", "post_AntennaPointing",
        "post_PassDirection", "post_RSP_Path_Number",
        "post_inc_near", "post_inc_far", "post_sar_time_ms",
        "match_total_dt_s", "match_n_candidates",
    ]
    return ee.Dictionary.fromLists(keys, [None] * len(keys))


def _coordinated_match(
    pre_target_ms, post_target_ms, pt: ee.Geometry,
    window_days: int, require_dual_pol: bool,
) -> ee.Dictionary:
    """Find the (PassDirection, RSP_Path_Number) key minimising |pre_dt|+|post_dt|."""
    pre_coll = _candidate_collection(pre_target_ms, pt, window_days, require_dual_pol)
    post_coll = _candidate_collection(post_target_ms, pt, window_days, require_dual_pol)

    # Tag each image with its key.
    pre_tagged = pre_coll.map(lambda img: img.set("_key", _key_str(img)))
    post_tagged = post_coll.map(lambda img: img.set("_key", _key_str(img)))

    pre_keys = ee.List(pre_tagged.aggregate_array("_key")).distinct()
    post_keys = ee.List(post_tagged.aggregate_array("_key")).distinct()
    common = pre_keys.filter(ee.Filter.inList("item", post_keys))
    n_common = common.size()

    def _per_key(key):
        key = ee.String(key)
        pre_best = ee.Image(
            pre_tagged.filter(ee.Filter.eq("_key", key)).sort("abs_dt_ms").first()
        )
        post_best = ee.Image(
            post_tagged.filter(ee.Filter.eq("_key", key)).sort("abs_dt_ms").first()
        )
        total_dt = ee.Number(pre_best.get("abs_dt_ms")).add(
            ee.Number(post_best.get("abs_dt_ms"))
        )
        return ee.Feature(None, {
            "_key": key,
            "_total_dt": total_dt,
            "_pre_id": pre_best.get("system:index"),
            "_post_id": post_best.get("system:index"),
        })

    candidates = ee.FeatureCollection(common.map(_per_key))
    best = ee.Feature(candidates.sort("_total_dt").first())
    pre_id = best.get("_pre_id")
    post_id = best.get("_post_id")

    pre_img = ee.Image(pre_tagged.filter(ee.Filter.eq("system:index", pre_id)).first())
    post_img = ee.Image(post_tagged.filter(ee.Filter.eq("system:index", post_id)).first())

    pre_sample = ee.Dictionary(_sample_at_point(pre_img, pt))
    post_sample = ee.Dictionary(_sample_at_point(post_img, pt))
    pre_meta = _meta_dict(pre_img)
    post_meta = _meta_dict(post_img)

    matched = ee.Dictionary({
        "pre_HH_dn": pre_sample.get("HH"),
        "pre_HV_dn": pre_sample.get("HV"),
        "pre_BeamID": pre_meta.get("BeamID"),
        "pre_AntennaPointing": pre_meta.get("AntennaPointing"),
        "pre_PassDirection": pre_meta.get("PassDirection"),
        "pre_RSP_Path_Number": pre_meta.get("RSP_Path_Number"),
        "pre_inc_near": pre_meta.get("inc_near"),
        "pre_inc_far": pre_meta.get("inc_far"),
        "pre_sar_time_ms": pre_meta.get("time_ms"),
        "post_HH_dn": post_sample.get("HH"),
        "post_HV_dn": post_sample.get("HV"),
        "post_BeamID": post_meta.get("BeamID"),
        "post_AntennaPointing": post_meta.get("AntennaPointing"),
        "post_PassDirection": post_meta.get("PassDirection"),
        "post_RSP_Path_Number": post_meta.get("RSP_Path_Number"),
        "post_inc_near": post_meta.get("inc_near"),
        "post_inc_far": post_meta.get("inc_far"),
        "post_sar_time_ms": post_meta.get("time_ms"),
        "match_total_dt_s": ee.Number(best.get("_total_dt")).divide(1000),
        "match_n_candidates": n_common,
    })

    null = _null_pair_dict().set("match_n_candidates", n_common)
    return ee.Dictionary(ee.Algorithms.If(n_common.gt(0), matched, null))


def _make_sample_one(window_days: int, require_dual_pol: bool):
    def _sample_one(feat):
        feat = ee.Feature(feat)
        pt = feat.geometry()
        d = _coordinated_match(
            feat.get("pre_target_ms"), feat.get("post_target_ms"),
            pt, window_days, require_dual_pol,
        )
        return feat.set(d)
    return _sample_one


# ---------------------------------------------------------------------------
# DN -> dB
# ---------------------------------------------------------------------------


def _dn_to_db(dn) -> float:
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
    require_dual_pol: bool,
    out_path: Path,
    log_path: Path,
) -> None:
    if out_path.exists():
        log.info("[%s] skip (exists)", chunk_id)
        return

    t0 = time.time()
    try:
        fc = _build_pair_fc(chunk_df)
        sampled = fc.map(_make_sample_one(window_days, require_dual_pol))
        raw = sampled.getInfo()
    except ee.ee_exception.EEException as e:
        msg = str(e)
        log.warning("[%s] EE error: %s", chunk_id, msg)
        append_sar_log(log_path, {
            "chunk_id": chunk_id, "n_in": len(chunk_df),
            "latency_s": f"{time.time() - t0:.1f}",
            "status": "ee_error", "error": msg,
        })
        return

    by_idx: dict[int, dict] = {}
    for feat in raw.get("features", []):
        props = feat.get("properties", {}) or {}
        ridx = props.get("row_idx")
        if ridx is not None:
            by_idx[int(ridx)] = props

    rows = []
    for idx, row in chunk_df.iterrows():
        p = by_idx.get(int(idx), {})
        out_row = row.to_dict()
        for f in SAR_FIELDS:
            out_row[f] = p.get(f)
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
        n_cand = out_row.get("match_n_candidates")
        has_pre = not pd.isna(out_row["pre_HV_db"])
        has_post = not pd.isna(out_row["post_HV_db"])
        if has_pre and has_post:
            out_row["sar_status"] = "ok"
        elif n_cand in (None, 0):
            out_row["sar_status"] = "no_pass_path_match"
        else:
            out_row["sar_status"] = "no_match"
        rows.append(out_row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    n_pre = int(df["pre_HV_db"].notna().sum())
    n_post = int(df["post_HV_db"].notna().sum())
    n_pp = int((df["sar_status"] == "ok").sum())
    latency = time.time() - t0
    append_sar_log(log_path, {
        "chunk_id": chunk_id, "n_in": len(chunk_df),
        "n_pre_matched": n_pre, "n_post_matched": n_post,
        "n_pass_path_matched": n_pp,
        "latency_s": f"{latency:.1f}", "status": "ok", "error": "",
    })
    log.info(
        "[%s] ok n_in=%d pre=%d post=%d pp_ok=%d (%.1fs)",
        chunk_id, len(chunk_df), n_pre, n_post, n_pp, latency,
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
    p.add_argument("--out", type=Path, default=Path("GEDI/Overlap/outputs/pairs_with_sar_v2.parquet"))
    p.add_argument("--chunks-dir", type=Path, default=Path("GEDI/Overlap/outputs/sar_chunks_v2"))
    p.add_argument("--log", type=Path, default=Path("GEDI/Overlap/outputs/sar_log_v2.csv"))
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-dual-pol", action="store_true",
                   help="Do NOT require HV band (allows HH-only single-pol scenes).")
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
        log.error("pairs.parquet missing 'pre_delta_time' column.")
        return 3

    require_dual_pol = not args.no_dual_pol
    n = len(pairs)
    n_chunks = (n + args.batch_size - 1) // args.batch_size

    def _run(i: int) -> None:
        chunk_id = f"chunk_{i:05d}"
        out_path = args.chunks_dir / f"{chunk_id}.parquet"
        chunk = pairs.iloc[i * args.batch_size : (i + 1) * args.batch_size]
        process_chunk(
            chunk_id, chunk, args.window_days, require_dual_pol,
            out_path, args.log,
        )

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
