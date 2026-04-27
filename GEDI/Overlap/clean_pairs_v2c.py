"""Clean pairs_with_sar_v2 → pairs_with_sar_v2c (A1 + B2).

Two failure modes diagnosed from chip review of the v2 dataset motivate
this cleaning pass:

A1. **Edge-of-patch geolocation noise.** Some GEDI shots fall within
    one Hansen pixel (~30 m) of a disturbance edge. Tiny lat/lon errors
    flip a shot in/out of the disturbed pixel, producing large random
    ΔHV with no real biomass change.

    Test: at the GEDI pre footprint, sample Hansen ``lossyear`` over a
    30 m radius circle and require ``min == max``. Any pixel-disagreement
    inside the buffer marks the site as "edge".

B2. **PALSAR-2 post-date precedes the disturbance year.** A few sites
    have ``hansen_lossyear ∈ {20, 21, 22}`` but a PALSAR-2 post scene
    from early-mid 2022, i.e. potentially before the actual loss event.

    Test: drop rows where ``post_sar_date.year <= 2000 + hansen_loss_max``
    when ``hansen_loss_max > 0``. (Keep stricter inequality so post-year
    must strictly exceed the latest loss year in the buffer.)

Subcommands
-----------
``annotate``
    Sample Hansen lossyear over a 30 m buffer at each pair's pre point
    via Earth Engine in chunked batches and write
    ``pairs_with_sar_v2c_annotated.parquet`` plus chunk shards.

``filter``
    Apply A1 + B2 rules to the annotated parquet, write
    ``pairs_with_sar_v2c.parquet`` (passing rows only) and
    ``pairs_with_sar_v2c_audit.csv`` (every row + drop reason).

``all``
    Run ``annotate`` then ``filter``.

Run::

    python GEDI/Overlap/clean_pairs_v2c.py annotate --ee-project dyce-biomass
    python GEDI/Overlap/clean_pairs_v2c.py filter
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
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_pairs_v2c")


HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"

ANNOT_FIELDS = ("hansen_loss_min", "hansen_loss_max", "hansen_loss_mode")

LOG_COLUMNS = ("chunk_id", "n_in", "n_annotated", "latency_s", "status", "error")

_LOG_LOCK = threading.Lock()


def append_log(log_path: Path, row: dict) -> None:
    write_header = not log_path.exists()
    with _LOG_LOCK:
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


def _build_fc(chunk_df: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for idx, row in chunk_df.iterrows():
        pt = ee.Geometry.Point([float(row["pre_lon"]), float(row["pre_lat"])])
        feats.append(ee.Feature(pt, {"row_idx": int(idx)}))
    return ee.FeatureCollection(feats)


def _make_sample_one(buffer_m: float):
    img = ee.Image(HANSEN_ASSET).select("lossyear")
    reducer = (
        ee.Reducer.min()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.mode(), sharedInputs=True)
    )

    def _one(feat):
        feat = ee.Feature(feat)
        geom = feat.geometry().buffer(buffer_m)
        d = img.reduceRegion(
            reducer=reducer,
            geometry=geom,
            scale=30,
            bestEffort=True,
            maxPixels=1e6,
        )
        return feat.set({
            "hansen_loss_min": d.get("lossyear_min"),
            "hansen_loss_max": d.get("lossyear_max"),
            "hansen_loss_mode": d.get("lossyear_mode"),
        })

    return _one


def process_chunk(
    chunk_id: str,
    chunk_df: pd.DataFrame,
    buffer_m: float,
    out_path: Path,
    log_path: Path,
) -> None:
    if out_path.exists():
        log.info("[%s] skip (exists)", chunk_id)
        return
    t0 = time.time()
    try:
        fc = _build_fc(chunk_df)
        sampled = fc.map(_make_sample_one(buffer_m))
        raw = sampled.getInfo()
    except ee.ee_exception.EEException as e:
        msg = str(e)
        log.warning("[%s] EE error: %s", chunk_id, msg)
        append_log(log_path, {
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
    n_ok = 0
    for idx, row in chunk_df.iterrows():
        p = by_idx.get(int(idx), {})
        out_row = row.to_dict()
        for f in ANNOT_FIELDS:
            out_row[f] = p.get(f)
        if p.get("hansen_loss_min") is not None:
            n_ok += 1
        rows.append(out_row)

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    latency = time.time() - t0
    append_log(log_path, {
        "chunk_id": chunk_id, "n_in": len(chunk_df),
        "n_annotated": n_ok,
        "latency_s": f"{latency:.1f}", "status": "ok", "error": "",
    })
    log.info("[%s] ok n_in=%d annotated=%d (%.1fs)",
             chunk_id, len(chunk_df), n_ok, latency)


def concat_chunks(chunks_dir: Path, master_path: Path) -> int:
    parts = sorted(chunks_dir.glob("*.parquet"))
    if not parts:
        log.warning("no chunk Parquets in %s", chunks_dir)
        return 0
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(master_path, index=False)
    log.info("wrote %s with %d rows from %d chunks", master_path, len(df), len(parts))
    return len(df)


def cmd_annotate(args: argparse.Namespace) -> int:
    if not args.input.exists():
        log.error("input not found: %s", args.input)
        return 2

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ee.Initialize(
        project=args.ee_project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    log.info("Earth Engine initialized (project=%s)", args.ee_project)

    df = pd.read_parquet(args.input).reset_index(drop=True)
    df["_v2c_row"] = df.index.astype(int)
    log.info("loaded %d rows from %s", len(df), args.input)

    # Only annotate rows with successful SAR sampling — others will be dropped
    # later anyway and there's no point hitting EE for them.
    mask_ok = df["sar_status"] == "ok"
    work = df.loc[mask_ok].copy()
    log.info("rows with sar_status=ok: %d", len(work))

    if args.limit > 0:
        work = work.iloc[: args.limit]
        log.info("limited to first %d rows for smoke test", len(work))

    n = len(work)
    n_chunks = (n + args.chunk_size - 1) // args.chunk_size

    def _run(i: int) -> None:
        chunk_id = f"chunk_{i:05d}"
        out_path = args.chunks_dir / f"{chunk_id}.parquet"
        chunk = work.iloc[i * args.chunk_size : (i + 1) * args.chunk_size]
        process_chunk(chunk_id, chunk, args.buffer_m, out_path, args.log)

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

    annotated_ok = pd.concat(
        [pd.read_parquet(p) for p in sorted(args.chunks_dir.glob("*.parquet"))],
        ignore_index=True,
    ) if any(args.chunks_dir.iterdir()) else pd.DataFrame()

    # Merge annotation columns back into full df (non-ok rows get NaN).
    if len(annotated_ok) > 0:
        # Use the synthetic _v2c_row index as join key — the annotated chunks
        # carry it through from row.to_dict(), so this avoids duplicates that
        # would arise from joining on (lat, lon, time).
        annot_subset = annotated_ok[["_v2c_row"] + list(ANNOT_FIELDS)]
        df = df.merge(annot_subset, on="_v2c_row", how="left")
    else:
        for f in ANNOT_FIELDS:
            df[f] = pd.NA

    df = df.drop(columns=["_v2c_row"])

    df.to_parquet(args.out, index=False)
    log.info("wrote %s (%d rows; %d annotated)",
             args.out, len(df), int(df["hansen_loss_min"].notna().sum()))
    return 0


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def cmd_filter(args: argparse.Namespace) -> int:
    if not args.input.exists():
        log.error("input not found: %s", args.input)
        return 2

    df = pd.read_parquet(args.input)
    log.info("loaded %d rows from %s", len(df), args.input)

    for f in ANNOT_FIELDS:
        if f not in df.columns:
            log.error("missing column %s — run `annotate` first", f)
            return 3

    # Numeric coercion (EE can return ints, pyarrow may store as object).
    lo = pd.to_numeric(df["hansen_loss_min"], errors="coerce")
    hi = pd.to_numeric(df["hansen_loss_max"], errors="coerce")

    # A1: edge-of-patch — drop where min != max.
    edge = (lo.notna() & hi.notna()) & (lo != hi)

    # B2: PALSAR-2 post-date timing — drop where post_sar_date.year is not
    # strictly greater than 2000 + hansen_loss_max (only when there was loss).
    has_loss = hi.notna() & (hi > 0)
    post_year = pd.to_datetime(df["post_sar_date"], utc=True, errors="coerce").dt.year
    timing = has_loss & (post_year <= 2000 + hi)

    # Reason precedence: edge first (geolocation issue is upstream), then timing.
    reason = pd.Series([None] * len(df), dtype=object, index=df.index)
    reason[edge] = "edge"
    reason[timing & ~edge] = "timing"

    drop_mask = edge | timing
    passes = ~drop_mask & (df["sar_status"] == "ok") & lo.notna()

    df_out = df.copy()
    df_out["v2c_drop_reason"] = reason
    df_out["passes_v2c"] = passes

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.loc[passes].drop(columns=["v2c_drop_reason", "passes_v2c"]).to_parquet(
        args.out, index=False,
    )
    log.info("wrote %s (%d passing rows)", args.out, int(passes.sum()))

    audit_cols = [
        "pre_lon", "pre_lat", "pre_delta_time", "post_delta_time",
        "pre_sar_date", "post_sar_date",
        "pre_HV_db", "post_HV_db", "pre_agbd", "post_agbd",
        "sar_status",
        "hansen_loss_min", "hansen_loss_max", "hansen_loss_mode",
        "v2c_drop_reason", "passes_v2c",
    ]
    audit_cols = [c for c in audit_cols if c in df_out.columns]
    df_out[audit_cols].to_csv(args.audit, index=False)
    log.info("wrote %s (%d rows)", args.audit, len(df_out))

    n_ok = int((df["sar_status"] == "ok").sum())
    n_edge = int(edge.sum())
    n_timing = int((timing & ~edge).sum())
    n_pass = int(passes.sum())
    log.info(
        "summary: sar_ok=%d  edge=%d  timing=%d  passes_v2c=%d (%.1f%% of sar_ok)",
        n_ok, n_edge, n_timing, n_pass,
        100.0 * n_pass / max(n_ok, 1),
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    out_dir = Path("GEDI/Overlap/outputs")

    a = sub.add_parser("annotate", help="Sample Hansen lossyear stats per pair via EE")
    a.add_argument("--input", type=Path, default=out_dir / "pairs_with_sar_v2.parquet")
    a.add_argument("--out", type=Path, default=out_dir / "pairs_with_sar_v2c_annotated.parquet")
    a.add_argument("--chunks-dir", type=Path, default=out_dir / "_v2c_chunks")
    a.add_argument("--log", type=Path, default=out_dir / "v2c_annotate_log.csv")
    a.add_argument("--buffer-m", type=float, default=30.0)
    a.add_argument("--chunk-size", type=int, default=200)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--limit", type=int, default=0, help="0 = all rows; >0 = first N")
    a.add_argument("--ee-project", default="dyce-biomass")

    f = sub.add_parser("filter", help="Apply A1 (edge) + B2 (timing) rules")
    f.add_argument("--input", type=Path, default=out_dir / "pairs_with_sar_v2c_annotated.parquet")
    f.add_argument("--out", type=Path, default=out_dir / "pairs_with_sar_v2c.parquet")
    f.add_argument("--audit", type=Path, default=out_dir / "pairs_with_sar_v2c_audit.csv")

    al = sub.add_parser("all", help="annotate + filter")
    al.add_argument("--input", type=Path, default=out_dir / "pairs_with_sar_v2.parquet")
    al.add_argument("--annotated", type=Path,
                    default=out_dir / "pairs_with_sar_v2c_annotated.parquet")
    al.add_argument("--out", type=Path, default=out_dir / "pairs_with_sar_v2c.parquet")
    al.add_argument("--audit", type=Path, default=out_dir / "pairs_with_sar_v2c_audit.csv")
    al.add_argument("--chunks-dir", type=Path, default=out_dir / "_v2c_chunks")
    al.add_argument("--log", type=Path, default=out_dir / "v2c_annotate_log.csv")
    al.add_argument("--buffer-m", type=float, default=30.0)
    al.add_argument("--chunk-size", type=int, default=200)
    al.add_argument("--workers", type=int, default=4)
    al.add_argument("--limit", type=int, default=0)
    al.add_argument("--ee-project", default="dyce-biomass")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "annotate":
        return cmd_annotate(args)
    if args.cmd == "filter":
        return cmd_filter(args)
    if args.cmd == "all":
        a_args = argparse.Namespace(
            input=args.input, out=args.annotated,
            chunks_dir=args.chunks_dir, log=args.log,
            buffer_m=args.buffer_m, chunk_size=args.chunk_size,
            workers=args.workers, limit=args.limit, ee_project=args.ee_project,
        )
        rc = cmd_annotate(a_args)
        if rc != 0:
            return rc
        f_args = argparse.Namespace(
            input=args.annotated, out=args.out, audit=args.audit,
        )
        return cmd_filter(f_args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
