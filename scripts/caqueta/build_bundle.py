#!/usr/bin/env python
"""Package the Caquetá inputs as a small demo bundle for the notebooks.

The bundle lets ``notebooks/11_disturbance_detection.ipynb`` (and notebooks 07,
09, 10) run on Colab without NASA Earthdata or Earth Engine credentials. It holds,
all on the 20 m analysis grid (EPSG:32618):

- HV backscatter (12 dates, dB x 100 as int16)
- HH coherence at 80 m (reprojected to 20 m, nearest) and 20 m (21 pairs, x 250
  as uint8, 0 = no data)
- RADD alert date, confidence, and primary-forest baseline
- Sentinel-2 true-colour RGB (JPEG, quality 92) on the clearest dates
- case outlines and attributes from ``select_cases.py`` (ring reference), with
  each case's mean Sentinel-2 NBR on the dates usable for it
- the source granule IDs

Values are quantized: 0.01 dB for backscatter and 0.004 for coherence, well
below the noise of either. Run after ``select_cases.py``:

    python scripts/caqueta/build_bundle.py

Writes ``local_examples/caqueta/bundle/caqueta_demo_v1.npz`` and prints its
SHA-256 for ``nice_sar/datasets.py``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging

import analysis
import config
import data
import numpy as np
from PIL import Image

logger = logging.getLogger("build_bundle")

OUT = config.OUT_DIR / "bundle" / "caqueta_demo_v1.npz"
CASES = config.OUT_DIR / "04_cases" / "cases_forest_ring"
RGB_STRETCH = (0.0, 0.15)  # surface reflectance mapped to 0-255


def _db_int16(stack: np.ndarray) -> np.ndarray:
    out = np.round(stack * 100)
    out[~np.isfinite(out)] = -32768
    return out.astype("int16")


def _coh_uint8(stack: np.ndarray) -> np.ndarray:
    out = np.clip(np.round(stack * 250), 1, 250)
    out[~np.isfinite(stack)] = 0
    return out.astype("uint8")


def _iso(days: np.ndarray) -> np.ndarray:
    return np.array([config.to_date(d).isoformat() for d in days])


def rgb_scenes(s2: data.S2Stack) -> tuple[np.ndarray, list[np.ndarray]]:
    """True-colour RGB at 20 m (JPEG bytes) on dates that are at least 85% clear and haze-free."""
    scenes = {s.day: s for s in data.s2_scenes()}
    days, rgbs = [], []
    for t, day in enumerate(s2.days):
        clear = s2.clear[t]
        if clear.mean() < 0.85 or np.median(s2.blue[t][clear]) > config.S2_MAX_BLUE:
            continue
        rgb10 = data.s2_rgb(scenes[day], stretch=RGB_STRETCH)
        h, w = rgb10.shape[0] // 2, rgb10.shape[1] // 2
        rgb20 = rgb10[: h * 2, : w * 2].reshape(h, 2, w, 2, 3).mean(axis=(1, 3))
        days.append(day)
        buf = io.BytesIO()
        Image.fromarray(np.round(rgb20 * 255).astype("uint8")).save(buf, "JPEG", quality=92)
        rgbs.append(np.frombuffer(buf.getvalue(), dtype="uint8"))
    logger.info("RGB dates: %s", ", ".join(_iso(np.array(days))))
    return np.array(days), rgbs


def cases(s2: data.S2Stack, shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Case outlines, attributes and NBR series from the ring-reference run."""
    npz = np.load(CASES.with_suffix(".npz"))
    with open(CASES.with_suffix(".csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    outlines = {}
    for r in rows:
        cid = r["case_id"]
        outlines[f"case_{cid}_rows"] = npz[f"{cid}_rows"].astype("int16")
        outlines[f"case_{cid}_cols"] = npz[f"{cid}_cols"].astype("int16")
        days, nbr = analysis.case_nbr(s2, npz[f"{cid}_rows"], npz[f"{cid}_cols"], shape)
        r["nbr_dates"] = list(_iso(np.asarray(days)))
        r["nbr"] = [round(float(v), 4) for v in nbr]
    return outlines, rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ds = data.load()
    assert np.array_equal(ds.coh80.ref, ds.coh20.ref)
    assert np.array_equal(ds.coh80.sec, ds.coh20.sec)
    s2 = data.load_s2_stack()
    rgb_days, rgb = rgb_scenes(s2)
    outlines, case_rows = cases(s2, ds.grid.shape)
    sf = ds.masks["stable_forest"]
    forest_nbr = [
        (config.to_date(d).isoformat(), round(float(np.nanmedian(s2.nbr[t][sf])), 4))
        for t, d in enumerate(s2.days)
        if s2.usable(t)
    ]
    manifest = json.loads(config.MANIFEST.read_text())

    meta = {
        "name": "caqueta_demo_v1",
        "description": "NISAR PROVISIONAL subsets over Caquetá, Colombia (frame 083 D 088)",
        "aoi_wgs84": list(config.AOI),
        "crs": ds.grid.crs.to_string(),
        "transform": list(ds.grid.transform)[:6],
        "shape": list(ds.grid.shape),
        "pixel_m": config.PIXEL_M,
        "looks": config.LOOKS,
        "encoding": {
            "hv": "dB x 100 (int16), -32768 = no data",
            "coh20, coh80": "coherence x 250 (uint8), 0 = no data",
            "s2_rgb_<i>": "JPEG bytes of the i-th date in s2_rgb_dates",
            "radd_alert_date": "days since 2025-01-01 (int16), -9999 = no alert",
        },
        "sources": {
            "NISAR": "NASA/JPL/ISRO NISAR L2 GCOV and GUNW, PROVISIONAL (open data), "
            "streamed with nice_sar.io.subset.subset_product",
            "RADD": "Wageningen University RADD alerts (CC BY 4.0), "
            "projects/radar-wur/raddalert/v1 via Google Earth Engine",
            "Sentinel-2": "Contains modified Copernicus Sentinel data (2025-2026), "
            "COPERNICUS/S2_SR_HARMONIZED with Cloud Score+",
        },
        "granules": sorted({f["granule_id"] for f in manifest["files"]}),
        "forest_nbr": forest_nbr,
        "cases": case_rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        meta=np.array(json.dumps(meta)),
        hv=_db_int16(ds.hv.values),
        hv_dates=_iso(ds.hv.days),
        coh80=_coh_uint8(ds.coh80.values),
        coh20=_coh_uint8(ds.coh20.values),
        pair_ref=_iso(ds.coh80.ref),
        pair_sec=_iso(ds.coh80.sec),
        radd_alert_date=ds.radd.alert_date.astype("int16"),
        radd_conf=ds.radd.conf.astype("uint8"),
        radd_forest=ds.radd.forest.astype("uint8"),
        s2_rgb_dates=_iso(rgb_days),
        **{f"s2_rgb_{i}": b for i, b in enumerate(rgb)},
        **outlines,
    )
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    logger.info("Wrote %s (%.1f MB)", OUT, OUT.stat().st_size / 1e6)
    logger.info("SHA-256 %s", digest)


if __name__ == "__main__":
    main()
