"""Tests for nice_sar.datasets (synthetic bundle, no network)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nice_sar import datasets


def _write_bundle(path: Path) -> Path:
    h, w = 20, 30
    rgb = np.zeros((h, w, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG")
    case = {"case_id": "forest_clearing_1", "category": "forest_clearing"}
    case |= {"nbr_dates": ["2025-12-05"], "nbr": [0.6]}
    meta = {
        "crs": "EPSG:32618",
        "transform": [20.0, 0.0, 500000.0, 0.0, -20.0, 100000.0],
        "shape": [h, w],
        "pixel_m": 20.0,
        "forest_nbr": [["2025-12-05", 0.62]],
        "cases": [case],
    }
    hv = np.full((2, h, w), -1000, np.int16)
    hv[1, 0, 0] = -32768
    coh = np.full((3, h, w), 100, np.uint8)
    coh[0, 0, 0] = 0
    alert = np.full((h, w), -9999, np.int16)
    alert[5, 5] = 365
    conf = np.zeros((h, w), np.uint8)
    conf[5, 5] = 3
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        hv=hv,
        hv_dates=np.array(["2025-12-09", "2026-01-02"]),
        coh80=coh,
        coh20=coh,
        pair_ref=np.array(["2025-11-03", "2025-11-27", "2025-12-09"]),
        pair_sec=np.array(["2025-11-27", "2025-12-09", "2025-12-21"]),
        radd_alert_date=alert,
        radd_conf=conf,
        radd_forest=np.ones((h, w), np.uint8),
        s2_rgb_dates=np.array(["2025-12-05"]),
        s2_rgb_0=np.frombuffer(buf.getvalue(), np.uint8),
        case_forest_clearing_1_rows=np.array([1, 2], np.int16),
        case_forest_clearing_1_cols=np.array([3, 4], np.int16),
    )
    return path


def test_load_decodes_units(tmp_path: Path) -> None:
    demo = datasets.load_caqueta_demo(_write_bundle(tmp_path / "b.npz"))
    assert demo.shape == (20, 30)
    assert demo.hv[0, 0, 0] == pytest.approx(-10.0)
    assert np.isnan(demo.hv[1, 0, 0])
    assert demo.coh80[1, 0, 0] == pytest.approx(0.4)
    assert np.isnan(demo.coh80[0, 0, 0])
    assert demo.radd_alert[5, 5] == np.datetime64("2026-01-01")
    assert np.isnat(demo.radd_alert[0, 0])
    assert demo.s2_rgb.shape == (1, 20, 30, 3)
    assert demo.pair_mid[0] == np.datetime64("2025-11-15")
    assert not demo.stable_forest()[5, 5]
    case = demo.cases["forest_clearing_1"]
    assert case.mask(demo.shape).sum() == 2
    assert case.area_ha == pytest.approx(0.08)
    assert demo.cases_in("forest_clearing") == [case]
    assert demo.transform.a == 20.0


def test_fetch_verifies_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _write_bundle(tmp_path / "src.npz")
    monkeypatch.setattr(datasets, "_SHA256", datasets._sha256(src))
    out = datasets.fetch_caqueta_demo(tmp_path / "cache", url=str(src))
    assert out.exists()
    assert datasets.fetch_caqueta_demo(tmp_path / "cache", url="unused") == out

    monkeypatch.setattr(datasets, "_SHA256", "0" * 64)
    with pytest.raises(OSError):
        datasets.fetch_caqueta_demo(tmp_path / "other", url=str(src))
