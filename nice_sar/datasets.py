"""Small demo datasets for the tutorials.

:func:`load_caqueta_demo` downloads (once) and loads a compact bundle of NISAR
PROVISIONAL subsets over a deforestation frontier in Caquetá, Colombia, with RADD
alerts, Sentinel-2 imagery and clearing outlines on the same 20 m grid. It lets
the tutorials run without NASA Earthdata or Earth Engine credentials. The bundle
is built by ``scripts/caqueta/build_bundle.py`` and published as a GitHub release
asset.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CAQUETA_AOI",
    "CAQUETA_FRAME",
    "CaquetaCase",
    "CaquetaDemo",
    "fetch_caqueta_demo",
    "load_caqueta_demo",
]

#: Study area in Caquetá, Colombia: west, south, east, north (WGS84).
CAQUETA_AOI = (-74.36, 0.76, -74.16, 0.92)

#: NISAR track, direction and frame covering the study area. Frame 083 D 088 is a
#: supplemental PROVISIONAL frame, processed back to November 2025.
CAQUETA_FRAME = {"track": 83, "direction": "D", "frame": 88}

_BUNDLE = "caqueta_demo_v1.npz"
_URL = f"https://github.com/bullocke/nice-sar/releases/download/data-caqueta-v1/{_BUNDLE}"
_SHA256 = "3e1ae83234f332450f05d3cf3c990b8f2932bd9dcff88b3cbe7c009067f549b9"


def _cache_dir() -> Path:
    root = os.environ.get("NICE_SAR_DATA")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "nice-sar"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_caqueta_demo(dest: str | Path | None = None, url: str = _URL) -> Path:
    """Download the Caquetá demo bundle if it is not already cached.

    Args:
        dest: Directory to store the bundle in. Defaults to ``$NICE_SAR_DATA``
            or ``~/.cache/nice-sar``.
        url: Download URL (override to use a mirror or a local build).

    Returns:
        Path to the verified ``.npz`` bundle.

    Raises:
        OSError: If the download fails or the checksum does not match.
    """
    folder = Path(dest) if dest is not None else _cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _BUNDLE
    if path.exists() and _sha256(path) == _SHA256:
        return path
    tmp = path.with_suffix(".part")
    logger.info("Downloading %s", url)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(url, tmp)
    digest = _sha256(tmp)
    if digest != _SHA256:
        tmp.unlink(missing_ok=True)
        raise OSError(f"Checksum mismatch for {url}: {digest}")
    tmp.replace(path)
    logger.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


@dataclass
class CaquetaCase:
    """One case study: a clearing outline or a control area.

    Attributes:
        case_id: Identifier, e.g. ``"forest_clearing_1"``.
        category: Case category, e.g. ``"forest_clearing"``, ``"stable_forest"``.
        rows: Pixel rows of the outline on the 20 m grid.
        cols: Pixel columns of the outline.
        attrs: Attributes from the case selection (dates, dips, HV step, ...).
        nbr_dates: Dates with a usable Sentinel-2 view of the case.
        nbr: Case-mean Sentinel-2 normalized burn ratio on those dates.
    """

    case_id: str
    category: str
    rows: np.ndarray
    cols: np.ndarray
    attrs: dict[str, Any] = field(default_factory=dict)
    nbr_dates: np.ndarray = field(default_factory=lambda: np.array([], "datetime64[D]"))
    nbr: np.ndarray = field(default_factory=lambda: np.array([]))

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean mask of the case on a grid of ``shape``."""
        out = np.zeros(shape, bool)
        out[self.rows, self.cols] = True
        return out

    @property
    def area_ha(self) -> float:
        """Area in hectares (20 m pixels)."""
        return self.rows.size * 400 / 1e4

    def window(self, pad: int, shape: tuple[int, int]) -> tuple[slice, slice]:
        """Square window around the case, padded by ``pad`` pixels."""
        r0, r1 = self.rows.min(), self.rows.max()
        c0, c1 = self.cols.min(), self.cols.max()
        half = max(r1 - r0, c1 - c0) // 2 + pad
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        rs = slice(max(rc - half, 0), min(rc + half + 1, shape[0]))
        cs = slice(max(cc - half, 0), min(cc + half + 1, shape[1]))
        return rs, cs


@dataclass
class CaquetaDemo:
    """The Caquetá demo bundle, decoded to physical units.

    Stacks are ``(T, H, W)`` on a 20 m grid; dates are ``numpy.datetime64[D]``.

    Attributes:
        hv: HV backscatter, dB.
        hv_dates: Acquisition dates of ``hv``.
        coh80: HH coherence from the 80 m GUNW layer, shown on the 20 m grid.
        coh20: HH coherence from the 20 m GUNW layer.
        pair_ref: First date of each coherence pair.
        pair_sec: Second date of each coherence pair.
        radd_alert: RADD alert date per pixel (NaT where none).
        radd_conf: RADD confidence (0 none, 2 low, 3 high).
        radd_forest: RADD primary humid forest baseline (bool).
        s2_rgb: Sentinel-2 true colour, ``(N, H, W, 3)`` uint8, clear dates only.
        s2_rgb_dates: Dates of ``s2_rgb``.
        forest_nbr_dates: Dates of ``forest_nbr``.
        forest_nbr: Median Sentinel-2 NBR of stable forest on clear dates.
        cases: Case studies keyed by ID.
        meta: Grid, sources, granule IDs and encoding notes.
    """

    hv: np.ndarray
    hv_dates: np.ndarray
    coh80: np.ndarray
    coh20: np.ndarray
    pair_ref: np.ndarray
    pair_sec: np.ndarray
    radd_alert: np.ndarray
    radd_conf: np.ndarray
    radd_forest: np.ndarray
    s2_rgb: np.ndarray
    s2_rgb_dates: np.ndarray
    forest_nbr_dates: np.ndarray
    forest_nbr: np.ndarray
    cases: dict[str, CaquetaCase]
    meta: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (rows, columns)."""
        return (int(self.hv.shape[1]), int(self.hv.shape[2]))

    @property
    def pixel_m(self) -> float:
        """Pixel size in metres."""
        return float(self.meta["pixel_m"])

    @property
    def transform(self) -> Any:
        """Affine transform of the grid."""
        from affine import Affine

        return Affine(*self.meta["transform"])

    @property
    def crs(self) -> str:
        """Coordinate reference system of the grid."""
        return str(self.meta["crs"])

    @property
    def pair_mid(self) -> np.ndarray:
        """Midpoint date of each coherence pair."""
        return np.asarray(self.pair_ref + (self.pair_sec - self.pair_ref) // 2)

    def stable_forest(self, buffer_m: float = 100.0) -> np.ndarray:
        """RADD forest never alerted and at least ``buffer_m`` from any alert."""
        from scipy import ndimage

        alerted = self.radd_conf > 0
        dist = ndimage.distance_transform_edt(~alerted) * self.pixel_m
        return np.asarray(self.radd_forest & ~alerted & (dist >= buffer_m))

    def cases_in(self, category: str) -> list[CaquetaCase]:
        """Cases of one category, in selection order."""
        return [c for c in self.cases.values() if c.category == category]


def _dates(a: Any) -> np.ndarray:
    return np.asarray(a, dtype="datetime64[D]")


def load_caqueta_demo(path: str | Path | None = None) -> CaquetaDemo:
    """Load the Caquetá demo bundle, downloading it first if needed.

    Args:
        path: A local bundle file. Defaults to :func:`fetch_caqueta_demo`.

    Returns:
        :class:`CaquetaDemo`.
    """
    path = Path(path) if path is not None else fetch_caqueta_demo()
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))

        def db(key: str) -> np.ndarray:
            v: np.ndarray = z[key].astype("float32")
            v[z[key] == -32768] = np.nan
            return v / 100

        def coh(key: str) -> np.ndarray:
            v: np.ndarray = z[key].astype("float32") / 250
            v[z[key] == 0] = np.nan
            return v

        from PIL import Image

        rgb_dates = _dates(z["s2_rgb_dates"])
        rgb = np.stack(
            [
                np.asarray(Image.open(io.BytesIO(z[f"s2_rgb_{i}"].tobytes())))
                for i in range(len(rgb_dates))
            ]
        )
        alert = z["radd_alert_date"].astype("int64")
        radd_alert = np.where(
            alert > -9999,
            np.datetime64("2025-01-01") + alert.astype("timedelta64[D]"),
            np.datetime64("NaT"),
        ).astype("datetime64[D]")

        cases = {}
        for row in meta["cases"]:
            cid = row["case_id"]
            cases[cid] = CaquetaCase(
                case_id=cid,
                category=row["category"],
                rows=z[f"case_{cid}_rows"].astype(int),
                cols=z[f"case_{cid}_cols"].astype(int),
                attrs=row,
                nbr_dates=_dates(row.get("nbr_dates", [])),
                nbr=np.asarray(row.get("nbr", []), dtype=float),
            )

        demo = CaquetaDemo(
            hv=db("hv"),
            hv_dates=_dates(z["hv_dates"]),
            coh80=coh("coh80"),
            coh20=coh("coh20"),
            pair_ref=_dates(z["pair_ref"]),
            pair_sec=_dates(z["pair_sec"]),
            radd_alert=radd_alert,
            radd_conf=z["radd_conf"].copy(),
            radd_forest=z["radd_forest"] == 1,
            s2_rgb=rgb,
            s2_rgb_dates=rgb_dates,
            forest_nbr_dates=_dates([d for d, _ in meta["forest_nbr"]]),
            forest_nbr=np.array([v for _, v in meta["forest_nbr"]], dtype=float),
            cases=cases,
            meta=meta,
        )
    logger.info(
        "Caquetá demo: %d HV dates, %d coherence pairs, %d cases",
        len(demo.hv_dates),
        len(demo.pair_ref),
        len(demo.cases),
    )
    return demo
