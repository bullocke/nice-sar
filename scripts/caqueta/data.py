"""Load the Caquetá NISAR subsets, RADD reference, and Sentinel-2 chips as arrays.

Everything is placed on one 20 m grid (EPSG:32618), the grid of the GCOV and
20 m GUNW subsets. The 80 m coherence is reprojected onto it with nearest
neighbour, so each 80 m cell becomes a 4 x 4 block of identical 20 m pixels.
Sentinel-2 is stored at 10 m on the same origin (exactly 2 x 2 S2 pixels per
NISAR pixel), so NISAR pixel indices convert to S2 indices by multiplying by 2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import config
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class Grid:
    """The 20 m analysis grid."""

    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    shape: tuple[int, int]


@dataclass
class DateStack:
    """Backscatter stack: ``values[t]`` is the image acquired on day ``days[t]``."""

    values: np.ndarray  # (T, H, W), dB
    days: np.ndarray  # (T,) days since EPOCH


@dataclass
class PairStack:
    """Coherence stack: ``values[p]`` spans day ``ref[p]`` to day ``sec[p]``."""

    values: np.ndarray  # (P, H, W), coherence magnitude 0-1
    ref: np.ndarray  # (P,)
    sec: np.ndarray  # (P,)

    @property
    def span(self) -> np.ndarray:
        """Temporal baseline of each pair in days."""
        return self.sec - self.ref


@dataclass
class Radd:
    """RADD alert layers on the analysis grid (see fetch_radd.py)."""

    alert_date: np.ndarray  # days since EPOCH, -9999 = no alert
    conf: np.ndarray  # 0 none, 2 low, 3 high
    forest: np.ndarray  # 1 = RADD primary humid forest baseline


@dataclass
class Dataset:
    """All inputs for the analysis on one grid."""

    grid: Grid
    hh: DateStack
    hv: DateStack
    coh20: PairStack
    coh80: PairStack
    radd: Radd
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def first_day(self) -> int:
        return int(min(self.hh.days.min(), self.coh20.ref.min()))

    @property
    def last_day(self) -> int:
        return int(max(self.hh.days.max(), self.coh20.sec.max()))


def read_on_grid(path: Path, grid: Grid, resampling=Resampling.nearest) -> np.ndarray:
    """Read band 1 of ``path`` onto ``grid`` as float32 with invalid values as NaN.

    Non-positive values are treated as invalid: backscatter power and coherence
    magnitude are both strictly positive where the product has data.
    """
    with rasterio.open(path) as src:
        if src.transform == grid.transform and src.shape == grid.shape:
            data = src.read(1).astype("float32")
            nodata = src.nodata
        else:
            with WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.shape[1],
                height=grid.shape[0],
                resampling=resampling,
            ) as vrt:
                data = vrt.read(1).astype("float32")
                nodata = vrt.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data) | (data <= 0)] = np.nan
    return data


def _manifest() -> list[dict]:
    return json.loads(config.MANIFEST.read_text())["files"]


def load(min_valid: float = 0.5) -> Dataset:
    """Load all NISAR layers, RADD, and pixel-class masks.

    Args:
        min_valid: Drop layers with less than this fraction of valid pixels
            (e.g. the partial-frame 2025-11-15 GCOV has no data over the AOI).
    """
    with rasterio.open(config.RADD_TIF) as src:
        alert_date, conf, forest = src.read()
        grid = Grid(src.crs, src.transform, src.shape)
    radd = Radd(alert_date.astype(float), conf, forest)

    backscatter: dict[str, list] = {"HH": [], "HV": []}
    coherence: dict[str, list] = {"coh20": [], "coh80": []}
    for entry in _manifest():
        path = config.DATA_DIR / entry["file"]
        ref = config.to_day(entry["start"])
        data = read_on_grid(path, grid)
        if np.isfinite(data).mean() < min_valid:
            logger.info("Skipping %s (%.0f%% valid)", path.name, 100 * np.isfinite(data).mean())
            continue
        if entry["product"] == "GCOV":
            pol = path.stem.split("_")[-1]
            backscatter[pol].append((ref, 10 * np.log10(data)))
        else:
            sec = config.to_day(entry["secondary_start"])
            kind = "coh20" if path.stem.endswith("coh20m") else "coh80"
            coherence[kind].append((ref, sec, data))

    def date_stack(items: list) -> DateStack:
        items = sorted(items, key=lambda x: x[0])
        return DateStack(np.stack([v for _, v in items]), np.array([d for d, _ in items]))

    def pair_stack(items: list) -> PairStack:
        items = sorted(items, key=lambda x: x[0])
        return PairStack(
            np.stack([v for *_, v in items]),
            np.array([r for r, *_ in items]),
            np.array([s for _, s, _ in items]),
        )

    ds = Dataset(
        grid=grid,
        hh=date_stack(backscatter["HH"]),
        hv=date_stack(backscatter["HV"]),
        coh20=pair_stack(coherence["coh20"]),
        coh80=pair_stack(coherence["coh80"]),
        radd=radd,
    )
    ds.masks = pixel_classes(ds)
    logger.info(
        "Loaded HH %d dates, HV %d dates, %d coherence pairs",
        len(ds.hh.days),
        len(ds.hv.days),
        len(ds.coh20.ref),
    )
    return ds


def pixel_classes(ds: Dataset) -> dict[str, np.ndarray]:
    """Reference pixel classes from RADD.

    - ``disturbed``: high-confidence alert inside the NISAR series, within RADD's
      primary-forest baseline.
    - ``stable_forest``: forest baseline, never alerted, and at least
      STABLE_BUFFER_M from any alert (avoids edges and unmapped small clearings).
    - ``pre_series_pasture``: alerted at least PRE_SERIES_MARGIN_D days before the
      first NISAR date, i.e. already cleared (pasture or regrowth) during the series.
    """
    r = ds.radd
    alerted = r.conf > 0
    dist_m = ndimage.distance_transform_edt(~alerted) * config.PIXEL_M
    in_window = (r.alert_date >= ds.first_day) & (r.alert_date <= ds.last_day)
    masks = {
        "disturbed": (r.conf == 3) & (r.forest == 1) & in_window,
        "stable_forest": (r.forest == 1) & ~alerted & (dist_m >= config.STABLE_BUFFER_M),
        "pre_series_pasture": alerted & (r.alert_date < ds.first_day - config.PRE_SERIES_MARGIN_D),
    }
    for name, mask in masks.items():
        logger.info("%s: %d px (%.1f%%)", name, mask.sum(), 100 * mask.mean())
    return masks


# --- Sentinel-2 -------------------------------------------------------------------


@dataclass
class S2Scene:
    """One Sentinel-2 date on the 10 m grid (2x the NISAR grid)."""

    day: int
    path: Path


def s2_scenes() -> list[S2Scene]:
    """List Sentinel-2 GeoTIFFs written by fetch_sentinel2.py, sorted by date."""
    scenes = [
        S2Scene(config.to_day(p.stem.split("_")[1]), p)
        for p in sorted(config.S2_DIR.glob("S2_*.tif"))
    ]
    return sorted(scenes, key=lambda s: s.day)


def s2_rgb(
    scene: S2Scene,
    window: tuple[slice, slice] | None = None,
    bands: tuple[str, str, str] = ("B4", "B3", "B2"),
    stretch: tuple[float, float] = (0.0, 0.15),
) -> np.ndarray:
    """Return an RGB array (H, W, 3) in 0-1 for a NISAR-grid window.

    Args:
        scene: Sentinel-2 scene.
        window: (row slice, col slice) on the 20 m NISAR grid; converted to the
            10 m S2 grid by doubling. ``None`` reads the full AOI.
        bands: Band names for R, G, B. Use ("B12", "B8", "B4") for false colour,
            where bare soil and burns appear magenta/brown and vegetation green;
            widen ``stretch`` to about (0, 0.35) for that combination.
        stretch: Surface-reflectance range mapped to 0-1.
    """
    with rasterio.open(scene.path) as src:
        names = list(src.descriptions)
        win = None
        if window is not None:
            rows, cols = window
            win = rasterio.windows.Window(
                cols.start * 2,
                rows.start * 2,
                (cols.stop - cols.start) * 2,
                (rows.stop - rows.start) * 2,
            )
        rgb = np.stack([src.read(names.index(b) + 1, window=win) for b in bands], -1)
        clear = src.read(names.index("clear") + 1, window=win)
    rgb = (rgb.astype("float32") / 10000 - stretch[0]) / (stretch[1] - stretch[0])
    rgb = np.clip(rgb, 0, 1)
    rgb[clear == 0] = 0.85  # cloud / shadow shown as light gray
    return rgb


def s2_clear_fraction(scene: S2Scene, window: tuple[slice, slice] | None = None) -> float:
    """Fraction of clear S2 pixels in a NISAR-grid window."""
    with rasterio.open(scene.path) as src:
        names = list(src.descriptions)
        win = None
        if window is not None:
            rows, cols = window
            win = rasterio.windows.Window(
                cols.start * 2,
                rows.start * 2,
                (cols.stop - cols.start) * 2,
                (rows.stop - rows.start) * 2,
            )
        return float((src.read(names.index("clear") + 1, window=win) > 0).mean())


def s2_usable(scene: S2Scene, window: tuple[slice, slice] | None = None) -> bool:
    """Whether a scene is clear and haze-free enough to show in a window.

    Requires at least ``S2_MIN_CLEAR`` clear pixels and a median blue (B2)
    reflectance of clear pixels at or below ``S2_MAX_BLUE``; haze and smoke
    brighten the blue band far more than they affect the cloud score.
    """
    with rasterio.open(scene.path) as src:
        names = list(src.descriptions)
        win = None
        if window is not None:
            rows, cols = window
            win = rasterio.windows.Window(
                cols.start * 2,
                rows.start * 2,
                (cols.stop - cols.start) * 2,
                (rows.stop - rows.start) * 2,
            )
        clear = src.read(names.index("clear") + 1, window=win) > 0
        if clear.mean() < config.S2_MIN_CLEAR:
            return False
        blue = src.read(names.index("B2") + 1, window=win)
    return float(np.median(blue[clear])) <= config.S2_MAX_BLUE


@dataclass
class S2Stack:
    """All Sentinel-2 dates aggregated to the 20 m NISAR grid, held in memory.

    ``nbr`` is the normalized burn ratio (B8 - B12) / (B8 + B12): about 0.6 for
    intact forest here and below 0.3 for felled, burned, or bare ground, which
    makes it a sharper clearing indicator than NDVI. Values are NaN where fewer
    than half of the four 10 m pixels are clear.
    """

    days: np.ndarray  # (T,)
    nbr: np.ndarray  # (T, H, W) float32
    clear: np.ndarray  # (T, H, W) bool
    blue: np.ndarray  # (T, H, W) float32, B2 reflectance x 10000

    def usable(self, t: int, window: tuple[slice, slice] | None = None) -> bool:
        """Clear and haze-free enough (see ``s2_usable``), evaluated at 20 m."""
        w = window or (slice(None), slice(None))
        clear = self.clear[t][w]
        if clear.mean() < config.S2_MIN_CLEAR:
            return False
        return float(np.median(self.blue[t][w][clear])) <= config.S2_MAX_BLUE


def _block_mean(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    return a[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def load_s2_stack() -> S2Stack:
    """Read every Sentinel-2 file once and aggregate to the 20 m grid."""
    days, nbr, clear, blue = [], [], [], []
    for scene in s2_scenes():
        with rasterio.open(scene.path) as src:
            names = list(src.descriptions)
            b = {n: src.read(names.index(n) + 1).astype("float32") for n in ("B2", "B8", "B12")}
            c = src.read(names.index("clear") + 1) > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (b["B8"] - b["B12"]) / (b["B8"] + b["B12"])
        v[~c] = np.nan
        frac = _block_mean(c.astype("float32"))
        with np.errstate(invalid="ignore"):
            n20 = np.nanmean(
                v[: v.shape[0] // 2 * 2, : v.shape[1] // 2 * 2].reshape(
                    v.shape[0] // 2, 2, v.shape[1] // 2, 2
                ),
                axis=(1, 3),
            )
        n20[frac < 0.5] = np.nan
        days.append(scene.day)
        nbr.append(n20.astype("float32"))
        clear.append(frac >= 0.5)
        blue.append(_block_mean(b["B2"]))
    logger.info("Loaded %d Sentinel-2 dates onto the 20 m grid", len(days))
    return S2Stack(np.array(days), np.stack(nbr), np.stack(clear), np.stack(blue))
