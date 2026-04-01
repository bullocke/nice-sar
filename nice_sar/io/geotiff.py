"""GeoTIFF export and inspection utilities.

Write numpy arrays and RGB composites to GeoTIFF format with metadata sidecars,
and read existing GeoTIFFs back into numpy arrays.
Public functions:

- :func:`export_geotiff` — Write a single-band float GeoTIFF with optional JSON sidecar
- :func:`write_rgb_geotiff_uint8` — Write a 3-band uint8 RGB GeoTIFF with per-band stretch
- :func:`read_geotiff` — Read a GeoTIFF into a numpy array with CRS/transform metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import Affine

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)


def export_geotiff(
    data: np.ndarray,
    output_path: str | Path,
    transform: Affine,
    crs: CRS,
    nodata_value: float = np.nan,
    description: str = "",
) -> None:
    """Export a numpy array to a single-band GeoTIFF with optional JSON sidecar.

    Args:
        data: 2D array to write.
        output_path: Destination file path.
        transform: Affine transform for georeferencing.
        crs: Coordinate reference system.
        nodata_value: Value to use for missing data.
        description: Band description metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = data.shape
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "IF_SAFER",
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
        if description:
            dst.set_band_description(1, description)
            dst.update_tags(1, description=description)

    logger.info("Exported: %s", output_path.name)
    _write_sidecar(output_path, {"description": description}, band_count=1)


def write_rgb_geotiff_uint8(
    rgb_stack: np.ndarray,
    output_path: str | Path,
    transform: Affine,
    crs: CRS,
    band_descriptions: list[str] | None = None,
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Write a 3-band uint8 GeoTIFF from a float32 RGB stack with percentile stretch.

    Applies 2-98% percentile stretch per band and records stretch parameters in tags.

    Args:
        rgb_stack: Array of shape ``(3, H, W)`` in float32.
        output_path: Destination file path.
        transform: Affine transform.
        crs: Coordinate reference system.
        band_descriptions: Optional list of 3 band descriptions.
        extra_tags: Additional tags to write.
    """
    if rgb_stack.ndim != 3 or rgb_stack.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) array, got shape {rgb_stack.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = rgb_stack.shape[1:]

    # If input is already uint8 (e.g. from make_rgb), skip stretching
    if rgb_stack.dtype == np.uint8:
        stretched_arr = rgb_stack
        stretches = [(0.0, 255.0)] * 3
    else:
        stretched = []
        stretches = []
        for b in range(3):
            u8, lo, hi = _percentile_stretch_uint8(rgb_stack[b])
            stretched.append(u8)
            stretches.append((lo, hi))
        stretched_arr = np.stack(stretched, axis=0)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 3,
        "dtype": np.uint8,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "photometric": "RGB",
        "bigtiff": "IF_SAFER",
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stretched_arr)
        if band_descriptions:
            for i, d in enumerate(band_descriptions, start=1):
                dst.set_band_description(i, d)
        tag_dict = {
            f"band{i}_stretch_min": str(stretches[i - 1][0]) for i in (1, 2, 3)
        }
        tag_dict.update(
            {f"band{i}_stretch_max": str(stretches[i - 1][1]) for i in (1, 2, 3)}
        )
        if extra_tags:
            tag_dict.update(extra_tags)
        dst.update_tags(**tag_dict)

    logger.info("Exported: %s", output_path.name)

    _write_sidecar(
        output_path,
        {
            "band_stretches": {
                "R": {"p2": stretches[0][0], "p98": stretches[0][1]},
                "G": {"p2": stretches[1][0], "p98": stretches[1][1]},
                "B": {"p2": stretches[2][0], "p98": stretches[2][1]},
            }
        },
        band_count=3,
    )


def read_band(filepath: str | Path) -> tuple[ArrayFloat32, dict]:
    """Read a single-band GeoTIFF and return data + profile.

    Args:
        filepath: Path to the GeoTIFF.

    Returns:
        Tuple of (data array as float32, rasterio profile dict).
    """
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile_stretch_uint8(
    arr: np.ndarray, p_low: float = 2, p_high: float = 98
) -> tuple[np.ndarray, float, float]:
    """Stretch a float array to uint8 via percentiles."""
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.uint8), np.nan, np.nan
    lo, hi = np.percentile(arr[valid], [p_low, p_high])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr[valid])), float(np.nanmax(arr[valid]))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8), lo, hi
    scaled = (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-12)
    u8 = (scaled * 255.0).astype(np.uint8)
    return u8, float(lo), float(hi)


def _write_sidecar(
    raster_path: Path, extra_info: dict | None = None, band_count: int = 1
) -> None:
    """Write a JSON sidecar for a raster file."""
    info: dict = {
        "output": str(raster_path),
        "band_count": band_count,
    }
    base = raster_path.name.lower()
    if "sigma0" in base:
        info["calibration"] = "sigma0"
    elif any(k in base for k in ("gamma", "db", "linear")):
        info["calibration"] = "gamma0"
    else:
        info["calibration"] = "unknown"

    if extra_info:
        info.update(extra_info)

    sidecar = raster_path.with_suffix(raster_path.suffix + ".json")
    with open(sidecar, "w") as f:
        json.dump(info, f, indent=2)
