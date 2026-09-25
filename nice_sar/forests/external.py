"""External forest mask raster ingestion and alignment utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from nice_sar._types import PathType
from nice_sar.io.geotiff import read_band
from nice_sar.io.products import read_gcov

logger = logging.getLogger(__name__)


def _normalize_raster_source(source: PathType) -> str:
    path_str = str(source)
    if path_str.startswith("/vsicurl/") or path_str.startswith("/vsis3/"):
        return path_str
    if path_str.startswith(("http://", "https://")):
        return f"/vsicurl/{path_str}"
    if path_str.startswith("s3://"):
        return f"/vsis3/{path_str.removeprefix('s3://')}"
    return path_str


def _make_data_array(
    data: np.ndarray,
    *,
    attrs: dict[str, Any],
    template: xr.DataArray | None = None,
    name: str = "forest_mask_source",
) -> xr.DataArray:
    if template is not None:
        return xr.DataArray(
            data,
            dims=template.dims,
            coords=template.coords,
            attrs={**template.attrs, **attrs},
            name=name,
        )

    return xr.DataArray(data, dims=["y", "x"], attrs=attrs, name=name)


def _target_grid_from_reference(
    target: PathType | xr.DataArray,
    *,
    frequency: str = "A",
    polarization: str = "HV",
) -> tuple[CRS, Affine, tuple[int, int], xr.DataArray | None]:
    if isinstance(target, xr.DataArray):
        if "crs" not in target.attrs or "transform" not in target.attrs:
            raise ValueError("Target DataArray must define 'crs' and 'transform' attrs.")
        crs = CRS.from_user_input(target.attrs["crs"])
        transform = Affine(*target.attrs["transform"])
        return crs, transform, target.shape, target

    target_path = str(target)
    if target_path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(_normalize_raster_source(target)) as src:
            crs = CRS.from_user_input(src.crs)
            transform = src.transform
            shape = (src.height, src.width)
        return crs, transform, shape, None

    target_da = read_gcov(target, frequency=frequency, polarization=polarization)
    crs = CRS.from_user_input(target_da.attrs["crs"])
    transform = Affine(*target_da.attrs["transform"])
    return crs, transform, target_da.shape, target_da


def read_external_raster(
    source: PathType,
    *,
    band: int = 1,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Read an external raster source into an xarray DataArray.

    Args:
        source: Local path or remote raster URL.
        band: Band number to read.

    Returns:
        Tuple containing the DataArray and a metadata dictionary.
    """
    path_str = _normalize_raster_source(source)
    with rasterio.open(path_str) as src:
        data = src.read(band).astype(np.float32)
        attrs = {
            "crs": str(src.crs) if src.crs is not None else None,
            "transform": tuple(src.transform),
            "nodata": src.nodata,
            "source": str(source),
            "band": band,
        }
        meta = {
            "source": str(source),
            "band": band,
            "shape": data.shape,
            "crs": attrs["crs"],
            "transform": attrs["transform"],
            "nodata": src.nodata,
        }
    return _make_data_array(data, attrs=attrs), meta


def align_external_raster(
    source: PathType,
    *,
    target: PathType | xr.DataArray,
    band: int = 1,
    frequency: str = "A",
    polarization: str = "HV",
    resampling: str = "nearest",
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Align an external raster to a target NISAR or raster grid.

    Args:
        source: Source raster path or URL.
        target: Target raster path, NISAR GCOV source path, or target DataArray.
        band: Source raster band.
        frequency: Frequency to use when *target* is a NISAR path.
        polarization: Polarization to use when *target* is a NISAR path.
        resampling: Rasterio resampling mode.

    Returns:
        Tuple containing the aligned DataArray and alignment metadata.
    """
    dst_crs, dst_transform, dst_shape, template = _target_grid_from_reference(
        target,
        frequency=frequency,
        polarization=polarization,
    )
    dst_height, dst_width = dst_shape
    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    resampling_enum = getattr(Resampling, resampling)

    with rasterio.open(_normalize_raster_source(source)) as src:
        src_data = src.read(band).astype(np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling_enum,
        )
        meta = {
            "source": str(source),
            "band": band,
            "resampling": resampling,
            "source_crs": str(src.crs) if src.crs is not None else None,
            "target_crs": str(dst_crs),
            "target_shape": dst_shape,
        }

    attrs = {
        "crs": str(dst_crs),
        "transform": tuple(dst_transform),
        "source": str(source),
        "band": band,
        "aligned_to_target": True,
    }
    return _make_data_array(dst, attrs=attrs, template=template), meta


def raster_to_mask(
    data: np.ndarray | xr.DataArray,
    *,
    threshold: float | None = None,
    mask_values: tuple[float, ...] | None = None,
    invert: bool = False,
) -> np.ndarray | xr.DataArray:
    """Convert a raster layer into a boolean forest mask.

    Args:
        data: Input raster values.
        threshold: If provided, pixels greater than or equal to this value are forest.
        mask_values: If provided, exact values that should be treated as forest.
        invert: If True, invert the final mask.

    Returns:
        Boolean mask with the same shape as *data*.
    """
    values = np.asarray(data)
    valid = np.isfinite(values)
    if mask_values is not None:
        mask = valid & np.isin(values, np.asarray(mask_values))
    elif threshold is not None:
        mask = valid & (values >= threshold)
    else:
        mask = valid & (values != 0)

    if invert:
        mask = valid & ~mask

    if not isinstance(data, xr.DataArray):
        return mask

    attrs = dict(data.attrs)
    attrs.update({"mask_method": "external_raster"})
    return xr.DataArray(mask, dims=data.dims, coords=data.coords, attrs=attrs, name="forest_mask")


def read_mask_raster(
    source: PathType,
    *,
    band: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a local GeoTIFF mask with the simple GeoTIFF utility path."""
    if str(source).startswith(("http://", "https://", "s3://")):
        da, meta = read_external_raster(source, band=band)
        return np.asarray(da), meta
    data, profile = read_band(Path(source))
    meta = {
        "source": str(source),
        "band": band,
        "shape": data.shape,
        "crs": str(profile.get("crs")) if profile.get("crs") is not None else None,
        "transform": tuple(profile["transform"]) if profile.get("transform") is not None else None,
        "nodata": profile.get("nodata"),
    }
    return data, meta