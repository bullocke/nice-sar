"""High-level forest mask generation API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import xarray as xr
from rasterio.transform import Affine

from nice_sar._types import PathType
from nice_sar.forests.external import align_external_raster, raster_to_mask, read_external_raster
from nice_sar.forests.gcov import (
    gcov_hv_threshold,
    gcov_hv_threshold_jaxa_biomass,
    gcov_hv_threshold_margin,
    gcov_hv_threshold_ramachandran,
)
from nice_sar.forests.registry import (
    get_method_spec,
    list_method_specs,
    list_provider_specs,
)
from nice_sar.io.geotiff import read_band
from nice_sar.io.products import read_gcov

logger = logging.getLogger(__name__)

_GCOV_METHODS = {
    "gcov_hv_threshold": gcov_hv_threshold,
    "gcov_hv_threshold_ramachandran": gcov_hv_threshold_ramachandran,
    "gcov_hv_threshold_jaxa_biomass": gcov_hv_threshold_jaxa_biomass,
}


@dataclass
class ForestMaskResult:
    """Standard output container for forest masking workflows.

    Attributes:
        mask: Forest mask output as a boolean array or DataArray.
        method: Method identifier used to generate the mask.
        source_kind: Broad source category such as ``"gcov"`` or ``"external_raster"``.
        units: Units used internally for the classification threshold.
        threshold: Threshold value used by the method, if any.
        metadata: Free-form metadata and provenance details.
        confidence: Optional confidence or margin layer associated with the mask.
    """

    mask: np.ndarray | xr.DataArray
    method: str
    source_kind: str
    units: str
    threshold: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: np.ndarray | xr.DataArray | None = None


def _wrap_raster_as_dataarray(data: np.ndarray, profile: dict[str, Any]) -> xr.DataArray:
    attrs = {}
    if profile.get("crs") is not None:
        attrs["crs"] = str(profile["crs"])
    if profile.get("transform") is not None:
        attrs["transform"] = tuple(profile["transform"])
    if profile.get("nodata") is not None:
        attrs["nodata"] = profile["nodata"]
    return xr.DataArray(data, dims=["y", "x"], attrs=attrs, name="hv_backscatter")


def _load_gcov_like_source(
    source: PathType | np.ndarray | xr.DataArray,
    *,
    frequency: str = "A",
    polarization: str = "HV",
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> tuple[np.ndarray | xr.DataArray, dict[str, Any], str]:
    if isinstance(source, xr.DataArray):
        return source, dict(source.attrs), "dataarray"

    if isinstance(source, np.ndarray):
        return source, {}, "array"

    source_str = str(source)
    if source_str.lower().endswith((".tif", ".tiff")):
        data, profile = read_band(Path(source_str))
        return _wrap_raster_as_dataarray(data, profile), profile, "raster"

    da = read_gcov(
        source,
        frequency=frequency,
        polarization=polarization,
        filesystem=filesystem,
    )
    return da, dict(da.attrs), "gcov"


def list_forest_mask_methods(include_unimplemented: bool = True) -> list[dict[str, object]]:
    """List forest-mask methods registered in the package."""
    return list_method_specs(include_unimplemented=include_unimplemented)


def list_forest_mask_providers(include_unimplemented: bool = True) -> list[dict[str, object]]:
    """List external forest-mask providers registered in the package."""
    return list_provider_specs(include_unimplemented=include_unimplemented)


def load_external_forest_mask(
    source: PathType,
    *,
    target: PathType | xr.DataArray | None = None,
    band: int = 1,
    frequency: str = "A",
    polarization: str = "HV",
    resampling: str = "nearest",
    threshold: float | None = None,
    mask_values: tuple[float, ...] | None = None,
    invert: bool = False,
) -> ForestMaskResult:
    """Load an external forest mask raster and optionally align it to a target grid.

    Args:
        source: Forest mask raster path or URL.
        target: Optional target raster or GCOV source defining the output grid.
        band: Raster band to use.
        frequency: Frequency used if *target* is a GCOV source path.
        polarization: Polarization used if *target* is a GCOV source path.
        resampling: Rasterio resampling mode when aligning to a target grid.
        threshold: Optional numeric threshold used to convert the raster to a mask.
        mask_values: Optional discrete values treated as forest.
        invert: Whether to invert the derived mask.

    Returns:
        :class:`ForestMaskResult` for the loaded mask.
    """
    if target is None:
        aligned, meta = read_external_raster(source, band=band)
    else:
        aligned, meta = align_external_raster(
            source,
            target=target,
            band=band,
            frequency=frequency,
            polarization=polarization,
            resampling=resampling,
        )
    mask = raster_to_mask(aligned, threshold=threshold, mask_values=mask_values, invert=invert)
    return ForestMaskResult(
        mask=mask,
        method="external_raster",
        source_kind="external_raster",
        units="mask",
        threshold=threshold,
        metadata={
            **meta,
            "mask_values": list(mask_values) if mask_values is not None else None,
            "invert": invert,
        },
        confidence=None,
    )


def generate_forest_mask(
    source: PathType | np.ndarray | xr.DataArray,
    *,
    method: str = "gcov_hv_threshold",
    frequency: str = "A",
    polarization: str = "HV",
    units: str | None = None,
    threshold_db: float | None = None,
    target: PathType | xr.DataArray | None = None,
    band: int = 1,
    threshold: float | None = None,
    mask_values: tuple[float, ...] | None = None,
    invert: bool = False,
    resampling: str = "nearest",
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> ForestMaskResult:
    """Generate a forest mask from GCOV backscatter or an external raster source.

    Args:
        source: NISAR GCOV path, external raster path, or an in-memory array.
        method: Forest mask method identifier.
        frequency: Frequency to use when reading GCOV.
        polarization: Polarization to use when reading GCOV.
        units: Units for numeric input arrays or raster backscatter values.
        threshold_db: Threshold override for GCOV HV threshold methods.
        target: Optional target grid for the external raster workflow.
        band: Raster band for the external raster workflow.
        threshold: Numeric threshold for external rasters.
        mask_values: Discrete values treated as forest for external rasters.
        invert: Whether to invert the external raster mask.
        resampling: Rasterio resampling mode for alignment.
        filesystem: Optional filesystem for remote GCOV reads.

    Returns:
        :class:`ForestMaskResult` describing the generated mask.
    """
    spec = get_method_spec(method)
    if not spec.implemented:
        refs = ", ".join(spec.references) if spec.references else "no references available"
        raise NotImplementedError(
            f"Forest mask method {method!r} is registered but not yet implemented. References: {refs}."
        )

    if method == "external_raster":
        return load_external_forest_mask(
            source,
            target=target,
            band=band,
            frequency=frequency,
            polarization=polarization,
            resampling=resampling,
            threshold=threshold,
            mask_values=mask_values,
            invert=invert,
        )

    hv_data, attrs, source_kind = _load_gcov_like_source(
        source,
        frequency=frequency,
        polarization=polarization,
        filesystem=filesystem,
    )
    method_func = _GCOV_METHODS[method]
    resolved_threshold = threshold_db if threshold_db is not None else spec.default_threshold_db
    if resolved_threshold is None:
        raise ValueError(f"Method {method!r} requires a threshold_db value.")

    if method == "gcov_hv_threshold":
        mask = method_func(hv_data, threshold_db=resolved_threshold, units=units)
    else:
        mask = method_func(hv_data, units=units)

    confidence = gcov_hv_threshold_margin(hv_data, threshold_db=resolved_threshold, units=units)
    metadata = {
        "frequency": frequency,
        "polarization": polarization,
        "threshold_db": resolved_threshold,
        "source_attrs": attrs,
    }
    return ForestMaskResult(
        mask=mask,
        method=method,
        source_kind=source_kind,
        units="db",
        threshold=resolved_threshold,
        metadata=metadata,
        confidence=confidence,
    )