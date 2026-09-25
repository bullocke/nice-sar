"""GCOV-based forest mask methods."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from nice_sar._types import ArrayFloat32
from nice_sar.preprocess.calibration import linear_to_db

logger = logging.getLogger(__name__)

_LINEAR_UNITS = {"linear", "linear_power", "power", "gamma0", "sigma0"}
_DB_UNITS = {"db", "decibel", "decibels", "dB"}


def _normalize_units(units: str | None) -> str | None:
    if units is None:
        return None
    normalized = units.strip()
    lowered = normalized.lower()
    if lowered in _LINEAR_UNITS:
        return "linear_power"
    if lowered in {value.lower() for value in _DB_UNITS}:
        return "db"
    return lowered


def infer_backscatter_units(data: np.ndarray | xr.DataArray, units: str | None = None) -> str:
    """Infer backscatter units from explicit metadata or array values.

    Args:
        data: Input backscatter array.
        units: Explicit units override.

    Returns:
        Normalized units string, either ``"db"`` or ``"linear_power"``.
    """
    normalized = _normalize_units(units)
    if normalized is not None:
        return normalized

    attrs_units = None
    if isinstance(data, xr.DataArray):
        attrs_units = _normalize_units(str(data.attrs.get("units"))) if "units" in data.attrs else None
    if attrs_units is not None:
        return attrs_units

    values = np.asarray(data)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "linear_power"
    if np.nanmin(finite) < 0:
        return "db"
    return "linear_power"


def _to_db(data: np.ndarray | xr.DataArray, units: str | None = None) -> np.ndarray:
    detected_units = infer_backscatter_units(data, units=units)
    values = np.asarray(data, dtype=np.float32)
    if detected_units == "db":
        return values.astype(np.float32)
    return linear_to_db(values)


def _wrap_like(
    template: np.ndarray | xr.DataArray,
    values: np.ndarray,
    *,
    name: str,
    attrs: dict[str, object],
) -> np.ndarray | xr.DataArray:
    if not isinstance(template, xr.DataArray):
        return values

    merged_attrs = dict(template.attrs)
    merged_attrs.update(attrs)
    return xr.DataArray(
        values,
        dims=template.dims,
        coords=template.coords,
        attrs=merged_attrs,
        name=name,
    )


def gcov_hv_threshold(
    hv: np.ndarray | xr.DataArray,
    *,
    threshold_db: float = -20.0,
    units: str | None = None,
) -> np.ndarray | xr.DataArray:
    """Generate a forest mask by thresholding GCOV HV backscatter.

    Pixels with HV backscatter greater than or equal to ``threshold_db`` are
    marked as forest. Inputs may be provided either in dB or linear power.

    Args:
        hv: HV backscatter image.
        threshold_db: Forest threshold in dB.
        units: Input units. If omitted, units are inferred from xarray attrs
            or from array values.

    Returns:
        Boolean forest mask with the same shape as *hv*.
    """
    hv_db = _to_db(hv, units=units)
    valid = np.isfinite(hv_db)
    mask = valid & (hv_db >= threshold_db)
    return _wrap_like(
        hv,
        mask,
        name="forest_mask",
        attrs={
            "mask_method": "gcov_hv_threshold",
            "threshold_db": float(threshold_db),
            "input_units": infer_backscatter_units(hv, units=units),
        },
    )


def gcov_hv_threshold_margin(
    hv: np.ndarray | xr.DataArray,
    *,
    threshold_db: float = -20.0,
    units: str | None = None,
) -> ArrayFloat32 | xr.DataArray:
    """Return the dB margin above or below a GCOV HV forest threshold."""
    hv_db = _to_db(hv, units=units)
    margin = (hv_db - threshold_db).astype(np.float32)
    wrapped = _wrap_like(
        hv,
        margin,
        name="forest_mask_margin_db",
        attrs={
            "mask_method": "gcov_hv_threshold",
            "threshold_db": float(threshold_db),
            "units": "db_margin",
        },
    )
    return wrapped


def gcov_hv_threshold_ramachandran(
    hv: np.ndarray | xr.DataArray,
    *,
    units: str | None = None,
) -> np.ndarray | xr.DataArray:
    """Generate a GCOV HV forest mask using the -20 dB literature preset."""
    return gcov_hv_threshold(hv, threshold_db=-20.0, units=units)


def gcov_hv_threshold_jaxa_biomass(
    hv: np.ndarray | xr.DataArray,
    *,
    units: str | None = None,
) -> np.ndarray | xr.DataArray:
    """Generate a GCOV HV forest mask using the -13 dB biomass-style preset."""
    return gcov_hv_threshold(hv, threshold_db=-13.0, units=units)