"""Polarimetric indices and parameters for SAR analysis.

Computes derived polarimetric quantities from backscatter measurements
including SPAN, RFDI, volume proxy, and multi-polarimetric indices.
"""

from __future__ import annotations

import numpy as np

from nice_sar._types import ArrayFloat32
from nice_sar.preprocess.calibration import linear_to_db


def compute_span(
    hh: np.ndarray | None = None,
    hv: np.ndarray | None = None,
    vv: np.ndarray | None = None,
) -> ArrayFloat32 | None:
    """Compute total power (SPAN) from available polarizations.

    Uses the appropriate formula depending on available channels:
    - Quad-pol: HH + VV + 2*HV
    - Dual-pol HH/VV: HH + VV
    - Dual-pol HH/HV: HH + HV

    Args:
        hh: HH backscatter (linear power).
        hv: HV backscatter (linear power).
        vv: VV backscatter (linear power).

    Returns:
        SPAN array in linear units, or ``None`` if insufficient inputs.
    """
    if hh is not None and vv is not None and hv is not None:
        return (hh + vv + 2.0 * hv).astype(np.float32)
    if hh is not None and vv is not None:
        return (hh + vv).astype(np.float32)
    if hh is not None and hv is not None:
        return (hh + hv).astype(np.float32)
    if vv is not None and hv is not None:
        return (vv + hv).astype(np.float32)
    for arr in (hh, vv, hv):
        if arr is not None:
            return arr.astype(np.float32)
    return None


def compute_rfdi(
    hh: np.ndarray | None = None,
    hv: np.ndarray | None = None,
    vv: np.ndarray | None = None,
) -> ArrayFloat32 | None:
    """Compute Radar Forest Degradation Index.

    RFDI = (co-pol - cross-pol) / (co-pol + cross-pol), ranging from -1 to +1.
    Positive values indicate surface/double-bounce dominant (non-forest),
    negative values indicate volume scattering (forest).

    Args:
        hh: HH backscatter (linear).
        hv: HV backscatter (linear).
        vv: VV backscatter (linear).

    Returns:
        RFDI array clipped to [-1, 1], or ``None`` if insufficient inputs.
    """
    if hv is None:
        return None

    if hh is not None and vv is not None:
        copol = 0.5 * (hh + vv)
    elif hh is not None:
        copol = hh
    elif vv is not None:
        copol = vv
    else:
        return None

    rfdi = (copol - hv) / (copol + hv + 1e-10)
    return np.clip(rfdi, -1.0, 1.0).astype(np.float32)


def volume_proxy(hh: np.ndarray, hv: np.ndarray) -> ArrayFloat32:
    """Compute dual-pol volume scattering proxy.

    Defined as HV / (HH + HV), ranges from 0 (surface) to ~0.5 (volume).

    Args:
        hh: HH backscatter (linear).
        hv: HV backscatter (linear).

    Returns:
        Volume proxy array clipped to [0, 1].
    """
    vp = hv / (hh + hv + 1e-10)
    return np.clip(vp, 0.0, 1.0).astype(np.float32)


def compute_indices(
    hh: np.ndarray,
    hv: np.ndarray | None = None,
    vv: np.ndarray | None = None,
) -> dict[str, ArrayFloat32]:
    """Compute a suite of polarimetric indices.

    Args:
        hh: HH backscatter (linear).
        hv: HV backscatter (linear), optional.
        vv: VV backscatter (linear), optional.

    Returns:
        Dictionary of index name → array. Available indices depend on inputs:
        ``hh_hv_ratio``, ``hh_hv_ratio_db``, ``rvi``, ``biomass_index``,
        ``biomass_index_db``.
    """
    indices: dict[str, ArrayFloat32] = {}

    if hv is not None:
        ratio = hh / (hv + 1e-10)
        indices["hh_hv_ratio"] = ratio.astype(np.float32)
        indices["hh_hv_ratio_db"] = linear_to_db(ratio)

    if hv is not None and vv is not None:
        rvi = (4.0 * hv) / (hh + vv + 2.0 * hv + 1e-10)
        indices["rvi"] = rvi.astype(np.float32)

    if hv is not None:
        biomass = hh + hv
        indices["biomass_index"] = biomass.astype(np.float32)
        indices["biomass_index_db"] = linear_to_db(biomass)

    return indices
