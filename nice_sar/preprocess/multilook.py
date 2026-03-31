"""Spatial multilooking for SAR data.

Provides power-domain averaging, complex multilooking, covariance matrix
multilooking, ENL estimation, and look-factor calculation utilities for
NISAR products.
"""

from __future__ import annotations

import logging

import numpy as np

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)


def multilook(data: np.ndarray, looks_y: int = 2, looks_x: int = 2) -> ArrayFloat32:
    """Apply spatial multilooking (averaging) to reduce speckle.

    Averages pixels in non-overlapping blocks of size ``(looks_y, looks_x)``.

    Args:
        data: 2D input array.
        looks_y: Number of looks in the y (range) direction.
        looks_x: Number of looks in the x (azimuth) direction.

    Returns:
        Multilooked array with reduced dimensions.
    """
    rows, cols = data.shape
    # Trim to exact multiples
    trim_r = (rows // looks_y) * looks_y
    trim_c = (cols // looks_x) * looks_x
    trimmed = data[:trim_r, :trim_c].astype(np.float32)

    reshaped = trimmed.reshape(trim_r // looks_y, looks_y, trim_c // looks_x, looks_x)
    result: ArrayFloat32 = np.nanmean(reshaped, axis=(1, 3)).astype(np.float32)
    return result


def multilook_complex(
    data: np.ndarray, looks_y: int = 2, looks_x: int = 2
) -> np.ndarray:
    """Coherent multilooking of complex SAR data.

    Averages complex values (preserving phase information) in non-overlapping
    blocks. Useful for interferogram formation and coherence estimation.

    Args:
        data: 2D complex input array (e.g. SLC or interferogram).
        looks_y: Number of looks in the y direction.
        looks_x: Number of looks in the x direction.

    Returns:
        Complex multilooked array with reduced dimensions.
    """
    rows, cols = data.shape
    trim_r = (rows // looks_y) * looks_y
    trim_c = (cols // looks_x) * looks_x
    trimmed = data[:trim_r, :trim_c].astype(np.complex64)

    reshaped = trimmed.reshape(trim_r // looks_y, looks_y, trim_c // looks_x, looks_x)
    return np.nanmean(reshaped, axis=(1, 3)).astype(np.complex64)


def multilook_covariance(
    covariances: dict[str, np.ndarray],
    looks_y: int = 2,
    looks_x: int = 2,
) -> dict[str, np.ndarray]:
    """Multilook a dictionary of polarimetric covariance terms.

    Real-valued diagonal terms are averaged in the power domain.
    Complex off-diagonal terms are averaged coherently.

    Args:
        covariances: Dictionary with keys like ``"HHHH"``, ``"HHHV"`` etc.
            from :func:`nice_sar.io.products.read_quad_covariances`.
        looks_y: Number of looks in the y direction.
        looks_x: Number of looks in the x direction.

    Returns:
        Dictionary with same keys but multilooked arrays.
    """
    out: dict[str, np.ndarray] = {}
    for key, arr in covariances.items():
        if np.iscomplexobj(arr):
            out[key] = multilook_complex(arr, looks_y, looks_x)
        else:
            out[key] = multilook(arr, looks_y, looks_x)
    return out


def estimate_enl(data: np.ndarray, window: int = 15) -> ArrayFloat32:
    """Estimate the Equivalent Number of Looks (ENL) from intensity data.

    Computes ENL as ``mean² / variance`` in a sliding window, which equals
    the number of independent samples for fully-developed speckle.

    Args:
        data: 2D intensity (power) array.
        window: Size of the square estimation window.

    Returns:
        ENL map with same shape as input.
    """
    from scipy.ndimage import uniform_filter

    intensity = data.astype(np.float64)
    mean = uniform_filter(intensity, size=window, mode="nearest")
    mean_sq = uniform_filter(intensity**2, size=window, mode="nearest")
    variance = mean_sq - mean**2
    variance = np.maximum(variance, 1e-10)
    enl: ArrayFloat32 = (mean**2 / variance).astype(np.float32)
    return enl


def compute_look_factors(
    native_az: float,
    native_rg: float,
    target_posting: float,
) -> tuple[int, int]:
    """Compute multilook factors to reach a target ground posting.

    Args:
        native_az: Native azimuth pixel spacing in metres.
        native_rg: Native range pixel spacing in metres.
        target_posting: Desired output posting in metres.

    Returns:
        Tuple of ``(looks_y, looks_x)`` — at least 1 in each dimension.
    """
    looks_y = max(1, round(target_posting / native_az))
    looks_x = max(1, round(target_posting / native_rg))
    logger.info(
        "Look factors for %.1f m posting: (%d, %d) from native (%.1f, %.1f) m",
        target_posting,
        looks_y,
        looks_x,
        native_az,
        native_rg,
    )
    return looks_y, looks_x
