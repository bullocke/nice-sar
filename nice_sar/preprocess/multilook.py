"""Spatial multilooking for SAR data."""

from __future__ import annotations

import numpy as np

from nice_sar._types import ArrayFloat32


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
    return np.nanmean(reshaped, axis=(1, 3)).astype(np.float32)
