"""Display utilities for SAR data visualization.

Provides normalization, stretching, and conversion functions used
across all visualization modules.
"""

from __future__ import annotations

import numpy as np

from nice_sar._types import ArrayFloat32, ArrayUInt8


def percentile_stretch(
    arr: np.ndarray, p_low: float = 2, p_high: float = 98
) -> ArrayFloat32:
    """Stretch array to [0, 1] range using percentile clipping.

    Invalid (NaN/Inf) pixels become 0.

    Args:
        arr: Input array.
        p_low: Lower percentile for clipping.
        p_high: Upper percentile for clipping.

    Returns:
        Normalized array in [0, 1] range as float32.
    """
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    stretched = np.where(valid, stretched, 0)
    return stretched.astype(np.float32)


def gamma_correct(arr: np.ndarray, gamma: float = 0.5) -> ArrayFloat32:
    """Apply gamma correction to a [0, 1] array.

    Values < 1 brighten dark areas; values > 1 darken bright areas.

    Args:
        arr: Input array with values in [0, 1].
        gamma: Gamma exponent.

    Returns:
        Gamma-corrected array.
    """
    result: ArrayFloat32 = np.power(np.clip(arr, 0, 1), gamma).astype(np.float32)
    return result


def to_uint8(arr: np.ndarray) -> ArrayUInt8:
    """Convert [0, 1] float array to [0, 255] uint8.

    NaN values are mapped to 0 (black).

    Args:
        arr: Input array with values in [0, 1].

    Returns:
        Array as uint8 in [0, 255].
    """
    result = np.clip(arr, 0, 1) * 255
    result = np.nan_to_num(result, nan=0.0)
    out: ArrayUInt8 = result.astype(np.uint8)
    return out
