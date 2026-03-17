"""Radiometric calibration utilities.

Functions for converting between backscatter scales and applying
radiometric corrections to SAR data.
"""

from __future__ import annotations

import numpy as np

from nice_sar._types import ArrayFloat32


def linear_to_db(data: np.ndarray, fill_value: float = np.nan) -> ArrayFloat32:
    """Convert linear power backscatter to decibels.

    Args:
        data: Array of linear power values.
        fill_value: Value to assign where input is non-positive.

    Returns:
        Backscatter in dB scale (10 * log10).
    """
    db_data = np.full_like(data, fill_value, dtype=np.float32)
    valid = data > 0
    db_data[valid] = 10.0 * np.log10(data[valid])
    return db_data


def db_to_linear(data: np.ndarray) -> ArrayFloat32:
    """Convert decibel backscatter to linear power.

    Args:
        data: Array of dB values.

    Returns:
        Linear power values (10^(dB/10)).
    """
    return np.power(10.0, data / 10.0).astype(np.float32)


def compute_sigma0(
    gamma0: np.ndarray, rtc_factor: np.ndarray | None
) -> ArrayFloat32 | None:
    """Convert gamma-nought to sigma-nought using RTC factor.

    Args:
        gamma0: Gamma-nought backscatter (linear units).
        rtc_factor: Radiometric terrain correction factor from GCOV product.

    Returns:
        Sigma-nought backscatter, or ``None`` if ``rtc_factor`` is ``None``.
    """
    if rtc_factor is None:
        return None
    result: ArrayFloat32 = (gamma0 * rtc_factor).astype(np.float32)
    return result


def power_transform(data: np.ndarray, gamma: float = 0.5) -> ArrayFloat32:
    """Apply power-law transform for visualization.

    Normalizes to per-image min/max, applies gamma, then rescales.

    Args:
        data: Input array (non-negative values expected).
        gamma: Exponent for power transform. Values < 1 brighten dark areas.

    Returns:
        Power-transformed array.
    """
    valid_mask = np.isfinite(data) & (data >= 0)
    output = np.full_like(data, np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return output

    valid_data = data[valid_mask]
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    if max_val <= min_val:
        return output

    normalized = (data - min_val) / (max_val - min_val)
    transformed = np.power(np.clip(normalized, 0, 1), gamma)
    output = np.asarray(
        transformed * (max_val - min_val) + min_val, dtype=np.float32,
    )
    return output
