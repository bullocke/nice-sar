"""Speckle filters for SAR data.

Provides classic Lee and directional Refined Lee speckle filters
operating on linear-scale backscatter data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from nice_sar._types import ArrayFloat32


def lee_filter(data: np.ndarray, window_size: int = 5) -> ArrayFloat32:
    """Classic Lee additive speckle filter.

    Args:
        data: 2D array of linear backscatter values.
        window_size: Size of the square filter window.

    Returns:
        Filtered array with reduced speckle noise.
    """
    filtered = data.astype(np.float32).copy()
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return filtered

    # Replace NaN with 0 for filtering, use valid count for correct mean/var
    safe = np.where(valid_mask, filtered, 0.0)
    ones = valid_mask.astype(np.float32)

    # Count of valid pixels in each window
    wsum = ndimage.uniform_filter(ones, size=window_size, mode="reflect") * window_size**2
    wsum = np.maximum(wsum, 1.0)

    # Local mean and variance (NaN-safe via zero-fill + count normalization)
    local_sum = ndimage.uniform_filter(safe, size=window_size, mode="reflect") * window_size**2
    local_mean = np.asarray(local_sum / wsum, dtype=np.float64)

    sq_filter = ndimage.uniform_filter(safe**2, size=window_size, mode="reflect")
    local_sq_sum = sq_filter * window_size**2
    local_var = np.maximum(np.asarray(local_sq_sum / wsum, dtype=np.float64) - local_mean**2, 0.0)

    overall_var = np.nanvar(filtered[valid_mask]) + 1e-10
    weight = local_var / (local_var + overall_var)
    filtered = np.asarray(
        (local_mean + weight * (filtered - local_mean)), dtype=np.float32,
    )
    filtered[~valid_mask] = np.nan
    return filtered


def refined_lee_filter(img: np.ndarray, win: int = 7) -> ArrayFloat32:
    """Refined Lee speckle filter with directional edge preservation.

    Uses four principal directions (0, 45, 90, 135 degrees) to find the
    local direction of minimum variance, then applies Lee filtering
    preferentially along that direction.

    Args:
        img: 2D array of linear backscatter values.
        win: Filter window size (minimum 5).

    Returns:
        Filtered array with edge-preserving speckle reduction.
    """
    if win < 5:
        win = 5
    img = img.astype(np.float32)
    valid = np.isfinite(img)
    if not np.any(valid):
        return img

    # Directional gradient kernels for 0°, 45°, 90°, 135°
    kernels = [
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32),
    ]
    grads = [np.abs(ndimage.convolve(img, k, mode="nearest")) for k in kernels]
    dir_idx = np.argmin(np.stack(grads, axis=0), axis=0)

    # Standard Lee statistics
    mean: NDArray[np.floating[Any]] = np.asarray(
        ndimage.uniform_filter(img, size=win, mode="nearest"), dtype=np.float64
    )
    sqmean: NDArray[np.floating[Any]] = np.asarray(
        ndimage.uniform_filter(img * img, size=win, mode="nearest"), dtype=np.float64
    )
    var = np.maximum(sqmean - mean * mean, 0.0)
    noise_var = np.nanmedian(var[valid]) + 1e-10
    w = var / (var + noise_var)

    def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)

    # Directional neighbor averages
    neigh = np.zeros_like(img)
    neigh[dir_idx == 0] = (
        (_shift(img, 0, -1)[dir_idx == 0] + _shift(img, 0, 1)[dir_idx == 0]) * 0.5
    )
    neigh[dir_idx == 1] = (
        (_shift(img, -1, 1)[dir_idx == 1] + _shift(img, 1, -1)[dir_idx == 1]) * 0.5
    )
    neigh[dir_idx == 2] = (
        (_shift(img, -1, 0)[dir_idx == 2] + _shift(img, 1, 0)[dir_idx == 2]) * 0.5
    )
    neigh[dir_idx == 3] = (
        (_shift(img, -1, -1)[dir_idx == 3] + _shift(img, 1, 1)[dir_idx == 3]) * 0.5
    )

    lee = mean + w * (img - mean)
    refined = np.asarray(0.5 * lee + 0.5 * neigh, dtype=np.float32)
    refined[~valid] = np.nan
    return refined
