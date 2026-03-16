"""Texture feature extraction for SAR data.

Computes GLCM-surrogate texture features using rank filters and
local statistics from backscatter imagery.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage
from skimage.filters import rank
from skimage.morphology import disk

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)


def compute_glcm_texture(
    data: np.ndarray, window_size: int = 11, levels: int = 32
) -> dict[str, ArrayFloat32] | None:
    """Compute texture features using rank filters on quantized data.

    Produces entropy, mean, variance, and range texture maps as
    efficient surrogates for full GLCM computation.

    Args:
        data: 2D backscatter array (typically in dB).
        window_size: Diameter of the disk-shaped filter footprint.
        levels: Number of quantization levels for the input.

    Returns:
        Dictionary with keys ``entropy``, ``mean``, ``variance``, ``range``,
        or ``None`` if no valid data exists.
    """
    logger.info("Computing texture features (window=%d)", window_size)
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return None

    valid_data = data[valid_mask]
    data_min, data_max = np.percentile(valid_data, [2, 98])
    normalized = np.clip((data - data_min) / (data_max - data_min + 1e-10), 0, 1)
    scaled = (normalized * (levels - 1)).astype(np.uint8)
    fp = disk(window_size // 2)

    textures: dict[str, ArrayFloat32] = {}
    try:
        textures["entropy"] = rank.entropy(scaled, fp).astype(np.float32)
        textures["mean"] = rank.mean(scaled, fp).astype(np.float32)
        scaled_sq = (scaled.astype(np.float32) ** 2).astype(np.uint8)
        mean_sq = rank.mean(scaled_sq, fp).astype(np.float32)
        textures["variance"] = mean_sq - (textures["mean"] ** 2)
        textures["range"] = rank.gradient(scaled, fp).astype(np.float32)
        for k in textures:
            textures[k][~valid_mask] = np.nan
    except Exception:
        logger.warning("Could not compute all texture features", exc_info=True)
        return None

    return textures


def compute_local_contrast_homogeneity(
    data: np.ndarray, window_size: int = 11
) -> tuple[ArrayFloat32, ArrayFloat32]:
    """Compute local contrast (std) and homogeneity (1/(1+CV)).

    Args:
        data: 2D array (typically dB backscatter).
        window_size: Size of the filter window.

    Returns:
        Tuple of (contrast, homogeneity) arrays.
    """
    valid_mask = np.isfinite(data)
    contrast = ndimage.generic_filter(
        data, np.nanstd, size=window_size, mode="constant", cval=np.nan
    )
    mean = ndimage.generic_filter(
        data, np.nanmean, size=window_size, mode="constant", cval=np.nan
    )
    homogeneity = 1.0 / (1.0 + contrast / (mean + 1e-10))
    contrast[~valid_mask] = np.nan
    homogeneity[~valid_mask] = np.nan
    return contrast.astype(np.float32), homogeneity.astype(np.float32)
