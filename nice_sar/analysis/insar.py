"""InSAR processing utilities for NISAR GSLC and GUNW products.

Provides functions for interferogram formation from GSLC pairs, coherence
estimation, phase-to-displacement conversion, ionospheric phase correction,
and coherence masking. Works with both GSLC-derived interferograms and
pre-computed GUNW products.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)

# NISAR L-band wavelength in metres
_L_BAND_WAVELENGTH_M = 0.24


@dataclass
class InSARResult:
    """Container for interferogram formation results.

    Attributes:
        interferogram: Complex interferogram (reference × conj(secondary)).
        coherence: Estimated coherence magnitude in [0, 1].
        amplitude: Geometric mean amplitude of the SLC pair.
    """

    interferogram: np.ndarray
    coherence: ArrayFloat32
    amplitude: ArrayFloat32


def form_interferogram(
    reference: np.ndarray,
    secondary: np.ndarray,
    coherence_window: int = 5,
) -> InSARResult:
    """Form an interferogram from a coregistered GSLC pair.

    Computes the complex interferogram as ``reference × conj(secondary)``
    and estimates coherence in a sliding window.

    Args:
        reference: Complex 2D array (reference GSLC).
        secondary: Complex 2D array (secondary GSLC), same shape.
        coherence_window: Window size for coherence estimation.

    Returns:
        :class:`InSARResult` with interferogram, coherence, and amplitude.
    """
    ifg = (reference * np.conj(secondary)).astype(np.complex64)
    coherence = estimate_coherence(reference, secondary, window=coherence_window)
    amp = np.sqrt(np.abs(reference) * np.abs(secondary)).astype(np.float32)
    return InSARResult(interferogram=ifg, coherence=coherence, amplitude=amp)


def estimate_coherence(
    reference: np.ndarray,
    secondary: np.ndarray,
    window: int = 5,
) -> ArrayFloat32:
    """Estimate interferometric coherence magnitude.

    Coherence is computed as:
    ``|⟨ref × conj(sec)⟩| / sqrt(⟨|ref|²⟩ × ⟨|sec|²⟩)``

    where ⟨·⟩ denotes spatial averaging over the given window.

    Args:
        reference: Complex 2D array.
        secondary: Complex 2D array, same shape.
        window: Spatial averaging window size.

    Returns:
        Coherence magnitude array in [0, 1].
    """
    cross = reference * np.conj(secondary)
    cross_mean_r = uniform_filter(cross.real.astype(np.float64), size=window, mode="nearest")
    cross_mean_i = uniform_filter(cross.imag.astype(np.float64), size=window, mode="nearest")
    cross_mean = cross_mean_r + 1j * cross_mean_i

    pow_ref = uniform_filter(
        (np.abs(reference) ** 2).astype(np.float64), size=window, mode="nearest"
    )
    pow_sec = uniform_filter(
        (np.abs(secondary) ** 2).astype(np.float64), size=window, mode="nearest"
    )

    denom = np.sqrt(pow_ref * pow_sec)
    denom = np.maximum(denom, 1e-10)
    coh: ArrayFloat32 = np.clip(np.abs(cross_mean) / denom, 0.0, 1.0).astype(np.float32)
    return coh


def phase_to_displacement(
    phase: np.ndarray,
    wavelength: float = _L_BAND_WAVELENGTH_M,
    incidence_angle: float | np.ndarray | None = None,
) -> ArrayFloat32:
    """Convert unwrapped interferometric phase to line-of-sight displacement.

    ``d_LOS = -λ × φ / (4π)``

    Positive displacement = motion toward the satellite (uplift/advance).

    If *incidence_angle* is provided (radians), converts to vertical:
    ``d_vert = d_LOS / cos(θ)``

    Args:
        phase: Unwrapped phase array in radians.
        wavelength: Radar wavelength in metres. Default is NISAR L-band (0.24 m).
        incidence_angle: Incidence angle in radians for vertical projection.

    Returns:
        Displacement array in metres.
    """
    d_los = (-wavelength * phase / (4.0 * np.pi)).astype(np.float32)
    if incidence_angle is not None:
        cos_theta = np.cos(incidence_angle)
        cos_theta = np.where(np.abs(cos_theta) < 1e-6, 1e-6, cos_theta)
        result: ArrayFloat32 = (d_los / cos_theta).astype(np.float32)
        return result
    return d_los


def apply_ionospheric_correction(
    phase: np.ndarray,
    iono_screen: np.ndarray,
) -> ArrayFloat32:
    """Subtract the ionospheric phase screen from unwrapped phase.

    NISAR GUNW products include an ionosphere phase screen estimated via
    the split-spectrum method. This function applies it as a simple
    subtraction.

    Args:
        phase: Unwrapped phase array in radians.
        iono_screen: Ionospheric phase screen in radians (same shape).

    Returns:
        Corrected phase in radians.
    """
    result: ArrayFloat32 = (phase - iono_screen).astype(np.float32)
    return result


def mask_by_coherence(
    data: np.ndarray,
    coherence: np.ndarray,
    threshold: float = 0.3,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Mask data where coherence falls below a threshold.

    Args:
        data: Array to mask (any dtype).
        coherence: Coherence magnitude array in [0, 1].
        threshold: Minimum coherence to retain. Default 0.3.
        fill_value: Value to insert where coherence is low.

    Returns:
        Masked copy of *data*.
    """
    out = data.astype(np.float32).copy()
    out[coherence < threshold] = fill_value
    return out
