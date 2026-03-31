"""Time series analysis for SAR change detection.

Implements four change-detection methods over temporal stacks of NISAR
backscatter or derived indices:

1. **CUSUM** — Cumulative sum change-point detection
2. **Coefficient of Variation (CoV)** — Temporal variability mapping
3. **Backscatter thresholding** — Rule-based classification (e.g. inundation)
4. **Harmonic fitting** — Seasonal model with residual anomaly detection

All functions accept 3-D numpy arrays of shape ``(T, H, W)`` where *T* is
the number of time steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


@dataclass
class CUSUMResult:
    """CUSUM change-point detection output.

    Attributes:
        magnitude: Maximum cumulative sum value at each pixel (change magnitude).
        change_index: Time index of the detected change point (0-based).
            ``-1`` if no change is detected.
        cusum_series: Full cumulative sum series ``(T, H, W)``.
    """

    magnitude: ArrayFloat32
    change_index: np.ndarray
    cusum_series: np.ndarray


def cusum(
    stack: np.ndarray,
    baseline_count: int | None = None,
    threshold: float | None = None,
) -> CUSUMResult:
    """Cumulative sum change-point detection on a temporal stack.

    Computes residuals from the baseline mean then accumulates them over
    time.  The position of the maximum cumulative sum indicates the most
    likely change-point timing.

    Args:
        stack: 3-D array ``(T, H, W)`` of backscatter or index values.
        baseline_count: Number of initial time steps to use as the stable
            baseline for computing the mean. Defaults to ``T // 2``.
        threshold: Minimum ``R_sum_max`` to declare a change. Pixels below
            this are assigned ``change_index = -1``. If ``None``, all pixels
            are returned without masking.

    Returns:
        :class:`CUSUMResult` with magnitude, change index, and full series.
    """
    T, H, W = stack.shape
    if baseline_count is None:
        baseline_count = max(1, T // 2)

    baseline_mean = np.nanmean(stack[:baseline_count], axis=0)  # (H, W)
    residuals = stack - baseline_mean[np.newaxis, :, :]  # (T, H, W)
    cs = np.nancumsum(residuals, axis=0).astype(np.float32)  # (T, H, W)

    magnitude = np.nanmax(np.abs(cs), axis=0).astype(np.float32)
    change_index = np.argmax(np.abs(cs), axis=0).astype(np.int32)

    if threshold is not None:
        change_index[magnitude < threshold] = -1

    return CUSUMResult(magnitude=magnitude, change_index=change_index, cusum_series=cs)


# ---------------------------------------------------------------------------
# Coefficient of Variation
# ---------------------------------------------------------------------------


def coefficient_of_variation(stack: np.ndarray) -> ArrayFloat32:
    """Compute temporal Coefficient of Variation (CoV) per pixel.

    ``CoV = σ / μ`` computed over the time dimension. High CoV indicates
    dynamic surfaces (e.g. agriculture, inundation).

    Args:
        stack: 3-D array ``(T, H, W)`` of backscatter values (linear power).

    Returns:
        2-D CoV map of shape ``(H, W)``.
    """
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    cov: ArrayFloat32 = np.where(mean > 1e-10, std / mean, 0.0).astype(np.float32)
    return cov


# ---------------------------------------------------------------------------
# Backscatter thresholding
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Result of backscatter thresholding classification.

    Attributes:
        mask: Boolean mask where the condition is met.
        fraction: Fraction of time steps where the condition is met per pixel.
    """

    mask: np.ndarray
    fraction: ArrayFloat32


def backscatter_threshold(
    stack: np.ndarray,
    low: float | None = None,
    high: float | None = None,
    min_fraction: float = 0.0,
) -> ThresholdResult:
    """Classify pixels by backscatter intensity thresholds.

    For each time step, pixels with values between *low* and *high* are
    flagged.  The temporal fraction of flagged steps is returned, along
    with a mask where the fraction exceeds *min_fraction*.

    This is useful for inundation mapping (see implementation guide for
    suggested thresholds: HH×HV < 5e-5 for open water).

    Args:
        stack: 3-D array ``(T, H, W)`` of backscatter values (linear).
        low: Lower bound (inclusive). ``None`` = no lower bound.
        high: Upper bound (inclusive). ``None`` = no upper bound.
        min_fraction: Minimum temporal fraction to set pixel as True.

    Returns:
        :class:`ThresholdResult` with boolean mask and temporal fraction.
    """
    if low is None:
        low = -np.inf
    if high is None:
        high = np.inf

    in_range = (stack >= low) & (stack <= high)  # (T, H, W)
    fraction = np.nanmean(in_range.astype(np.float32), axis=0).astype(np.float32)
    mask = fraction >= min_fraction

    return ThresholdResult(mask=mask, fraction=fraction)


# ---------------------------------------------------------------------------
# Harmonic fitting
# ---------------------------------------------------------------------------


@dataclass
class HarmonicResult:
    """Result of harmonic model fitting.

    Attributes:
        intercept: Mean (offset) term per pixel.
        amplitude: Amplitude of the annual harmonic per pixel.
        phase: Phase of the annual harmonic per pixel (radians).
        residuals: Residuals ``(T, H, W)`` — observations minus fitted model.
        rmse: Root mean squared error of the fit per pixel.
    """

    intercept: ArrayFloat32
    amplitude: ArrayFloat32
    phase: ArrayFloat32
    residuals: np.ndarray
    rmse: ArrayFloat32


def harmonic_fit(
    stack: np.ndarray,
    times: np.ndarray,
    period: float = 365.25,
) -> HarmonicResult:
    """Fit a first-order harmonic (seasonal) model to a temporal stack.

    The model is: ``y(t) = a₀ + a₁ cos(2πt/P) + a₂ sin(2πt/P)``

    where ``P`` is the period (default annual). Residuals can be used for
    anomaly / change detection (z-scores, Bayesian updating, etc.).

    Args:
        stack: 3-D array ``(T, H, W)`` of backscatter or index values.
        times: 1-D array of length *T* with time values in the same
            units as *period* (e.g. day-of-year or fractional years).
        period: Period of the harmonic cycle (default 365.25 days).

    Returns:
        :class:`HarmonicResult` with fitted parameters and residuals.
    """
    T, H, W = stack.shape
    omega = 2.0 * np.pi / period

    # Design matrix: [1, cos(ωt), sin(ωt)]
    A = np.column_stack([
        np.ones(T),
        np.cos(omega * times),
        np.sin(omega * times),
    ])  # (T, 3)

    # Reshape stack to (T, N)
    N = H * W
    Y = stack.reshape(T, N)

    # Least-squares solve: A @ coeffs = Y → coeffs = (A^T A)^-1 A^T Y
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)  # (3, N)

    a0 = coeffs[0].reshape(H, W).astype(np.float32)
    a1 = coeffs[1].reshape(H, W).astype(np.float32)
    a2 = coeffs[2].reshape(H, W).astype(np.float32)

    amplitude = np.sqrt(a1**2 + a2**2).astype(np.float32)
    phase = np.arctan2(a2, a1).astype(np.float32)

    fitted = (A @ coeffs).reshape(T, H, W)  # (T, H, W)
    residuals = (stack - fitted).astype(np.float32)
    rmse = np.sqrt(np.nanmean(residuals**2, axis=0)).astype(np.float32)

    return HarmonicResult(
        intercept=a0,
        amplitude=amplitude,
        phase=phase,
        residuals=residuals,
        rmse=rmse,
    )
