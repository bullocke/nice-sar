"""Tests for nice_sar.analysis.insar."""

from __future__ import annotations

import numpy as np
import pytest

from nice_sar.analysis.insar import (
    InSARResult,
    apply_ionospheric_correction,
    estimate_coherence,
    form_interferogram,
    mask_by_coherence,
    phase_to_displacement,
)


@pytest.fixture()
def slc_pair() -> tuple[np.ndarray, np.ndarray]:
    """Coregistered, correlated complex SLC pair (64x64)."""
    rng = np.random.default_rng(50)
    amp = rng.exponential(0.1, size=(64, 64)).astype(np.float32)
    phase_common = rng.uniform(-np.pi, np.pi, size=(64, 64)).astype(np.float32)
    noise = rng.normal(0, 0.1, size=(64, 64)).astype(np.float32)
    ref = (amp * np.exp(1j * phase_common)).astype(np.complex64)
    sec = (amp * np.exp(1j * (phase_common + noise))).astype(np.complex64)
    return ref, sec


class TestFormInterferogram:
    def test_returns_result(self, slc_pair: tuple) -> None:
        ref, sec = slc_pair
        result = form_interferogram(ref, sec)
        assert isinstance(result, InSARResult)

    def test_interferogram_shape(self, slc_pair: tuple) -> None:
        ref, sec = slc_pair
        result = form_interferogram(ref, sec)
        assert result.interferogram.shape == (64, 64)

    def test_interferogram_complex(self, slc_pair: tuple) -> None:
        ref, sec = slc_pair
        result = form_interferogram(ref, sec)
        assert np.iscomplexobj(result.interferogram)

    def test_coherence_range(self, slc_pair: tuple) -> None:
        ref, sec = slc_pair
        result = form_interferogram(ref, sec)
        assert np.all(result.coherence >= 0)
        assert np.all(result.coherence <= 1)


class TestEstimateCoherence:
    def test_identical_inputs_high_coherence(self) -> None:
        rng = np.random.default_rng(51)
        data = (rng.exponential(0.1, (32, 32)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (32, 32)))).astype(np.complex64)
        coh = estimate_coherence(data, data, window=5)
        assert np.all(coh > 0.95)

    def test_uncorrelated_low_coherence(self) -> None:
        rng = np.random.default_rng(52)
        ref = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))).astype(np.complex64)
        sec = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))).astype(np.complex64)
        coh = estimate_coherence(ref, sec, window=11)
        assert np.median(coh) < 0.3

    def test_output_shape(self, slc_pair: tuple) -> None:
        ref, sec = slc_pair
        coh = estimate_coherence(ref, sec, window=5)
        assert coh.shape == ref.shape


class TestPhaseToDisplacement:
    def test_zero_phase_zero_displacement(self) -> None:
        phase = np.zeros((10, 10), dtype=np.float32)
        disp = phase_to_displacement(phase)
        np.testing.assert_allclose(disp, 0.0, atol=1e-10)

    def test_2pi_phase(self) -> None:
        phase = np.full((5, 5), 2 * np.pi, dtype=np.float32)
        disp = phase_to_displacement(phase, wavelength=0.24)
        expected = -0.24 * 2 * np.pi / (4 * np.pi)  # = -0.12 m
        np.testing.assert_allclose(disp, expected, atol=1e-5)

    def test_vertical_projection(self) -> None:
        phase = np.full((5, 5), np.pi, dtype=np.float32)
        los = phase_to_displacement(phase, wavelength=0.24)
        vert = phase_to_displacement(phase, wavelength=0.24, incidence_angle=np.deg2rad(30))
        # Vertical should be larger in magnitude (dividing by cos(30°))
        assert np.all(np.abs(vert) > np.abs(los))


class TestApplyIonosphericCorrection:
    def test_subtraction(self) -> None:
        phase = np.full((10, 10), 3.0, dtype=np.float32)
        iono = np.full((10, 10), 0.5, dtype=np.float32)
        corrected = apply_ionospheric_correction(phase, iono)
        np.testing.assert_allclose(corrected, 2.5, atol=1e-6)

    def test_dtype(self) -> None:
        phase = np.ones((5, 5), dtype=np.float64)
        iono = np.zeros((5, 5), dtype=np.float64)
        result = apply_ionospheric_correction(phase, iono)
        assert result.dtype == np.float32


class TestMaskByCoherence:
    def test_masks_low_coherence(self) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        coh = np.full((10, 10), 0.1, dtype=np.float32)
        masked = mask_by_coherence(data, coh, threshold=0.3)
        assert np.all(np.isnan(masked))

    def test_retains_high_coherence(self) -> None:
        data = np.ones((10, 10), dtype=np.float32) * 5.0
        coh = np.full((10, 10), 0.8, dtype=np.float32)
        masked = mask_by_coherence(data, coh, threshold=0.3)
        np.testing.assert_allclose(masked, 5.0)

    def test_custom_fill_value(self) -> None:
        data = np.ones((5, 5), dtype=np.float32)
        coh = np.zeros((5, 5), dtype=np.float32)
        masked = mask_by_coherence(data, coh, threshold=0.5, fill_value=-9999.0)
        np.testing.assert_allclose(masked, -9999.0)
