"""Tests for nice_sar.analysis.polarimetry."""

from __future__ import annotations

import numpy as np

from nice_sar.analysis.polarimetry import (
    compute_indices,
    compute_rfdi,
    compute_span,
    volume_proxy,
)


class TestComputeSpan:
    def test_basic(self, hh_linear: np.ndarray, hv_linear: np.ndarray) -> None:
        span = compute_span(hh=hh_linear, hv=hv_linear)
        assert span.shape == hh_linear.shape
        assert (span >= 0).all()

    def test_with_vv(
        self, hh_linear: np.ndarray, hv_linear: np.ndarray, vv_linear: np.ndarray,
    ) -> None:
        span = compute_span(hh=hh_linear, hv=hv_linear, vv=vv_linear)
        expected = hh_linear + 2 * hv_linear + vv_linear
        np.testing.assert_allclose(span, expected, rtol=1e-5)


class TestComputeRfdi:
    def test_range(self, hh_linear: np.ndarray, hv_linear: np.ndarray) -> None:
        rfdi = compute_rfdi(hh_linear, hv_linear)
        assert rfdi.min() >= -1.0
        assert rfdi.max() <= 1.0


class TestVolumeProxy:
    def test_range(self, hh_linear: np.ndarray, hv_linear: np.ndarray) -> None:
        vp = volume_proxy(hh_linear, hv_linear)
        assert vp.min() >= 0.0
        assert vp.max() <= 1.0


class TestComputeIndices:
    def test_keys(self, hh_linear: np.ndarray, hv_linear: np.ndarray) -> None:
        indices = compute_indices(hh_linear, hv=hv_linear)
        assert "hh_hv_ratio" in indices
        assert "biomass_index" in indices
