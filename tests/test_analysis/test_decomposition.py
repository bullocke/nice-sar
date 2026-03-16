"""Tests for nice_sar.analysis.decomposition."""

from __future__ import annotations

import numpy as np

from nice_sar.analysis.decomposition import (
    build_coherency_matrix,
    compute_pauli_rgb,
)


def _make_quad_covariances(
    rng: np.random.Generator, n: int
) -> dict[str, np.ndarray]:
    """Build a synthetic covariance dict for testing."""
    hh = rng.exponential(0.05, (n, n)).astype(np.float32)
    hv = rng.exponential(0.01, (n, n)).astype(np.float32)
    vv = rng.exponential(0.04, (n, n)).astype(np.float32)
    return {
        "HHHH": hh,
        "HVHV": hv,
        "VVVV": vv,
        "HHHV": (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(
            np.complex64
        )
        * 0.01,
        "HHVV": (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(
            np.complex64
        )
        * 0.01,
        "HVVV": (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(
            np.complex64
        )
        * 0.01,
    }


class TestBuildCoherencyMatrix:
    def test_shape(self) -> None:
        rng = np.random.default_rng(0)
        covs = _make_quad_covariances(rng, 50)
        T = build_coherency_matrix(covs, window=3)
        assert T.shape == (3, 3, 50, 50)

    def test_hermitian(self) -> None:
        rng = np.random.default_rng(1)
        covs = _make_quad_covariances(rng, 20)
        T = build_coherency_matrix(covs, window=3)
        # Check hermitian: T[i,j,:,:] = conj(T[j,i,:,:])
        T_h = np.conj(np.transpose(T, (1, 0, 2, 3)))
        np.testing.assert_allclose(T, T_h, atol=1e-5)


class TestComputePauliRgb:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(2)
        hh = rng.exponential(0.05, (30, 30)).astype(np.float32)
        hv = rng.exponential(0.01, (30, 30)).astype(np.float32)
        vv = rng.exponential(0.04, (30, 30)).astype(np.float32)
        rgb = compute_pauli_rgb(hh, hv, vv)
        assert rgb.shape == (3, 30, 30)
        assert rgb.dtype == np.float32
