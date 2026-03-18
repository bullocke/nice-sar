"""Tests for nice_sar.analysis.decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from nice_sar.analysis.decomposition import (
    build_coherency_matrix,
    cloude_pottier,
    compute_pauli_rgb,
    freeman_durden,
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


# ---------------------------------------------------------------------------
# Freeman-Durden 3-component decomposition
# ---------------------------------------------------------------------------


def _build_coherency(rng: np.random.Generator, n: int = 30) -> np.ndarray:
    """Build a coherency matrix from synthetic covariances for testing."""
    covs = _make_quad_covariances(rng, n)
    return build_coherency_matrix(covs, window=3)


class TestFreemanDurden:
    """Tests for freeman_durden() decomposition."""

    def test_output_shapes(self) -> None:
        T = _build_coherency(np.random.default_rng(10), n=40)
        Ps, Pd, Pv = freeman_durden(T)
        assert Ps.shape == (40, 40)
        assert Pd.shape == (40, 40)
        assert Pv.shape == (40, 40)

    def test_output_dtype(self) -> None:
        T = _build_coherency(np.random.default_rng(11))
        Ps, Pd, Pv = freeman_durden(T)
        assert Ps.dtype == np.float32
        assert Pd.dtype == np.float32
        assert Pv.dtype == np.float32

    def test_non_negative(self) -> None:
        T = _build_coherency(np.random.default_rng(12), n=50)
        Ps, Pd, Pv = freeman_durden(T)
        assert np.all(Ps >= 0)
        assert np.all(Pd >= 0)
        assert np.all(Pv >= 0)

    def test_power_conservation(self) -> None:
        """Ps + Pd + Pv should approximate the total SPAN (T11 + T22 + T33)."""
        rng = np.random.default_rng(13)
        T = _build_coherency(rng, n=60)
        Ps, Pd, Pv = freeman_durden(T)
        total_decomp = Ps + Pd + Pv
        span = T[0, 0].real + T[1, 1].real + T[2, 2].real
        # The decomposition redistributes power; total should be on the same order.
        # Allow generous tolerance — the simplified F-D model is approximate.
        ratio = total_decomp / (span + 1e-12)
        assert np.nanmedian(ratio) > 0.3, "Decomposed power too small vs SPAN"
        assert np.nanmedian(ratio) < 3.0, "Decomposed power too large vs SPAN"

    def test_volume_dominance_for_high_cross_pol(self) -> None:
        """When cross-pol is large relative to co-pol, volume should dominate."""
        rng = np.random.default_rng(14)
        n = 20
        # Set HVHV >> HHHH, VVVV to simulate volume scattering
        covs = _make_quad_covariances(rng, n)
        covs["HVHV"] = np.full((n, n), 0.5, dtype=np.float32)
        covs["HHHH"] = np.full((n, n), 0.01, dtype=np.float32)
        covs["VVVV"] = np.full((n, n), 0.01, dtype=np.float32)
        T = build_coherency_matrix(covs, window=3)
        Ps, Pd, Pv = freeman_durden(T)
        # Volume power should generally exceed surface and double-bounce
        assert np.mean(Pv) > np.mean(Ps)
        assert np.mean(Pv) > np.mean(Pd)


# ---------------------------------------------------------------------------
# Cloude-Pottier H/A/alpha decomposition
# ---------------------------------------------------------------------------


class TestCloudePottier:
    """Tests for cloude_pottier() decomposition."""

    def test_output_shapes(self) -> None:
        T = _build_coherency(np.random.default_rng(20), n=30)
        H, A, alpha = cloude_pottier(T)
        assert H.shape == (30, 30)
        assert A.shape == (30, 30)
        assert alpha.shape == (30, 30)

    def test_output_dtype(self) -> None:
        T = _build_coherency(np.random.default_rng(21))
        H, A, alpha = cloude_pottier(T)
        assert H.dtype == np.float32
        assert A.dtype == np.float32
        assert alpha.dtype == np.float32

    def test_entropy_range(self) -> None:
        """Entropy H ∈ [0, 1]."""
        T = _build_coherency(np.random.default_rng(22), n=50)
        H, _, _ = cloude_pottier(T)
        valid = H[np.isfinite(H)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_anisotropy_range(self) -> None:
        """Anisotropy A ∈ [0, 1]."""
        T = _build_coherency(np.random.default_rng(23), n=50)
        _, A, _ = cloude_pottier(T)
        valid = A[np.isfinite(A)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_alpha_range(self) -> None:
        """Alpha angle ∈ [0°, 90°]."""
        T = _build_coherency(np.random.default_rng(24), n=50)
        _, _, alpha = cloude_pottier(T)
        valid = alpha[np.isfinite(alpha)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 90.0)

    def test_identity_coherency_low_entropy(self) -> None:
        """Scaled identity T → single dominant mechanism → low entropy."""
        n = 20
        T = np.zeros((3, 3, n, n), dtype=np.complex64)
        # Identity-like: equal eigenvalues → H should be near 1 (max entropy)
        # But a single dominant: T11 >> T22, T33 → low H
        T[0, 0] = 1.0
        T[1, 1] = 0.01
        T[2, 2] = 0.01
        H, A, alpha = cloude_pottier(T)
        valid_H = H[np.isfinite(H)]
        # Single dominant eigenvalue → low entropy
        assert np.mean(valid_H) < 0.5, f"Expected low entropy, got {np.mean(valid_H):.2f}"


# ---------------------------------------------------------------------------
# Edge cases for both decompositions
# ---------------------------------------------------------------------------


class TestDecompositionEdgeCases:
    """Edge-case tests for freeman_durden and cloude_pottier."""

    def test_freeman_durden_zero_coherency(self) -> None:
        """All-zero coherency → zero power everywhere."""
        T = np.zeros((3, 3, 10, 10), dtype=np.complex64)
        Ps, Pd, Pv = freeman_durden(T)
        np.testing.assert_array_equal(Ps, 0.0)
        np.testing.assert_array_equal(Pd, 0.0)
        np.testing.assert_array_equal(Pv, 0.0)

    def test_cloude_pottier_zero_coherency(self) -> None:
        """All-zero coherency → output filled (may be NaN or near-zero)."""
        T = np.zeros((3, 3, 10, 10), dtype=np.complex64)
        H, A, alpha = cloude_pottier(T)
        assert H.shape == (10, 10)
        assert A.shape == (10, 10)

    def test_freeman_durden_nan_input(self) -> None:
        """NaN in coherency → NaN in output, no crash."""
        T = np.full((3, 3, 5, 5), np.nan, dtype=np.complex64)
        Ps, Pd, Pv = freeman_durden(T)
        assert Ps.shape == (5, 5)
        # NaN input → NaN output
        assert np.all(np.isnan(Ps))

    def test_cloude_pottier_nan_input(self) -> None:
        """NaN in coherency → NaN output pixels, no crash."""
        T = np.full((3, 3, 5, 5), np.nan, dtype=np.complex64)
        H, A, alpha = cloude_pottier(T)
        assert H.shape == (5, 5)
        assert np.all(np.isnan(H))

    def test_single_pixel(self) -> None:
        """Both decompositions work on a single pixel."""
        T = np.zeros((3, 3, 1, 1), dtype=np.complex64)
        T[0, 0] = 0.5
        T[1, 1] = 0.1
        T[2, 2] = 0.05
        Ps, Pd, Pv = freeman_durden(T)
        assert Ps.shape == (1, 1)
        H, A, alpha = cloude_pottier(T)
        assert H.shape == (1, 1)
