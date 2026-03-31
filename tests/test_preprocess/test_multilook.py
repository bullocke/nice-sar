"""Tests for nice_sar.preprocess.multilook."""

from __future__ import annotations

import numpy as np

from nice_sar.preprocess.multilook import (
    compute_look_factors,
    estimate_enl,
    multilook,
    multilook_complex,
    multilook_covariance,
)


class TestMultilook:
    def test_output_shape(self) -> None:
        data = np.ones((100, 100), dtype=np.float32)
        result = multilook(data, looks_y=2, looks_x=2)
        assert result.shape == (50, 50)

    def test_asymmetric_looks(self) -> None:
        data = np.ones((100, 120), dtype=np.float32)
        result = multilook(data, looks_y=4, looks_x=3)
        assert result.shape == (25, 40)

    def test_averaging(self) -> None:
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = multilook(data, looks_y=2, looks_x=2)
        assert result.shape == (2, 2)
        # Top-left 2x2 block: (0+1+4+5)/4 = 2.5
        np.testing.assert_allclose(result[0, 0], 2.5)

    def test_trims_remainder(self) -> None:
        data = np.ones((11, 13), dtype=np.float32)
        result = multilook(data, looks_y=3, looks_x=4)
        # 11 // 3 = 3, 13 // 4 = 3
        assert result.shape == (3, 3)

    def test_handles_nan(self) -> None:
        data = np.ones((4, 4), dtype=np.float32)
        data[0, 0] = np.nan
        result = multilook(data, looks_y=2, looks_x=2)
        # nanmean of [nan, 1, 1, 1] = 1.0
        np.testing.assert_allclose(result[0, 0], 1.0)


class TestMultilookComplex:
    def test_output_shape(self) -> None:
        data = np.ones((8, 8), dtype=np.complex64)
        result = multilook_complex(data, looks_y=2, looks_x=2)
        assert result.shape == (4, 4)

    def test_preserves_phase(self) -> None:
        # All pixels have same phase → mean should preserve it
        phase = np.pi / 4
        data = np.exp(1j * phase) * np.ones((4, 4), dtype=np.complex64)
        result = multilook_complex(data, looks_y=2, looks_x=2)
        np.testing.assert_allclose(np.angle(result[0, 0]), phase, atol=1e-6)

    def test_complex_dtype(self) -> None:
        data = np.ones((4, 4), dtype=np.complex64)
        result = multilook_complex(data, looks_y=2, looks_x=2)
        assert result.dtype == np.complex64


class TestMultilookCovariance:
    def test_processes_real_and_complex(self) -> None:
        cov = {
            "HHHH": np.ones((8, 8), dtype=np.float32),
            "HHHV": np.ones((8, 8), dtype=np.complex64) * (1 + 0.5j),
        }
        result = multilook_covariance(cov, looks_y=2, looks_x=2)
        assert result["HHHH"].shape == (4, 4)
        assert result["HHHV"].shape == (4, 4)
        assert not np.iscomplexobj(result["HHHH"])
        assert np.iscomplexobj(result["HHHV"])


class TestEstimateEnl:
    def test_output_shape(self) -> None:
        data = np.random.default_rng(42).exponential(1.0, size=(50, 50)).astype(np.float32)
        enl = estimate_enl(data, window=7)
        assert enl.shape == (50, 50)

    def test_constant_input_high_enl(self) -> None:
        data = np.ones((30, 30), dtype=np.float32)
        enl = estimate_enl(data, window=5)
        # Constant → variance ~0 → ENL very high
        assert np.all(enl > 100)

    def test_dtype(self) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        enl = estimate_enl(data, window=5)
        assert enl.dtype == np.float32


class TestComputeLookFactors:
    def test_basic(self) -> None:
        ly, lx = compute_look_factors(native_az=7.0, native_rg=3.0, target_posting=80.0)
        assert ly == 11  # round(80/7)
        assert lx == 27  # round(80/3)

    def test_minimum_one(self) -> None:
        ly, lx = compute_look_factors(native_az=100.0, native_rg=100.0, target_posting=30.0)
        assert ly >= 1
        assert lx >= 1
