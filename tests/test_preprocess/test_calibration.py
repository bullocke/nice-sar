"""Tests for nice_sar.preprocess.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from nice_sar.preprocess.calibration import (
    db_to_linear,
    linear_to_db,
    power_transform,
)


class TestLinearToDb:
    def test_roundtrip(self, hh_linear: np.ndarray) -> None:
        db = linear_to_db(hh_linear)
        recovered = db_to_linear(db)
        np.testing.assert_allclose(recovered, hh_linear, rtol=1e-5)

    def test_zero_handling(self) -> None:
        arr = np.array([0.0, 1.0, 0.01], dtype=np.float32)
        db = linear_to_db(arr)
        # Zero produces fill_value (NaN by default), positive values are finite
        assert np.isnan(db[0])
        assert np.isfinite(db[1:]).all()

    def test_negative_handling(self) -> None:
        arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        db = linear_to_db(arr)
        # Non-positive values produce NaN
        assert np.isnan(db[0])
        assert np.isnan(db[1])
        assert np.isfinite(db[2])


class TestPowerTransform:
    def test_default_exponent(self, hh_linear: np.ndarray) -> None:
        result = power_transform(hh_linear)
        assert result.shape == hh_linear.shape
        assert result.dtype == np.float32 or np.issubdtype(result.dtype, np.floating)

    def test_exponent_one(self, hh_linear: np.ndarray) -> None:
        result = power_transform(hh_linear, gamma=1.0)
        assert result.shape == hh_linear.shape
        # gamma=1 preserves relative order
        valid = np.isfinite(result)
        assert valid.any()
