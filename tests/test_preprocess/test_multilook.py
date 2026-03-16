"""Tests for nice_sar.preprocess.multilook."""

from __future__ import annotations

import numpy as np

from nice_sar.preprocess.multilook import multilook


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
