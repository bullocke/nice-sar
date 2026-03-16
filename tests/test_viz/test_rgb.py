"""Tests for nice_sar.viz.rgb composites."""

from __future__ import annotations

import numpy as np
import pytest

from nice_sar.viz.rgb import list_rgb_methods, make_rgb


class TestMakeRgb:
    @pytest.mark.parametrize("method", list_rgb_methods())
    def test_all_methods_produce_valid_output(
        self, method: str, hh_linear: np.ndarray, hv_linear: np.ndarray,
        hh_db: np.ndarray, hv_db: np.ndarray,
    ) -> None:
        from nice_sar.viz.rgb import _RGB_METHODS

        entry = _RGB_METHODS[method]
        if entry["input"] == "db":
            rgb, names = make_rgb(hh_db, hv_db, method=method)
        else:
            rgb, names = make_rgb(hh_linear, hv_linear, method=method)

        assert rgb.shape == (3, 100, 100)
        assert rgb.dtype == np.uint8
        assert len(names) == 3

    def test_invalid_method(self, hh_linear: np.ndarray, hv_linear: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown RGB method"):
            make_rgb(hh_linear, hv_linear, method="nonexistent")


class TestListMethods:
    def test_returns_sorted(self) -> None:
        methods = list_rgb_methods()
        assert methods == sorted(methods)
        assert len(methods) == 12
