"""Tests for nice_sar.analysis.disturbance."""

from __future__ import annotations

import numpy as np
import pytest

from nice_sar.analysis.disturbance import (
    boxcar_db,
    coherence_dips,
    coherence_floor,
    disturbance_index,
    forest_change_noise,
    forest_ring,
    hv_step_dating,
    own_history_change,
    reference_series,
)


class TestCoherenceFloor:
    def test_known_values(self) -> None:
        assert coherence_floor(112) == pytest.approx(0.0837, abs=1e-3)
        assert coherence_floor(18) == pytest.approx(0.2103, abs=1e-3)

    def test_decreases_with_looks(self) -> None:
        assert coherence_floor(4) > coherence_floor(18) > coherence_floor(112)

    def test_rejects_zero_looks(self) -> None:
        with pytest.raises(ValueError):
            coherence_floor(0)


class TestHVStep:
    def test_dates_injected_step(self) -> None:
        rng = np.random.default_rng(1)
        dates = np.arange(12) * 24.0
        stack = rng.normal(-10.0, 0.3, (12, 6, 6))
        stack[7:, 2:4, 2:4] -= 2.5  # clearing between image 6 and 7
        res = hv_step_dating(stack, dates)
        assert np.all(res.t0[2:4, 2:4] == dates[6])
        assert np.all(res.t1[2:4, 2:4] == dates[7])
        assert np.all(res.step_db[2:4, 2:4] > 2.0)
        assert np.all(res.abrupt_frac[2:4, 2:4] > 0.5)
        assert np.nanmax(res.step_db[0]) < 1.0

    def test_single_series(self) -> None:
        series = np.array([-10.0, -10.1, -9.9, -12.5, -12.4, -12.6])
        res = hv_step_dating(series, np.arange(6.0))
        assert res.t0 == 2.0 and res.t1 == 3.0
        assert res.step_db == pytest.approx(2.5, abs=0.05)

    def test_respects_min_dates_each_side(self) -> None:
        series = np.array([-10.0, -13.0, -13.0, -13.0, -13.0])
        res = hv_step_dating(series, np.arange(5.0), min_dates_each_side=2)
        assert res.index >= 1

    def test_date_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            hv_step_dating(np.zeros((4, 2, 2)), np.arange(3.0))

    def test_boxcar_averages_in_linear_power(self) -> None:
        img = np.full((1, 5, 5), -10.0)
        img[0, 2, 2] = 0.0  # 10x brighter in linear power
        out = boxcar_db(img, size=3)
        expected = 10 * np.log10((8 * 0.1 + 1.0) / 9)
        assert out[0, 2, 2] == pytest.approx(expected, abs=1e-4)


class TestForestRing:
    def test_geometry(self) -> None:
        area = np.zeros((41, 41), bool)
        area[20, 20] = True
        ring = forest_ring(area, np.ones_like(area), inner_px=4, outer_px=15)
        assert not ring[20, 20]
        assert not ring[20, 24]  # within the inner gap
        assert ring[20, 25] and ring[20, 35]
        assert not ring[20, 36]  # beyond the outer edge

    def test_excludes_non_forest(self) -> None:
        area = np.zeros((41, 41), bool)
        area[20, 20] = True
        forest = np.ones_like(area)
        forest[:, 30:] = False
        ring = forest_ring(area, forest)
        assert not ring[:, 30:].any()
        assert ring.any()

    def test_empty_area(self) -> None:
        assert not forest_ring(np.zeros((5, 5), bool), np.ones((5, 5), bool)).any()


class TestSeries:
    def test_reference_series(self) -> None:
        stack = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
        mask = np.zeros((3, 3), bool)
        mask[0, :2] = True
        np.testing.assert_allclose(reference_series(stack, mask, "mean"), [0.5, 9.5])

    def test_reference_series_all_nan(self) -> None:
        stack = np.full((2, 2, 2), np.nan)
        assert np.isnan(reference_series(stack, np.ones((2, 2), bool))).all()

    def test_own_history_change(self) -> None:
        change = own_history_change(np.array([0.0, 0.0, 0.0, -0.3, 0.0]), baseline_pairs=3)
        assert np.isnan(change[:2]).all()
        assert change[3] == pytest.approx(-0.3)
        assert change[4] == pytest.approx(0.0)

    def test_own_history_change_skips_nans(self) -> None:
        change = own_history_change(np.array([0.1, np.nan, 0.1, 0.1, -0.2]))
        assert change[4] == pytest.approx(-0.3)


def _synthetic_scene(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coherence stack with patchy rain on the west half and a clearing."""
    rng = np.random.default_rng(seed)
    p, h, w = 14, 80, 80
    weather = rng.normal(0.0, 0.08, p)
    stack = 0.35 + weather[:, None, None] + rng.normal(0, 0.03, (p, h, w))
    stack[6, :, :40] -= 0.15  # rain in the west half only
    clearing = np.zeros((h, w), bool)
    clearing[30:36, 10:16] = True
    stack[8][clearing] -= 0.2  # felling during pair 8
    stack[9:, clearing] += 0.3  # non-forest afterwards
    forest = ~clearing
    return np.clip(stack, 0, 1), forest, clearing


class TestDips:
    def test_ring_flags_clearing_not_rain(self) -> None:
        stack, forest, clearing = _synthetic_scene()
        ring = forest_ring(clearing, forest)
        ref = reference_series(stack, ring, "mean")
        area = reference_series(stack, clearing, "mean")
        noise = forest_change_noise(stack, forest, int(clearing.sum()), "ring", n_samples=40)
        res = coherence_dips(area, ref, noise.change_sd, threshold=3.0, low_forest_coh=None)
        assert res.flagged[8]
        assert not res.flagged[6]  # the rain pair is shared with the ring
        assert res.deepest == 8
        assert res.flagged.sum() == 1

    def test_ring_noise_below_scene_noise(self) -> None:
        stack, forest, _ = _synthetic_scene()
        ring = forest_change_noise(stack, forest, 36, "ring", n_samples=40)
        scene = forest_change_noise(stack, forest, 36, "scene", n_samples=40)
        assert ring.change_sd < scene.change_sd

    def test_low_forest_pairs_masked(self) -> None:
        ref = np.array([0.4, 0.4, 0.4, 0.2, 0.4])
        area = np.array([0.4, 0.4, 0.4, 0.0, 0.4])
        res = coherence_dips(area, ref, 0.02, low_forest_coh=0.3)
        assert res.masked[3] and not res.flagged.any()

    def test_noise_rejects_tiny_forest(self) -> None:
        stack = np.ones((3, 4, 4))
        with pytest.raises(ValueError):
            forest_change_noise(stack, np.zeros((4, 4), bool), 9)


class TestDisturbanceIndex:
    @pytest.mark.parametrize("method", ["scene", "pixmean"])
    def test_detects_sustained_drop(self, method: str) -> None:
        rng = np.random.default_rng(5)
        stack = rng.normal(-10.0, 0.3, (10, 40, 40))
        stack[6:, 5:10, 5:10] -= 3.0
        forest = np.ones((40, 40), bool)
        res = disturbance_index(stack, forest, start=3, method=method, threshold=3.0)
        assert res.detected[5:10, 5:10].mean() > 0.9
        assert np.all(res.first[5:10, 5:10][res.detected[5:10, 5:10]] == 6)
        assert res.detected.mean() < 0.03
        dates = np.arange(10) * 12.0
        assert np.nanmin(res.detection_date(dates)) >= 36.0

    def test_single_exceedance_not_confirmed(self) -> None:
        rng = np.random.default_rng(6)
        stack = rng.normal(0.3, 0.02, (8, 30, 30))
        stack[5, :5, :5] -= 0.3  # one-off dip
        res = disturbance_index(stack, np.ones((30, 30), bool), start=2, n_consecutive=2)
        assert not res.detected[:5, :5].any()

    def test_rise_detector(self) -> None:
        rng = np.random.default_rng(7)
        stack = rng.normal(0.3, 0.02, (8, 30, 30))
        stack[4:, :5, :5] += 0.3
        res = disturbance_index(stack, np.ones((30, 30), bool), start=2, sign=1)
        assert res.detected[:5, :5].all()

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError):
            disturbance_index(np.zeros((3, 20, 20)), np.ones((20, 20), bool), 1, method="x")
