"""Forest disturbance signals in L-band backscatter and coherence time series.

These methods come from an analysis of PROVISIONAL NISAR data over a
deforestation frontier in Caquetá, Colombia (frame 083 D 088, Nov 2025 to Sep
2026). They separate three things that all lower forest coherence:

1. **Normal forest decorrelation.** Over 12 days, L-band HH coherence of intact
   tropical forest is already low (about 0.2-0.5) and varies from pair to pair
   with rain and wind.
2. **Disturbance between the two dates of a pair.** Felling or burning lowers
   coherence below the area's own recent level.
3. **Change after disturbance.** Once the land is non-forest, coherence rises
   and stays well above forest, while HV backscatter drops by about 2 dB.

The building blocks are:

- :func:`coherence_floor`: the value a fully decorrelated pair still reads.
- :func:`hv_step_dating`: date an abrupt HV decrease with a single-step fit.
- :func:`forest_ring`: intact forest around an area, used as a local weather
  reference (rain is patchy, so nearby forest is a better reference than the
  whole scene).
- :func:`reference_series`: per-pair mean or median over a reference mask.
- :func:`own_history_change`: each pair's departure from the area's own
  recent pairs.
- :func:`forest_change_noise`: how much intact forest of the same size varies
  by chance, which calibrates a dip threshold in units of sigma.
- :func:`coherence_dips`: flag pairs that dip below the area's own history.
- :func:`disturbance_index`: an **experimental** forward-running Disturbance
  Index for whole images.

Arrays are ``(T, H, W)`` stacks (one image per date or pair) or ``(T,)`` series.
Dates can be any increasing numbers, such as days since an epoch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import ndimage
from scipy.special import gammaln

logger = logging.getLogger(__name__)

__all__ = [
    "DIResult",
    "DipResult",
    "HVStepResult",
    "NoiseEstimate",
    "boxcar_db",
    "coherence_dips",
    "coherence_floor",
    "disturbance_index",
    "forest_change_noise",
    "forest_ring",
    "hv_step_dating",
    "own_history_change",
    "reference_series",
]


# ---------------------------------------------------------------------------
# Coherence estimator floor
# ---------------------------------------------------------------------------


def coherence_floor(n_looks: int) -> float:
    """Expected coherence magnitude of a fully decorrelated pair.

    Coherence estimated from ``N`` independent looks is biased high, so even a
    pair with zero true coherence reads ``E|γ̂| = (√π / 2) · Γ(N) / Γ(N + ½)``.
    NISAR GUNW coherence uses 6 × 3 = 18 looks at 20 m and 16 × 7 = 112 looks at
    80 m, which gives floors of about 0.21 and 0.084. Intact tropical forest at
    20 m is close to its floor, which leaves little room for a disturbance dip.

    Args:
        n_looks: Number of independent looks.

    Returns:
        The expected coherence magnitude when the true coherence is zero.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be >= 1")
    return float(np.sqrt(np.pi) / 2 * np.exp(gammaln(n_looks) - gammaln(n_looks + 0.5)))


# ---------------------------------------------------------------------------
# HV step dating
# ---------------------------------------------------------------------------


@dataclass
class HVStepResult:
    """Single-step fit to an HV backscatter time series.

    Attributes:
        step_db: ``mean(HV before) - mean(HV after)`` at the best split, in dB.
            A clearing gives a positive step (forest about -10 dB, pasture about
            -12.5 dB in Caquetá).
        t0: Date of the last image before the step (NaN where no fit).
        t1: Date of the first image after the step (NaN where no fit).
        index: Index of the last image before the step (-1 where no fit).
        abrupt_frac: Share of the step carried by the single interval at the
            split (1 = all in one interval, lower = gradual).
    """

    step_db: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    index: np.ndarray
    abrupt_frac: np.ndarray


def boxcar_db(stack_db: np.ndarray, size: int = 3) -> np.ndarray:
    """Boxcar-filter each image of a dB stack in linear power, then return dB.

    Averaging in linear power, not dB, is the correct way to reduce speckle.
    NaNs are filled with the image median first so they do not spread.

    Args:
        stack_db: ``(T, H, W)`` backscatter in dB.
        size: Filter window in pixels.

    Returns:
        Filtered stack in dB, same shape as the input.
    """
    stack_db = np.asarray(stack_db)
    if not np.issubdtype(stack_db.dtype, np.floating):
        stack_db = stack_db.astype("float32")
    out = np.empty_like(stack_db)
    for i, img in enumerate(stack_db):
        lin = 10 ** (img / 10)
        lin = np.where(np.isfinite(lin), lin, np.nanmedian(lin))
        out[i] = 10 * np.log10(ndimage.uniform_filter(lin, size))
    return out


def hv_step_dating(
    hv_db: np.ndarray,
    dates: np.ndarray,
    min_dates_each_side: int = 2,
) -> HVStepResult:
    """Date an abrupt decrease in HV backscatter with a single-step fit.

    For each split between consecutive dates ``i`` and ``i + 1`` that keeps at
    least ``min_dates_each_side`` dates on each side, the step is
    ``mean(hv[:i+1]) - mean(hv[i+1:])``; the split with the largest step wins.
    Dating uses backscatter only, so it is independent of coherence and can be
    used to test coherence without circularity.

    Args:
        hv_db: HV backscatter in dB, ``(T,)`` for one series or ``(T, H, W)``
            for a stack (filter it first with :func:`boxcar_db` for pixels).
        dates: ``(T,)`` increasing acquisition dates.
        min_dates_each_side: Fewest dates allowed on either side of the split.

    Returns:
        :class:`HVStepResult` with arrays shaped like one image of ``hv_db``.
    """
    hv = np.asarray(hv_db)
    if not np.issubdtype(hv.dtype, np.floating):
        hv = hv.astype("float64")
    mean = np.nanmean if np.isnan(hv).any() else np.mean
    dates = np.asarray(dates, dtype="float64")
    n = hv.shape[0]
    if dates.shape != (n,):
        raise ValueError("dates must have one entry per image")
    k = min_dates_each_side
    shape = hv.shape[1:]
    best = np.full(shape, -np.inf)
    idx = np.full(shape, -1)
    for i in range(k - 1, n - k):
        step = mean(hv[: i + 1], 0) - mean(hv[i + 1 :], 0)
        better = step > best
        best = np.where(better, step, best)
        idx = np.where(better, i, idx)
    valid = idx >= 0
    safe = np.clip(idx, 0, n - 2)
    t0 = np.where(valid, dates[safe], np.nan)
    t1 = np.where(valid, dates[safe + 1], np.nan)
    at_split = (
        np.take_along_axis(hv, safe[None, ...], 0)[0]
        - np.take_along_axis(hv, (safe + 1)[None, ...], 0)[0]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        abrupt = np.where(valid & (best > 0), at_split / best, np.nan)
    return HVStepResult(
        step_db=np.where(valid, best, np.nan),
        t0=t0,
        t1=t1,
        index=idx,
        abrupt_frac=abrupt,
    )


# ---------------------------------------------------------------------------
# Weather references
# ---------------------------------------------------------------------------


def forest_ring(
    area: np.ndarray,
    forest: np.ndarray,
    inner_px: int = 4,
    outer_px: int = 15,
) -> np.ndarray:
    """Intact forest in a ring around an area, used as a local weather reference.

    The ring runs from ``inner_px`` to ``outer_px`` pixels outside the area. The
    default (80-300 m at 20 m pixels) leaves one 80 m coherence cell between the
    area and the ring, so no ring cell straddles the area's edge. Rain is
    patchy: forest within a few hundred metres shares the weather of the area
    much better than the scene-wide forest median does, which lowers the noise
    of coherence comparisons 2-3 fold.

    Args:
        area: Boolean ``(H, W)`` mask of the area (e.g. a clearing outline).
        forest: Boolean ``(H, W)`` mask of intact forest allowed in the ring.
        inner_px: Gap between the area and the ring, in pixels.
        outer_px: Outer edge of the ring, in pixels from the area.

    Returns:
        Boolean ``(H, W)`` ring mask.
    """
    area = np.asarray(area, bool)
    if not area.any():
        return np.zeros(area.shape, bool)
    rows, cols = np.nonzero(area)
    h, w = area.shape
    pad = outer_px + 1
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, h)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, w)
    crop = (slice(r0, r1), slice(c0, c1))
    local = area[crop]
    ring = ndimage.binary_dilation(local, iterations=outer_px)
    if inner_px > 0:
        ring &= ~ndimage.binary_dilation(local, iterations=inner_px)
    else:
        ring &= ~local
    ring &= np.asarray(forest, bool)[crop]
    out = np.zeros(area.shape, bool)
    out[crop] = ring
    return out


def reference_series(
    stack: np.ndarray,
    mask: np.ndarray,
    stat: Literal["median", "mean"] = "median",
) -> np.ndarray:
    """Per-image statistic of a stack over a reference mask.

    With the stable-forest mask of a whole scene this gives the scene reference
    (median). With a :func:`forest_ring` mask it gives the local reference
    (mean). Subtracting it removes variation shared with the reference forest,
    such as a rainy pair that lowers coherence everywhere.

    Args:
        stack: ``(T, H, W)`` stack.
        mask: Boolean ``(H, W)`` reference mask.
        stat: ``"median"`` or ``"mean"``.

    Returns:
        ``(T,)`` series; NaN where the mask has no valid pixels.
    """
    func = np.nanmedian if stat == "median" else np.nanmean
    vals = np.asarray(stack)[:, np.asarray(mask, bool)]
    out = np.full(vals.shape[0], np.nan)
    has = np.isfinite(vals).any(1)
    out[has] = func(vals[has], 1)
    return out


def own_history_change(
    series: np.ndarray,
    baseline_pairs: int = 3,
    min_baseline: int = 2,
) -> np.ndarray:
    """Change of each value from the median of the preceding values.

    Apply it to a weather-corrected series (area coherence minus a reference).
    The baseline is the median of up to ``baseline_pairs`` preceding finite
    values, so the change measures a departure from the area's own recent
    state, whatever that state is (forest, pasture or regrowth). Burning
    already-cleared land can drop coherence a long way while it stays at forest
    level; comparing with intact forest would miss that.

    Args:
        series: ``(T,)`` weather-corrected series.
        baseline_pairs: Number of preceding values in the baseline.
        min_baseline: Fewest finite preceding values needed.

    Returns:
        ``(T,)`` change; NaN where the baseline is too short.
    """
    series = np.asarray(series, dtype="float64")
    out = np.full(series.shape, np.nan)
    for i in range(min_baseline, len(series)):
        prev = series[max(0, i - baseline_pairs) : i]
        prev = prev[np.isfinite(prev)]
        if prev.size >= min_baseline:
            out[i] = series[i] - np.median(prev)
    return out


@dataclass
class NoiseEstimate:
    """Chance variation of intact-forest areas of a given size.

    Attributes:
        level_sd: SD of area coherence minus the reference.
        change_sd: SD of :func:`own_history_change` of that difference. Divide
            a change by this to express it in sigma.
        n_samples: Number of forest squares sampled.
        side_px: Side of each sampled square, in pixels.
    """

    level_sd: float
    change_sd: float
    n_samples: int
    side_px: int


def forest_change_noise(
    stack: np.ndarray,
    forest: np.ndarray,
    n_px: int,
    reference: Literal["scene", "ring"] = "ring",
    ring_forest: np.ndarray | None = None,
    n_samples: int = 150,
    baseline_pairs: int = 3,
    inner_px: int = 4,
    outer_px: int = 15,
    min_ring_px: int = 50,
    seed: int = 0,
) -> NoiseEstimate:
    """Estimate how much intact forest of a given size varies by chance.

    Samples random squares of about ``n_px`` pixels lying entirely inside
    ``forest``. For each square, the per-pair area mean minus the reference
    (scene-wide forest median, or the square's own forest ring) is one
    "level" series, and its :func:`own_history_change` is one "change" series.
    The pooled SDs say how far an undisturbed area wanders by chance, which
    calibrates the dip threshold: in Caquetá, 3 sigma against the ring flagged
    a false dip in 4-8% of intact-forest areas over a 10-month series.

    Args:
        stack: ``(P, H, W)`` coherence stack.
        forest: Boolean ``(H, W)`` stable-forest mask; squares are drawn from it
            and it defines the scene reference.
        n_px: Area of the case in pixels (the square side is ``sqrt(n_px)``).
        reference: ``"scene"`` or ``"ring"``.
        ring_forest: Forest allowed in rings (defaults to ``forest``).
        n_samples: Number of squares.
        baseline_pairs: Passed to :func:`own_history_change`.
        inner_px: Ring inner gap.
        outer_px: Ring outer edge.
        min_ring_px: Rings smaller than this fall back to the scene reference.
        seed: Random seed.

    Returns:
        :class:`NoiseEstimate`.
    """
    forest = np.asarray(forest, bool)
    ring_forest = forest if ring_forest is None else np.asarray(ring_forest, bool)
    side = max(2, int(round(np.sqrt(n_px))))
    rng = np.random.default_rng(seed)
    fits = ndimage.binary_erosion(forest, structure=np.ones((side, side)))
    ys, xs = np.nonzero(fits)
    if ys.size == 0:
        raise ValueError(f"No {side} x {side} square fits inside the forest mask")
    pick = rng.choice(ys.size, min(n_samples, ys.size), replace=False)
    scene = reference_series(stack, forest, "median")
    h = side // 2
    levels, changes = [], []
    for y, x in zip(ys[pick], xs[pick], strict=True):
        win = (slice(y - h, y - h + side), slice(x - h, x - h + side))
        ref = scene
        if reference == "ring":
            sq = np.zeros(forest.shape, bool)
            sq[win] = True
            ring = forest_ring(sq, ring_forest, inner_px, outer_px)
            if ring.sum() >= min_ring_px:
                ref = reference_series(stack, ring, "mean")
        d = np.nanmean(stack[:, win[0], win[1]], axis=(1, 2)) - ref
        levels.append(d)
        changes.append(own_history_change(d, baseline_pairs))
    return NoiseEstimate(
        level_sd=float(np.nanstd(np.stack(levels))),
        change_sd=float(np.nanstd(np.stack(changes))),
        n_samples=int(pick.size),
        side_px=side,
    )


# ---------------------------------------------------------------------------
# Coherence dips
# ---------------------------------------------------------------------------


@dataclass
class DipResult:
    """Coherence dips of one area against its own history.

    Attributes:
        delta: Area coherence minus the reference, per pair.
        change: :func:`own_history_change` of ``delta``.
        sigma: ``change / change_sd``.
        flagged: Boolean per pair, a dip below ``-threshold`` sigma.
        masked: Boolean per pair, reference forest too decorrelated to test
            (a clearing cannot lower coherence that is already near the floor).
        threshold: The sigma threshold used.
    """

    delta: np.ndarray
    change: np.ndarray
    sigma: np.ndarray
    flagged: np.ndarray
    masked: np.ndarray
    threshold: float

    @property
    def deepest(self) -> int | None:
        """Index of the most negative change, or None if there is none."""
        if not np.isfinite(self.sigma).any():
            return None
        return int(np.nanargmin(self.sigma))


def coherence_dips(
    area_series: np.ndarray,
    reference: np.ndarray,
    change_sd: float,
    threshold: float = 3.0,
    baseline_pairs: int = 3,
    low_forest_coh: float | None = 0.30,
) -> DipResult:
    """Flag pairs where an area's coherence dips below its own recent history.

    1. ``delta = area - reference`` removes weather shared with the reference
       forest (use a :func:`forest_ring` reference where possible).
    2. ``change = own_history_change(delta)``.
    3. A pair is flagged when ``change < -threshold * change_sd`` and the
       reference coherence is at least ``low_forest_coh``.

    In Caquetá, 80 m coherence of forest clearings fell 13-18 sigma below its
    own history with a ring reference, against 4-5 sigma with the scene-wide
    reference.

    Args:
        area_series: ``(P,)`` mean coherence of the area per pair.
        reference: ``(P,)`` reference forest coherence per pair.
        change_sd: Noise of the change for this area size
            (:attr:`NoiseEstimate.change_sd`).
        threshold: Dip threshold in sigma (3.0 with a ring reference, 2.5 with
            the scene reference).
        baseline_pairs: Passed to :func:`own_history_change`.
        low_forest_coh: Pairs with reference coherence below this are masked.
            ``None`` disables masking.

    Returns:
        :class:`DipResult`.
    """
    area_series = np.asarray(area_series, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    delta = area_series - reference
    change = own_history_change(delta, baseline_pairs)
    sigma = change / change_sd
    masked = (
        np.zeros(delta.shape, bool)
        if low_forest_coh is None
        else np.nan_to_num(reference, nan=0.0) < low_forest_coh
    )
    flagged = np.nan_to_num(sigma, nan=0.0) < -threshold
    flagged &= ~masked
    return DipResult(delta, change, sigma, flagged, masked, threshold)


# ---------------------------------------------------------------------------
# Experimental: forward-running Disturbance Index
# ---------------------------------------------------------------------------


@dataclass
class DIResult:
    """Output of :func:`disturbance_index`.

    Attributes:
        first: ``(H, W)`` index of the image that starts the confirmed run of
            exceedances, -1 where nothing was detected.
        n_forest: ``(T,)`` size of the monitored forest population per image.
        di: ``(T, H, W)`` Disturbance Index per image (NaN before ``start``).
    """

    first: np.ndarray
    n_forest: np.ndarray
    di: np.ndarray = field(repr=False)

    @property
    def detected(self) -> np.ndarray:
        """Boolean map of detected pixels."""
        return self.first >= 0

    def detection_date(self, dates: np.ndarray) -> np.ndarray:
        """Map of detection dates (NaN where nothing was detected)."""
        out = np.full(self.first.shape, np.nan)
        m = self.detected
        out[m] = np.asarray(dates, dtype="float64")[self.first[m]]
        return out


def _scene_z(v: np.ndarray, pop: np.ndarray, min_pop: int) -> np.ndarray:
    vals = v[pop & np.isfinite(v)]
    if vals.size < min_pop:
        return np.full(v.shape, np.nan)
    return np.asarray((v - vals.mean()) / vals.std())


def disturbance_index(
    stack: np.ndarray,
    forest: np.ndarray,
    start: int,
    method: Literal["scene", "pixmean"] = "pixmean",
    threshold: float = 3.0,
    n_consecutive: int = 2,
    sign: Literal[-1, 1] = -1,
    min_pop: int = 200,
) -> DIResult:
    """Forward-running Disturbance Index (DI). **Experimental.**

    Images from index ``start`` on are processed in date order. Each image is
    standardized against the current forest population (``forest`` minus
    pixels already detected):

    - ``"scene"``: ``(x - mean_F) / sd_F``, the classic DI.
    - ``"pixmean"``: subtract each pixel's own mean of the earlier non-exceeding
      images first, then standardize the anomalies over the forest. This
      removes fixed spatial patterns such as the scene-wide gradients seen in
      coherence.

    A pixel exceeds when ``sign * DI >= threshold`` and is detected after
    ``n_consecutive`` valid exceedances in a row (missing images neither count
    nor break the run). Detected pixels leave the forest population.

    Use ``sign=-1`` for drops (HV backscatter, coherence dips) and ``sign=+1``
    for rises (coherence after clearing). In Caquetá the best DI detector so far,
    80 m coherence rise with ``"pixmean"``, detected 26% of reference clearings
    on time, so treat the output as exploratory.

    Args:
        stack: ``(T, H, W)`` images (backscatter in dB or coherence).
        forest: Boolean ``(H, W)`` forest mask at the start of monitoring.
        start: Index of the first monitored image; earlier images only build
            pixel history.
        method: ``"scene"`` or ``"pixmean"``.
        threshold: Exceedance threshold in DI units.
        n_consecutive: Exceedances in a row needed to confirm.
        sign: -1 flags drops, +1 flags rises.
        min_pop: Fewest valid forest pixels needed to standardize an image.

    Returns:
        :class:`DIResult`.
    """
    stack = np.asarray(stack, dtype="float64")
    forest = np.asarray(forest, bool)
    n, *shape = stack.shape
    hist_n = np.zeros(shape)
    hist_s = np.zeros(shape)
    for i in range(start):
        ok = np.isfinite(stack[i])
        hist_n[ok] += 1
        hist_s[ok] += stack[i][ok]

    first = np.full(shape, -1, dtype=int)
    run = np.zeros(shape, dtype=int)
    run_start = np.full(shape, -1, dtype=int)
    n_forest = np.zeros(n, dtype=int)
    di_all = np.full(stack.shape, np.nan)

    for t in range(start, n):
        x = stack[t]
        valid = np.isfinite(x)
        pop = forest & (first < 0)
        n_forest[t] = int(pop.sum())
        if method == "scene":
            di = _scene_z(x, pop, min_pop)
        elif method == "pixmean":
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(hist_n > 0, hist_s / hist_n, np.nan)
            di = _scene_z(x - mean, pop, min_pop)
        else:
            raise ValueError(f"Unknown method {method!r}")
        di_all[t] = di
        has = pop & valid & np.isfinite(di)
        exc = has & (sign * di >= threshold)
        calm = has & ~exc
        run_start[exc & (run == 0)] = t
        run[exc] += 1
        confirmed = exc & (run >= n_consecutive)
        first[confirmed] = run_start[confirmed]
        run[calm] = 0
        add = (calm | (valid & ~forest)) & np.isfinite(x)
        hist_n[add] += 1
        hist_s[add] += x[add]

    logger.debug("DI %s: %d pixels detected", method, int((first >= 0).sum()))
    return DIResult(first=first, n_forest=n_forest, di=di_all.astype("float32"))
