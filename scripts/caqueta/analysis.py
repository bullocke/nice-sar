"""Methods for separating disturbance decorrelation from normal forest decorrelation.

The problem: an L-band HH coherence value for a forest pixel over 12-24 days is
already low (temporal and volume decorrelation) and varies strongly from pair to
pair with weather. A clearing between the two dates of a pair should push it
lower still, toward the estimator floor. This module provides the pieces used
to test that:

1. ``hv_event_dating``: date each clearing independently of coherence, from the
   step down in HV backscatter (forest ~-10 dB -> pasture ~-12.5 dB).
2. ``forest_normalize``: remove pair-to-pair variation that is common to the
   whole scene (rain, wind) by subtracting each pair's stable-forest median.
3. ``classify_pairs``: place each pair before, inside, or after a pixel's event.
4. ``spanning_test``: compare in-event coherence with the pixel's own pre-event
   history against a matched null (see the function docstring).
5. ``coherence_floor``: expected coherence magnitude when the true coherence is
   zero, for a given number of looks (the lowest value a pair can read).
6. Patch and case utilities for the case-study figures.

Day numbers are days since ``config.EPOCH`` throughout.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import config
import numpy as np
from data import Dataset, DateStack, PairStack, S2Stack
from scipy import ndimage
from scipy.special import gammaln
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


# --- Coherence estimator floor -------------------------------------------------------


def coherence_floor(n_looks: int) -> float:
    """Expected |coherence| estimate when the true coherence is zero.

    For ``N`` independent looks, E|γ̂| = (√π / 2) · Γ(N) / Γ(N + ½). Coherence
    estimated from few looks is biased high, so even a completely decorrelated
    pair never reads zero. Looks in the NISAR GUNW products are oversampled, so
    the effective N is somewhat smaller and the true floor somewhat higher.

    Args:
        n_looks: Number of looks (``config.LOOKS``: 18 at 20 m, 112 at 80 m).
    """
    return float(np.sqrt(np.pi) / 2 * np.exp(gammaln(n_looks) - gammaln(n_looks + 0.5)))


# --- 1. HV event dating --------------------------------------------------------------


@dataclass
class EventDating:
    """Per-pixel HV step and the acquisition bracket in which it occurs.

    ``t0`` / ``t1`` are the last HV date before and the first HV date after the
    step (NaN where no step could be fitted); ``step_db`` is mean(HV before) minus
    mean(HV after), so a clearing gives a positive step.
    """

    t0: np.ndarray
    t1: np.ndarray
    step_db: np.ndarray
    hv_smooth: np.ndarray  # (T, H, W) boxcar-filtered HV, dB


def smooth_db(stack: DateStack, size: int = config.HV_BOXCAR_PX) -> np.ndarray:
    """Boxcar-filter each image in linear power, then return dB.

    Averaging in linear power (not dB) is the correct way to reduce speckle.
    NaNs are filled with the image median first so they do not spread.
    """
    out = np.empty_like(stack.values)
    for i, img in enumerate(stack.values):
        lin = 10 ** (img / 10)
        lin = np.where(np.isfinite(lin), lin, np.nanmedian(lin))
        out[i] = 10 * np.log10(ndimage.uniform_filter(lin, size))
    return out


def hv_event_dating(hv: DateStack) -> EventDating:
    """Date an abrupt HV decrease per pixel with a single-step (change-point) fit.

    For each candidate split between consecutive HV dates i and i+1 (keeping at
    least ``HV_MIN_DATES_EACH_SIDE`` dates on each side), compute
    ``mean(HV[:i+1]) - mean(HV[i+1:])`` and keep the split with the largest step.
    This dating uses backscatter only, so testing coherence against it is not
    circular.
    """
    smooth = smooth_db(hv)
    n = len(hv.days)
    k = config.HV_MIN_DATES_EACH_SIDE
    best = np.full(smooth.shape[1:], -np.inf, dtype="float32")
    idx = np.full(smooth.shape[1:], -1)
    for i in range(k - 1, n - k):
        step = smooth[: i + 1].mean(0) - smooth[i + 1 :].mean(0)
        better = step > best
        best[better] = step[better]
        idx[better] = i
    valid = idx >= 0
    t0 = np.where(valid, hv.days[np.clip(idx, 0, n - 1)], np.nan)
    t1 = np.where(valid, hv.days[np.clip(idx + 1, 0, n - 1)], np.nan)
    return EventDating(t0.astype(float), t1.astype(float), best, smooth)


def event_pixels(ds: Dataset, ev: EventDating) -> np.ndarray:
    """Core clearing pixels with a confident HV step near their RADD date.

    - RADD high-confidence disturbance inside the series (``masks['disturbed']``)
    - at least ``CORE_EROSION_PX`` pixels from the edge of any cleared area, so
      80 m cells are not mixed with forest
    - HV step >= ``HV_STEP_MIN_DB``
    - HV bracket ends within ``HV_RADD_MAX_OFFSET_D`` days of the RADD alert
    """
    dist = ds.masks["disturbed"]
    cleared = dist | (ds.radd.alert_date > -9999)
    core = ndimage.binary_erosion(cleared, iterations=config.CORE_EROSION_PX) & dist
    near = np.abs(ev.t1 - ds.radd.alert_date) <= config.HV_RADD_MAX_OFFSET_D
    return core & (ev.step_db >= config.HV_STEP_MIN_DB) & near


# --- 2. Forest normalization ---------------------------------------------------------


def forest_normalize(pairs: PairStack, stable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each pair's stable-forest median coherence.

    Returns ``(anomaly, forest_median)``: anomaly has the same shape as
    ``pairs.values``; forest_median has one value per pair.
    """
    med = np.array([np.nanmedian(v[stable]) for v in pairs.values])
    return pairs.values - med[:, None, None], med


# --- 3. Pair classification ----------------------------------------------------------


def classify_pairs(pairs: PairStack, t0: np.ndarray, t1: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean (P, H, W) masks placing each pair relative to each pixel's bracket.

    - ``pre``: pair ends at least ``PRE_PAIR_BUFFER_D`` days before t0 (forest)
    - ``inside``: pair lies within [t0, t1] (one of these spans the clearing)
    - ``post1``: first pair starting at or after t1 (freshly cleared)
    - ``post2``: later pairs (established clearing)
    """
    r = pairs.ref[:, None, None]
    s = pairs.sec[:, None, None]
    return {
        "pre": s <= t0[None] - config.PRE_PAIR_BUFFER_D,
        "inside": (r >= t0[None]) & (s <= t1[None]),
        "post1": (r >= t1[None]) & (r < t1[None] + 24),
        "post2": r >= t1[None] + 24,
    }


# --- 4. Spanning-pair test -----------------------------------------------------------


@dataclass
class SpanningResult:
    """Outcome of ``spanning_test`` for one coherence resolution."""

    kind: str
    n_pixels: int
    n_one_pair_brackets: int
    n_two_pair_brackets: int
    median_observed: float
    median_null: float
    auc: float
    null_p05: float
    frac_below_null_p05: float
    same_pixel_noise_sd: float
    class_medians: dict[str, float]
    single_pairs: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def spanning_test(
    ds: Dataset, ev: EventDating, events: np.ndarray, kind: str
) -> tuple[SpanningResult, dict[str, np.ndarray]]:
    """Is coherence lower in the pair that spans a clearing than the pixel's own history?

    Steps, per event pixel with at least ``MIN_PRE_PAIRS`` pre-event pairs:

    1. anomaly = coherence minus the same pair's stable-forest median
    2. baseline = mean anomaly over the pixel's pre-event pairs
    3. observed = min(anomaly - baseline) over the pairs inside the HV bracket.
       A 24-day bracket usually holds two 12-day pairs; exactly one spans the
       clearing, the other is still forest or already cleared, so the minimum
       picks the spanning pair.
    4. matched null = the minimum of the same number of the pixel's own
       pre-event anomalies, each taken leave-one-out (so it is not compared with
       a mean that includes itself). Taking a minimum in both cases removes the
       bias of selecting the lowest value.

    AUC is the probability that an observed value is lower than a null value
    (0.5 = no signal, 1 = perfect separation).

    Returns the result and the per-pixel arrays (for plotting).
    """
    pairs: PairStack = getattr(ds, kind)
    anom, forest_med = forest_normalize(pairs, ds.masks["stable_forest"])
    cls = classify_pairs(pairs, ev.t0, ev.t1)
    rng = np.random.default_rng(config.SEED)

    observed, null, n_inside, noise = [], [], [], []
    for y, x in zip(*np.nonzero(events), strict=True):
        a = anom[:, y, x]
        finite = np.isfinite(a)
        pre = np.flatnonzero(cls["pre"][:, y, x] & finite)
        ins = np.flatnonzero(cls["inside"][:, y, x] & finite)
        if len(pre) < config.MIN_PRE_PAIRS or len(ins) == 0:
            continue
        baseline = a[pre].mean()
        loo = a[pre] - (a[pre].sum() - a[pre]) / (len(pre) - 1)
        observed.append((a[ins] - baseline).min())
        null.append(rng.choice(loo, len(ins), replace=False).min())
        n_inside.append(len(ins))
        noise.append(rng.choice(loo))
    observed = np.array(observed)
    null = np.array(null)
    n_inside = np.array(n_inside)

    auc = mannwhitneyu(null, observed).statistic / (len(observed) * len(null))
    p05 = float(np.percentile(null, 5))

    # Class medians of (anomaly - own baseline), pooled over pixel-pairs
    baseline_all = np.nanmean(np.where(cls["pre"], anom, np.nan), 0)
    ok = events & ((cls["pre"] & np.isfinite(anom)).sum(0) >= config.MIN_PRE_PAIRS)
    class_medians = {}
    for name, mask in cls.items():
        v = np.where(mask & ok[None], anom - baseline_all[None], np.nan)
        class_medians[name] = float(np.nanmedian(v))

    # Cleanest test: single 24-day pairs that fill a whole bracket
    single = []
    for p in np.flatnonzero(pairs.span == 24):
        m = ok & (ev.t0 == pairs.ref[p]) & (ev.t1 == pairs.sec[p])
        if m.sum() < 50:
            continue
        single.append(
            {
                "pair": f"{config.to_date(pairs.ref[p])}/{config.to_date(pairs.sec[p])}",
                "n_pixels": int(m.sum()),
                "median_coherence": float(np.nanmedian(pairs.values[p][m])),
                "forest_median": float(forest_med[p]),
                "median_anomaly_vs_own_baseline": float(np.nanmedian(anom[p][m] - baseline_all[m])),
            }
        )

    result = SpanningResult(
        kind=kind,
        n_pixels=len(observed),
        n_one_pair_brackets=int((n_inside == 1).sum()),
        n_two_pair_brackets=int((n_inside == 2).sum()),
        median_observed=float(np.median(observed)),
        median_null=float(np.median(null)),
        auc=float(auc),
        null_p05=p05,
        frac_below_null_p05=float((observed < p05).mean()),
        same_pixel_noise_sd=float(np.std(noise)),
        class_medians=class_medians,
        single_pairs=single,
    )
    logger.info(
        "%s: n=%d, observed %.3f vs null %.3f, AUC %.2f",
        kind,
        result.n_pixels,
        result.median_observed,
        result.median_null,
        result.auc,
    )
    return result, {"observed": observed, "null": null, "n_inside": n_inside}


# --- 6. Patches and case studies -----------------------------------------------------
#
# Coherence-based disturbance signals follow a sequence: a small dip from early
# degradation, a large dip when the forest is felled or the land is burned, then a
# large, sustained rise once the surface is non-forest. Dips are therefore measured
# against the case's OWN recent history (after removing weather with the stable-
# forest reference), not against intact forest: burned pasture can fall a long way
# while staying at forest level.


@dataclass
class Patch:
    """A clearing, burn, degradation patch, or control used as a case study."""

    case_id: str
    category: str
    rows: np.ndarray  # pixel row indices of the outline
    cols: np.ndarray  # pixel col indices
    delineation: str  # "Sentinel-2", "RADD seed", or "control"
    state_before: str  # "forest", "non-forest", or "unknown"
    t0: float  # HV bracket start (NaN for controls)
    t1: float
    radd_day: float
    hv_step_db: float
    optical_start: float  # optical clearing interval from NBR (NaN if none)
    optical_end: float
    dips: str  # flagged coherence dips: "YYYY-MM-DD/YYYY-MM-DD:sigma;..."
    deepest_sigma: float  # deepest own-history change in 80 m coherence, in sigma
    event_start: float  # earliest flagged dip or HV bracket start
    event_end: float
    spanning_forest_coh: float  # lowest forest coherence in pairs spanning the event
    noise_sd: float  # forest level noise (80 m) for this area
    change_sd: float  # forest own-history change noise (80 m) for this area
    rep_pixel: tuple[int, int]

    @property
    def n_px(self) -> int:
        return len(self.rows)

    @property
    def area_ha(self) -> float:
        return self.n_px * config.PIXEL_M**2 / 1e4

    @property
    def center(self) -> tuple[int, int]:
        return int(np.round(self.rows.mean())), int(np.round(self.cols.mean()))


def patch_series(ds: Dataset, rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
    """Patch-mean time series (mean in linear power for backscatter)."""
    out = {}
    for name, stack in (("HH", ds.hh), ("HV", ds.hv)):
        lin = 10 ** (stack.values[:, rows, cols] / 10)
        out[name] = 10 * np.log10(np.nanmean(lin, 1))
    for name in ("coh20", "coh80"):
        out[name] = np.nanmean(getattr(ds, name).values[:, rows, cols], 1)
    return out


_FOREST_CACHE: dict[int, dict[str, np.ndarray]] = {}
_NOISE_CACHE: dict[tuple, tuple[float, float]] = {}


def forest_reference(ds: Dataset) -> dict[str, np.ndarray]:
    """Stable-forest median per date (HH, HV, dB) and per pair (coh20, coh80).

    Used to remove variation common to the whole scene (e.g. canopy moisture after
    rain; the 21 Dec - 2 Jan pair is low everywhere), and drawn as the reference
    in the figures.
    """
    key = id(ds)
    if key not in _FOREST_CACHE:
        sf = ds.masks["stable_forest"]
        _FOREST_CACHE[key] = {
            name: np.array([np.nanmedian(v[sf]) for v in getattr(ds, attr).values])
            for name, attr in (("HH", "hh"), ("HV", "hv"), ("coh20", "coh20"), ("coh80", "coh80"))
        }
    return _FOREST_CACHE[key]


def own_history_change(delta: np.ndarray) -> np.ndarray:
    """Change of each pair from the median of the case's previous pairs.

    ``delta`` is the case's coherence minus the stable-forest median for the same
    pair (weather removed). The baseline is the median of up to
    ``CHANGE_BASELINE_PAIRS`` preceding pairs (at least two), so the change
    measures a departure from the case's own recent state, whatever that state
    is (forest, pasture, or regrowth). NaN for the first two pairs.
    """
    out = np.full(len(delta), np.nan)
    k = config.CHANGE_BASELINE_PAIRS
    for i in range(2, len(delta)):
        prev = delta[max(0, i - k) : i]
        prev = prev[np.isfinite(prev)]
        if prev.size >= 2:
            out[i] = delta[i] - np.median(prev)
    return out


def forest_noise(ds: Dataset, n_px: int, kind: str = "coh80") -> tuple[float, float]:
    """Noise of the level and of the own-history change for an area of ``n_px``.

    Samples ``NOISE_SAMPLES`` random squares of about ``n_px`` pixels lying
    entirely in stable forest. For each square, computes its forest-normalized
    coherence per pair and the own-history change. Returns the pooled standard
    deviations ``(level_sd, change_sd)``: how far an intact-forest area of this
    size wanders from the forest reference, and from its own recent median, by
    chance. Both barely shrink with area because forest coherence varies
    coherently across space, not only as estimation noise.
    """
    side = max(2, int(round(np.sqrt(n_px))))
    key = (id(ds), side, kind)
    if key not in _NOISE_CACHE:
        rng = np.random.default_rng(config.SEED)
        fits = ndimage.binary_erosion(ds.masks["stable_forest"], structure=np.ones((side, side)))
        ys, xs = np.nonzero(fits)
        pick = rng.choice(len(ys), min(config.NOISE_SAMPLES, len(ys)), replace=False)
        ref = forest_reference(ds)[kind]
        vals = getattr(ds, kind).values
        h = side // 2
        levels, changes = [], []
        for y, x in zip(ys[pick], xs[pick], strict=True):
            d = np.nanmean(vals[:, y - h : y - h + side, x - h : x - h + side], axis=(1, 2)) - ref
            levels.append(d)
            changes.append(own_history_change(d))
        _NOISE_CACHE[key] = (
            float(np.nanstd(np.stack(levels))),
            float(np.nanstd(np.stack(changes))),
        )
    return _NOISE_CACHE[key]


def forest_noise_sd(ds: Dataset, n_px: int, kind: str = "coh80") -> float:
    """Level noise only (see ``forest_noise``)."""
    return forest_noise(ds, n_px, kind)[0]


def hv_step(ds: Dataset, series: dict[str, np.ndarray]) -> dict:
    """Single-step fit to the forest-normalized patch-mean HV series.

    Returns ``step`` (mean before - mean after, dB), the HV dates ``t0``/``t1``
    on either side of the best split, and ``abrupt_frac``: the share of the step
    carried by the single interval at the split.
    """
    hv = series["HV"] - forest_reference(ds)["HV"]
    days = ds.hv.days
    k = config.HV_MIN_DATES_EACH_SIDE
    step, i = max((hv[: i + 1].mean() - hv[i + 1 :].mean(), i) for i in range(k - 1, len(hv) - k))
    abrupt = float((hv[i] - hv[i + 1]) / step) if step > 0 else np.nan
    return {
        "t0": float(days[i]),
        "t1": float(days[i + 1]),
        "step": float(step),
        "abrupt_frac": abrupt,
    }


def case_nbr(
    s2: S2Stack, rows: np.ndarray, cols: np.ndarray, shape
) -> tuple[np.ndarray, np.ndarray]:
    """Case-mean Sentinel-2 NBR on dates usable for the case.

    A date counts if the area around the case (bounding box + 10 px) is clear and
    haze-free and at least 80% of the case pixels are clear.
    """
    win = (
        slice(max(rows.min() - 10, 0), min(rows.max() + 11, shape[0])),
        slice(max(cols.min() - 10, 0), min(cols.max() + 11, shape[1])),
    )
    days, vals = [], []
    for t in range(len(s2.days)):
        if not s2.usable(t, win):
            continue
        v = s2.nbr[t][rows, cols]
        if np.isfinite(v).mean() >= config.S2_MIN_CLEAR:
            days.append(s2.days[t])
            vals.append(float(np.nanmean(v)))
    return np.array(days), np.array(vals)


def optical_drop(days: np.ndarray, nbr: np.ndarray) -> tuple[float, float]:
    """Optical clearing interval: last forest-like NBR date before the first cleared date.

    Finds the first date NBR is <= ``NBR_CLEARED_MAX`` after having been forest-like
    (>= ``NBR_FOREST_MIN``), and returns (last forest-like date before it, that
    date). Later regrowth back to forest-like values is ignored. NaN, NaN if the
    case is never forest-like, or never cleared afterwards.
    """
    seen_forest = None
    for i in range(len(days)):
        if nbr[i] >= config.NBR_FOREST_MIN:
            seen_forest = i
        elif nbr[i] <= config.NBR_CLEARED_MAX and seen_forest is not None:
            return float(days[seen_forest]), float(days[i])
    return np.nan, np.nan


def delineate_clearing(
    s2: S2Stack, rows: np.ndarray, cols: np.ndarray, t0: float, t1: float, shape
) -> tuple[np.ndarray, np.ndarray] | None:
    """Outline the whole disturbed area around a seed from a Sentinel-2 NBR drop.

    RADD alerts fragment disturbances (dates are patchy and lag), so the case
    outline is taken from Sentinel-2 instead:

    1. "before" = the last two usable images at least ``PRE_GAP_D`` days before
       the HV bracket; "after" = the first two usable images after it (usable =
       at least 80% clear and haze-free around the seed).
    2. A pixel is disturbed if its NBR dropped by at least ``NBR_DROP_MIN`` (max
       of the two before, min of the two after, which bridges small cloud gaps)
       and ended low (<= ``NBR_CLEARED_MAX``). No forest-like starting value is
       required, so burns of already-cleared land are outlined too.
    3. Remove isolated pixels (morphological opening) and keep the 8-connected
       regions touching the seed.

    Returns global (rows, cols), or ``None`` if there are not enough usable
    images or the outline is too small or too large.
    """
    pad = config.DELINEATE_PAD_PX
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, shape[1])
    win = (slice(r0, r1), slice(c0, c1))
    seed_win = (
        slice(max(rows.min() - 10, 0), rows.max() + 11),
        slice(max(cols.min() - 10, 0), cols.max() + 11),
    )
    usable = [t for t in range(len(s2.days)) if s2.usable(t, seed_win)]
    pre = [t for t in usable if s2.days[t] <= t0 - config.PRE_GAP_D][-2:]
    post = [t for t in usable if s2.days[t] >= t1][:2]
    if not pre or not post:
        return None
    with np.errstate(all="ignore"):
        before = np.nanmax(s2.nbr[pre][:, win[0], win[1]], axis=0)
        after = np.nanmin(s2.nbr[post][:, win[0], win[1]], axis=0)
    disturbed = (after <= config.NBR_CLEARED_MAX) & (before - after >= config.NBR_DROP_MIN)
    disturbed = ndimage.binary_opening(disturbed)
    labels, _ = ndimage.label(disturbed, structure=np.ones((3, 3)))
    seed = np.zeros(disturbed.shape, bool)
    seed[rows - r0, cols - c0] = True
    ids = np.unique(labels[ndimage.binary_dilation(seed)])
    ids = ids[ids > 0]
    obj = np.isin(labels, ids)
    if not config.MIN_PATCH_PX <= obj.sum() <= config.MAX_OBJECT_PX:
        return None
    rr, cc = np.nonzero(obj)
    return rr + r0, cc + c0


def describe(ds: Dataset, s2: S2Stack, rows: np.ndarray, cols: np.ndarray) -> dict:
    """Everything used to categorize and plot a case (see ``select_cases``).

    - HV step and bracket (forest-normalized case-mean HV)
    - own-history change of 80 m coherence per pair and the flagged dips
      (change below ``-DIP_SIGMA`` x the forest change noise for this area)
    - the event: from the earliest flagged dip (or the HV bracket, if no dip) to
      the end of the deepest dip (or the bracket)
    - land state before the event: forest if the case-mean NBR before it is
      >= ``NBR_FOREST_MIN``, non-forest if lower; if there is no usable optical
      image, non-forest when coherence sat more than 2 sigma above forest
    - optical clearing interval, and the lowest forest coherence among pairs
      spanning the clearing (a clearing in a pair where the forest itself is near
      the floor cannot produce a dip)
    """
    s = patch_series(ds, rows, cols)
    hv = hv_step(ds, s)
    level_sd, change_sd = forest_noise(ds, len(rows))
    ref = forest_reference(ds)["coh80"]
    pairs = ds.coh80
    delta = s["coh80"] - ref
    change = own_history_change(delta)
    flagged = np.flatnonzero(change < -config.DIP_SIGMA * change_sd)
    deepest = int(np.nanargmin(change)) if np.isfinite(change).any() else None

    if flagged.size:
        main = int(flagged[np.argmin(change[flagged])])
        event_start = float(pairs.ref[flagged].min())
        event_end = float(pairs.sec[main])
    else:
        main = None
        event_start, event_end = hv["t0"], hv["t1"]

    nd, nv = case_nbr(s2, rows, cols, ds.grid.shape)
    opt0, opt1 = optical_drop(nd, nv)
    before_nbr = nv[nd < event_start]
    if before_nbr.size:
        state = "forest" if np.median(before_nbr) >= config.NBR_FOREST_MIN else "non-forest"
    else:
        pre_level = delta[pairs.sec <= event_start]
        pre_level = pre_level[np.isfinite(pre_level)]
        if pre_level.size >= 2:
            state = "non-forest" if np.median(pre_level) > 2 * level_sd else "forest"
        else:
            state = "unknown"

    span_lo, span_hi = (opt0, opt1) if np.isfinite(opt0) else (hv["t0"], hv["t1"])
    spanning = (pairs.ref < span_hi) & (pairs.sec > span_lo)
    return {
        "hv": hv,
        "delta": delta,
        "change": change,
        "flagged": flagged,
        "main": main,
        "deepest_sigma": float(change[deepest] / change_sd) if deepest is not None else np.nan,
        "event_start": event_start,
        "event_end": event_end,
        "state": state,
        "optical": (opt0, opt1),
        "cleared": np.isfinite(opt0) or hv["step"] >= config.PATCH_STEP_DB,
        "spanning_forest_coh": float(ref[spanning].min()) if spanning.any() else np.nan,
        "level_sd": level_sd,
        "change_sd": change_sd,
    }


CATEGORY_ORDER = (
    "forest_clearing",
    "cleared_land_disturbance",
    "degradation",
    "clearing_in_low_coherence_pair",
    "clearing_without_dip",
    "radd_only",
)


def categorize(m: dict) -> str | None:
    """Assign a category from ``describe`` output (first match wins).

    - ``forest_clearing``: forest before, a flagged coherence dip, and evidence
      of clearing (optical NBR drop or HV step >= PATCH_STEP_DB)
    - ``cleared_land_disturbance``: non-forest before and a flagged dip (e.g.
      burning of felled vegetation or pasture)
    - ``degradation``: forest before, a flagged dip, no clearing evidence
    - ``clearing_in_low_coherence_pair``: forest cleared, no flagged dip, and the
      pairs spanning the clearing had forest coherence below LOW_FOREST_COH (the
      forest was already near the floor, so a dip cannot show)
    - ``clearing_without_dip``: forest cleared, no flagged dip, forest coherent
    - ``radd_only``: no flagged dip, no clearing evidence, HV step < PATCH_NO_STEP_DB
    """
    dip = m["flagged"].size > 0
    if dip and m["state"] == "forest" and m["cleared"]:
        return "forest_clearing"
    if dip and m["state"] == "non-forest":
        return "cleared_land_disturbance"
    if dip and m["state"] == "forest":
        return "degradation"
    if not dip and m["state"] == "forest" and m["cleared"]:
        if m["spanning_forest_coh"] < config.LOW_FOREST_COH:
            return "clearing_in_low_coherence_pair"
        return "clearing_without_dip"
    if not dip and not m["cleared"] and m["hv"]["step"] < config.PATCH_NO_STEP_DB:
        return "radd_only"
    return None


def select_cases(ds: Dataset, s2: S2Stack) -> list[Patch]:
    """Pick case studies with transparent, seeded rules.

    1. **Seeds**: connected groups of core RADD high-confidence pixels whose alert
       falls between the same two consecutive HV dates.
    2. **Outline**: each seed is grown with ``delineate_clearing`` (Sentinel-2 NBR
       drop); if that fails, the seed is kept.
    3. **Describe** (``describe``) and **categorize** (``categorize``).
    4. Within each category, Sentinel-2 outlines come first, then the strongest
       examples (deepest dip in sigma, or largest HV step for no-dip categories);
       case centres are at least ``CASE_MIN_SPACING_M`` apart and outlines never
       overlap. Controls are seeded 1 ha squares of stable forest and of land
       cleared before the series.
    """
    rng = np.random.default_rng(config.SEED)
    dist = ds.masks["disturbed"]
    cleared = dist | (ds.radd.alert_date > -9999)
    core = ndimage.binary_erosion(cleared, iterations=1) & dist
    bins = np.digitize(ds.radd.alert_date, ds.hv.days)

    candidates = []
    for b in np.unique(bins[core]):
        labels, n = ndimage.label(core & (bins == b))
        for lab in range(1, n + 1):
            rows, cols = np.nonzero(labels == lab)
            if not config.MIN_PATCH_PX <= len(rows) <= 1000:
                continue
            seed_hv = hv_step(ds, patch_series(ds, rows, cols))
            obj = delineate_clearing(s2, rows, cols, seed_hv["t0"], seed_hv["t1"], ds.grid.shape)
            how = "Sentinel-2"
            if obj is None:
                obj, how = (rows, cols), "RADD seed"
            candidates.append((obj[0], obj[1], how, describe(ds, s2, *obj)))
    logger.info(
        "%d candidates (%d outlined from Sentinel-2)",
        len(candidates),
        sum(c[2] == "Sentinel-2" for c in candidates),
    )

    def strength(m: dict, cat: str) -> float:
        if cat in ("clearing_in_low_coherence_pair", "clearing_without_dip"):
            return -m["hv"]["step"]
        if cat == "radd_only":
            return m["hv"]["step"]
        return m["deepest_sigma"]

    chosen: list[Patch] = []
    taken = np.zeros(ds.grid.shape, bool)

    def available(rows: np.ndarray, cols: np.ndarray) -> bool:
        r, c = rows.mean(), cols.mean()
        far = all(
            np.hypot(r - p.center[0], c - p.center[1]) * config.PIXEL_M >= config.CASE_MIN_SPACING_M
            for p in chosen
        )
        return far and not taken[rows, cols].any()

    for cat in CATEGORY_ORDER:
        pool = [c for c in candidates if categorize(c[3]) == cat]
        pool.sort(key=lambda c: (c[2] != "Sentinel-2", strength(c[3], cat)))
        k = 0
        for rows, cols, how, m in pool:
            if k == config.CASES_PER_CATEGORY:
                break
            if not available(rows, cols):
                continue
            k += 1
            chosen.append(_make_patch(ds, f"{cat}_{k}", cat, rows, cols, how, m))
            taken[rows, cols] = True
        logger.info("%s: %d candidates, %d chosen", cat, len(pool), k)

    for cat, mask in (
        ("stable_forest", ds.masks["stable_forest"]),
        ("pre_series_pasture", ds.masks["pre_series_pasture"]),
    ):
        full = ndimage.binary_erosion(mask, structure=np.ones((5, 5)))
        ys, xs = np.nonzero(full)
        k = 0
        for j in rng.permutation(len(ys)):
            if k == config.CASES_PER_CATEGORY:
                break
            rr, cc = np.meshgrid(
                np.arange(ys[j] - 2, ys[j] + 3), np.arange(xs[j] - 2, xs[j] + 3), indexing="ij"
            )
            rr, cc = rr.ravel(), cc.ravel()
            if not available(rr, cc):
                continue
            k += 1
            m = describe(ds, s2, rr, cc)
            chosen.append(_make_patch(ds, f"{cat}_{k}", cat, rr, cc, "control", m, control=True))
            taken[rr, cc] = True
    return chosen


def _make_patch(
    ds: Dataset, case_id: str, cat: str, rows, cols, how: str, m: dict, control: bool = False
) -> Patch:
    # representative pixel: HV step (3x3-smoothed) closest to the case median
    sub = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    hv = smooth_db(DateStack(ds.hv.values[:, sub[0], sub[1]], ds.hv.days))
    i = int(np.searchsorted(ds.hv.days, m["hv"]["t0"]))
    pix_step = hv[: i + 1].mean(0) - hv[i + 1 :].mean(0)
    vals = pix_step[rows - rows.min(), cols - cols.min()]
    j = int(np.nanargmin(np.abs(vals - np.nanmedian(vals))))
    radd = ds.radd.alert_date[rows, cols]
    in_series = radd[radd >= ds.first_day]
    radd = in_series if in_series.size else radd[radd > -9999]
    pairs = ds.coh80
    dips = ";".join(
        f"{config.to_date(pairs.ref[p])}/{config.to_date(pairs.sec[p])}:"
        f"{m['change'][p] / m['change_sd']:.1f}"
        for p in m["flagged"]
    )
    nan = np.nan
    return Patch(
        case_id=case_id,
        category=cat,
        rows=rows,
        cols=cols,
        delineation=how,
        state_before=m["state"],
        t0=nan if control else m["hv"]["t0"],
        t1=nan if control else m["hv"]["t1"],
        radd_day=float(np.median(radd)) if radd.size else nan,
        hv_step_db=m["hv"]["step"],
        optical_start=m["optical"][0],
        optical_end=m["optical"][1],
        dips="" if control else dips,
        deepest_sigma=m["deepest_sigma"],
        event_start=nan if control else m["event_start"],
        event_end=nan if control else m["event_end"],
        spanning_forest_coh=nan if control else m["spanning_forest_coh"],
        noise_sd=m["level_sd"],
        change_sd=m["change_sd"],
        rep_pixel=(int(rows[j]), int(cols[j])),
    )


def timing_sensitivity(
    ds: Dataset,
    ev: EventDating,
    events: np.ndarray,
    kind: str,
    baseline_gap_d: int = 36,
    min_baseline: int = 4,
) -> list[dict]:
    """Where relative to the HV bracket does coherence drop?

    Same matched-null design as ``spanning_test``, but the baseline uses only
    pairs ending at least ``baseline_gap_d`` days before the bracket, so the
    pair(s) just before the bracket can be tested too. Three windows are compared
    on the same pixels:

    - the pair(s) in the 24 days before the bracket (dip before the HV drop,
      e.g. felling before the slash is burned)
    - the pair(s) inside the bracket
    - both together

    Only pixels with at least as many baseline pairs as pairs in the widest
    window are used, so every window is tested on the same pixels.
    """
    pairs: PairStack = getattr(ds, kind)
    anom, _ = forest_normalize(pairs, ds.masks["stable_forest"])
    r = pairs.ref[:, None, None]
    s = pairs.sec[:, None, None]
    t0, t1 = ev.t0[None], ev.t1[None]
    windows = {
        "24 d before bracket": (r >= t0 - 24) & (s <= t0),
        "inside bracket": (r >= t0) & (s <= t1),
        "24 d before + inside": (r >= t0 - 24) & (s <= t1),
    }
    base = s <= t0 - baseline_gap_d
    widest = windows["24 d before + inside"]
    out = []
    for name, mask in windows.items():
        rng = np.random.default_rng(config.SEED)
        obs, null = [], []
        for y, x in zip(*np.nonzero(events), strict=True):
            a = anom[:, y, x]
            finite = np.isfinite(a)
            b = np.flatnonzero(base[:, y, x] & finite)
            wide = np.flatnonzero(widest[:, y, x] & finite)
            w = np.flatnonzero(mask[:, y, x] & finite)
            if len(b) < max(min_baseline, len(wide)) or len(w) == 0:
                continue
            loo = a[b] - (a[b].sum() - a[b]) / (len(b) - 1)
            obs.append((a[w] - a[b].mean()).min())
            null.append(rng.choice(loo, len(w), replace=False).min())
        obs, null = np.array(obs), np.array(null)
        out.append(
            {
                "window": name,
                "n_pixels": len(obs),
                "median_observed": float(np.median(obs)),
                "median_null": float(np.median(null)),
                "auc": float(mannwhitneyu(null, obs).statistic / (len(obs) * len(null))),
            }
        )
    return out


def local_ring(ds: Dataset, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Surrounding intact forest: a ring from RING_INNER_PX to RING_OUTER_PX around the case.

    The inner gap (one 80 m cell) keeps 80 m coherence cells that straddle the
    outline out of the ring; RADD-alerted pixels and non-forest are excluded.
    """
    mask = np.zeros(ds.grid.shape, bool)
    mask[rows, cols] = True
    ring = ndimage.binary_dilation(mask, iterations=config.RING_OUTER_PX)
    ring &= ~ndimage.binary_dilation(mask, iterations=config.RING_INNER_PX)
    return ring & (ds.radd.alert_date <= -9999) & (ds.radd.forest == 1)


def coherence_context(ds: Dataset, rows: np.ndarray, cols: np.ndarray, before_day: float) -> dict:
    """Case vs surrounding-forest coherence before the disturbance and at its minimum.

    For 80 m and 20 m: the mean case coherence over pairs ending on or before
    ``before_day`` (and the same for the surrounding ring), and the pair with the
    lowest case coherence, with the ring's value in that pair. Comparing with the
    local ring shows whether the case is darker than nearby forest, independent of
    scene-wide weather.
    """
    ring = local_ring(ds, rows, cols)
    out = {}
    for kind in ("coh80", "coh20"):
        pairs: PairStack = getattr(ds, kind)
        inside = np.nanmean(pairs.values[:, rows, cols], 1)
        around = np.array([np.nanmean(v[ring]) if ring.any() else np.nan for v in pairs.values])
        pre = pairs.sec <= before_day
        i = int(np.nanargmin(inside))
        out[kind] = {
            "n_pre": int(pre.sum()),
            "pre_inside": float(np.nanmean(inside[pre])) if pre.any() else np.nan,
            "pre_ring": float(np.nanmean(around[pre])) if pre.any() else np.nan,
            "min_inside": float(inside[i]),
            "min_ring": float(around[i]),
            "min_ref": int(pairs.ref[i]),
            "min_sec": int(pairs.sec[i]),
        }
    return out
