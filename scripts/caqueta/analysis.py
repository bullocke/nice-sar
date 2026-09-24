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
from data import Dataset, DateStack, PairStack
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


@dataclass
class Patch:
    """A clearing (or control area) used as a case study."""

    case_id: str
    category: str
    rows: np.ndarray  # pixel row indices
    cols: np.ndarray  # pixel col indices
    t0: float  # HV bracket start (NaN for controls)
    t1: float
    radd_day: float
    hv_step_db: float
    hv_abrupt_frac: float
    coh80_z: float  # in-bracket min vs own pre-event baseline, in pre-event SDs
    coh80_diff: float  # same, in coherence units
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

    def window(self, min_size: int = 60, pad: int = 12) -> tuple[slice, slice]:
        """Square NISAR-grid window around the patch (at least ``min_size`` px)."""
        h = max(np.ptp(self.rows), np.ptp(self.cols)) + 2 * pad
        size = max(min_size, h)
        r, c = self.center
        return slice(r - size // 2, r - size // 2 + size), slice(
            c - size // 2, c - size // 2 + size
        )


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


def forest_reference(ds: Dataset) -> dict[str, np.ndarray]:
    """Stable-forest median per date (HH, HV, dB) and per pair (coh20, coh80).

    Used to remove variation common to the whole scene (e.g. canopy moisture after
    rain), and drawn as the gray reference in the figures.
    """
    key = id(ds)
    if key not in _FOREST_CACHE:
        sf = ds.masks["stable_forest"]
        _FOREST_CACHE[key] = {
            name: np.array([np.nanmedian(v[sf]) for v in getattr(ds, attr).values])
            for name, attr in (("HH", "hh"), ("HV", "hv"), ("coh20", "coh20"), ("coh80", "coh80"))
        }
    return _FOREST_CACHE[key]


def _patch_metrics(ds: Dataset, rows: np.ndarray, cols: np.ndarray) -> dict:
    """Describe a patch from its own forest-normalized patch-mean series.

    - ``step``: largest mean(before) - mean(after) HV step over all splits (dB)
    - ``t0``/``t1``: HV dates on either side of that split
    - ``abrupt_frac``: share of the step carried by the single interval at the
      split (1 = one-date drop; small = decline spread over several dates)
    - ``z``: 80 m coherence, lowest pair inside [t0, t1] relative to the patch's
      pre-event pairs, in units of their standard deviation (``diff`` is the
      same quantity in coherence units)
    """
    s = patch_series(ds, rows, cols)
    ref = forest_reference(ds)
    hv, days = s["HV"] - ref["HV"], ds.hv.days
    k = config.HV_MIN_DATES_EACH_SIDE
    steps = [(hv[: i + 1].mean() - hv[i + 1 :].mean(), i) for i in range(k - 1, len(hv) - k)]
    step, i = max(steps)
    t0, t1 = float(days[i]), float(days[i + 1])
    abrupt_frac = float((hv[i] - hv[i + 1]) / step) if step > 0 else np.nan
    pairs = ds.coh80
    a = s["coh80"] - ref["coh80"]
    pre = pairs.sec <= t0 - config.PRE_PAIR_BUFFER_D
    ins = (pairs.ref >= t0) & (pairs.sec <= t1)
    z = diff = np.nan
    if pre.sum() >= config.MIN_PRE_PAIRS and ins.any():
        diff = a[ins].min() - a[pre].mean()
        z = diff / a[pre].std(ddof=1)
    return {
        "t0": t0,
        "t1": t1,
        "step": float(step),
        "abrupt_frac": abrupt_frac,
        "z": float(z),
        "diff": float(diff),
    }


def select_cases(ds: Dataset) -> list[Patch]:
    """Pick case-study patches in six categories with transparent, seeded rules.

    Candidate clearings are connected groups of core RADD-disturbed pixels whose
    RADD alert falls between the same two consecutive HV dates (so each patch is
    one event). Each candidate is then described by its own forest-normalized
    patch-mean series (``_patch_metrics``). Patch-mean HV steps are smaller than
    pixel steps (a patch mixes fully and partly cleared pixels), so patch
    thresholds are lower than ``HV_STEP_MIN_DB``:

    - ``dip_detected``: step >= PATCH_STEP_DB, abrupt, and z <= DIP_Z
    - ``no_dip``: step >= PATCH_STEP_DB, abrupt, and z >= 0
    - ``gradual_decline``: step >= PATCH_STEP_DB but not abrupt (the split
      interval carries < GRADUAL_FRAC of the step), e.g. understory clearing
      weeks before felling
    - ``radd_only``: step < PATCH_NO_STEP_DB (RADD alert, little NISAR response)
    - ``stable_forest`` and ``pre_series_pasture``: seeded 5 x 5 px (1 ha) controls

    Within each category, patches are ranked by how strongly they express it and
    must be 1-40 ha and at least 1 km from every other case.
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
            candidates.append((rows, cols, _patch_metrics(ds, rows, cols)))
    logger.info("%d candidate clearings", len(candidates))

    def category(m: dict) -> str | None:
        big = m["step"] >= config.PATCH_STEP_DB
        abrupt = m["abrupt_frac"] >= config.ABRUPT_FRAC
        if big and abrupt and m["z"] <= config.DIP_Z:
            return "dip_detected"
        if big and abrupt and m["z"] >= 0:
            return "no_dip"
        if big and m["abrupt_frac"] < config.GRADUAL_FRAC:
            return "gradual_decline"
        if m["step"] < config.PATCH_NO_STEP_DB:
            return "radd_only"
        return None

    rank_key = {
        # rank by coherence units, not z: z is unstable with only 4-5 pre pairs
        "dip_detected": lambda m: m["diff"],
        "no_dip": lambda m: -m["step"],
        "gradual_decline": lambda m: -m["step"],
        "radd_only": lambda m: m["step"],
    }
    chosen: list[Patch] = []

    def far_enough(r: int, c: int) -> bool:
        return all(
            np.hypot(r - p.center[0], c - p.center[1]) * config.PIXEL_M >= 1000 for p in chosen
        )

    for cat, key in rank_key.items():
        pool = [c for c in candidates if category(c[2]) == cat]
        pool.sort(key=lambda c: key(c[2]))
        k = 0
        for rows, cols, m in pool:
            if k == config.CASES_PER_CATEGORY:
                break
            r, c = int(rows.mean()), int(cols.mean())
            if not far_enough(r, c):
                continue
            k += 1
            chosen.append(_make_patch(ds, f"{cat}_{k}", cat, rows, cols, m))
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
            r, c = ys[j], xs[j]
            if not far_enough(r, c):
                continue
            rr, cc = np.meshgrid(np.arange(r - 2, r + 3), np.arange(c - 2, c + 3), indexing="ij")
            k += 1
            chosen.append(
                _make_patch(
                    ds,
                    f"{cat}_{k}",
                    cat,
                    rr.ravel(),
                    cc.ravel(),
                    _patch_metrics(ds, rr.ravel(), cc.ravel()),
                    control=True,
                )
            )
    return chosen


def _make_patch(
    ds: Dataset, case_id: str, cat: str, rows, cols, m: dict, control: bool = False
) -> Patch:
    # representative pixel: HV step closest to the patch's (pixel-level, smoothed)
    hv = smooth_db(
        DateStack(
            ds.hv.values[:, rows.min() : rows.max() + 1, cols.min() : cols.max() + 1], ds.hv.days
        )
    )
    i = int(np.searchsorted(ds.hv.days, m["t0"]))
    pix_step = hv[: i + 1].mean(0) - hv[i + 1 :].mean(0)
    vals = pix_step[rows - rows.min(), cols - cols.min()]
    j = int(np.nanargmin(np.abs(vals - np.nanmedian(vals))))
    radd = ds.radd.alert_date[rows, cols]
    radd = radd[radd > -9999]
    return Patch(
        case_id=case_id,
        category=cat,
        rows=rows,
        cols=cols,
        t0=np.nan if control else m["t0"],
        t1=np.nan if control else m["t1"],
        radd_day=float(np.median(radd)) if radd.size else np.nan,
        hv_step_db=m["step"],
        hv_abrupt_frac=m["abrupt_frac"],
        coh80_z=m["z"],
        coh80_diff=m["diff"],
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
