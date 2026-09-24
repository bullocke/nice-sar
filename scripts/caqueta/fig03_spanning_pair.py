#!/usr/bin/env python
"""Test whether the pair spanning a clearing decorrelates beyond normal forest noise.

Pixel-level test (``analysis.spanning_test``) on core clearing pixels dated by
their HV step. Writes to ``local_examples/caqueta/03_spanning_pair/``:

- ``anomaly_vs_null.png``: cumulative distributions of the coherence change in
  the pair spanning the clearing (relative to each pixel's own pre-event mean)
  against a matched null from the same pixels' pre-event pairs; 80 m and 20 m.
- ``pair_map.png``: Sentinel-2 before and after, and the 80 m coherence of the
  single 24-day pair 19 Feb - 15 Mar 2026, zoomed to where most pixels dated to
  that bracket occur (outlined).
- ``results.json`` and ``README.md`` with the numbers.

Usage:
    python scripts/caqueta/fig03_spanning_pair.py
"""

# ruff: noqa: E501  (long lines are Markdown in README templates)
from __future__ import annotations

import json
import logging

import analysis
import config
import data
import matplotlib.pyplot as plt
import numpy as np
import style
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = config.OUT_DIR / "03_spanning_pair"
PAIR = ("2026-02-19", "2026-03-15")  # the cleanest single 24-day bracket
ZOOM_PX = 300  # 6 km


def ecdf(ax, values, color, label, lw=2.5, ls="-"):
    v = np.sort(values)
    ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=lw, ls=ls, label=label)


def plot_anomaly_vs_null(arrays: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, kind, color, title in (
        (axes[0], "coh80", style.COH80, "80 m"),
        (axes[1], "coh20", style.COH20, "20 m"),
    ):
        a = arrays[kind]
        ecdf(ax, a["observed"], color, "Pair spanning clearing")
        ecdf(ax, a["null"], style.FOREST_REF, "Same pixels, pre-event", ls="--")
        ax.axvline(0, color=style.INK_2, lw=1)
        ax.set_title(title)
        ax.set_xlim(-0.6, 0.4)
        ax.set_xlabel("Coherence change vs own pre-event mean")
        ax.grid(axis="x", color=style.GRID)
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].legend(loc="upper left")
    style.save(fig, OUT / "anomaly_vs_null.png")


def plot_pair_map(ds: data.Dataset, ev: analysis.EventDating, events: np.ndarray) -> None:
    t0, t1 = config.to_day(PAIR[0]), config.to_day(PAIR[1])
    dated = events & (ev.t0 == t0) & (ev.t1 == t1)
    # zoom to the 6 km window holding the most pixels dated to this bracket
    density = ndimage.uniform_filter(dated.astype(float), ZOOM_PX, mode="constant")
    r, c = np.unravel_index(np.argmax(density), density.shape)
    h, w = ds.grid.shape
    r0 = int(np.clip(r - ZOOM_PX // 2, 0, h - ZOOM_PX))
    c0 = int(np.clip(c - ZOOM_PX // 2, 0, w - ZOOM_PX))
    win = (slice(r0, r0 + ZOOM_PX), slice(c0, c0 + ZOOM_PX))

    scenes = data.s2_scenes()

    def clearest(lo: int, hi: int) -> data.S2Scene:
        cands = [s for s in scenes if lo <= s.day <= hi]
        usable = [s for s in cands if data.s2_usable(s, win)] or cands
        return max(usable, key=lambda s: (data.s2_clear_fraction(s, win), -abs(s.day - lo)))

    before = clearest(t0 - 30, t0)
    after = clearest(t1, t1 + 30)
    p = int(np.flatnonzero((ds.coh80.ref == t0) & (ds.coh80.sec == t1))[0])
    coh = ds.coh80.values[p][win]
    outline = np.kron(dated[win].astype(float), np.ones((2, 2)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, scene in ((axes[0], before), (axes[1], after)):
        ax.imshow(data.s2_rgb(scene, win), interpolation="nearest")
        ax.contour(outline, levels=[0.5], colors=style.OUTLINE, linewidths=1)
        ax.set_title(config.to_date(scene.day).strftime("%d %b %Y"))
    im = axes[2].imshow(coh, cmap="gray", vmin=0, vmax=0.7, interpolation="nearest")
    axes[2].contour(dated[win].astype(float), levels=[0.5], colors=style.OUTLINE, linewidths=1)
    axes[2].set_title("Coherence 80 m, 19 Feb - 15 Mar")
    for ax in axes:
        style.image_axes(ax)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=12)
    style.save(fig, OUT / "pair_map.png")
    return {
        "zoom_window_rows_cols": [r0, c0, ZOOM_PX],
        "s2_before": str(config.to_date(before.day)),
        "s2_after": str(config.to_date(after.day)),
        "pixels_outlined": int(dated[win].sum()),
    }


def write_readme(results: dict, map_info: dict) -> None:
    r80, r20 = results["coh80"], results["coh20"]
    single = {s["pair"]: s for s in r80["single_pairs"]}
    s = single.get(f"{PAIR[0]}/{PAIR[1]}", {})
    timing_rows = "\n".join(
        f"| {a['window']} | {a['n_pixels']} | {a['auc']:.2f} | {b['auc']:.2f} |"
        for a, b in zip(results["timing"]["coh80"], results["timing"]["coh20"], strict=True)
    )
    text = f"""# Does the pair spanning a clearing decorrelate beyond forest noise?

## Method

1. **Date each clearing from HV, not coherence.** For every pixel, fit one step to
   the HV series (3x3 boxcar in linear power): the split between two consecutive
   dual-pol dates with the largest mean(before) - mean(after). Keep core RADD
   clearing pixels (at least 40 m from any clearing edge) with a step of at least
   {config.HV_STEP_MIN_DB} dB whose bracket ends within {config.HV_RADD_MAX_OFFSET_D} days of the RADD alert.
   Dating from HV keeps the coherence test independent.
2. **Remove weather.** Subtract each pair's stable-forest median coherence.
3. **Own baseline.** For pixels with at least {config.MIN_PRE_PAIRS} pairs ending at least
   {config.PRE_PAIR_BUFFER_D} days before the bracket, take their mean as the pixel's baseline.
4. **Observed.** The lowest pair inside the bracket minus the baseline. A 24-day
   bracket usually holds two 12-day pairs; one spans the clearing and the other
   does not, so the minimum picks the spanning pair.
5. **Matched null.** The minimum of the same number of the pixel's own pre-event
   pairs, each relative to the mean of the others (leave-one-out). Using a minimum
   on both sides removes the bias of picking the lowest value.

## Results

| | 80 m | 20 m |
|---|--:|--:|
| Pixels tested | {r80["n_pixels"]} | {r20["n_pixels"]} |
| Median change, spanning pair | {r80["median_observed"]:+.3f} | {r20["median_observed"]:+.3f} |
| Median change, matched null | {r80["median_null"]:+.3f} | {r20["median_null"]:+.3f} |
| AUC (spanning lower than null) | {r80["auc"]:.2f} | {r20["auc"]:.2f} |
| Same-pixel pair-to-pair SD | {r80["same_pixel_noise_sd"]:.3f} | {r20["same_pixel_noise_sd"]:.3f} |
| Expected coherence at zero (looks) | {results["floor"]["coh80"]:.3f} ({config.LOOKS["coh80"]}) | {results["floor"]["coh20"]:.3f} ({config.LOOKS["coh20"]}) |

AUC is the probability that a spanning-pair value is lower than a null value
(0.5 = no signal). RADD alerts come a median of {results["radd_minus_bracket_end_days"]:+.0f} days after the end of
the HV bracket, which is why aligning on RADD dates hid the dip.

**Single 24-day pair {PAIR[0]} to {PAIR[1]}** ({s.get("n_pixels", 0)} pixels whose whole bracket
is this one pair): median 80 m coherence {s.get("median_coherence", float("nan")):.2f} against {s.get("forest_median", float("nan")):.2f} for stable
forest in the same pair.

## Interpretation

Across many pixels the spanning pair is clearly lower (about -0.1 at 80 m), and
close to the estimator floor in the cleanest case. For a single pixel and a single
pair the drop is about the size of that pixel's normal pair-to-pair variation, so
it cannot be detected reliably from one nearest-neighbour pair. The 20 m product
shows a weaker effect because its floor (about {results["floor"]["coh20"]:.2f}) sits close to forest coherence.

## Timing: does coherence drop before the HV step?

Same test, but the baseline only uses pairs ending at least 36 days before the
bracket, so the pair(s) just before it can be tested as well (same pixels in
every row).

| Window | Pixels | 80 m AUC | 20 m AUC |
|---|--:|--:|--:|
{timing_rows}

Searching the 24 days before the bracket together with the bracket separates best.
In most clearings the drop coincides with the HV step, but in some (e.g.
`../04_cases/no_dip/no_dip_1`) coherence falls one cycle earlier, when the optical
chips already show felled vegetation.

## Caution: which stage of clearing does the HV step date?

In `pair_map.png`, many pixels HV-dated to the 19 Feb - 15 Mar bracket already
look brown (felled vegetation) in the Sentinel-2 image from before the bracket.
Clearing here typically proceeds as understory slashing, felling, drying, then
burning. Felled trunks can keep HV volume scattering until they are burned or
removed, so the HV step, and the coherence dip dated with it, may often mark
**burning of the slash** rather than felling. This is a hypothesis to test, e.g.
by comparing the HV bracket with the first optical date showing brown vegetation
for each clearing, and with active-fire detections.

## Figures

- `anomaly_vs_null.png`: curves further left mean lower coherence than the pixel's
  own history. The gap between the solid and dashed curves is the signal.
- `pair_map.png`: Sentinel-2 {map_info["s2_before"]} and {map_info["s2_after"]}, and 80 m coherence for
  the 24-day pair; yellow outlines are pixels HV-dated to this bracket
  ({map_info["pixels_outlined"]} in the 6 km window).

Regenerate: `python scripts/caqueta/fig03_spanning_pair.py`.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = data.load()
    ev = analysis.hv_event_dating(ds.hv)
    events = analysis.event_pixels(ds, ev)
    lag = ds.radd.alert_date[events] - ev.t1[events]

    results: dict = {
        "event_pixels": int(events.sum()),
        "radd_minus_bracket_end_days": float(np.median(lag)),
        "floor": {k: analysis.coherence_floor(n) for k, n in config.LOOKS.items()},
    }
    arrays = {}
    for kind in ("coh80", "coh20"):
        res, arr = analysis.spanning_test(ds, ev, events, kind)
        results[kind] = res.to_dict()
        arrays[kind] = arr
    results["timing"] = {
        kind: analysis.timing_sensitivity(ds, ev, events, kind) for kind in ("coh80", "coh20")
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    plot_anomaly_vs_null(arrays)
    map_info = plot_pair_map(ds, ev, events)
    write_readme(results, map_info)
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
