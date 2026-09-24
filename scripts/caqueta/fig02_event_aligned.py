#!/usr/bin/env python
"""Event-aligned composites: all RADD clearings stacked on their alert date.

Every disturbed pixel (RADD high-confidence alert during the NISAR series) is
aligned on its RADD alert date. For each 12-day bin of time relative to the
alert, the median and interquartile range over all pixel-dates are plotted.
Backscatter uses the acquisition date; coherence uses the pair midpoint. Stable
forest and pasture cleared before the series are shown as horizontal references
(median over all dates or pairs).

Writes to ``local_examples/caqueta/02_event_aligned/``:

- ``backscatter.png``: HH and HV
- ``coherence_80m_vs_20m.png``: 80 m and 20 m on the same y-axis range, with
  the zero-coherence estimator floor

Usage:
    python scripts/caqueta/fig02_event_aligned.py
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT = config.OUT_DIR / "02_event_aligned"
BIN_D = 12
MAX_REL_D = 180
MIN_PER_BIN = 200


def aligned(ds: data.Dataset, values: np.ndarray, days: np.ndarray) -> dict:
    """Bin disturbed-pixel values by (layer day - RADD alert day)."""
    dist = ds.masks["disturbed"]
    alert = ds.radd.alert_date[dist]
    rel = np.concatenate([d - alert for d in days])
    val = np.concatenate([v[dist] for v in values])
    ok = np.isfinite(val) & (np.abs(rel) <= MAX_REL_D)
    edges = np.arange(-MAX_REL_D, MAX_REL_D + BIN_D, BIN_D) - BIN_D / 2
    idx = np.digitize(rel[ok], edges)
    out = {"centre": [], "median": [], "q25": [], "q75": [], "n": []}
    for i in range(1, len(edges)):
        v = val[ok][idx == i]
        if v.size < MIN_PER_BIN:
            continue
        out["centre"].append((edges[i - 1] + edges[i]) / 2)
        out["median"].append(float(np.median(v)))
        out["q25"].append(float(np.percentile(v, 25)))
        out["q75"].append(float(np.percentile(v, 75)))
        out["n"].append(int(v.size))
    refs = {}
    for name in ("stable_forest", "pre_series_pasture"):
        m = ds.masks[name]
        refs[name] = float(np.nanmedian([np.nanmedian(v[m]) for v in values]))
    return {**{k: np.array(v) for k, v in out.items()}, "refs": refs}


def panel(ax, res: dict, color: str, ylabel: str) -> None:
    ax.fill_between(res["centre"], res["q25"], res["q75"], color=color, alpha=0.2, lw=0)
    ax.plot(res["centre"], res["median"], color=color, marker="o", ms=4)
    for name, ls, label in (
        ("stable_forest", "-", "Forest"),
        ("pre_series_pasture", "--", "Pasture"),
    ):
        y = res["refs"][name]
        ax.axhline(y, color=style.FOREST_REF if ls == "-" else style.PASTURE_REF, lw=1.5, ls=ls)
        ax.text(MAX_REL_D + 3, y, label, va="center", fontsize=12, color=style.INK_2)
    ax.axvline(0, color=style.INK_2, lw=1, ls=":")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-MAX_REL_D - 6, MAX_REL_D + 6)


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = data.load()
    res = {
        "HH": aligned(ds, ds.hh.values, ds.hh.days),
        "HV": aligned(ds, ds.hv.values, ds.hv.days),
        "coh80": aligned(ds, ds.coh80.values, (ds.coh80.ref + ds.coh80.sec) / 2),
        "coh20": aligned(ds, ds.coh20.values, (ds.coh20.ref + ds.coh20.sec) / 2),
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    panel(axes[0], res["HH"], style.HH, "HH (dB)")
    panel(axes[1], res["HV"], style.HV, "HV (dB)")
    axes[1].set_xlabel("Days from RADD alert")
    style.save(fig, OUT / "backscatter.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, kind, color, label in (
        (axes[0], "coh80", style.COH80, "Coherence, 80 m"),
        (axes[1], "coh20", style.COH20, "Coherence, 20 m"),
    ):
        panel(ax, res[kind], color, label)
        floor = analysis.coherence_floor(config.LOOKS[kind])
        ax.axhline(floor, color=style.FLOOR, lw=1.5, ls=":")
        ax.text(MAX_REL_D + 3, floor, "Floor", va="center", fontsize=12, color=style.INK_2)
    axes[0].set_ylim(0, 0.8)
    axes[1].set_xlabel("Days from RADD alert (pair midpoint)")
    style.save(fig, OUT / "coherence_80m_vs_20m.png")

    summary = {
        k: {
            "refs": v["refs"],
            "bins": {
                f"{c:+.0f}": round(m, 3) for c, m in zip(v["centre"], v["median"], strict=True)
            },
        }
        for k, v in res.items()
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    def ref(k: str, name: str) -> str:
        return f"{res[k]['refs'][name]:.2f}"

    text = f"""# Event-aligned composites

All {int(ds.masks["disturbed"].sum())} RADD-disturbed pixels (high-confidence alerts during the
NISAR series, inside RADD's forest baseline) are aligned on their RADD alert date
and binned in {BIN_D}-day steps. Lines are medians, shading the interquartile range over
all pixel-dates in the bin. Coherence is placed at the midpoint of each pair.
Horizontal lines are the median stable-forest and pre-series-pasture values.

| | Forest | Pasture |
|---|--:|--:|
| HH (dB) | {ref("HH", "stable_forest")} | {ref("HH", "pre_series_pasture")} |
| HV (dB) | {ref("HV", "stable_forest")} | {ref("HV", "pre_series_pasture")} |
| Coherence 80 m | {ref("coh80", "stable_forest")} | {ref("coh80", "pre_series_pasture")} |
| Coherence 20 m | {ref("coh20", "stable_forest")} | {ref("coh20", "pre_series_pasture")} |

## What the figures show

- **HV** falls about 2 dB from forest to pasture level, starting about 50 days
  *before* the RADD date. RADD alerts lag the NISAR HV drop (median about 2 weeks,
  see `../03_spanning_pair/`), and some clearing starts with understory removal.
- **HH** rises slightly after clearing.
- **Coherence** rises from forest level to well above pasture level after clearing.
  No dip is visible at day 0 here: alignment on the lagging RADD date puts many
  "spanning" pairs after the clearing, which hides the dip. The HV-dated test in
  `../03_spanning_pair/` recovers it.
- **80 m vs 20 m**: same pattern, but 20 m reads about 0.07 higher throughout
  and its zero-coherence floor ({analysis.coherence_floor(config.LOOKS["coh20"]):.2f}, {config.LOOKS["coh20"]} looks) sits close to forest
  coherence, leaving little room for a decorrelation dip. 80 m ({config.LOOKS["coh80"]} looks) has a
  floor of {analysis.coherence_floor(config.LOOKS["coh80"]):.2f}.

`summary.json` has the binned medians. Regenerate:
`python scripts/caqueta/fig02_event_aligned.py`.
"""
    (OUT / "README.md").write_text(text)
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
