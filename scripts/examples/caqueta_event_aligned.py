#!/usr/bin/env python
"""Event-aligned NISAR backscatter and coherence over Caquetá disturbances.

Aligns every disturbed pixel on its RADD alert date and summarizes GCOV
backscatter (HH, HV) and GUNW HH coherence as a function of time relative to
the disturbance. Stable forest and land cleared before the NISAR series are
shown as references drawn from the same acquisitions and pairs.

Pixel classes (20 m, from ``RADD_reference.tif``):

- ``disturbed``: high-confidence RADD alert inside the NISAR series window,
  within RADD's primary forest baseline
- ``stable forest``: forest baseline, no alert, and >= 100 m from any alert
- ``cleared before series``: alert (any confidence) at least 90 days before
  the first NISAR acquisition

For coherence, each pair is placed at the midpoint of its two dates relative to
the alert date. Pairs with a midpoint near zero span the disturbance. RADD
alert dates can lag the actual event by one or two Sentinel-1 revisits, so the
spanning signal may sit at slightly negative offsets.

Usage:
    python scripts/examples/caqueta_event_aligned.py
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EPOCH = date(2025, 1, 1)  # RADD_reference alert_date day 0
BIN = 12  # days
MAX_REL = 180  # days either side of the alert

COLORS = {
    "disturbed": "#2a78d6",
    "cleared before series": "#eb6834",
    "stable forest": "#1baf7a",
}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e4e3df"}


@dataclass
class Layer:
    """One GeoTIFF subset with its acquisition day(s) relative to EPOCH."""

    path: Path
    kind: str  # "HH", "HV", "coh20", "coh80"
    day: float  # acquisition (GCOV) or pair midpoint (GUNW)
    ref_day: int
    sec_day: int | None


def _day(iso: str) -> int:
    return (date.fromisoformat(iso[:10]) - EPOCH).days


def load_layers(root: Path) -> list[Layer]:
    manifest = json.loads((root / "manifest.json").read_text())
    layers = []
    for entry in manifest["files"]:
        path = root / entry["file"]
        ref = _day(entry["start"])
        if entry["product"] == "GCOV":
            kind = path.stem.split("_")[-1]
            layers.append(Layer(path, kind, ref, ref, None))
        else:
            sec = _day(entry["secondary_start"])
            kind = "coh20" if path.stem.endswith("coh20m") else "coh80"
            layers.append(Layer(path, kind, (ref + sec) / 2, ref, sec))
    return layers


def pixel_classes(radd_path: Path, first_day: int, last_day: int) -> dict:
    with rasterio.open(radd_path) as src:
        alert_date, conf, forest = src.read()
        grid = {"crs": src.crs, "transform": src.transform, "shape": src.shape}
    alerted = conf > 0
    dist_px = ndimage.distance_transform_edt(~alerted)
    disturbed = (conf == 3) & (forest == 1)
    disturbed &= (alert_date >= first_day) & (alert_date <= last_day)
    classes = {
        "disturbed": disturbed,
        "stable forest": (forest == 1) & ~alerted & (dist_px * 20 >= 100),
        "cleared before series": alerted & (alert_date < first_day - 90),
    }
    for name, mask in classes.items():
        logger.info("%s: %d pixels (%.1f%%)", name, mask.sum(), 100 * mask.mean())
    return {"classes": classes, "alert_date": alert_date, "grid": grid}


def _read(path: Path, grid: dict) -> np.ndarray:
    """Read a band onto the 20 m reference grid (80 m layers: nearest neighbour)."""
    with rasterio.open(path) as src:
        if src.transform == grid["transform"] and src.shape == grid["shape"]:
            data = src.read(1).astype("float32")
            nodata = src.nodata
        else:
            with WarpedVRT(
                src,
                crs=grid["crs"],
                transform=grid["transform"],
                width=grid["shape"][1],
                height=grid["shape"][0],
                resampling=Resampling.nearest,
            ) as vrt:
                data = vrt.read(1).astype("float32")
                nodata = vrt.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data) | (data <= 0)] = np.nan
    return data


def event_aligned(layers: list[Layer], kind: str, info: dict, grid: dict) -> dict:
    """Median and IQR of disturbed pixels per relative-time bin, plus references."""
    alert = info["alert_date"].astype(float)
    dist = info["classes"]["disturbed"]
    edges = np.arange(-MAX_REL, MAX_REL + BIN, BIN) - BIN / 2
    rel_all, val_all = [], []
    refs: dict[str, list[float]] = {"stable forest": [], "cleared before series": []}
    series = [lyr for lyr in layers if lyr.kind == kind]
    for lyr in series:
        data = _read(lyr.path, grid)
        if kind in ("HH", "HV"):
            data = 10 * np.log10(data)
        rel_all.append(lyr.day - alert[dist])
        val_all.append(data[dist])
        for name in refs:
            refs[name].append(float(np.nanmedian(data[info["classes"][name]])))
    rel = np.concatenate(rel_all)
    val = np.concatenate(val_all)
    ok = np.isfinite(val) & (np.abs(rel) <= MAX_REL)
    idx = np.digitize(rel[ok], edges)
    centers, med, q25, q75, count = [], [], [], [], []
    for i in range(1, len(edges)):
        v = val[ok][idx == i]
        if v.size < 200:
            continue
        centers.append((edges[i - 1] + edges[i]) / 2)
        med.append(np.median(v))
        q25.append(np.percentile(v, 25))
        q75.append(np.percentile(v, 75))
        count.append(v.size)
    return {
        "centers": np.array(centers),
        "median": np.array(med),
        "q25": np.array(q25),
        "q75": np.array(q75),
        "count": np.array(count),
        "refs": {k: np.array(v) for k, v in refs.items()},
        "n_layers": len(series),
    }


def plot(results: dict, out: Path) -> None:
    panels = [
        ("HH", "HH backscatter (dB)"),
        ("HV", "HV backscatter (dB)"),
        ("coh20", "HH coherence, 20 m"),
        ("coh80", "HH coherence, 80 m"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 11), sharex=True, dpi=150)
    for ax, (kind, label) in zip(axes, panels, strict=True):
        r = results[kind]
        for name in ("stable forest", "cleared before series"):
            ref = r["refs"][name]
            lo, mid, hi = np.nanpercentile(ref, [25, 50, 75])
            ax.axhspan(lo, hi, color=COLORS[name], alpha=0.15, lw=0)
            ax.axhline(mid, color=COLORS[name], lw=2)
            ax.text(MAX_REL + 4, mid, name, color=INK["secondary"], fontsize=7.5, va="center")
        ax.fill_between(
            r["centers"], r["q25"], r["q75"], color=COLORS["disturbed"], alpha=0.2, lw=0
        )
        ax.plot(
            r["centers"],
            r["median"],
            color=COLORS["disturbed"],
            lw=2,
            marker="o",
            ms=4,
            label="disturbed pixels (median, IQR)",
        )
        ax.axvline(0, color=INK["secondary"], lw=1, ls="--")
        ax.set_ylabel(label, fontsize=9, color=INK["primary"])
        ax.grid(axis="y", color=INK["grid"], lw=0.6)
        ax.tick_params(labelsize=8, colors=INK["secondary"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            f"{r['n_layers']} {'pairs' if kind.startswith('coh') else 'dates'}",
            fontsize=8,
            color=INK["secondary"],
            loc="right",
        )
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[-1].set_xlabel(
        "Days relative to RADD alert (GCOV: acquisition date; GUNW: pair midpoint)",
        fontsize=9,
        color=INK["primary"],
    )
    fig.suptitle(
        "NISAR response to forest disturbance, Caquetá (frame 083 D 088)",
        fontsize=11,
        color=INK["primary"],
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig(out, bbox_inches="tight")
    logger.info("Wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("NISAR_Data/caqueta"))
    args = parser.parse_args()

    layers = load_layers(args.root)
    days = [lyr.ref_day for lyr in layers] + [
        lyr.sec_day for lyr in layers if lyr.sec_day is not None
    ]
    info = pixel_classes(args.root / "RADD_reference.tif", min(days), max(days))

    results = {}
    for kind in ("HH", "HV", "coh20", "coh80"):
        results[kind] = event_aligned(layers, kind, info, info["grid"])
        r = results[kind]
        logger.info(
            "%s: %d layers; disturbed median by bin: %s",
            kind,
            r["n_layers"],
            ", ".join(f"{c:+.0f}d={m:.2f}" for c, m in zip(r["centers"], r["median"], strict=True)),
        )
        for name, ref in r["refs"].items():
            logger.info("  %s median %.2f", name, np.nanmedian(ref))
    plot(results, args.root / "event_aligned.png")


if __name__ == "__main__":
    main()
