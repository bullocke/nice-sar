"""Render the v2c (cleaned) ΔHV vs ΔAGBD headline figure.

Reads ``outputs/pairs_with_sar_v2c.parquet`` (the v2 dataset with A1+B2
cleaning applied by ``clean_pairs_v2c.py``) and writes figures to
``outputs/figures/v2c/``.

This is a thin wrapper around ``make_figure_v2.plot_headline_v2`` and the
v1 filter cascade — only the input parquet, output directory, and figure
title change.

Run::

    python GEDI/Overlap/make_figure_v2c.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figure import (  # noqa: E402
    add_derived,
    apply_filters,
    fit_ols,
    plot_supplementary,
    trim_outliers_mad,
    _ci_band,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_figure_v2c")


def plot_headline_v2c(df: pd.DataFrame, fit: dict, out_path: Path) -> None:
    """ΔHV (dB) vs ΔAGBD scatter with v2c (Hansen-edge + timing-cleaned) title."""
    x = df["delta_hv_db"].to_numpy()
    y = df["delta_agbd"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    if len(x) > 500:
        ax.hexbin(x, y, gridsize=40, mincnt=1, cmap="Greys", alpha=0.6, zorder=1)
    ax.scatter(x, y, s=12, alpha=0.45, color="#1f77b4", edgecolor="none", zorder=2)

    if len(x) >= 30:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 8)))
        if len(edges) >= 3:
            centers = 0.5 * (edges[:-1] + edges[1:])
            med = np.full(len(centers), np.nan)
            p25 = np.full(len(centers), np.nan)
            p75 = np.full(len(centers), np.nan)
            for i in range(len(centers)):
                if i == len(centers) - 1:
                    m = (x >= edges[i]) & (x <= edges[i + 1])
                else:
                    m = (x >= edges[i]) & (x < edges[i + 1])
                if m.sum() >= 3:
                    med[i] = np.median(y[m])
                    p25[i] = np.percentile(y[m], 25)
                    p75[i] = np.percentile(y[m], 75)
            ax.errorbar(
                centers, med,
                yerr=np.vstack([med - p25, p75 - med]),
                fmt="o", color="black", markersize=7, lw=1.5, capsize=3,
                zorder=5, label="Binned median ± IQR",
            )

    if fit["n"] >= 3:
        xs, lo, hi = _ci_band(x, fit)
        ax.fill_between(xs, lo, hi, color="#d62728", alpha=0.18, zorder=3,
                        label="95% CI")
        ys = fit["slope"] * xs + fit["intercept"]
        ax.plot(xs, ys, color="#d62728", lw=2.0, zorder=4,
                label=(f"OLS: slope={fit['slope']:.1f} ± {fit['stderr_slope']:.1f}\n"
                       f"R² = {fit['r_squared']:.2f}, n = {fit['n']}"))

    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="grey", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel(r"$\Delta\sigma^0_{HV}$ (dB), post − pre", fontsize=11)
    ax.set_ylabel(r"$\Delta$AGBD (Mg ha$^{-1}$), post − pre", fontsize=11)
    ax.set_title(
        "Per-pixel pre/post change: PALSAR-2 L-band $\\sigma^0_{HV}$ vs GEDI AGBD\n"
        "Hansen-disturbed Brazilian Amazon — same pass+path, "
        "Hansen-edge + timing cleaned",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", out_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pairs-with-sar", type=Path,
        default=Path("GEDI/Overlap/outputs/pairs_with_sar_v2c.parquet"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("GEDI/Overlap/outputs/figures/v2c"),
    )
    p.add_argument("--min-sensitivity", type=float, default=0.95)
    p.add_argument("--min-pre-agbd", type=float, default=50.0)
    p.add_argument("--no-full-power-beams", action="store_true")
    p.add_argument("--no-require-same-pointing", action="store_true")
    p.add_argument("--require-same-beam", action="store_true")
    p.add_argument("--mad-k", type=float, default=3.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pairs_with_sar.exists():
        log.error("input not found: %s", args.pairs_with_sar)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.pairs_with_sar)
    log.info("loaded %d rows from %s", len(df), args.pairs_with_sar)

    decisions: list[str] = []
    decisions.append(f"input: {args.pairs_with_sar}")
    decisions.append(
        f"args: min_sensitivity={args.min_sensitivity}, "
        f"min_pre_agbd={args.min_pre_agbd}, "
        f"full_power_beams={not args.no_full_power_beams}, "
        f"require_same_pointing={not args.no_require_same_pointing}, "
        f"require_same_beam={args.require_same_beam}, "
        f"mad_k={args.mad_k}"
    )
    decisions.append(
        "v2c cleaning (applied upstream by clean_pairs_v2c.py): "
        "A1 Hansen lossyear homogeneous within 30 m of pre footprint "
        "(min == max); B2 post_sar_date.year > 2000 + hansen_loss_max."
    )

    df = apply_filters(
        df,
        min_sensitivity=args.min_sensitivity,
        min_pre_agbd=args.min_pre_agbd,
        full_power_beams=not args.no_full_power_beams,
        require_same_pointing=not args.no_require_same_pointing,
        decisions=decisions,
    )
    if args.require_same_beam:
        n0 = len(df)
        df = df[df["pre_BeamID"] == df["post_BeamID"]].reset_index(drop=True)
        decisions.append(f"require_same_beam: n={len(df)} (was {n0})")
    df = add_derived(df)
    df = trim_outliers_mad(df, "delta_hv_db", k=args.mad_k, decisions=decisions)

    if len(df) < 10:
        log.error("after filters n=%d < 10; aborting", len(df))
        (args.out_dir / "decisions_log.md").write_text(
            "# v2c decisions log\n\n" + "\n".join(f"- {d}" for d in decisions)
        )
        return 4

    fits = {
        "delta_hv_db": fit_ols(df["delta_hv_db"].to_numpy(), df["delta_agbd"].to_numpy()),
        "delta_hh_db": fit_ols(df["delta_hh_db"].to_numpy(), df["delta_agbd"].to_numpy()),
        "delta_rfdi": fit_ols(df["delta_rfdi"].to_numpy(), df["delta_agbd"].to_numpy()),
    }

    plot_headline_v2c(df, fits["delta_hv_db"],
                      args.out_dir / "delta_hv_vs_delta_agbd_v2c")
    plot_supplementary(df, fits, args.out_dir / "supplementary_dhh_drfdi_v2c")

    (args.out_dir / "fit_stats.json").write_text(json.dumps(fits, indent=2))
    log.info("wrote %s", args.out_dir / "fit_stats.json")

    (args.out_dir / "decisions_log.md").write_text(
        "# v2c decisions log\n\n" + "\n".join(f"- {d}" for d in decisions)
    )
    log.info("wrote %s", args.out_dir / "decisions_log.md")

    f = fits["delta_hv_db"]
    log.info(
        "v2c headline: ΔHV slope=%+.2f (SE %.2f), R²=%.3f, n=%d",
        f["slope"], f["stderr_slope"], f["r_squared"], f["n"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
