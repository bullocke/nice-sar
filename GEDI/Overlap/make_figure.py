"""Filter, fit, and plot ΔAGBD vs ΔSAR for the proposal figure.

Reads ``pairs_with_sar.parquet`` (output of ``sample_sar.py``), applies post-hoc
quality filters, fits OLS regression, and writes:

- ``outputs/figures/delta_hv_vs_delta_agbd.png`` and ``.pdf`` (headline)
- ``outputs/figures/supplementary_dhh_drfdi.png`` and ``.pdf``
- ``outputs/figures/fit_stats.json``
- ``outputs/figures/decisions_log.md`` (filter cascade + n at each step)

Default filters (configurable):

- ``sensitivity ≥ 0.95`` for both pre and post shots
- ``pre_AGBD ≥ 50`` Mg/ha (restricts to "biomass loss from forest")
- Full-power beams only: ``BEAM0101, BEAM0110, BEAM1000, BEAM1011``
- Same antenna pointing pre vs post (``--require-same-pointing``)
- 3 * MAD outlier trim on ΔHV(dB)

Run::

    python GEDI/Overlap/make_figure.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_figure")

# GEDI full-power beams. The L4A INDEX exposes the integer encoding
# (BEAM0101=5, BEAM0110=6, BEAM1000=8, BEAM1011=11), so accept both.
FULL_POWER_BEAMS = {5, 6, 8, 11, "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    min_sensitivity: float,
    min_pre_agbd: float,
    full_power_beams: bool,
    require_same_pointing: bool,
    decisions: list[str],
) -> pd.DataFrame:
    """Apply the filter cascade in order. Records n surviving each step."""
    n0 = len(df)
    decisions.append(f"start: n={n0}")

    # Drop rows missing SAR or AGBD entirely.
    keep = (
        df["pre_HV_db"].notna() & df["post_HV_db"].notna()
        & df["pre_HH_db"].notna() & df["post_HH_db"].notna()
        & df["pre_agbd"].notna() & df["post_agbd"].notna()
    )
    df = df[keep].copy()
    decisions.append(f"complete pre/post SAR + AGBD: n={len(df)}")

    if min_sensitivity > 0:
        df = df[(df["pre_sensitivity"] >= min_sensitivity)
                & (df["post_sensitivity"] >= min_sensitivity)]
        decisions.append(
            f"sensitivity >= {min_sensitivity:.2f} (both): n={len(df)}"
        )

    if min_pre_agbd > 0:
        df = df[df["pre_agbd"] >= min_pre_agbd]
        decisions.append(f"pre_agbd >= {min_pre_agbd:g} Mg/ha: n={len(df)}")

    if full_power_beams:
        df = df[df["pre_beam"].isin(FULL_POWER_BEAMS)
                & df["post_beam"].isin(FULL_POWER_BEAMS)]
        decisions.append(f"full-power beams (both): n={len(df)}")

    if require_same_pointing:
        df = df[df["pre_AntennaPointing"] == df["post_AntennaPointing"]]
        decisions.append(f"same antenna pointing: n={len(df)}")

    return df.reset_index(drop=True)


def trim_outliers_mad(df: pd.DataFrame, col: str, k: float, decisions: list[str]) -> pd.DataFrame:
    """Drop rows with |col - median| > k * MAD."""
    if len(df) == 0:
        return df
    x = df[col].to_numpy()
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        decisions.append(f"MAD trim on {col}: skipped (MAD=0); n={len(df)}")
        return df
    keep = np.abs(x - med) <= k * mad
    out = df[keep].reset_index(drop=True)
    decisions.append(
        f"MAD trim on {col} ({k:g} * MAD={mad:.3f}): n={len(out)}"
    )
    return out


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ΔAGBD, ΔHV(dB), ΔHH(dB), and ΔRFDI."""
    df = df.copy()
    df["delta_agbd"] = df["post_agbd"] - df["pre_agbd"]
    df["delta_hv_db"] = df["post_HV_db"] - df["pre_HV_db"]
    df["delta_hh_db"] = df["post_HH_db"] - df["pre_HH_db"]
    # RFDI in linear power. dB -> linear: 10**(dB/10).
    pre_hh = 10.0 ** (df["pre_HH_db"] / 10.0)
    pre_hv = 10.0 ** (df["pre_HV_db"] / 10.0)
    post_hh = 10.0 ** (df["post_HH_db"] / 10.0)
    post_hv = 10.0 ** (df["post_HV_db"] / 10.0)
    df["pre_rfdi"] = (pre_hh - pre_hv) / (pre_hh + pre_hv)
    df["post_rfdi"] = (post_hh - post_hv) / (post_hh + post_hv)
    df["delta_rfdi"] = df["post_rfdi"] - df["pre_rfdi"]
    return df


# ---------------------------------------------------------------------------
# Fit + plot
# ---------------------------------------------------------------------------


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS via scipy.stats.linregress. Returns slope/intercept/R2/n/p/stderr."""
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "stderr_slope": float(res.stderr),
        "stderr_intercept": float(res.intercept_stderr),
        "n": int(len(x)),
    }


def _ci_band(x: np.ndarray, fit: dict, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """95% CI band for the regression line."""
    n = fit["n"]
    if n < 3:
        return np.array([]), np.array([]), np.array([])
    xs = np.linspace(np.min(x), np.max(x), 200)
    ys = fit["slope"] * xs + fit["intercept"]
    se_slope = fit["stderr_slope"]
    se_int = fit["stderr_intercept"]
    # Simple CI on line: combine via x-mean offset. This is approximate but fine
    # for a proposal figure.
    x_mean = float(np.mean(x))
    se_y = np.sqrt(se_int ** 2 + (xs - x_mean) ** 2 * se_slope ** 2)
    t = stats.t.ppf(1 - alpha / 2, df=n - 2)
    return xs, ys - t * se_y, ys + t * se_y


def plot_headline(df: pd.DataFrame, fit: dict, out_path: Path) -> None:
    """ΔHV (dB) vs ΔAGBD scatter with OLS fit and 95% CI."""
    x = df["delta_hv_db"].to_numpy()
    y = df["delta_agbd"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    if len(x) > 500:
        ax.hexbin(x, y, gridsize=40, mincnt=1, cmap="Greys", alpha=0.6, zorder=1)
    ax.scatter(x, y, s=12, alpha=0.45, color="#1f77b4", edgecolor="none", zorder=2)

    # Binned medians overlay. Per-pixel scatter is dominated by speckle and
    # geolocation noise; binning along ΔHV reveals the underlying gradient.
    if len(x) >= 30:
        edges = np.quantile(x, np.linspace(0, 1, 8))
        edges = np.unique(edges)
        if len(edges) >= 3:
            centers = 0.5 * (edges[:-1] + edges[1:])
            med = np.full(len(centers), np.nan)
            p25 = np.full(len(centers), np.nan)
            p75 = np.full(len(centers), np.nan)
            for i in range(len(centers)):
                m = (x >= edges[i]) & (x <= edges[i + 1] if i == len(centers) - 1 else x < edges[i + 1])
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

    # Fit line + CI band.
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
        "over Hansen-disturbed Brazilian Amazon forest",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", out_path)


def plot_supplementary(df: pd.DataFrame, fits: dict, out_path: Path) -> None:
    """Two-panel: ΔHH(dB) and ΔRFDI vs ΔAGBD on shared y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    y = df["delta_agbd"].to_numpy()

    panels = [
        ("delta_hh_db", r"$\Delta\sigma^0_{HH}$ (dB)", "ΔHH", fits.get("delta_hh_db")),
        ("delta_rfdi", r"$\Delta$RFDI", "ΔRFDI", fits.get("delta_rfdi")),
    ]
    for ax, (col, xlabel, _short, fit) in zip(axes, panels):
        x = df[col].to_numpy()
        ax.scatter(x, y, s=12, alpha=0.45, color="#2ca02c", edgecolor="none")
        if fit and fit["n"] >= 3:
            xs, lo, hi = _ci_band(x, fit)
            ax.fill_between(xs, lo, hi, color="#d62728", alpha=0.18)
            ys = fit["slope"] * xs + fit["intercept"]
            ax.plot(xs, ys, color="#d62728", lw=2.0,
                    label=(f"slope={fit['slope']:.2g}\n"
                           f"R² = {fit['r_squared']:.2f}, n = {fit['n']}"))
            ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\Delta$AGBD (Mg ha$^{-1}$)", fontsize=11)
    fig.suptitle("Supplementary: ΔHH and ΔRFDI vs GEDI ΔAGBD", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pairs-with-sar", type=Path,
        default=Path("GEDI/Overlap/outputs/pairs_with_sar.parquet"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("GEDI/Overlap/outputs/figures"),
    )
    p.add_argument("--min-sensitivity", type=float, default=0.95)
    p.add_argument("--min-pre-agbd", type=float, default=50.0)
    p.add_argument("--no-full-power-beams", action="store_true",
                   help="disable the full-power beams filter")
    p.add_argument("--no-require-same-pointing", action="store_true",
                   help="disable the same-pointing filter")
    p.add_argument("--mad-k", type=float, default=3.0,
                   help="MAD multiplier for ΔHV outlier trim (set 0 to skip)")
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
        f"mad_k={args.mad_k}"
    )

    df = apply_filters(
        df,
        min_sensitivity=args.min_sensitivity,
        min_pre_agbd=args.min_pre_agbd,
        full_power_beams=not args.no_full_power_beams,
        require_same_pointing=not args.no_require_same_pointing,
        decisions=decisions,
    )
    df = add_derived(df)
    if args.mad_k > 0:
        df = trim_outliers_mad(df, "delta_hv_db", args.mad_k, decisions)

    if len(df) < 3:
        log.error("only %d rows after filtering — cannot fit", len(df))
        (args.out_dir / "decisions_log.md").write_text(
            "# Decisions log\n\n" + "\n".join(f"- {d}" for d in decisions) + "\n"
            "\n**ABORTED**: fewer than 3 rows after filtering.\n"
        )
        return 4

    fits = {
        "delta_hv_db": fit_ols(df["delta_hv_db"].to_numpy(), df["delta_agbd"].to_numpy()),
        "delta_hh_db": fit_ols(df["delta_hh_db"].to_numpy(), df["delta_agbd"].to_numpy()),
        "delta_rfdi": fit_ols(df["delta_rfdi"].to_numpy(), df["delta_agbd"].to_numpy()),
    }

    # Headline + supplementary.
    plot_headline(df, fits["delta_hv_db"], args.out_dir / "delta_hv_vs_delta_agbd")
    plot_supplementary(df, fits, args.out_dir / "supplementary_dhh_drfdi")

    # Save fit stats.
    (args.out_dir / "fit_stats.json").write_text(
        json.dumps(fits, indent=2) + "\n"
    )
    log.info("wrote %s", args.out_dir / "fit_stats.json")

    # Save decisions log.
    decisions.append("")
    decisions.append("## Fit results")
    for k, v in fits.items():
        decisions.append(
            f"- {k}: slope={v['slope']:+.3g} (SE {v['stderr_slope']:.3g}), "
            f"intercept={v['intercept']:+.3g}, R²={v['r_squared']:.3f}, "
            f"p={v['p_value']:.2e}, n={v['n']}"
        )
    (args.out_dir / "decisions_log.md").write_text(
        "# Decisions log\n\n" + "\n".join(f"- {d}" if not d.startswith("#") and d else d for d in decisions) + "\n"
    )
    log.info("wrote %s", args.out_dir / "decisions_log.md")
    log.info(
        "headline: ΔHV slope=%+.2f (SE %.2f), R²=%.3f, n=%d",
        fits["delta_hv_db"]["slope"],
        fits["delta_hv_db"]["stderr_slope"],
        fits["delta_hv_db"]["r_squared"],
        fits["delta_hv_db"]["n"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
