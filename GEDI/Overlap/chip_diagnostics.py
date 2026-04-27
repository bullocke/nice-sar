"""Chip diagnostics for v2 ΔHV scatter — pick 5 sites and download S2/Hansen/PALSAR-2 chips.

Three subcommands:

- ``select``  : stratified-sample 5 pairs from the v2 filtered set, label them
                on a copy of the v2 scatter, write ``chip_sites.parquet``.
- ``download``: for each site, download S2 (pre+post median composites),
                Hansen (lossyear 20-22), and PALSAR-2 ScanSAR HV (the exact
                pre+post scenes used in v2) as GeoTIFFs via wxee.
- ``panel``   : render a 1x5 matplotlib panel per site overlaying both GEDI
                shots on every chip.
- ``all``     : run select → download → panel.

Site IDs are zero-padded master-pairs.parquet row indices (e.g. ``0037``).

Run::

    python GEDI/Overlap/chip_diagnostics.py select --seed 42
    python GEDI/Overlap/chip_diagnostics.py download
    python GEDI/Overlap/chip_diagnostics.py panel
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import wxee  # noqa: F401  (registers .wx accessor)
from matplotlib.patches import Circle
from rasterio.transform import xy as rio_xy
from scipy import stats

# Reuse v1 / v2 filter logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figure import add_derived, apply_filters, fit_ols, trim_outliers_mad  # noqa: E402
from make_figure_v2 import plot_headline_v2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chip_diagnostics")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S2_ASSET = "COPERNICUS/S2_SR_HARMONIZED"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
SCANSAR_ASSET = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"

# SCL classes to mask out (cloud_shadow, cloud_med, cloud_high, thin_cirrus).
S2_BAD_SCL = (3, 8, 9, 10)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _utm_epsg(lon: float, lat: float) -> str:
    """Auto-pick UTM zone EPSG for a lon/lat point."""
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _chip_geom(lon: float, lat: float, chip_m: float) -> ee.Geometry:
    """Square WGS84 polygon centered on (lon, lat) with side length ``chip_m``."""
    half = chip_m / 2.0
    dlat = half / 110_540.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return ee.Geometry.Polygon(
        [[
            [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat],
            [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat],
        ]]
    )


# ---------------------------------------------------------------------------
# Filter cascade replication (matches make_figure_v2 defaults)
# ---------------------------------------------------------------------------


def _v2_filtered(parquet: Path) -> pd.DataFrame:
    """Apply make_figure_v2's default filter cascade and return the resulting df.

    Preserves the master ``pairs.parquet`` row index in column ``pair_id``.
    """
    df = pd.read_parquet(parquet).reset_index().rename(columns={"index": "pair_id"})
    log.info("loaded %d rows from %s", len(df), parquet)
    decisions: list[str] = []
    df = apply_filters(
        df,
        min_sensitivity=0.95,
        min_pre_agbd=50.0,
        full_power_beams=True,
        require_same_pointing=True,
        decisions=decisions,
    )
    df = add_derived(df)
    df = trim_outliers_mad(df, "delta_hv_db", k=3.0, decisions=decisions)
    log.info("after v2 filters: n=%d", len(df))
    return df


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def _stratify_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    exclude_pair_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Pick n=5 sites: 1 signal, 1 noise, 1 false_alarm, n-3 random.

    Strata defined relative to the OLS fit of ΔAGBD ~ ΔHV. Within each stratum
    we pick the row with the median residual (i.e. typical, not extreme).

    ``exclude_pair_ids`` removes already-selected sites from the candidate pool
    so subsequent calls return new picks (used by --append).
    """
    rng = np.random.default_rng(seed)
    fit = stats.linregress(df["delta_hv_db"], df["delta_agbd"])
    pred = fit.slope * df["delta_hv_db"] + fit.intercept
    df = df.assign(_residual=df["delta_agbd"].to_numpy() - pred.to_numpy())
    if exclude_pair_ids:
        df = df[~df["pair_id"].isin(exclude_pair_ids)]
        log.info("excluded %d already-selected pair_ids; %d candidates remain",
                 len(exclude_pair_ids), len(df))

    p20_dhv = df["delta_hv_db"].quantile(0.20)
    p20_dagbd = df["delta_agbd"].quantile(0.20)
    p40_abs_dhv = df["delta_hv_db"].abs().quantile(0.40)
    p50_dagbd = df["delta_agbd"].quantile(0.50)

    def _median_row(subset: pd.DataFrame) -> pd.Series | None:
        if len(subset) == 0:
            return None
        idx = (subset["_residual"] - subset["_residual"].median()).abs().idxmin()
        return subset.loc[idx]

    picks: dict[str, pd.Series] = {}
    used: set[int] = set()

    sig = df[(df["delta_hv_db"] <= p20_dhv) & (df["delta_agbd"] <= p20_dagbd)]
    r = _median_row(sig)
    if r is not None:
        picks["signal"] = r
        used.add(int(r["pair_id"]))

    noise = df[(df["delta_hv_db"].abs() <= p40_abs_dhv) & (df["delta_agbd"] <= p20_dagbd)]
    noise = noise[~noise["pair_id"].isin(used)]
    r = _median_row(noise)
    if r is not None:
        picks["noise"] = r
        used.add(int(r["pair_id"]))

    fa = df[(df["delta_hv_db"] <= p20_dhv) & (df["delta_agbd"] >= p50_dagbd)]
    fa = fa[~fa["pair_id"].isin(used)]
    r = _median_row(fa)
    if r is not None:
        picks["false_alarm"] = r
        used.add(int(r["pair_id"]))

    remaining = df[~df["pair_id"].isin(used)]
    n_random = max(0, n - len(picks))
    if len(remaining) > 0 and n_random > 0:
        random_idx = rng.choice(remaining.index.to_numpy(), size=min(n_random, len(remaining)), replace=False)
        for i, idx in enumerate(random_idx):
            picks[f"random_{i + 1}"] = remaining.loc[idx]

    out = pd.DataFrame(picks).T.reset_index().rename(columns={"index": "stratum"})
    out["pair_id"] = out["pair_id"].astype(int)
    out["site_id"] = out["pair_id"].apply(lambda x: f"{x:04d}")
    log.info("selected %d sites: %s", len(out), dict(out["stratum"].value_counts()))
    return out


def _label_v2_figure(df_filtered: pd.DataFrame, sites: pd.DataFrame, out_path: Path) -> None:
    """Re-render the v2 scatter with `#NNNN` labels next to the selected sites."""
    fit = fit_ols(df_filtered["delta_hv_db"].to_numpy(), df_filtered["delta_agbd"].to_numpy())
    plot_headline_v2(df_filtered, fit, out_path.with_suffix(""))
    # Re-open the rendered figure is wasteful — just regenerate with annotations.
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    x = df_filtered["delta_hv_db"].to_numpy()
    y = df_filtered["delta_agbd"].to_numpy()
    ax.scatter(x, y, s=12, alpha=0.35, color="#1f77b4", edgecolor="none")
    # Highlight selected sites + label.
    sel_x = sites["delta_hv_db"].to_numpy(dtype=float)
    sel_y = sites["delta_agbd"].to_numpy(dtype=float)
    ax.scatter(sel_x, sel_y, s=80, facecolor="none", edgecolor="#d62728", lw=2.0, zorder=5)
    for _, row in sites.iterrows():
        ax.annotate(
            f"#{row['site_id']}",
            (float(row["delta_hv_db"]), float(row["delta_agbd"])),
            xytext=(7, 5), textcoords="offset points",
            fontsize=9, color="#d62728", fontweight="bold",
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(r"$\Delta\sigma^0_{HV}$ (dB), post − pre", fontsize=11)
    ax.set_ylabel(r"$\Delta$AGBD (Mg ha$^{-1}$), post − pre", fontsize=11)
    ax.set_title(
        "v2 scatter with diagnostic chip sites labeled\n"
        f"OLS slope={fit['slope']:.1f} ± {fit['stderr_slope']:.1f}, "
        f"R²={fit['r_squared']:.3f}, n={fit['n']}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", out_path)


def cmd_select(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _v2_filtered(args.input)
    sites_path = args.out_dir / "chip_sites.parquet"

    existing: pd.DataFrame | None = None
    exclude: set[int] = set()
    if args.append and sites_path.exists():
        existing = pd.read_parquet(sites_path)
        exclude = set(existing["pair_id"].astype(int).tolist())
        log.info("--append: loaded %d existing sites from %s", len(existing), sites_path)

    new_sites = _stratify_sample(df, n=args.n, seed=args.seed, exclude_pair_ids=exclude)

    if existing is not None and len(existing) > 0:
        # Re-label strata of new picks to avoid clashes (signal_2, etc.).
        used_strata = set(existing["stratum"].astype(str).tolist())
        renamed: list[str] = []
        for s in new_sites["stratum"].tolist():
            cand = s
            i = 2
            while cand in used_strata:
                cand = f"{s}_{i}"
                i += 1
            renamed.append(cand)
            used_strata.add(cand)
        new_sites = new_sites.assign(stratum=renamed)
        combined = pd.concat([existing, new_sites.drop(columns=["_residual"], errors="ignore")],
                             ignore_index=True)
        combined = combined.drop_duplicates(subset="pair_id", keep="first")
    else:
        combined = new_sites.drop(columns=["_residual"], errors="ignore")

    combined.to_parquet(sites_path, index=False)
    log.info("wrote %s (%d sites total, %d new)", sites_path, len(combined), len(new_sites))

    # Re-render labeled scatter using all sites with delta values pulled from filtered df.
    label_df = combined.merge(
        df[["pair_id", "delta_hv_db", "delta_agbd"]], on="pair_id",
        how="left", suffixes=("", "_y"),
    )
    for col in ("delta_hv_db", "delta_agbd"):
        if f"{col}_y" in label_df.columns:
            label_df[col] = label_df[f"{col}_y"]
            label_df = label_df.drop(columns=[f"{col}_y"])
    _label_v2_figure(df, label_df, args.out_dir / "v2_figure_labeled")
    return 0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def _s2_composite(geom: ee.Geometry, center_ms: float, window_days: int) -> ee.Image:
    """Cloud-masked S2 SR median composite within ±window_days of center_ms."""
    target = ee.Number(center_ms)
    half = ee.Number(window_days * 86400 * 1000)
    start = ee.Date(target.subtract(half))
    end = ee.Date(target.add(half))
    coll = (
        ee.ImageCollection(S2_ASSET)
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    )

    def _mask(img: ee.Image) -> ee.Image:
        scl = img.select("SCL")
        bad = scl.eq(S2_BAD_SCL[0])
        for c in S2_BAD_SCL[1:]:
            bad = bad.Or(scl.eq(c))
        return img.updateMask(bad.Not()).select(["B4", "B3", "B2"]).divide(10000)

    return coll.map(_mask).median()


def _hansen_loss_mask(geom: ee.Geometry) -> ee.Image:
    """1 where Hansen lossyear ∈ {20, 21, 22}, 0 elsewhere, masked outside."""
    img = ee.Image(HANSEN_ASSET)
    ly = img.select("lossyear")
    mask = ly.eq(20).Or(ly.eq(21)).Or(ly.eq(22))
    return mask.updateMask(mask).rename("hansen_loss")


def _exact_scansar(time_ms: float, pass_dir: str | None, rsp: float | None) -> ee.Image | None:
    """Try to retrieve the exact ScanSAR scene used in v2.

    Returns a single-band HV image (in DN, caller converts to dB), or None.
    """
    if pd.isna(time_ms):
        return None
    target = ee.Number(time_ms)
    coll = ee.ImageCollection(SCANSAR_ASSET).filter(
        ee.Filter.eq("system:time_start", target)
    )
    if pass_dir is not None and not pd.isna(pass_dir):
        coll = coll.filter(ee.Filter.eq("PassDirection", str(pass_dir)))
    if rsp is not None and not pd.isna(rsp):
        coll = coll.filter(ee.Filter.eq("RSP_Path_Number", int(rsp)))
    n = coll.size().getInfo()
    if n == 0:
        # Fallback: nearest in time within ±1 day.
        win = ee.Number(86400 * 1000)
        coll2 = (
            ee.ImageCollection(SCANSAR_ASSET)
            .filterDate(ee.Date(target.subtract(win)), ee.Date(target.add(win)))
        )
        if pass_dir is not None and not pd.isna(pass_dir):
            coll2 = coll2.filter(ee.Filter.eq("PassDirection", str(pass_dir)))
        if rsp is not None and not pd.isna(rsp):
            coll2 = coll2.filter(ee.Filter.eq("RSP_Path_Number", int(rsp)))
        if coll2.size().getInfo() == 0:
            return None
        return ee.Image(coll2.first()).select("HV")
    return ee.Image(coll.first()).select("HV")


def _download(image: ee.Image, geom: ee.Geometry, scale: float, crs: str,
              out_dir: Path, name: str) -> Path | None:
    """Download a single chip via wxee; return path or None on failure."""
    out_path = out_dir / f"{name}.tif"
    if out_path.exists():
        log.info("    skip %s (exists)", out_path.name)
        return out_path
    try:
        # wxee requires system:time_start; set a dummy value for composites/masks.
        img = image.set("system:time_start", 0)
        files = img.wx.to_tif(  # type: ignore[attr-defined]
            out_dir=str(out_dir),
            description=name,
            region=geom,
            scale=scale,
            crs=crs,
            progress=False,
        )
        # wxee returns a list of paths; rename to drop .time.<...> suffix.
        actual = Path(files[0]) if isinstance(files, (list, tuple)) else Path(files)
        if actual.name != out_path.name:
            actual.rename(out_path)
        return out_path
    except Exception as e:
        log.warning("    failed %s: %s", name, e)
        return None


def cmd_download(args: argparse.Namespace) -> int:
    sites = pd.read_parquet(args.sites)
    log.info("loaded %d sites from %s", len(sites), args.sites)
    if args.pair_ids:
        keep = [int(x) for x in args.pair_ids.split(",")]
        sites = sites[sites["pair_id"].isin(keep)].reset_index(drop=True)
        log.info("filtered to %d sites by --pair-ids", len(sites))

    ee.Initialize(
        project=args.ee_project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    log.info("Earth Engine initialized (project=%s)", args.ee_project)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in sites.iterrows():
        site_id = row["site_id"]
        site_dir = args.out_dir / f"site_{site_id}"
        site_dir.mkdir(parents=True, exist_ok=True)
        lon, lat = float(row["pre_lon"]), float(row["pre_lat"])
        crs = _utm_epsg(lon, lat)
        geom = _chip_geom(lon, lat, args.chip_m)

        log.info("site #%s (stratum=%s, lon=%.4f lat=%.4f, CRS=%s)",
                 site_id, row["stratum"], lon, lat, crs)

        # S2 pre/post — 10 m
        pre_ms = float(row["pre_sar_time_ms"])
        post_ms = float(row["post_sar_time_ms"])
        s2_pre = _s2_composite(geom, pre_ms, args.s2_window_days)
        s2_post = _s2_composite(geom, post_ms, args.s2_window_days)
        _download(s2_pre, geom, scale=10, crs=crs, out_dir=site_dir, name="s2_pre")
        _download(s2_post, geom, scale=10, crs=crs, out_dir=site_dir, name="s2_post")

        # Hansen — 30 m
        hansen = _hansen_loss_mask(geom)
        _download(hansen, geom, scale=30, crs=crs, out_dir=site_dir, name="hansen")

        # PALSAR-2 HV pre/post — 25 m
        for label, t_ms, pass_dir, rsp in [
            ("palsar_pre", row["pre_sar_time_ms"], row.get("pre_PassDirection"),
             row.get("pre_RSP_Path_Number")),
            ("palsar_post", row["post_sar_time_ms"], row.get("post_PassDirection"),
             row.get("post_RSP_Path_Number")),
        ]:
            img = _exact_scansar(t_ms, pass_dir, rsp)
            if img is None:
                log.warning("    no ScanSAR scene for %s", label)
                continue
            _download(img, geom, scale=25, crs=crs, out_dir=site_dir, name=label)

    return 0


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------


def _read_chip(path: Path) -> tuple[np.ndarray, rasterio.Affine, str] | None:
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read()
        return arr, src.transform, src.crs.to_string()


def _stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def _palsar_hv_db(dn: np.ndarray) -> np.ndarray:
    """PALSAR-2 L2.2 calibration; matches sample_sar_v2 formula."""
    dn = np.asarray(dn, dtype=np.float64)
    out = np.full_like(dn, np.nan)
    valid = (dn > 0) & np.isfinite(dn)
    out[valid] = 10.0 * np.log10(dn[valid] ** 2) - 83.0
    return out


def _world_to_px(lon: float, lat: float, transform: rasterio.Affine, crs: str) -> tuple[float, float]:
    """Convert WGS84 lon/lat to pixel (col, row) in chip CRS."""
    from rasterio.warp import transform as warp_transform
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    inv = ~transform
    col, row = inv * (xs[0], ys[0])
    return col, row


def _draw_shots(ax, transform, crs, pre_lon, pre_lat, post_lon, post_lat, footprint_m: float = 12.5) -> None:
    """Draw pre (yellow) and post (cyan) GEDI shot circles in pixel coords."""
    pre_col, pre_row = _world_to_px(pre_lon, pre_lat, transform, crs)
    post_col, post_row = _world_to_px(post_lon, post_lat, transform, crs)
    px = abs(transform.a)  # pixel size in CRS units (m for UTM)
    radius_px = footprint_m / px
    ax.add_patch(Circle((pre_col, pre_row), radius_px, fill=False,
                        edgecolor="yellow", lw=1.8, zorder=5))
    ax.add_patch(Circle((post_col, post_row), radius_px, fill=False,
                        edgecolor="cyan", lw=1.8, zorder=5))


def _render_panel(site: pd.Series, site_dir: Path, out_path: Path,
                  global_stretch: tuple[float, float] | None) -> None:
    chips: dict[str, tuple[np.ndarray, rasterio.Affine, str] | None] = {
        name: _read_chip(site_dir / f"{name}.tif")
        for name in ("s2_pre", "s2_post", "hansen", "palsar_pre", "palsar_post")
    }
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.0))

    # S2 pre/post — RGB
    for ax, key, title in [
        (axes[0], "s2_pre", "S2 pre"),
        (axes[1], "s2_post", "S2 post"),
    ]:
        ax.set_title(title, fontsize=10)
        c = chips[key]
        if c is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        arr, t, crs = c  # bands first axis
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
        rgb = np.clip(rgb / 0.30, 0, 1)  # simple visual stretch on reflectance
        ax.imshow(rgb, origin="upper")
        _draw_shots(ax, t, crs, site["pre_lon"], site["pre_lat"],
                    site["post_lon"], site["post_lat"])
        ax.set_xticks([]); ax.set_yticks([])

    # S2 post + Hansen overlay
    ax = axes[2]
    ax.set_title("S2 post + Hansen loss", fontsize=10)
    c2 = chips["s2_post"]
    if c2 is not None:
        arr, t, crs = c2
        # Greyscale luminance.
        gs = np.clip((0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]) / 0.30, 0, 1)
        ax.imshow(gs, cmap="gray", origin="upper")
        # Overlay Hansen.
        h = chips["hansen"]
        if h is not None:
            harr, ht, hcrs = h
            # Hansen is 30 m; the chip already has it in matching CRS but shape differs.
            # Resample by simple block-resize to S2 grid.
            from rasterio.warp import reproject, Resampling
            dst = np.zeros(gs.shape, dtype=np.float32)
            reproject(
                source=harr[0].astype(np.float32),
                destination=dst,
                src_transform=ht, src_crs=hcrs,
                dst_transform=t, dst_crs=crs,
                resampling=Resampling.nearest,
            )
            mask = dst > 0.5
            rgba = np.zeros((*gs.shape, 4))
            rgba[..., 0] = 1.0
            rgba[..., 3] = np.where(mask, 0.55, 0.0)
            ax.imshow(rgba, origin="upper")
        _draw_shots(ax, t, crs, site["pre_lon"], site["pre_lat"],
                    site["post_lon"], site["post_lat"])
    ax.set_xticks([]); ax.set_yticks([])

    # PALSAR pre/post — HV dB
    pchips = {}
    for key in ("palsar_pre", "palsar_post"):
        c = chips[key]
        if c is None:
            pchips[key] = None
            continue
        arr, t, crs = c
        db = _palsar_hv_db(arr[0])
        pchips[key] = (db, t, crs)

    if global_stretch is not None:
        vmin, vmax = global_stretch
    else:
        cat = np.concatenate(
            [v[0][np.isfinite(v[0])].ravel() for v in pchips.values() if v is not None]
        ) if any(v is not None for v in pchips.values()) else np.array([])
        if cat.size > 0:
            vmin, vmax = float(np.percentile(cat, 2)), float(np.percentile(cat, 98))
        else:
            vmin, vmax = -22.0, -8.0

    for ax, key, label in [
        (axes[3], "palsar_pre", "PALSAR-2 HV pre"),
        (axes[4], "palsar_post", "PALSAR-2 HV post"),
    ]:
        time_ms = site["pre_sar_time_ms"] if key == "palsar_pre" else site["post_sar_time_ms"]
        date = pd.Timestamp(time_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d") if not pd.isna(time_ms) else "n/a"
        title = f"{label}\n{date}, {site.get('pre_PassDirection', '?')} path {int(site.get('pre_RSP_Path_Number', 0))}"
        ax.set_title(title, fontsize=9)
        v = pchips[key]
        if v is None:
            ax.text(0.5, 0.5, "no scene", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        db, t, crs = v
        im = ax.imshow(db, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        _draw_shots(ax, t, crs, site["pre_lon"], site["pre_lat"],
                    site["post_lon"], site["post_lat"])
        ax.set_xticks([]); ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes[3:5], orientation="horizontal",
                        fraction=0.05, pad=0.08, shrink=0.8)
    cbar.set_label(r"$\sigma^0_{HV}$ (dB)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Site #{site['site_id']} — stratum={site['stratum']}  |  "
        f"ΔHV={site['delta_hv_db']:+.2f} dB, ΔAGBD={site['delta_agbd']:+.0f} Mg/ha  |  "
        f"pre_AGBD={site['pre_agbd']:.0f}, post_AGBD={site['post_agbd']:.0f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", out_path)


def cmd_panel(args: argparse.Namespace) -> int:
    sites = pd.read_parquet(args.sites)
    if args.pair_ids:
        keep = [int(x) for x in args.pair_ids.split(",")]
        sites = sites[sites["pair_id"].isin(keep)].reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Need delta_agbd / delta_hv_db / pre_agbd / post_agbd in sites — recompute from filtered v2
    filt = _v2_filtered(args.input)
    sites_full = sites.merge(
        filt[["pair_id", "delta_agbd", "delta_hv_db", "pre_agbd", "post_agbd"]],
        on="pair_id", how="left", suffixes=("", "_y"),
    )
    for col in ("delta_agbd", "delta_hv_db", "pre_agbd", "post_agbd"):
        if f"{col}_y" in sites_full.columns:
            sites_full[col] = sites_full[f"{col}_y"]
            sites_full = sites_full.drop(columns=[f"{col}_y"])

    global_stretch = (args.vmin, args.vmax) if args.global_stretch else None
    for _, row in sites_full.iterrows():
        site_id = row["site_id"]
        site_dir = args.chips_dir / f"site_{site_id}"
        out_path = args.out_dir / f"site_{site_id}"
        _render_panel(row, site_dir, out_path, global_stretch)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    common_input = dict(type=Path, default=Path("GEDI/Overlap/outputs/pairs_with_sar_v2c.parquet"))
    common_diag = Path("GEDI/Overlap/outputs/diagnostics/v2c")

    sp = sub.add_parser("select", help="Pick 5 sites and label v2 figure")
    sp.add_argument("--input", **common_input)
    sp.add_argument("--out-dir", type=Path, default=common_diag)
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--append", action="store_true",
                    help="Append new picks to existing chip_sites.parquet, "
                         "excluding already-selected pair_ids.")

    sp = sub.add_parser("download", help="Download chips for selected sites")
    sp.add_argument("--sites", type=Path, default=common_diag / "chip_sites.parquet")
    sp.add_argument("--out-dir", type=Path, default=common_diag / "chips")
    sp.add_argument("--chip-m", type=float, default=500.0)
    sp.add_argument("--s2-window-days", type=int, default=180)
    sp.add_argument("--ee-project", default="dyce-biomass")
    sp.add_argument("--pair-ids", default="", help="Comma-separated subset")

    sp = sub.add_parser("panel", help="Render diagnostic panels")
    sp.add_argument("--sites", type=Path, default=common_diag / "chip_sites.parquet")
    sp.add_argument("--input", **common_input)
    sp.add_argument("--chips-dir", type=Path, default=common_diag / "chips")
    sp.add_argument("--out-dir", type=Path, default=common_diag / "panels")
    sp.add_argument("--global-stretch", action="store_true")
    sp.add_argument("--vmin", type=float, default=-22.0)
    sp.add_argument("--vmax", type=float, default=-8.0)
    sp.add_argument("--pair-ids", default="", help="Comma-separated subset")

    sp = sub.add_parser("all", help="Run select + download + panel with defaults")
    sp.add_argument("--input", **common_input)
    sp.add_argument("--out-dir", type=Path, default=common_diag)
    sp.add_argument("--chip-m", type=float, default=500.0)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--append", action="store_true",
                    help="Append new picks to existing chip_sites.parquet.")
    sp.add_argument("--s2-window-days", type=int, default=180)
    sp.add_argument("--ee-project", default="dyce-biomass")
    sp.add_argument("--global-stretch", action="store_true")
    sp.add_argument("--vmin", type=float, default=-22.0)
    sp.add_argument("--vmax", type=float, default=-8.0)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "download":
        return cmd_download(args)
    if args.cmd == "panel":
        return cmd_panel(args)
    if args.cmd == "all":
        # select
        sel_args = argparse.Namespace(
            input=args.input, out_dir=args.out_dir, n=args.n, seed=args.seed,
            append=args.append,
        )
        rc = cmd_select(sel_args)
        if rc != 0:
            return rc
        # download
        dl_args = argparse.Namespace(
            sites=args.out_dir / "chip_sites.parquet",
            out_dir=args.out_dir / "chips",
            chip_m=args.chip_m, s2_window_days=args.s2_window_days,
            ee_project=args.ee_project, pair_ids="",
        )
        rc = cmd_download(dl_args)
        if rc != 0:
            return rc
        # panel
        pn_args = argparse.Namespace(
            sites=args.out_dir / "chip_sites.parquet",
            input=args.input,
            chips_dir=args.out_dir / "chips",
            out_dir=args.out_dir / "panels",
            global_stretch=args.global_stretch,
            vmin=args.vmin, vmax=args.vmax, pair_ids="",
        )
        return cmd_panel(pn_args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
