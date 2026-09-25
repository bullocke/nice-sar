#!/usr/bin/env python
"""Compare rank-filter and GLCM texture features on NISAR GCOV data.

Loads an HV backscatter subset, computes both texture methods on a crop,
and generates a side-by-side comparison figure with matched stretches.

Usage:
    python scripts/texture_comparison.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nice_sar.io.geotiff import read_band
from nice_sar.preprocess.calibration import linear_to_db
from nice_sar.preprocess.texture import (
    compute_glcm_texture,
    compute_local_contrast_homogeneity,
    compute_rank_texture,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- Configuration --------------------------------------------------------
DATA_DIR = Path("NISAR_Data/locations/peru/GCOV")
HV_FILE = DATA_DIR / "GCOV_freqA_HV_HVHV_2026-01-17_subset.tif"
OUTPUT_DIR = Path("NISAR_Data/locations/peru/figures")
CROP_SIZE = 150  # pixels — keeps pure-Python GLCM tractable
WINDOW_SIZE = 11
LEVELS = 32


def load_crop() -> np.ndarray:
    """Load HV data, convert to dB, return a square crop."""
    if not HV_FILE.exists():
        msg = (
            f"Data file not found: {HV_FILE}\n"
            "Download it with:  nice-sar subset --help"
        )
        raise FileNotFoundError(msg)

    arr, _ = read_band(HV_FILE)
    hv_db = linear_to_db(arr)

    # Pick a crop from the top-right of the valid data area.
    # The left ~38 % of this scene is NaN (off-swath), so the array centre
    # sits near the swath edge with noise-floor artifacts.  Offset into
    # the interior of the valid swath instead.
    r0 = 200
    c0 = hv_db.shape[1] - 200 - CROP_SIZE
    crop = hv_db[r0 : r0 + CROP_SIZE, c0 : c0 + CROP_SIZE]
    finite = np.isfinite(crop).sum()
    logger.info(
        "Crop shape: %s  finite pixels: %d / %d",
        crop.shape,
        finite,
        crop.size,
    )
    return crop


def matched_vlim(
    arr_a: np.ndarray, arr_b: np.ndarray, pct: tuple[float, float] = (2, 98)
) -> tuple[float, float]:
    """Compute shared percentile stretch across two arrays."""
    combined = np.concatenate(
        [arr_a[np.isfinite(arr_a)].ravel(), arr_b[np.isfinite(arr_b)].ravel()]
    )
    if len(combined) == 0:
        return (0.0, 1.0)
    return tuple(np.percentile(combined, pct))  # type: ignore[return-value]


def main() -> None:
    crop = load_crop()

    # ---- Compute rank-filter textures ------------------------------------
    logger.info("Computing rank-filter textures...")
    t0 = time.perf_counter()
    rank_result = compute_rank_texture(crop, window_size=WINDOW_SIZE, levels=LEVELS)
    rank_time = time.perf_counter() - t0
    logger.info("Rank-filter done in %.2f s", rank_time)

    # ---- Compute local contrast / homogeneity ----------------------------
    logger.info("Computing local contrast / homogeneity...")
    t0 = time.perf_counter()
    local_contrast, local_homogeneity = compute_local_contrast_homogeneity(
        crop, window_size=WINDOW_SIZE
    )
    ch_time = time.perf_counter() - t0
    logger.info("Local contrast/homogeneity done in %.2f s", ch_time)

    # ---- Compute GLCM Haralick textures ----------------------------------
    glcm_features = [
        "contrast",
        "homogeneity",
        "entropy",
        "variance",
        "asm",
        "correlation",
        "sum_average",
    ]
    logger.info("Computing GLCM textures (this may take a minute)...")
    t0 = time.perf_counter()
    glcm_result = compute_glcm_texture(
        crop,
        window_size=WINDOW_SIZE,
        levels=LEVELS,
        features=glcm_features,
    )
    glcm_time = time.perf_counter() - t0
    logger.info("GLCM done in %.2f s", glcm_time)

    if rank_result is None or glcm_result is None:
        logger.error("One or both texture methods returned None — no valid data.")
        return

    # ---- Build the comparison figure -------------------------------------
    # 6 rows × 2 cols:
    #   Row 0: input dB           |  (blank or ASM)
    #   Row 1: rank entropy       |  GLCM entropy
    #   Row 2: rank variance      |  GLCM variance
    #   Row 3: local contrast     |  GLCM contrast
    #   Row 4: local homogeneity  |  GLCM homogeneity
    #   Row 5: GLCM correlation   |  GLCM ASM

    rows = [
        {
            "left_data": crop,
            "left_title": "Input HV (dB)",
            "right_data": glcm_result["asm"],
            "right_title": "GLCM: ASM",
            "cmap": ("gray", "magma"),
            "shared_stretch": False,
            "pct": (2, 98),
        },
        {
            "left_data": rank_result["entropy"],
            "left_title": "Rank: entropy",
            "right_data": glcm_result["entropy"],
            "right_title": "GLCM: entropy",
            "cmap": ("inferno", "inferno"),
            "shared_stretch": True,
            "pct": (2, 98),
        },
        {
            "left_data": rank_result["variance"],
            "left_title": "Rank: variance",
            "right_data": glcm_result["variance"],
            "right_title": "GLCM: variance",
            "cmap": ("viridis", "viridis"),
            "shared_stretch": False,  # different scales
            "pct": (2, 98),
        },
        {
            "left_data": local_contrast,
            "left_title": "Local: contrast",
            "right_data": glcm_result["contrast"],
            "right_title": "GLCM: contrast",
            "cmap": ("magma", "magma"),
            "shared_stretch": False,  # different scales (variance vs co-occurrence)
            "pct": (2, 98),
        },
        {
            "left_data": local_homogeneity,
            "left_title": "Local: homogeneity",
            "right_data": glcm_result["homogeneity"],
            "right_title": "GLCM: homogeneity",
            "cmap": ("cividis", "cividis"),
            "shared_stretch": False,  # different scales
            "pct": (2, 98),
        },
        {
            "left_data": rank_result["mean"],
            "left_title": "Rank: mean",
            "right_data": glcm_result["sum_average"],
            "right_title": "GLCM: sum average",
            "cmap": ("viridis", "viridis"),
            "shared_stretch": False,  # different scales
            "pct": (2, 98),
        },
        {
            "left_data": glcm_result["correlation"],
            "left_title": "GLCM: correlation",
            "right_data": glcm_result["asm"],
            "right_title": "GLCM: ASM",
            "cmap": ("coolwarm", "magma"),
            "shared_stretch": False,
            "pct": (2, 98),
        },
    ]

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.2 * len(rows)))
    fig.suptitle(
        f"Texture Comparison — Peru HV  (window={WINDOW_SIZE}, levels={LEVELS})\n"
        f"Rank: {rank_time:.2f}s  |  GLCM: {glcm_time:.2f}s",
        fontsize=13,
        y=1.01,
    )

    for i, row_cfg in enumerate(rows):
        ax_l, ax_r = axes[i]

        # --- Left panel ---
        left = row_cfg["left_data"]
        right = row_cfg["right_data"]

        if row_cfg["shared_stretch"]:
            vmin, vmax = matched_vlim(left, right, pct=row_cfg["pct"])
            vmin_l = vmin_r = vmin
            vmax_l = vmax_r = vmax
        else:
            fl = left[np.isfinite(left)]
            fr = right[np.isfinite(right)]
            vmin_l, vmax_l = (
                np.percentile(fl, row_cfg["pct"]) if len(fl) else (0, 1)
            )
            vmin_r, vmax_r = (
                np.percentile(fr, row_cfg["pct"]) if len(fr) else (0, 1)
            )

        cmap_l, cmap_r = row_cfg["cmap"]

        im_l = ax_l.imshow(left, cmap=cmap_l, vmin=vmin_l, vmax=vmax_l)
        ax_l.set_title(row_cfg["left_title"], fontsize=10)
        ax_l.axis("off")
        fig.colorbar(im_l, ax=ax_l, shrink=0.7, pad=0.02)

        im_r = ax_r.imshow(right, cmap=cmap_r, vmin=vmin_r, vmax=vmax_r)
        ax_r.set_title(row_cfg["right_title"], fontsize=10)
        ax_r.axis("off")
        fig.colorbar(im_r, ax=ax_r, shrink=0.7, pad=0.02)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "texture_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    logger.info("Figure saved → %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
