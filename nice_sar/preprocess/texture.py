"""Texture feature extraction for SAR data.

Provides two approaches to texture analysis:

* **Rank-filter textures** (``compute_rank_texture``) — fast first-order
  local statistics (entropy, mean, variance, range) using ``skimage.filters.rank``.
* **True GLCM textures** (``compute_glcm_texture``) — sliding-window
  Gray-Level Co-occurrence Matrix with all 13 Haralick features.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage
from skimage.filters import rank
from skimage.morphology import disk

from nice_sar._types import ArrayFloat32

logger = logging.getLogger(__name__)


def compute_rank_texture(
    data: np.ndarray, window_size: int = 11, levels: int = 32
) -> dict[str, ArrayFloat32] | None:
    """Compute texture features using rank filters on quantized data.

    A fast first-order approximation that computes local entropy, mean,
    variance, and range using ``skimage.filters.rank`` on a quantized
    image.  These capture local intensity distributions but **not**
    spatial co-occurrence structure.  For true second-order texture
    features see :func:`compute_glcm_texture`.

    Args:
        data: 2D backscatter array (typically in dB).
        window_size: Diameter of the disk-shaped filter footprint.
        levels: Number of quantization levels for the input.

    Returns:
        Dictionary with keys ``entropy``, ``mean``, ``variance``, ``range``,
        or ``None`` if no valid data exists.
    """
    logger.info("Computing texture features (window=%d)", window_size)
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return None

    valid_data = data[valid_mask]
    data_min, data_max = np.percentile(valid_data, [2, 98])
    normalized = np.clip((data - data_min) / (data_max - data_min + 1e-10), 0, 1)
    scaled = (normalized * (levels - 1)).astype(np.uint8)
    fp = disk(window_size // 2)

    textures: dict[str, ArrayFloat32] = {}
    try:
        textures["entropy"] = rank.entropy(scaled, fp).astype(np.float32)
        textures["mean"] = rank.mean(scaled, fp).astype(np.float32)
        # Variance from float local moments over the same disk; squaring the
        # uint8 levels in place would overflow for levels > 16.
        kernel = fp.astype(np.float64) / fp.sum()
        scaled_f = scaled.astype(np.float64)
        local_mean = ndimage.convolve(scaled_f, kernel, mode="reflect")
        local_sq = ndimage.convolve(scaled_f**2, kernel, mode="reflect")
        textures["variance"] = np.maximum(local_sq - local_mean**2, 0).astype(np.float32)
        textures["range"] = rank.gradient(scaled, fp).astype(np.float32)
        for k in textures:
            textures[k][~valid_mask] = np.nan
    except Exception:
        logger.warning("Could not compute all texture features", exc_info=True)
        return None

    return textures


def compute_local_contrast_homogeneity(
    data: np.ndarray, window_size: int = 11
) -> tuple[ArrayFloat32, ArrayFloat32]:
    """Compute local contrast (std) and homogeneity (1/(1+CV)).

    Args:
        data: 2D array (typically dB backscatter).
        window_size: Size of the filter window.

    Returns:
        Tuple of (contrast, homogeneity) arrays.
    """
    valid_mask = np.isfinite(data)
    contrast = ndimage.generic_filter(
        data, np.nanstd, size=window_size, mode="constant", cval=np.nan
    )
    mean = ndimage.generic_filter(data, np.nanmean, size=window_size, mode="constant", cval=np.nan)
    homogeneity = 1.0 / (1.0 + contrast / (mean + 1e-10))
    contrast[~valid_mask] = np.nan
    homogeneity[~valid_mask] = np.nan
    return contrast.astype(np.float32), homogeneity.astype(np.float32)


# ---------------------------------------------------------------------------
# True GLCM (Gray-Level Co-occurrence Matrix) Haralick texture features
# ---------------------------------------------------------------------------

#: All 13 Haralick feature names in canonical order.
HARALICK_FEATURES: list[str] = [
    "asm",
    "contrast",
    "correlation",
    "variance",
    "homogeneity",
    "sum_average",
    "sum_variance",
    "sum_entropy",
    "entropy",
    "difference_variance",
    "difference_entropy",
    "imc1",
    "imc2",
]


def _has_numba() -> bool:
    """Return True if numba is importable."""
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def _quantize(
    data: np.ndarray,
    levels: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Percentile-based quantization to ``[0, levels-1]`` uint8.

    Args:
        data: Input 2D array.
        levels: Number of output gray levels.
        valid_mask: Boolean mask of finite pixels.

    Returns:
        Quantized uint8 array.
    """
    valid_data = data[valid_mask]
    data_min, data_max = np.percentile(valid_data, [2, 98])
    normalized = np.clip((data - data_min) / (data_max - data_min + 1e-10), 0, 1)
    return (normalized * (levels - 1)).astype(np.uint8)


def _haralick_from_glcm(p: np.ndarray) -> np.ndarray:
    """Compute all 13 Haralick features from a normalized GLCM.

    Args:
        p: Normalized co-occurrence matrix of shape ``(L, L)`` that sums
           to 1.  If the matrix is all-zero an array of NaNs is returned.

    Returns:
        1-D float64 array of length 13 (order matches
        :data:`HARALICK_FEATURES`).
    """
    eps = 1e-15
    levels = p.shape[0]
    result = np.full(13, np.nan, dtype=np.float64)

    total = p.sum()
    if total < eps:
        return result

    p = p / total  # ensure normalized
    p_safe = np.where(p > 0, p, eps)  # avoid log(0)

    i_idx = np.arange(levels, dtype=np.float64)
    j_idx = np.arange(levels, dtype=np.float64)
    I, J = np.meshgrid(i_idx, j_idx, indexing="ij")  # noqa: E741

    # Marginals
    px = p.sum(axis=1)  # shape (L,)
    py = p.sum(axis=0)  # shape (L,)

    mu_x = (i_idx * px).sum()
    mu_y = (j_idx * py).sum()
    sig_x = np.sqrt(((i_idx - mu_x) ** 2 * px).sum())
    sig_y = np.sqrt(((j_idx - mu_y) ** 2 * py).sum())

    # p_{x+y} and p_{x-y} distributions
    p_xpy = np.zeros(2 * levels - 1, dtype=np.float64)
    p_xmy = np.zeros(levels, dtype=np.float64)
    for i in range(levels):
        for j in range(levels):
            p_xpy[i + j] += p[i, j]
            p_xmy[abs(i - j)] += p[i, j]

    # 1. ASM (Angular Second Moment / Energy²)
    result[0] = (p**2).sum()

    # 2. Contrast
    result[1] = ((I - J) ** 2 * p).sum()

    # 3. Correlation
    if sig_x > eps and sig_y > eps:
        result[2] = ((I - mu_x) * (J - mu_y) * p).sum() / (sig_x * sig_y)
    else:
        result[2] = 0.0

    # 4. Variance (Sum of Squares: Variance)
    mu = (I * p).sum()
    result[3] = ((I - mu) ** 2 * p).sum()

    # 5. Homogeneity (IDM — Inverse Difference Moment)
    result[4] = (p / (1.0 + (I - J) ** 2)).sum()

    # 6. Sum Average
    k_xpy = np.arange(2 * levels - 1, dtype=np.float64)
    result[5] = (k_xpy * p_xpy).sum()

    # 7. Sum Entropy
    p_xpy_safe = np.where(p_xpy > 0, p_xpy, eps)
    sum_entropy = -(p_xpy * np.log(p_xpy_safe)).sum()
    result[7] = sum_entropy

    # 8. Sum Variance (uses sum_entropy)
    result[6] = ((k_xpy - sum_entropy) ** 2 * p_xpy).sum()

    # 9. Entropy
    result[8] = -(p * np.log(p_safe)).sum()

    # 10. Difference Variance
    k_xmy = np.arange(levels, dtype=np.float64)
    diff_mean = (k_xmy * p_xmy).sum()
    result[9] = ((k_xmy - diff_mean) ** 2 * p_xmy).sum()

    # 11. Difference Entropy
    p_xmy_safe = np.where(p_xmy > 0, p_xmy, eps)
    result[10] = -(p_xmy * np.log(p_xmy_safe)).sum()

    # 12–13. Information Measures of Correlation (IMC1, IMC2)
    hxy = result[8]  # entropy of p(i,j)

    px_safe = np.where(px > 0, px, eps)
    py_safe = np.where(py > 0, py, eps)
    log_px = np.log(px_safe)
    log_py = np.log(py_safe)

    # HXY1: -sum p(i,j) * log(px(i) * py(j))
    hxy1 = 0.0
    for i in range(levels):
        for j in range(levels):
            if p[i, j] > 0:
                hxy1 -= p[i, j] * (log_px[i] + log_py[j])

    # HXY2: -sum px(i)*py(j) * log(px(i)*py(j))
    hxy2 = 0.0
    for i in range(levels):
        for j in range(levels):
            pij_ind = px[i] * py[j]
            if pij_ind > 0:
                hxy2 -= pij_ind * np.log(pij_ind)

    hx = -(px * np.log(px_safe)).sum()
    hy = -(py * np.log(py_safe)).sum()

    max_hxy = max(hx, hy)
    if max_hxy > eps:
        result[11] = (hxy - hxy1) / max_hxy  # IMC1
    else:
        result[11] = 0.0

    imc2_arg = 1.0 - np.exp(-2.0 * (hxy2 - hxy))
    result[12] = np.sqrt(max(imc2_arg, 0.0))  # IMC2

    return result


# ---- Numba-accelerated backend -------------------------------------------

_NUMBA_KERNEL = None  # lazily compiled


def _get_numba_kernel():  # type: ignore[no-untyped-def]
    """Return (or compile) the numba-jitted sliding-window GLCM kernel."""
    global _NUMBA_KERNEL  # noqa: PLW0603
    if _NUMBA_KERNEL is not None:
        return _NUMBA_KERNEL

    import numba as nb

    @nb.jit(nopython=True, cache=True)
    def _kernel(
        quantized: np.ndarray,
        valid: np.ndarray,
        rows: int,
        cols: int,
        half: int,
        levels: int,
        distances: np.ndarray,
        d_rows: np.ndarray,
        d_cols: np.ndarray,
        n_offsets: int,
    ) -> np.ndarray:
        """Numba kernel: sliding-window GLCM → 13 Haralick features."""
        n_feat = 13
        out = np.full((n_feat, rows, cols), np.nan, dtype=np.float64)
        eps = 1e-15
        glcm = np.zeros((levels, levels), dtype=np.float64)

        for r in range(rows):
            for c in range(cols):
                if not valid[r, c]:
                    continue

                # ---- Accumulate GLCM over all offsets ---
                glcm[:, :] = 0.0
                total = 0.0
                for oi in range(n_offsets):
                    dr = d_rows[oi]
                    dc = d_cols[oi]
                    r_lo = max(r - half, 0)
                    r_hi = min(r + half + 1, rows)
                    c_lo = max(c - half, 0)
                    c_hi = min(c + half + 1, cols)
                    for rr in range(r_lo, r_hi):
                        for cc in range(c_lo, c_hi):
                            nr = rr + dr
                            nc = cc + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and valid[rr, cc]
                                and valid[nr, nc]
                            ):
                                gi = quantized[rr, cc]
                                gj = quantized[nr, nc]
                                glcm[gi, gj] += 1.0
                                glcm[gj, gi] += 1.0
                                total += 2.0

                if total < eps:
                    continue

                # Normalize
                for i in range(levels):
                    for j in range(levels):
                        glcm[i, j] /= total

                # ---- Marginals ----
                px = np.zeros(levels, dtype=np.float64)
                py = np.zeros(levels, dtype=np.float64)
                for i in range(levels):
                    for j in range(levels):
                        px[i] += glcm[i, j]
                        py[j] += glcm[i, j]

                mu_x = 0.0
                mu_y = 0.0
                for i in range(levels):
                    mu_x += i * px[i]
                    mu_y += i * py[i]

                sig_x2 = 0.0
                sig_y2 = 0.0
                for i in range(levels):
                    sig_x2 += (i - mu_x) ** 2 * px[i]
                    sig_y2 += (i - mu_y) ** 2 * py[i]
                sig_x = np.sqrt(sig_x2)
                sig_y = np.sqrt(sig_y2)

                # p_{x+y}, p_{x-y}
                p_xpy = np.zeros(2 * levels - 1, dtype=np.float64)
                p_xmy = np.zeros(levels, dtype=np.float64)
                for i in range(levels):
                    for j in range(levels):
                        p_xpy[i + j] += glcm[i, j]
                        d_abs = i - j
                        if d_abs < 0:
                            d_abs = -d_abs
                        p_xmy[d_abs] += glcm[i, j]

                # ---- Features ----
                asm_val = 0.0
                contrast_val = 0.0
                corr_num = 0.0
                var_sum = 0.0
                hom_val = 0.0
                ent_val = 0.0
                mu_ij = 0.0

                for i in range(levels):
                    for j in range(levels):
                        pij = glcm[i, j]
                        asm_val += pij * pij
                        diff = float(i - j)
                        contrast_val += diff * diff * pij
                        corr_num += (i - mu_x) * (j - mu_y) * pij
                        mu_ij += i * pij
                        hom_val += pij / (1.0 + diff * diff)
                        if pij > eps:
                            ent_val -= pij * np.log(pij)

                for i in range(levels):
                    var_sum += (i - mu_ij) ** 2 * glcm[i, :].sum()

                out[0, r, c] = asm_val
                out[1, r, c] = contrast_val
                if sig_x > eps and sig_y > eps:
                    out[2, r, c] = corr_num / (sig_x * sig_y)
                else:
                    out[2, r, c] = 0.0
                out[3, r, c] = var_sum
                out[4, r, c] = hom_val

                # Sum average, sum entropy, sum variance
                sa = 0.0
                se = 0.0
                for k in range(2 * levels - 1):
                    sa += k * p_xpy[k]
                    if p_xpy[k] > eps:
                        se -= p_xpy[k] * np.log(p_xpy[k])
                sv = 0.0
                for k in range(2 * levels - 1):
                    sv += (k - se) ** 2 * p_xpy[k]
                out[5, r, c] = sa
                out[6, r, c] = sv
                out[7, r, c] = se

                out[8, r, c] = ent_val

                # Difference variance, difference entropy
                dm = 0.0
                de = 0.0
                for k in range(levels):
                    dm += k * p_xmy[k]
                dv = 0.0
                for k in range(levels):
                    dv += (k - dm) ** 2 * p_xmy[k]
                    if p_xmy[k] > eps:
                        de -= p_xmy[k] * np.log(p_xmy[k])
                out[9, r, c] = dv
                out[10, r, c] = de

                # IMC1, IMC2
                hx = 0.0
                hy = 0.0
                hxy1 = 0.0
                hxy2 = 0.0
                for i in range(levels):
                    if px[i] > eps:
                        hx -= px[i] * np.log(px[i])
                    if py[i] > eps:
                        hy -= py[i] * np.log(py[i])
                for i in range(levels):
                    for j in range(levels):
                        lpx = np.log(px[i]) if px[i] > eps else np.log(eps)
                        lpy = np.log(py[j]) if py[j] > eps else np.log(eps)
                        if glcm[i, j] > 0:
                            hxy1 -= glcm[i, j] * (lpx + lpy)
                        pij_ind = px[i] * py[j]
                        if pij_ind > 0:
                            hxy2 -= pij_ind * np.log(pij_ind)
                max_hxy = hx if hx > hy else hy
                if max_hxy > eps:
                    out[11, r, c] = (ent_val - hxy1) / max_hxy
                else:
                    out[11, r, c] = 0.0
                imc2_arg = 1.0 - np.exp(-2.0 * (hxy2 - ent_val))
                if imc2_arg > 0:
                    out[12, r, c] = np.sqrt(imc2_arg)
                else:
                    out[12, r, c] = 0.0

        return out

    _NUMBA_KERNEL = _kernel
    return _NUMBA_KERNEL


def _offset_arrays(
    distances: list[int],
    angles: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (distance, angle) pairs to pixel offset arrays.

    Returns:
        Tuple of (distances_arr, d_rows, d_cols) — all int arrays.
    """
    d_rows_list: list[int] = []
    d_cols_list: list[int] = []
    dist_list: list[int] = []
    for d in distances:
        for a in angles:
            dr = -int(round(d * np.sin(a)))
            dc = int(round(d * np.cos(a)))
            d_rows_list.append(dr)
            d_cols_list.append(dc)
            dist_list.append(d)
    return (
        np.array(dist_list, dtype=np.int64),
        np.array(d_rows_list, dtype=np.int64),
        np.array(d_cols_list, dtype=np.int64),
    )


def _glcm_sliding_python(
    quantized: np.ndarray,
    valid_mask: np.ndarray,
    levels: int,
    half: int,
    distances: list[int],
    angles: list[float],
) -> np.ndarray:
    """Pure-Python sliding-window GLCM using skimage for each window."""
    from skimage.feature import graycomatrix

    rows, cols = quantized.shape
    n_feat = 13
    out = np.full((n_feat, rows, cols), np.nan, dtype=np.float64)

    dist_arr = np.array(distances, dtype=int)
    ang_arr = np.array(angles, dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r, c]:
                continue

            r_lo = max(r - half, 0)
            r_hi = min(r + half + 1, rows)
            c_lo = max(c - half, 0)
            c_hi = min(c + half + 1, cols)
            window = quantized[r_lo:r_hi, c_lo:c_hi]
            w_valid = valid_mask[r_lo:r_hi, c_lo:c_hi]

            # Mask invalid pixels by setting to 0 and building
            # the GLCM only over valid pairs.
            w = window.copy()
            w[~w_valid] = 0

            glcm = graycomatrix(
                w,
                distances=dist_arr,
                angles=ang_arr,
                levels=levels,
                symmetric=True,
                normed=True,
            )

            # Average across all distance/angle combos → (L, L)
            p = glcm.mean(axis=(2, 3))
            p_sum = p.sum()
            if p_sum < 1e-15:
                continue
            p = p / p_sum

            feats = _haralick_from_glcm(p)
            for fi in range(n_feat):
                out[fi, r, c] = feats[fi]

    return out


def compute_glcm_texture(
    data: np.ndarray,
    window_size: int = 11,
    levels: int = 32,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    features: list[str] | None = None,
) -> dict[str, ArrayFloat32] | None:
    """Compute true GLCM Haralick texture features over a sliding window.

    For each pixel, a Gray-Level Co-occurrence Matrix is built from the
    surrounding ``window_size × window_size`` neighbourhood and all 13
    Haralick features are derived.  Features are averaged across the
    requested distance/angle combinations.

    When ``numba`` is installed the computation is JIT-compiled and
    significantly faster.  Otherwise a pure-Python fallback using
    :func:`skimage.feature.graycomatrix` is used (much slower on large
    images).

    Args:
        data: 2D backscatter array (typically in dB).
        window_size: Side length of the sliding window (must be odd).
        levels: Number of gray-level quantization bins.
        distances: Pixel-pair distances for the co-occurrence matrix.
            Defaults to ``[1]``.
        angles: Pixel-pair angles in radians.  Defaults to
            ``[0, π/4, π/2, 3π/4]``.
        features: Subset of feature names to return.  ``None`` returns
            all 13 Haralick features.  Valid names are listed in
            :data:`HARALICK_FEATURES`.

    Returns:
        Dictionary mapping feature names to 2D float32 arrays, or
        ``None`` if no valid data exists.

    Raises:
        ValueError: If an unknown feature name is requested.
    """
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    if features is not None:
        unknown = set(features) - set(HARALICK_FEATURES)
        if unknown:
            msg = f"Unknown Haralick feature(s): {unknown}"
            raise ValueError(msg)

    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return None

    half = window_size // 2
    quantized = _quantize(data, levels, valid_mask)

    if _has_numba():
        logger.info(
            "Computing GLCM texture (numba, window=%d, levels=%d)",
            window_size,
            levels,
        )
        dist_arr, dr_arr, dc_arr = _offset_arrays(distances, angles)
        kernel = _get_numba_kernel()
        raw = kernel(
            quantized,
            valid_mask,
            data.shape[0],
            data.shape[1],
            half,
            levels,
            dist_arr,
            dr_arr,
            dc_arr,
            len(dr_arr),
        )
    else:
        logger.info(
            "Computing GLCM texture (pure-python, window=%d, levels=%d)",
            window_size,
            levels,
        )
        logger.warning(
            "numba not installed — GLCM computation will be slow. "
            "Install numba for ~50-100x speedup: pip install numba"
        )
        raw = _glcm_sliding_python(quantized, valid_mask, levels, half, distances, angles)

    # Build output dict
    requested = features if features is not None else HARALICK_FEATURES
    textures: dict[str, ArrayFloat32] = {}
    for name in requested:
        idx = HARALICK_FEATURES.index(name)
        arr = raw[idx].astype(np.float32)
        arr[~valid_mask] = np.nan
        textures[name] = arr

    return textures
