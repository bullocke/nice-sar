# Texture Analysis: Rank Filters vs GLCM

`nice-sar` provides two families of texture-feature extraction for SAR
backscatter images.  This guide explains the theory, trade-offs, and
practical usage of each.

## Quick comparison

| | Rank filters | GLCM (Haralick) |
|---|---|---|
| **Function** | `compute_rank_texture()` | `compute_glcm_texture()` |
| **Features** | mean, variance, entropy, range | 13 Haralick features |
| **Order** | First-order (histogram) | Second-order (co-occurrence) |
| **Speed** | Milliseconds | Minutes (pure Python) or seconds (numba) |
| **Best for** | Quick screening, large images | Classification, detailed analysis |

## Background

### First-order texture (rank filters)

First-order features describe the *distribution* of pixel values inside a
local window, ignoring spatial arrangement.  Two windows with identical
histograms but different spatial patterns will produce the same features.

`compute_rank_texture()` uses `skimage.filters.rank` to efficiently compute
local mean, variance, entropy, and range over a circular structuring element.

### Second-order texture (GLCM)

The **Gray-Level Co-occurrence Matrix** (GLCM) tallies how often pairs of
pixel values appear at a given spatial offset (distance, angle).  From the
normalised co-occurrence matrix $P(i, j)$, Haralick (1973) defined 13
features that capture contrast, orderliness, and correlation of spatial
patterns.

For example, a smooth region produces a GLCM concentrated along the diagonal
(high homogeneity, high ASM, low contrast), while a high-frequency texture
produces off-diagonal entries (low homogeneity, low ASM, high contrast).

## Haralick features

The 13 features computed by `compute_glcm_texture()`:

| # | Name | Measures |
|---|------|----------|
| 1 | `asm` | Angular Second Moment — uniformity / orderliness |
| 2 | `contrast` | Local intensity variation |
| 3 | `correlation` | Linear dependency between gray levels |
| 4 | `variance` | Spread around the mean gray level |
| 5 | `homogeneity` | Inverse Difference Moment — closeness to the diagonal |
| 6 | `sum_average` | Mean of $P_{x+y}$ |
| 7 | `sum_variance` | Variance of $P_{x+y}$ |
| 8 | `sum_entropy` | Entropy of $P_{x+y}$ |
| 9 | `entropy` | Randomness of the GLCM |
| 10 | `difference_variance` | Variance of $P_{x-y}$ |
| 11 | `difference_entropy` | Entropy of $P_{x-y}$ |
| 12 | `imc1` | Information Measure of Correlation 1 |
| 13 | `imc2` | Information Measure of Correlation 2 |

## Usage

### Rank filters (fast)

```python
from nice_sar.preprocess.texture import compute_rank_texture

textures = compute_rank_texture(hv_db, window_size=11, levels=32)
# textures["mean"], textures["entropy"], ...
```

### GLCM (detailed)

```python
from nice_sar.preprocess.texture import compute_glcm_texture

textures = compute_glcm_texture(
    hv_db,
    window_size=11,
    levels=32,
    distances=[1],
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
)
# textures["contrast"], textures["homogeneity"], textures["entropy"], ...
```

Request a subset of features to save memory:

```python
textures = compute_glcm_texture(
    hv_db,
    window_size=11,
    features=["contrast", "homogeneity", "entropy"],
)
```

### Performance: numba acceleration

Without numba, the pure-Python sliding-window implementation is slow
(minutes on a 200×200 crop).  Install numba for ~50–100× speedup:

```bash
pip install "nice-sar[accel]"
# or directly:
pip install numba
```

The function detects numba at runtime and selects the faster backend
automatically.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 11 | Side length of the sliding window (must be odd) |
| `levels` | 32 | Number of quantisation bins — lower = faster, coarser |
| `distances` | `[1]` | Pixel offsets for co-occurrence pairs |
| `angles` | `[0, π/4, π/2, 3π/4]` | Directions for co-occurrence pairs |
| `features` | all 13 | Subset of feature names to return |

## When to use which method

- **Quick look or large scenes:** Use `compute_rank_texture()`. It runs in
  milliseconds and gives a good first impression of spatial heterogeneity.

- **Classification or change detection:** Use `compute_glcm_texture()`. The
  second-order features provide additional discriminating power — two land
  covers with similar backscatter distributions but different spatial patterns
  (e.g. plantation vs natural forest) may only be separable via GLCM.

- **Feature selection:** Many Haralick features are highly correlated.
  A correlation analysis (see the notebook `10_texture_comparison.ipynb`)
  can identify a compact, low-redundancy subset for your application.

## References

- Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). *Textural Features
  for Image Classification*. IEEE Transactions on Systems, Man, and
  Cybernetics, SMC-3(6), 610–621.

## See also

- [Notebook: Texture Comparison](notebooks.md) — interactive comparison with
  real NISAR GCOV data
- [API: preprocess](api/preprocess.md) — full function signatures and
  docstrings
