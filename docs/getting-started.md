# Getting Started

This guide walks through the core `nice-sar` workflow: authenticate, search, read, preprocess,
and visualize NISAR SAR data — all from the cloud.

## Prerequisites

1. A free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account
2. Python ≥ 3.10
3. `nice-sar` installed:

```bash
pip install nice-sar
```

## 1. Authenticate

`nice-sar` uses [`earthaccess`](https://earthaccess.readthedocs.io/) for NASA Earthdata
authentication. On first use it will prompt for your credentials and cache them in `~/.netrc`.

```python
import os

from nice_sar.auth import get_https_filesystem, get_s3_filesystem, login

login()
if os.environ.get("AWS_DEFAULT_REGION") == "us-west-2":
    fs = get_s3_filesystem()  # Direct S3 reads inside AWS us-west-2
    granule_access = "s3"
else:
    fs = get_https_filesystem()  # HTTPS streaming everywhere else
    granule_access = "https"
```

!!! tip
    For non-interactive environments (CI, SLURM), set `EARTHDATA_USERNAME` and
    `EARTHDATA_PASSWORD` environment variables or configure `~/.netrc`.

## 2. Search for Data

Search for NISAR GCOV products over an area of interest:

```python
from nice_sar.search import search_gcov, summarize_results

results = search_gcov(
    bbox=(-63.5, -10.0, -62.5, -9.0),  # Rondônia, Brazil
    start="2026-06-17",
    end="2026-09-30",
    max_results=20,
)
for s in summarize_results(results):
    print(s.maturity, s.crid, s.track, s.direction, s.frame, s.start, s.granule_id)
```

### Data maturity: PROVISIONAL vs BETA

ASF archives NISAR products in a separate collection for each data maturity, and
every search function takes a `maturity` argument:

| `maturity` | Collections | What it contains |
|---|---|---|
| `"provisional"` (default) | `NISAR_L2_GCOV_PROVISIONAL_V1`, ... | Calibrated, partially validated products (CRID `P05023`+). Forward processing from 2026-06-17, plus [supplemental frames](supplemental-frames.md) back to October 2025. |
| `"beta"` | `NISAR_L2_GCOV_BETA_V1`, ... | Pre-calibration products from the February 2026 release (acquisitions October 2025 to January 2026, CRIDs `X05007`-`X05010`). |
| `"validated"` | `NISAR_L2_GCOV_V1`, ... | Fully validated products; expected from the late-2026 reprocessing campaign. |
| `"any"` | all of the above | Every maturity, labeled in the results. |

```python
from nice_sar.search import search_nisar, nisar_short_names

beta = search_nisar("GCOV", bbox=aoi, maturity="beta")        # pre-calibration only
both = search_nisar("GUNW", bbox=aoi, maturity="any")         # mixed, labeled
series = search_nisar("GCOV", track=68, frame=93, direction="D")  # one frame

nisar_short_names("GUNW", "provisional")  # ['NISAR_L2_GUNW_PROVISIONAL_V1']
```

`track`, `frame`, and `direction` filter on the NISAR CMR attributes, so they work
with or without a bounding box. For granules already on disk,
`parse_granule_name(path).maturity` infers maturity from the CRID in the file name.
`search_earthdata(product_type=..., maturity=...)` accepts the same maturity values
for `earthaccess` users.

!!! warning "Mixing maturities"
    Differences between BETA and PROVISIONAL products can come from changes in the
    processing software rather than from the surface. Keep them in separate
    analyses. Also note:

    - There is a permanent instrument data gap from `2026-07-27T22:03:25Z` to
      `2026-08-10T00:55:27Z`.
    - PROVISIONAL polarimetric phase is not fully calibrated. For coherent GCOV
      analysis, multiply `HVVH` by e<sup>+j59°</sup> and `HHHV`, `HHVV`, `VHVV` by
      e<sup>-j59°</sup> (diagonal terms such as `HHHH` and `HVHV` are unaffected).
    - Some frames alternate between dual-pol (`DHDH`) and single-pol (`SHSH`)
      acquisitions. Check the polarization token in the granule name before
      building an HV time series.

    See the [PROVISIONAL known issues](nisar-docs/data-availability/provisional-known-issues.md)
    for the full list.

### From the command line

```bash
# Tabulate PROVISIONAL GUNW granules for one frame
nice-sar search --product GUNW --track 68 --frame 93 --direction D

# Compare maturities over an AOI (JSON output for scripting)
nice-sar search --product GCOV --bbox=-63.5,-10,-62.5,-9 --maturity any --json

# Download full HDF5 granules (skips files already downloaded)
nice-sar download --product GCOV --bbox=-63.5,-10,-62.5,-9 \
    --start 2026-06-18 --end 2026-06-19 --max-granules 1 -o NISAR_Data/GCOV
```

`scripts/examples/amazon_frontier_download.py` is a complete example. It selects
early and late dry-season GCOV and GUNW granules over Rondônia, downloads them, and
checks each file's track and frame.

## 3. Open and Read

Open a NISAR HDF5 file and read a polarization layer as a lazy `xarray.DataArray`:

```python
from nice_sar.auth import get_granule_url
from nice_sar.io import open_nisar, read_gcov

granule_url = get_granule_url(results[0], access=granule_access)
h5 = open_nisar(granule_url, filesystem=fs)
hh = read_gcov(h5, frequency="A", polarization="HH")
print(hh)  # xarray.DataArray with spatial coords
```

The data is backed by Dask — no pixels are loaded until you call `.values` or `.compute()`.

## 4. Preprocess

Apply calibration and speckle filtering:

```python
from nice_sar.preprocess import linear_to_db, lee_filter, multilook

# Convert to dB
hh_db = linear_to_db(hh.values)

# Speckle filter
hh_filtered = lee_filter(hh.values, window_size=7)

# Multilook (spatial averaging)
hh_ml = multilook(hh.values, looks_y=2, looks_x=2)
```

## 5. Visualize

Create RGB composites:

```python
from nice_sar.viz.rgb import rgb_standard_dualpol

hv = read_gcov(h5, frequency="A", polarization="HVHV")
rgb, bands = rgb_standard_dualpol(linear_to_db(hh.values), linear_to_db(hv.values))
```

Export to GeoTIFF:

```python
from nice_sar.io import export_geotiff

export_geotiff(hh_db, "hh_db.tif", transform=..., crs=...)
```

## 6. Polarimetric Analysis

For quad-pol data, run decompositions:

```python
from nice_sar.analysis import freeman_durden, cloude_pottier, build_coherency_matrix

T = build_coherency_matrix(covariances, window=5)
Ps, Pd, Pv = freeman_durden(T)
H, A, alpha = cloude_pottier(T)
```

## 7. Subset & Download

Don't need the full scene? Download only the pixels inside a bounding box:

```python
from nice_sar.io.subset import subset_product

tif_paths = subset_product(
    source=granule_url,
    product="GCOV",
    bbox=(-58.24, 4.40, -58.06, 4.57),  # Guyana AOI
    frequency="A",
    polarizations=["HH", "HV"],
    output_dir="my_subset",
    filesystem=fs,
    confirm=False,
)
```

The function streams only the required pixels from the remote HDF5 file and exports
each band as a GeoTIFF. It supports GCOV, GSLC, GUNW, and GOFF products.

The bbox can also be a GeoJSON dict, a GeoJSON file path, or a CSV string — see
:func:`nice_sar.io.bbox_parser.parse_bbox` for all formats.

A CLI interface is also available:

```bash
nice-sar subset \
    --bbox=-58.24,4.40,-58.06,4.57 \
    --product GCOV \
    --maturity provisional \
    --polarization HH --polarization HV \
    -o ./my_subset/
```

Use `--bbox=...` (with `=`) when the first coordinate is negative, so the value is
not mistaken for a flag. `subset` accepts the same `--maturity`, `--track`,
`--frame`, and `--direction` filters as `search`.

## Next Steps

- See the [Notebooks](notebooks.md) for interactive end-to-end examples
- Browse the [API Reference](api/auth.md) for full function documentation
- Check the [CHPC Guide](chpc-guide.md) for running on Utah CHPC
