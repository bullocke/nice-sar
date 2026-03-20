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
from nice_sar.search import search_gcov

results = search_gcov(
    bbox=(-112.1, 40.5, -111.7, 40.9),  # Salt Lake City area
    start="2024-12-01",
    end="2025-01-01",
    max_results=5,
)
print(f"Found {len(results)} granules")
```

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

## Next Steps

- See the [Notebooks](notebooks.md) for interactive end-to-end examples
- Browse the [API Reference](api/auth.md) for full function documentation
- Check the [CHPC Guide](chpc-guide.md) for running on Utah CHPC
