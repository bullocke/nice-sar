# nice-sar

**NISAR SAR data processing and analysis toolkit**

`nice-sar` is a Python package for cloud-native processing and analysis of Synthetic Aperture
Radar (SAR) data from the NASA-ISRO NISAR satellite. It provides tools for data discovery,
preprocessing, polarimetric analysis, and visualization — all designed for streaming directly
from NASA Earthdata Cloud (AWS S3).

## Features

- **Cloud-native access** — Stream NISAR HDF5 data directly from S3 using `earthaccess` and `s3fs`
- **Lazy loading** — Dask-backed `xarray.DataArray` for large rasters without memory limits
- **Preprocessing** — Calibration (linear ↔ dB), Lee/Refined Lee speckle filters, multilooking,
  GLCM texture
- **Polarimetric analysis** — Freeman-Durden and Cloude-Pottier decompositions, polarimetric indices
  (SPAN, RFDI, volume proxy), Pauli RGB
- **12 RGB composites** — Standard dual-pol, vegetation, forest-enhanced, urban, and more
- **GeoTIFF export** — Write georeferenced outputs with `rasterio`
- **HPC-ready** — SLURM scripts for Utah CHPC and other clusters

## Quick Example

```python
import os

import nice_sar

# Authenticate with NASA Earthdata
nice_sar.auth.login()

if os.environ.get("AWS_DEFAULT_REGION") == "us-west-2":
  fs = nice_sar.auth.get_s3_filesystem()
  granule_access = "s3"
else:
  fs = nice_sar.auth.get_https_filesystem()
  granule_access = "https"

# Search for GCOV products
results = nice_sar.search.search_gcov(
    bbox=(-112.1, 40.5, -111.7, 40.9),
    start="2024-12-01",
    end="2025-01-01",
)

# Open and read as xarray DataArray
granule_url = nice_sar.auth.get_granule_url(results[0], access=granule_access)
h5 = nice_sar.io.open_nisar(granule_url, filesystem=fs)
hh = nice_sar.io.read_gcov(h5, frequency="A", polarization="HHHH")

# Preprocess and visualize
from nice_sar.preprocess import linear_to_db, lee_filter
hh_filtered = lee_filter(hh.values, window_size=7)
hh_db = linear_to_db(hh_filtered)
```

## Installation

```bash
pip install nice-sar
```

For development:

```bash
git clone https://github.com/bullocke/nice-sar.git
cd nice-sar
pip install -e ".[dev]"
```

## Supported NISAR Products

| Product | Level | Description |
|---------|-------|-------------|
| RSLC | L1 | Range-Doppler Single Look Complex |
| GSLC | L1 | Geocoded Single Look Complex |
| GCOV | L2 | Geocoded Covariance Matrix |
| GUNW | L2 | Geocoded Unwrapped Interferogram |
| GOFF | L2 | Geocoded Pixel Offsets |

## License

MIT — see [LICENSE](https://github.com/bullocke/nice-sar/blob/main/LICENSE).
