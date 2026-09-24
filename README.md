# nice-sar

[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://bullocke.github.io/nice-sar)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **⚠️ Early Development — Not Ready for Production Use**
>
> This package is in the early stages of development. APIs are unstable, functionality is incomplete, and nothing has been validated against real NISAR data yet. Use at your own risk and expect breaking changes. Contributions and feedback are welcome.

A Python toolkit for processing and analyzing SAR data from the NASA-ISRO **NISAR** satellite.

## Features

- **Cloud-native access** to NISAR HDF5 products via NASA Earthdata Cloud (AWS S3)
- **Data discovery** through `earthaccess` and `asf_search`
- **Preprocessing**: radiometric calibration, speckle filtering, terrain correction
- **Polarimetric analysis**: decompositions (Freeman-Durden, Cloude-Pottier), indices (RFDI, RVI)
- **Visualization**: 12 RGB composite methods, amplitude/phase display, interactive maps
- **Supports all NISAR products**: RSLC, GCOV, GUNW, GSLC, GOFF, and Level-3 derivatives

## Installation

```bash
# Create environment
micromamba create -f environment.yml
micromamba activate nisar

# Install package in development mode
pip install -e ".[dev]"
```

## Quick start

```python
import nice_sar

# Authenticate with NASA Earthdata
nice_sar.auth.login()

# Read a GCOV product
ds = nice_sar.io.read_gcov("path/to/NISAR_L2_GCOV.h5", frequency="A", polarization="HH")

# Convert to dB and visualize
from nice_sar.preprocess.calibration import linear_to_db
from nice_sar.viz.display import percentile_stretch

db = linear_to_db(ds.values)
stretched = percentile_stretch(db)
```

## Finding and downloading data

Searches return calibrated **PROVISIONAL** products by default. Pass `maturity="beta"` for the February 2026 pre-calibration release, or `maturity="any"` for both:

```python
from nice_sar.search import search_nisar, summarize_results
from nice_sar.io.download import download_granules

results = search_nisar("GCOV", bbox=(-63.5, -10.0, -62.5, -9.0), start="2026-06-17")
for s in summarize_results(results):
    print(s.maturity, s.crid, s.track, s.frame, s.start)

download_granules(results[:1], "NISAR_Data/GCOV")
```

```bash
nice-sar search --product GUNW --bbox=-63.5,-10,-62.5,-9 --maturity any
nice-sar download --product GCOV --track 68 --frame 93 --direction D --max-granules 2 -o NISAR_Data/GCOV
```

See [Getting Started](docs/getting-started.md#data-maturity-provisional-vs-beta) for the maturity levels and the [supplemental PROVISIONAL frames](docs/supplemental-frames.md), which extend back to October 2025.

## Project structure

```
nice_sar/
├── auth/          # NASA Earthdata authentication
├── search/        # Data discovery (ASF, earthaccess)
├── io/            # HDF5 readers, GeoTIFF export, download
├── preprocess/    # Calibration, filtering, texture
├── analysis/      # Polarimetry, decompositions, InSAR
└── viz/           # RGB composites, display utilities, mapping
```

## License

MIT
