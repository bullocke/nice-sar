# nice-sar

[![Documentation](https://img.shields.io/badge/docs-bullocke.github.io%2Fnice--sar-teal)](https://bullocke.github.io/nice-sar)
[![CI](https://github.com/bullocke/nice-sar/actions/workflows/ci.yml/badge.svg)](https://github.com/bullocke/nice-sar/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/11_disturbance_detection.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Search, stream and analyse data from the NASA-ISRO **NISAR** L-band radar satellite in Python.

[![A forest clearing in Caquetá, Colombia, seen by Sentinel-2, NISAR HV backscatter and NISAR coherence](https://raw.githubusercontent.com/bullocke/nice-sar/main/docs/assets/gallery/11_disturbance_detection.jpg)](https://bullocke.github.io/nice-sar/tutorials/11_disturbance_detection/)

<sub>A 26 ha forest clearing in Caquetá, Colombia, in Sentinel-2 (top), NISAR HV backscatter (middle) and NISAR 80 m coherence (bottom). Coherence dips in the pair that spans the felling and rises once the land is open. From <a href="https://bullocke.github.io/nice-sar/tutorials/11_disturbance_detection/">tutorial 11</a>.</sub>

> [!WARNING]
> `nice-sar` is in early development and its API may change. NISAR products released so far are PROVISIONAL: calibrated but only partly validated.

## Features

- **Search** ASF by area, date, track, frame and data maturity; parse granule names; build download URLs.
- **Stream** only the pixels you need: read a 20 km window from a 2 GB GCOV or GUNW granule over HTTPS in seconds, or save it as GeoTIFF.
- **Preprocess**: dB conversion, Lee and refined Lee filters, multilooking, rank-filter and GLCM texture.
- **Polarimetry**: RFDI, RVI, volume proxy, Freeman-Durden and Cloude-Pottier decompositions, Pauli RGB, 12 RGB composites.
- **InSAR**: GUNW phase and coherence at 20 m and 80 m posting, coherence masking, displacement.
- **Forests**: HV-threshold forest masks and alignment of external masks.
- **Disturbance**: HV step dating, coherence dips measured against local forest and each area's own history, and an experimental Disturbance Index.

## Installation

```bash
pip install "nice-sar @ git+https://github.com/bullocke/nice-sar.git"
```

For development:

```bash
git clone https://github.com/bullocke/nice-sar.git
cd nice-sar
micromamba create -f environment.yml
micromamba activate nisar
pip install -e ".[dev]"
```

## Quick start

Stream NISAR HV backscatter over a deforestation frontier in Colombia. You need a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account; `login()` prompts for it once and stores it in `~/.netrc`.

```python
from nice_sar.auth import get_https_filesystem, login
from nice_sar.datasets import CAQUETA_AOI
from nice_sar.io.products import read_gcov
from nice_sar.preprocess.calibration import linear_to_db
from nice_sar.search import search_gcov, summarize_results

login()
fs = get_https_filesystem()

results = search_gcov(bbox=CAQUETA_AOI, start="2026-08-25", end="2026-09-05")
granule = summarize_results(results)[0]

# Read only the pixels inside the bounding box (about 20 s over HTTPS)
hv = read_gcov(granule.url, polarization="HV", filesystem=fs, bbox=CAQUETA_AOI)
hv_db = linear_to_db(hv.values)
```

Searches return calibrated PROVISIONAL products by default; pass `maturity="beta"` or `maturity="any"` for others. The command line offers the same:

```bash
nice-sar search --product GUNW --bbox=-74.36,0.76,-74.16,0.92
nice-sar subset --product GCOV --bbox=-74.36,0.76,-74.16,0.92 --start 2026-08-25 --end 2026-09-05 --polarization HV -o caqueta
```

## Tutorials

Twelve notebooks, all over the same study area in Caquetá, Colombia. Each opens in Google Colab and is [rendered with its figures](https://bullocke.github.io/nice-sar/notebooks/) on the documentation site.

| | Tutorial | |
|---|---|---|
| 00 | Data discovery: products, maturity and what is inside a granule | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/00_data_discovery.ipynb) |
| 01 | Search and download | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/01_search_and_download.ipynb) |
| 02 | Read GCOV backscatter | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/02_read_gcov.ipynb) |
| 03 | Speckle filtering and multilooking | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/03_preprocessing.ipynb) |
| 04 | RGB composites | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/04_rgb_composites.ipynb) |
| 05 | Polarimetry | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/05_polarimetric_analysis.ipynb) |
| 06 | InSAR coherence and phase | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/06_insar_coherence.ipynb) |
| 07 | Time-series change | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/07_timeseries_change.ipynb) |
| 08 | Subset download | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/08_subset_download.ipynb) |
| 09 | Forest masks | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/09_forest_masks.ipynb) |
| 10 | Texture | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/10_texture_comparison.ipynb) |
| 11 | Detecting tropical forest clearing | [Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/11_disturbance_detection.ipynb) |

## Documentation

- [Getting started](https://bullocke.github.io/nice-sar/getting-started/)
- [Gallery](https://bullocke.github.io/nice-sar/gallery/)
- [Forest disturbance in Caquetá](https://bullocke.github.io/nice-sar/disturbance/)
- [API reference](https://bullocke.github.io/nice-sar/api/auth/)

## Project structure

```text
nice_sar/
├── auth/          # NASA Earthdata authentication
├── search/        # Data discovery (ASF, earthaccess), maturity, granule names
├── io/            # HDF5 readers, windowed reads, GeoTIFF export, subsetting, download
├── preprocess/    # Calibration, filtering, multilooking, texture
├── analysis/      # Polarimetry, decompositions, InSAR, time series, disturbance
├── forests/       # Forest and non-forest masking workflows
├── viz/           # RGB composites, display utilities, mapping, figure style
└── datasets.py    # Demo data for the tutorials
```

## License

MIT
