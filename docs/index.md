---
hide:
  - navigation
---

# nice-sar

<p class="lead">Search, stream and analyse <strong>NISAR</strong> L-band radar data in Python, from a single HDF5 granule to a year of forest change.</p>

[Get started](getting-started.md){ .md-button .md-button--primary } [Browse the gallery](gallery.md){ .md-button } [Open a tutorial in Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/11_disturbance_detection.ipynb){ .md-button }

<figure class="hero" markdown>
[![A forest clearing in Caquetá, Colombia, seen by Sentinel-2, NISAR HV backscatter and NISAR coherence](assets/gallery/11_disturbance_detection.jpg)](tutorials/11_disturbance_detection.ipynb)
<figcaption>A 26 ha forest clearing in Caquetá, Colombia, seen by Sentinel-2 (top), NISAR HV backscatter (middle) and NISAR 80 m coherence (bottom). Coherence dips in the pair that spans the felling, 9 to 21 December 2025, and rises once the land is open. Red frames mark each sensor's first detection. From <a href="tutorials/11_disturbance_detection/">tutorial 11</a>.</figcaption>
</figure>

!!! warning "Early development, PROVISIONAL data"
    `nice-sar` is under active development and its API may change. NISAR products released so far are PROVISIONAL: calibrated but only partly validated. See [data maturity](getting-started.md#data-maturity-provisional-vs-beta).

## What you can do

<div class="grid cards" markdown>

-   :material-magnify: **Find data**

    ---

    Search ASF by area, date, track and frame, filter by data maturity, and parse granule names.

    [:octicons-arrow-right-24: Search and download](tutorials/01_search_and_download.ipynb)

-   :material-cloud-download-outline: **Stream only what you need**

    ---

    Read a 20 km window from a 2 GB granule over HTTPS in seconds, or save it as GeoTIFF.

    [:octicons-arrow-right-24: Read GCOV](tutorials/02_read_gcov.ipynb)

-   :material-image-filter-hdr: **Preprocess and visualise**

    ---

    Speckle filters, multilooking, texture, dual-pol RGB composites and polarimetric indices.

    [:octicons-arrow-right-24: Speckle and multilooking](tutorials/03_preprocessing.ipynb)

-   :material-sine-wave: **Interferometric coherence**

    ---

    Read GUNW phase and coherence at 20 m and 80 m posting and understand the coherence floor.

    [:octicons-arrow-right-24: InSAR coherence](tutorials/06_insar_coherence.ipynb)

-   :material-pine-tree: **Map forests**

    ---

    Build forest masks from HV backscatter and compare them with external products.

    [:octicons-arrow-right-24: Forest masks](tutorials/09_forest_masks.ipynb)

-   :material-axe: **Detect forest clearing**

    ---

    Date clearings from HV steps and coherence dips measured against each area's own history.

    [:octicons-arrow-right-24: Forest disturbance](tutorials/11_disturbance_detection.ipynb)

</div>

## Quick start

Install from GitHub (a PyPI release will follow):

```bash
pip install "nice-sar @ git+https://github.com/bullocke/nice-sar.git"
```

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

No account yet? The tutorials that need a time series use a 40 MB demo bundle instead:

```python
from nice_sar.datasets import load_caqueta_demo

demo = load_caqueta_demo()  # HV, coherence, RADD alerts and Sentinel-2 on one 20 m grid
```

## Tutorials

Twelve notebooks, each runnable in [Google Colab](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/) and rendered here with their figures. They all use the same study area, frame 083 D 088 over Caquetá, Colombia, so each one builds on the last. See the [tutorial overview](notebooks.md) or the [gallery](gallery.md).

## Supported NISAR products

| Product | Level | Reader | Description |
|---|---|---|---|
| GCOV | L2 | `read_gcov` | Geocoded polarimetric covariance (backscatter) |
| GUNW | L2 | `read_gunw` | Geocoded unwrapped interferogram, coherence and phase |
| GSLC | L2 | `read_gslc` | Geocoded single-look complex |
| GOFF | L2 | `read_goff` | Geocoded pixel offsets |
| RSLC | L1 | `read_rslc` | Range-Doppler single-look complex |

## License

MIT. See [LICENSE](https://github.com/bullocke/nice-sar/blob/main/LICENSE).
