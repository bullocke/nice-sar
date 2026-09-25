# Tutorials

Twelve notebooks take you from finding a NISAR granule to detecting forest clearing. They all use one study area: a 22 × 18 km window of NISAR frame 083 D 088 over a deforestation frontier in Caquetá, Colombia, imaged every 12 days since November 2025. Each tutorial is rendered here with its figures, and each opens in Google Colab.

!!! tip "Two ways to get data"
    - **Stream** (tutorials 00 to 08): a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account. `login()` asks for it once and stores it in `~/.netrc`. Each tutorial reads only a small window of a pinned granule, typically in 15 to 30 s per layer.
    - **Demo bundle** (tutorials 07, 09, 10 and 11): no account. `nice_sar.datasets.load_caqueta_demo()` downloads a 40 MB bundle of the same data once.

| | Tutorial | What you will learn | Data | Colab |
|---|---|---|---|---|
| 00 | [Data discovery](tutorials/00_data_discovery.ipynb) | NISAR products and data maturity, granule names, what is inside a GCOV file | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/00_data_discovery.ipynb) |
| 01 | [Search and download](tutorials/01_search_and_download.ipynb) | Search by area, date, track and frame; acquisition timelines; download options | No login | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/01_search_and_download.ipynb) |
| 02 | [Read GCOV](tutorials/02_read_gcov.ipynb) | Stream HH and HV backscatter for a window, convert to dB, export GeoTIFF | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/02_read_gcov.ipynb) |
| 03 | [Speckle and multilooking](tutorials/03_preprocessing.ipynb) | Lee and refined Lee filters, multilooking, equivalent number of looks | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/03_preprocessing.ipynb) |
| 04 | [RGB composites](tutorials/04_rgb_composites.ipynb) | Twelve dual-pol colour composites, gamma, RGB GeoTIFF export | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/04_rgb_composites.ipynb) |
| 05 | [Polarimetry](tutorials/05_polarimetric_analysis.ipynb) | Dual-pol indices over forest; quad-pol Pauli, entropy and alpha over Chicago | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/05_polarimetric_analysis.ipynb) |
| 06 | [InSAR coherence](tutorials/06_insar_coherence.ipynb) | GUNW phase and coherence at 20 m and 80 m, the coherence floor, ionosphere correction | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/06_insar_coherence.ipynb) |
| 07 | [Time-series change](tutorials/07_timeseries_change.ipynb) | Coefficient of variation, CUSUM change dates and harmonic fits on an HV stack | Bundle | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/07_timeseries_change.ipynb) |
| 08 | [Subset download](tutorials/08_subset_download.ipynb) | Bounding boxes in any format, size estimates, GeoTIFF subsets, multi-date change composite | Stream | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/08_subset_download.ipynb) |
| 09 | [Forest masks](tutorials/09_forest_masks.ipynb) | HV-threshold forest masks, accuracy against a reference, external masks | Bundle | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/09_forest_masks.ipynb) |
| 10 | [Texture](tutorials/10_texture_comparison.ipynb) | Rank-filter and GLCM texture, and which features separate forest from pasture | Bundle | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/10_texture_comparison.ipynb) |
| 11 | [Forest disturbance](tutorials/11_disturbance_detection.ipynb) | HV steps, coherence dips against local forest and each area's own history, an experimental Disturbance Index | Bundle | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/11_disturbance_detection.ipynb) |

## Running the tutorials

=== "Google Colab"

    Click a Colab badge. The first code cell installs `nice-sar` (about a minute). In streaming tutorials, `login()` asks for your Earthdata username and password; you can also set the `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables before calling it.

=== "Your computer"

    ```bash
    git clone https://github.com/bullocke/nice-sar.git
    cd nice-sar
    micromamba create -f environment.yml
    micromamba activate nisar
    pip install -e ".[notebooks]"
    jupyter lab notebooks/
    ```

=== "HPC (CHPC)"

    Run the notebooks interactively with Open OnDemand or headless with SLURM. See the [CHPC guide](chpc-guide.md).

## Keeping the figures current

The figures on this site are the saved outputs of the notebooks. `scripts/docs/run_notebooks.sh` re-executes every notebook with Earthdata credentials and copies the tagged figures into the documentation, and CI refuses to build the site if a notebook has errors, unexecuted cells or a figure that no longer matches its notebook.
