# Notebooks

Interactive Jupyter notebooks demonstrating `nice-sar` workflows. Each notebook is self-contained
with authentication, data search, and visualization.

All notebooks support three configurable study areas:

| Area | Region | Bbox |
|------|--------|------|
| `salt_lake_city` | Salt Lake City, UT | (-112.1, 40.5, -111.7, 40.9) |
| `cascades_wa` | Cascades, WA | (-121.8, 46.7, -121.4, 46.9) |
| `amazon` | Central Amazon, Brazil | (-60.2, -3.2, -59.8, -2.8) |

## Notebook Index

### 00 — Data Discovery

Search for NISAR products using `earthaccess` and browse available collections, temporal coverage,
and spatial extent.

### 01 — Search and Download

Use `asf_search` to find GCOV granules and download them locally, or stream directly from S3.

### 02 — Read GCOV

Open NISAR GCOV HDF5 files, explore the data hierarchy, extract polarization layers as
`xarray.DataArray`, and inspect metadata and georeferencing.

### 03 — Preprocessing

Apply calibration (linear ↔ dB), Lee and Refined Lee speckle filters, multilooking, power-law
transforms, and GLCM texture features.

### 04 — RGB Composites

Generate a gallery of 12 RGB composite methods for dual-pol and quad-pol NISAR data, including
standard, vegetation, forest-enhanced, urban, and Pauli decomposition composites.

### 05 — Polarimetric Analysis

Quad-pol decomposition workflow: polarimetric indices (SPAN, RFDI), coherency matrix construction,
Freeman-Durden three-component decomposition, Cloude-Pottier H/A/α, entropy-alpha plane
visualization, and Pauli RGB.

### 06 — InSAR and Coherence

Interferometric SAR workflows: read GUNW unwrapped-phase and coherence products, convert phase to
line-of-sight displacement, form interferograms from GSLC pairs, multilook complex data, estimate
coherence, apply ionospheric correction, and overlay results on interactive Folium maps.

### 07 — Time Series Change Detection

Temporal analysis of GCOV backscatter stacks: build data cubes, compute Coefficient of Variation
(CoV), run CUSUM change-point detection, apply backscatter thresholding for inundation mapping,
and fit harmonic models for seasonal baselines.

### 08 — Spatial Subset & Download

Download only the pixels you need: define a bounding box (tuple, GeoJSON, or file), stream just
those pixels from a remote NISAR HDF5 file, and export as GeoTIFFs. Demonstrates GCOV subsetting,
visualization of downloaded clips, speckle filtering, RGB composites, alternative bbox formats,
and CLI usage examples.

## Running Notebooks

### Local

```bash
pip install nice-sar[notebooks]
jupyter lab notebooks/
```

### Headless (CI / HPC)

```bash
bash scripts/run_notebook.sh notebooks/02_read_gcov.ipynb
```

### CHPC (SLURM)

```bash
sbatch scripts/submit_notebook.slurm notebooks/02_read_gcov.ipynb
```
