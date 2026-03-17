# Plan 1 — Implementation Progress

> Tracking progress against the original plan in [`Plan/plan1.md`](../Plan/plan1.md).
> Last updated: 2026-03-17

---

## Phase 1: Project Scaffolding and Tooling

### Step 1.1 — Git and Remote
- [x] `git init`
- [x] `git remote add origin https://github.com/bullocke/nice-sar.git`
- [x] `git branch -M main`
- [x] First commit after scaffolding complete
- [x] Pushed to GitHub (6 commits as of 2026-03-17)

### Step 1.2 — `pyproject.toml`
- [x] Build backend: hatchling (PEP 621)
- [x] `[project]`: name=`nice-sar`, version=`0.1.0`, license=MIT
- [x] `requires-python` set (changed to `>=3.10` for Colab compat; plan said `>=3.12`)
- [x] `[project.optional-dependencies]`: `dev`, `test`, `docs`, `notebooks`, `viz` groups
- [x] `[tool.ruff]`: line-length=100, target-version=py312, rules E/F/I/UP/B/SIM, google docstrings
- [x] `[tool.mypy]`: python_version=3.12, warn_return_any, ignore_missing_imports
- [x] `[tool.pytest.ini_options]`: testpaths=`["tests"]`, addopts=`"--cov=nice_sar"`

### Step 1.3 — `.gitignore`
- [x] Python: `__pycache__/`, `*.pyc`, `dist/`, `build/`, `*.egg-info`
- [x] Jupyter: `.ipynb_checkpoints/`
- [x] SAR data: `*.h5`, `*.hdf5`, `*.he5`, `*.tif`, `*.tiff`, `*.nc`, `*.nc4`
- [x] Large data dirs: `data/`, `GCOV_Data/`, `scratch/`
- [x] Env/credentials: `.env`, `.netrc`, `*.credentials`
- [x] IDE: `.idea/`

### Step 1.4 — `.github/copilot-instructions.md`
- [x] Domain context (geospatial SAR, NISAR, HDF5, Earthdata Cloud)
- [x] NISAR-specific HDF5 paths and structure
- [x] Architecture guidance (dask, xarray, earthaccess, asf_search)
- [x] Coding conventions (type hints, Google docstrings, logging, ruff)
- [x] Security rules (no hardcoded credentials)
- [x] Preferred patterns (S3 direct read, chunked processing)
- [x] Anti-patterns (no setup.py, no print, no hardcoded paths)

### Step 1.5 — `.pre-commit-config.yaml`
- [x] ruff (lint + format)
- [x] mypy
- [x] check-yaml
- [x] end-of-file-fixer
- [x] trailing-whitespace

### Step 1.6 — `.github/workflows/ci.yml`
- [x] Trigger on push/PR to main
- [x] checkout → setup-python → pip install → ruff → mypy → pytest

### Step 1.7 — Additional Scaffolding Files
- [x] `README.md` (with early-development disclaimer)
- [x] `LICENSE` (MIT)

---

## Phase 2: Environment and Dependencies

### Step 2.1 — `environment.yml`
- [x] `name: nisar`, `channels: [conda-forge]`
- [x] Conda deps: python, h5py, rasterio, xarray, dask, numpy, scipy, shapely, geopandas, pyproj, s3fs, boto3, fsspec, matplotlib, scikit-image
- [x] Pip deps: earthaccess, asf_search, `-e .`

### Step 2.2 — Optional Dependency Groups in `pyproject.toml`
- [x] `test`: pytest, pytest-cov, pytest-xdist
- [x] `docs`: mkdocs, mkdocs-material, mkdocstrings[python]
- [x] `notebooks`: jupyter, jupyterlab, ipywidgets
- [x] `viz`: folium, hvplot, holoviews, datashader
- [x] `dev`: all above + pre-commit, ruff, mypy

### Step 2.3 — Pin Strategy
- [x] Initially used `>=` lower bounds (plan spec)
- [x] Changed to unpinned deps to resolve Colab conflicts (practical modification)
- [x] Notebook install pins `fsspec<=2025.3.0` and `s3fs<=2025.3.0` for Colab compat

---

## Phase 3: Package Architecture

### Module Structure
- [x] `nice_sar/__init__.py` — version, top-level imports
- [x] `nice_sar/_types.py` — PathType, BBox, ArrayFloat32, ArrayFloat64, ArrayUInt8
- [x] `nice_sar/config.py` — logging setup with `get_logger()`

### `auth/`
- [x] `auth/earthdata.py` — `login()`, `get_s3_filesystem()`, `get_https_filesystem()`
  - Modified from plan: added `get_https_filesystem()` and NISAR-specific S3 endpoint (discovered during doc alignment)

### `search/`
- [x] `search/asf.py` — `search_nisar()`, `search_gcov()`
  - Modified from plan: uses `dataset='NISAR'` instead of `platform=PLATFORM.NISAR` (per official docs)
- [x] `search/earthdata.py` — `search_earthdata()` wrapper
  - Modified from plan: default short_name set to `NISAR_L2_GCOV_BETA_V1` (not `NISAR_L2_PR_GCOV`)
- [ ] `search/results.py` — SearchResult dataclass
  - Deferred: not needed yet; raw asf_search/earthaccess result objects suffice

### `io/`
- [x] `io/hdf5.py` — `open_nisar()` (local, S3, HTTPS), `get_frequencies()`, `get_polarizations()`
  - Modified from plan: supports HTTPS URLs in addition to S3
- [x] `io/products.py` — `read_gcov()`, `read_gcov_metadata()`, `get_projection_info()`, `read_quad_covariances()`
- [x] `io/geotiff.py` — `export_geotiff()`, `write_rgb_geotiff_uint8()`, `read_band()`
- [x] `io/download.py` — `download_url()`, internal `_get_asf_session()`

### `preprocess/`
- [x] `preprocess/calibration.py` — `linear_to_db()`, `db_to_linear()`, `compute_sigma0()`
- [x] `preprocess/filters.py` — `lee_filter()`, `refined_lee_filter()`
- [x] `preprocess/texture.py` — `compute_glcm_texture()`, `compute_glcm_contrast_homogeneity()`
- [x] `preprocess/multilook.py` — `multilook()` spatial averaging

### `analysis/`
- [x] `analysis/polarimetry.py` — `compute_span()`, `compute_rfdi()`, `volume_proxy()`, `compute_indices()`
- [x] `analysis/decomposition.py` — `build_coherency_matrix()`, `compute_pauli_rgb()`
  - Partial: Freeman-Durden and Cloude-Pottier decompositions not yet implemented (vectorized versions planned)
- [ ] `analysis/insar.py` — placeholder only (Milestone 4)
- [ ] `analysis/timeseries.py` — placeholder only (Milestone 4)

### `viz/`
- [x] `viz/rgb.py` — 12 RGB composite methods + `make_rgb()` dispatcher + `list_rgb_methods()`
  - Plan called for 11 methods; implementation has 12
- [x] `viz/display.py` — `percentile_stretch()`, `gamma_correct()`, `to_uint8()`
- [ ] `viz/mapping.py` — placeholder only (Milestone 4)

### Design Decisions (from plan)
- [x] Primary data container: `xarray.DataArray` backed by dask
- [x] Processing functions accept numpy; xarray wrapping at API boundary
- [x] S3-first access: `open_nisar()` detects `s3://` and `https://` paths
- [x] Product readers return xarray with CRS + coordinates

---

## Phase 4: Migration of Existing Code

### `download_from_asf.py` → `nice_sar/io/download.py`
- [x] Reuse `asf_search.download_url()` pattern
- [x] Remove hardcoded credentials (uses env vars / earthaccess)
- [x] Parameterized URL/path
- [x] Error handling + logging
- [ ] Batch downloads with progress bar (not yet implemented)

### `create_rgbs.py` → `nice_sar/viz/rgb.py` + `nice_sar/viz/display.py`
- [x] Migrate all RGB composite algorithms (12 methods)
- [x] Migrate `percentile_stretch()`, `gamma_correct()`, `to_uint8()` to display.py
- [x] Separate I/O from computation (readers in `io/geotiff.py`)
- [x] Accept numpy arrays as input (not just file paths)

### `prep_gcov.py` → Split Across Modules

| Source Functions | Target Module | Status |
|---|---|---|
| `read_gcov_metadata()`, `get_projection_info()` | `io/products.py` | ✅ Done |
| `export_geotiff()`, `write_rgb_geotiff_uint8()` | `io/geotiff.py` | ✅ Done |
| `linear_to_db()`, `compute_sigma0()` | `preprocess/calibration.py` | ✅ Done |
| `lee_filter()`, `refined_lee_filter()` | `preprocess/filters.py` | ✅ Done |
| `compute_glcm_texture()`, `compute_glcm_contrast_homogeneity()` | `preprocess/texture.py` | ✅ Done |
| `compute_span()`, `compute_rfdi()`, `volume_proxy()`, `compute_indices()` | `analysis/polarimetry.py` | ✅ Done |
| `freeman_durden_from_T()`, `cloude_pottier_from_T()` | `analysis/decomposition.py` | ⬜ Not yet (vectorized versions planned) |
| `make_dualpol_rgb()`, `compute_pauli_rgb()` | `viz/rgb.py` | ✅ Done |
| `process_gcov_file()` | Notebook workflow | ✅ Not library code — covered by notebook |

### Cross-Cutting Refactoring
- [x] Replace `print()` → `logging.getLogger(__name__)`
- [x] Add type hints to all functions
- [x] Remove hardcoded paths → `Path` parameters
- [x] Add Google-style docstrings
- [x] Consistent NaN/nodata handling

---

## Phase 5: Testing and Documentation

### Test Structure

| Test File | Target Module | Tests | Status |
|---|---|---|---|
| `tests/conftest.py` | Fixtures: synthetic GCOV, rng, sample arrays | — | ✅ Done |
| `tests/fixtures/create_synthetic_gcov.py` | Synthetic 64×64 NISAR HDF5 generator | — | ✅ Done |
| `tests/test_io/test_hdf5.py` | `io/hdf5.py` | 9 | ✅ Done |
| `tests/test_io/test_products.py` | `io/products.py` | 19 | ✅ Done |
| `tests/test_io/test_geotiff.py` | `io/geotiff.py` | 6 | ✅ Done |
| `tests/test_preprocess/test_calibration.py` | `preprocess/calibration.py` | 5 | ✅ Done |
| `tests/test_preprocess/test_filters.py` | `preprocess/filters.py` | 7 | ✅ Done |
| `tests/test_preprocess/test_multilook.py` | `preprocess/multilook.py` | 5 | ✅ Done |
| `tests/test_analysis/test_decomposition.py` | `analysis/decomposition.py` | 3 | ✅ Done |
| `tests/test_analysis/test_polarimetry.py` | `analysis/polarimetry.py` | 5 | ✅ Done |
| `tests/test_viz/test_display.py` | `viz/display.py` | 6 | ✅ Done |
| `tests/test_viz/test_rgb.py` | `viz/rgb.py` | 3 (parametrized → 14+) | ✅ Done |
| `tests/test_search/test_asf.py` | `search/asf.py` (mock-based) | — | ⬜ Not yet |

**Total: 79 tests passing, 69% coverage**

### Test Fixtures
- [x] Synthetic HDF5 in `conftest.py` — mimics `/science/LSAR/GCOV/grids/frequencyA/` with 64×64 arrays
- [ ] Sample NISAR data from JPL for integration tests (gitignored, CI-fetched)

### Documentation
- [ ] MkDocs + Material site
- [ ] `docs/` directory
- [ ] API reference via mkdocstrings
- [ ] Getting Started guide
- [ ] Tutorials / narrative docs
- [ ] Contributing guide

### Notebooks

| Notebook | Description | Status |
|---|---|---|
| `notebooks/00_data_discovery.ipynb` | Search, auth, read, validate GCOV from Earthdata Cloud | ✅ Created & tested in Colab |
| `notebooks/01_search_and_download.ipynb` | End-to-end search → download workflow | ⬜ Not yet |
| `notebooks/02_read_gcov.ipynb` | Read GCOV into xarray, explore structure | ⬜ Not yet |
| `notebooks/03_preprocessing.ipynb` | Calibration, filtering, multilooking | ⬜ Not yet |
| `notebooks/04_rgb_composites.ipynb` | Generate RGB composites from GCOV | ⬜ Not yet |
| `notebooks/05_polarimetric_analysis.ipynb` | Polarimetric indices + decomposition | ⬜ Not yet |
| `notebooks/06_insar_basics.ipynb` | InSAR processing (future) | ⬜ Not yet |

---

## Phase 6: Implementation Roadmap

### Milestone 1: "Read and See" — Core I/O + Visualization
- [x] Project scaffolding (all Phase 1 steps)
- [x] Environment setup (Phase 2)
- [x] `auth.earthdata` — login wrapper
- [x] `io.hdf5` + `io.products` — read GCOV into xarray from local/S3/HTTPS
- [x] `io.geotiff` — GeoTIFF export
- [x] `viz.display` — percentile_stretch, gamma_correct, to_uint8
- [x] `viz.rgb` — 12 RGB composite methods (plan called for 11)
- [x] `preprocess.calibration` — linear_to_db, sigma0
- [x] Tests for all above (79 passing, 69% coverage)
- [x] First push to GitHub
- [ ] Notebooks 02 (read_gcov) & 04 (RGB composites)

**Deliverable**: ✅ Can open NISAR GCOV (local/S3/HTTPS) → read bands → convert dB → RGB composites → export GeoTIFF. Validated against real NISAR data in Colab.

### Milestone 2: "Search and Process" — Data Discovery + Preprocessing
- [x] `search.asf` + `search.earthdata` — search wrappers
- [x] `io.download` — refactored downloader
- [x] `preprocess.filters` — Lee, Refined Lee
- [x] `preprocess.texture` — GLCM surrogates
- [ ] Tests for search (mock-based ASF tests)
- [ ] Notebooks 01 (search & download) & 03 (preprocessing)

**Deliverable**: Partially complete — code is implemented but notebooks and search tests are not yet done.

### Milestone 3: "Analyze" — Polarimetric Analysis
- [x] `analysis.polarimetry` — indices, SPAN, RFDI, volume proxy
- [x] `analysis.decomposition` — build_coherency_matrix, compute_pauli_rgb
- [ ] `analysis.decomposition` — Freeman-Durden (vectorized)
- [ ] `analysis.decomposition` — Cloude-Pottier (vectorized)
- [ ] Notebook 05 (polarimetric analysis)
- [ ] MkDocs documentation site deployed

**Deliverable**: Partially complete — basic polarimetric analysis works, but advanced decompositions and docs not yet done.

### Milestone 4: "Expand" — InSAR, Time Series, Interactive Viz
- [ ] `io/products.py` — expand to RSLC, GUNW, GOFF readers
- [ ] `analysis/insar.py` — interferogram generation, coherence
- [ ] `analysis/timeseries.py` — change detection
- [ ] `viz/mapping.py` — folium interactive maps
- [ ] `preprocess/multilook.py` — expand (basic implementation exists)
- [ ] CLI entry point (`cli.py`)
- [ ] Remaining notebooks (06)

**Deliverable**: Not started.

---

## Verifications (from plan)

| # | Verification | Status |
|---|---|---|
| 1 | `ruff check nice_sar/ tests/` — zero errors | ✅ Pass |
| 2 | `mypy nice_sar/` with `--ignore-missing-imports` | ⬜ Not run (ruff passes) |
| 3 | `pytest tests/ --cov=nice_sar` — ≥80% on implemented modules | ✅ 79 pass, 69% overall |
| 4 | Synthetic HDF5 round-trip: read_gcov returns valid xarray with coords + CRS | ✅ Pass |
| 5 | `percentile_stretch()` output ∈ [0,1]; `to_uint8()` dtype uint8 range [0,255] | ✅ Pass |
| 6 | RGB composites produce (H,W,3) arrays in [0,1] | ✅ Pass |
| 7 | GeoTIFF round-trip: write → rasterio read-back matches | ✅ Pass |
| 8 | `linear_to_db(0.01) ≈ -20.0`; `linear_to_db(1.0) = 0.0` | ✅ Pass |
| 9 | Freeman-Durden: Ps+Pd+Pv ≈ SPAN (power conservation) | ⬜ Not yet (F-D not implemented) |
| 10 | Notebooks execute without errors against real data | ✅ 00_data_discovery tested in Colab |
| 11 | CI workflow passes on GitHub Actions | ⬜ Not verified (workflow exists) |
| 12 | `pip install -e .` succeeds | ✅ Pass (tested in Colab) |

---

## Bug Log

Bugs discovered and fixed during implementation:

| Bug | Description | Status |
|---|---|---|
| BUG-001/002 | `read_gcov` file handle leak | ✅ Fixed (Phase 1) |
| BUG-004 | `asf_search` used `PLATFORM.NISAR` (not valid) | ✅ Fixed (use `dataset='NISAR'`) |
| BUG-006 | `write_rgb_geotiff_uint8` double-stretch | ✅ Fixed (Phase 1) |
| BUG-007 | `lee_filter` NaN propagation | ✅ Fixed (Phase 1) |
| BUG-008 | `read_quad_covariances` compound type auto-conversion | ✅ Fixed (Phase 2) |
| BUG-009 | `get_s3_filesystem()` used `daac='ASF'` (fails for NISAR) | ✅ Fixed (NISAR endpoint) |
| BUG-010 | Wrong default short_name `NISAR_L2_PR_GCOV` | ✅ Fixed (→ `NISAR_L2_GCOV_BETA_V1`) |
| BUG-011 | No HTTPS filesystem support for non-us-west-2 | ✅ Fixed (added `get_https_filesystem()`) |

---

## Next Steps

### Immediate (complete Milestones 1–2)
1. **Run mypy** on the full codebase and fix any type errors (Verification #2)
2. **Create notebook 02** (`read_gcov.ipynb`) — complete Milestone 1
3. **Create notebook 04** (`rgb_composites.ipynb`) — complete Milestone 1
4. **Add mock-based search tests** (`test_search/test_asf.py`) — complete Milestone 2 tests
5. **Create notebook 01** (`search_and_download.ipynb`)
6. **Create notebook 03** (`preprocessing.ipynb`)

### Medium-term (Milestone 3)
7. **Implement vectorized Freeman-Durden** decomposition in `analysis/decomposition.py`
8. **Implement vectorized Cloude-Pottier** decomposition in `analysis/decomposition.py`
9. **Create notebook 05** (`polarimetric_analysis.ipynb`)
10. **Set up MkDocs** documentation site with API reference

### Longer-term (Milestone 4)
11. Expand product readers (RSLC, GUNW, GOFF)
12. Implement `analysis/insar.py` and `analysis/timeseries.py`
13. Implement `viz/mapping.py` (folium interactive maps)
14. CLI entry point
15. Verify CI passes on GitHub Actions
