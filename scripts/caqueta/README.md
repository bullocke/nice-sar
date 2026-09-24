# Caquetá disturbance signals in NISAR backscatter and coherence

These scripts reproduce the analysis of how L-band backscatter and coherence respond to forest clearing in a 22 × 18 km deforestation hotspot in Caquetá, Colombia. The site sits in NISAR frame 083 D 088, a supplemental PROVISIONAL frame with data from November 2025. The central question is whether the image pair that spans a clearing decorrelates enough to separate it from normal forest decorrelation and from post-clearing pairs.

Inputs are kept in `NISAR_Data/caqueta/` and outputs in `local_examples/caqueta/`. Both folders are gitignored.

## Run order

```bash
# 1. Data (once; slow, needs credentials)
python scripts/caqueta/fetch_nisar_subsets.py      # NASA Earthdata login; ~45 min
python scripts/caqueta/fetch_radd.py --template NISAR_Data/caqueta/GCOV/GCOV_20251103_HH.tif   # Earth Engine
python scripts/caqueta/fetch_sentinel2.py          # Earth Engine; ~10 min

# 2. Analysis and figures (~5 min)
python scripts/caqueta/run_all.py
```

`run_all.py` runs the steps below in order. Each one can also be run on its own.

| Step | Output in `local_examples/caqueta/` | What it shows |
|---|---|---|
| `select_cases.py` | `04_cases/cases.csv`, `cases.npz` | Rule-based choice of 18 case studies in 6 categories |
| `fig01_site_overview.py` | `01_site/` | Sentinel-2 at the start and end of the series, RADD alert dates, case locations |
| `fig02_event_aligned.py` | `02_event_aligned/` | All clearings stacked on their RADD date: HH, HV, and 80 m vs 20 m coherence |
| `fig03_spanning_pair.py` | `03_spanning_pair/` | Pixel-level test: is the spanning pair lower than the pixel's own history? |
| `fig04_case_studies.py` | `04_cases/<category>/` | Per case: optical chips, backscatter, and coherence time series |

Each output folder has its own README, generated with the numbers from that run, explaining the method and how to read the figures.

## Modules

| File | Role |
|---|---|
| `config.py` | Paths, AOI, thresholds, GUNW look counts. Change thresholds here. |
| `data.py` | Loads every layer onto the 20 m NISAR grid (80 m coherence resampled with nearest neighbour), the RADD layers, the reference masks, and Sentinel-2 chips |
| `analysis.py` | Methods: HV event dating, forest normalization, pair classification, spanning-pair test with a matched null, coherence estimator floor, patches and case selection |
| `style.py` | Figure rules (text at least 12 pt, tight margins) and the fixed colour for each quantity |

## Method in brief

1. **Reference classes (RADD).** "Disturbed" means a high-confidence alert during the series inside RADD's forest baseline. "Stable forest" is baseline forest at least 100 m from any alert. "Pasture" is land alerted at least 90 days before the series began.
2. **Date clearings from HV.** A single step fit to each pixel's HV series gives the pair of consecutive dual-pol dates (usually 24 days apart) that bracket the drop. It uses backscatter only, so the coherence test built on it is not circular. RADD alerts come a median of about 14 days after the end of the HV bracket.
3. **Remove weather.** Rain and wind shift coherence across the whole scene, so each pair's stable-forest median coherence is subtracted.
4. **Compare with the pixel's own history.** The lowest pair inside the bracket, relative to the mean of the pixel's pre-event pairs, is compared with a matched null built from the same pixel's pre-event pairs.
5. **Estimator floor.** A fully decorrelated pair still reads about 0.08 at 80 m (112 looks) and 0.21 at 20 m (18 looks), because coherence estimated from few looks is biased high.

## Data sources

- **NISAR:** L2 GCOV and GUNW, PROVISIONAL collections (CRID P05023), streamed as subsets with `nice_sar.io.subset.subset_product`.
- **RADD:** forest disturbance alerts (Reiche et al. 2021, Environ. Res. Lett. 16:024005), from Earth Engine `projects/radar-wur/raddalert/v1`.
- **Sentinel-2:** L2A (`COPERNICUS/S2_SR_HARMONIZED`) with Cloud Score+ cloud masking (Pasquarella et al. 2023).
