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
| `select_cases.py` | `04_cases/cases.csv`, `cases.npz` | Rule-based choice of 18 case studies in 6 categories; clearing outlines from a Sentinel-2 NBR drop |
| `fig01_site_overview.py` | `01_site/` | Sentinel-2 at the start and end of the series, RADD alert dates, case locations |
| `fig02_event_aligned.py` | `02_event_aligned/` | All clearings stacked on their RADD date: HH, HV, and 80 m vs 20 m coherence |
| `fig03_spanning_pair.py` | `03_spanning_pair/` | Pixel-level test: is the spanning pair lower than the pixel's own history? |
| `fig04_case_studies.py` | `04_cases/<category>/` | Per case: Sentinel-2, HV, and coherence chips; NBR, backscatter, and coherence-minus-forest time series with HV-drop and coherence-dip bands |

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
5. **Case outlines, dips, and categories.** Each case is outlined from a Sentinel-2 NBR drop around a RADD seed. Coherence dips are measured against the case's **own recent history**: the weather-corrected 80 m coherence of a pair minus the median of the previous three pairs, flagged when it falls below a threshold calibrated on intact-forest areas of the same size. Cases are categorized by land state before the event (Sentinel-2 NBR) and by whether a dip occurred.
   - **Two weather references**, run separately: `scene` subtracts the stable-forest median over the whole study area (threshold 2.5σ); `ring` subtracts intact forest 80–300 m around each case (threshold 3.0σ). Rain is patchy, so the ring has 2–3× lower noise. Top forest clearings reach −13 to −18σ with the ring, against −4 to −5σ with the scene reference. Both thresholds give a false dip somewhere in the series for roughly 3–8% of intact-forest areas.
   - Outputs go to `04_cases/<category>/forest_scene/` and `04_cases/<category>/forest_ring/`, with separate case lists (`cases_forest_<reference>.csv`). The two references select different cases.
   - **Caveat:** the noise is calibrated on forest. Pasture varies more, and 5 of 8 pasture controls cross either threshold, so dips on non-forest land (including after a clearing) need a separate calibration.
6. **Estimator floor.** A fully decorrelated pair still reads about 0.08 at 80 m (112 looks) and 0.21 at 20 m (18 looks), because coherence estimated from few looks is biased high.

## Data sources

- **NISAR:** L2 GCOV and GUNW, PROVISIONAL collections (CRID P05023), streamed as subsets with `nice_sar.io.subset.subset_product`.
- **RADD:** forest disturbance alerts (Reiche et al. 2021, Environ. Res. Lett. 16:024005), from Earth Engine `projects/radar-wur/raddalert/v1`.
- **Sentinel-2:** L2A (`COPERNICUS/S2_SR_HARMONIZED`) with Cloud Score+ cloud masking (Pasquarella et al. 2023).
