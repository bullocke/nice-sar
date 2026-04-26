# GEDI L4A pre/post overlap — feasibility assessment

Test of whether co-located, quality-filtered GEDI L4A shot pairs that bracket a Hansen forest-loss event can be discovered programmatically over the Amazon, to support a PALSAR-2 SAR-change vs ΔAGBD calibration figure for the NISAR proposal.

All numbers below come from a small live test over a 0.5° × 0.5° AOI in the deforestation arc of southern Pará, Brazil, executed end-to-end in [notebook.ipynb](notebook.ipynb). No HDF5 files were downloaded.

## Bottom line

**Feasible.** The recommended workflow is:

1. **Filter** `LARSE/GEDI/GEDI04_A_002_INDEX` server-side on Earth Engine to the AOI and time window, producing a small list of per-granule asset IDs.
2. **Build pre/post FeatureCollections** by mapping each asset ID to `ee.FeatureCollection(table_id).filterBounds(AOI)` and merging.
3. **Apply quality filter** (`l4_quality_flag == 1`, `degrade_flag == 0`).
4. **Pre-filter by Hansen `lossyear`** to keep only shots on disturbed pixels (this is what shrinks the join from intractable to a few minutes).
5. **Self-join** pre × post with `ee.Filter.withinDistance` at 10 m and 25 m.
6. **Sample ScanSAR backscatter** (`JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR`) at each pair location pre and post.

The GEDI INDEX → per-granule FC pattern is required because `LARSE/GEDI/GEDI04_A_002` is an `IndexedFolder`, not a single FeatureCollection — `ee.FeatureCollection("LARSE/GEDI/GEDI04_A_002")` errors with `Expected asset to be a Collection, found 'IndexedFolder'`.

## Test parameters

| Setting | Value |
|---|---|
| AOI bounds (W, S, E, N) | (−55.5°, −7.0°, −55.0°, −6.5°) |
| AOI area | ≈ 3,070 km² |
| Pre window | 2019‑04‑18 → 2021‑01‑01 |
| Post window | 2022‑01‑01 → 2023‑03‑17 |
| Quality filter | `l4_quality_flag == 1` AND `degrade_flag == 0` |
| Disturbance filter | Hansen `lossyear` ∈ {20, 21, 22} (i.e. 2020‑2022) |
| Overlap thresholds | 10 m (strict, ≈ GEDI 1σ geolocation) and 25 m (footprint diameter) |
| GEE project | `dyce-biomass` (high-volume endpoint) |
| Hansen image | `UMD/hansen/global_forest_change_2024_v1_12` |

## Quantitative results

### GEDI access methods

| Method | Installable | Latency | pairs @ 25 m | pairs @ 10 m | disturbed pairs @ 25 m | quality flags | polygon AOI | returns shot points |
|---|---|---|---|---|---|---|---|---|
| **GEE shot-level via `LARSE/GEDI/GEDI04_A_002` INDEX → per-granule FC merge → disturbance prefilter → self-join** | yes | ≈ 9 min¹ | **18** | **18** | **18** | yes | yes | yes |
| GEE monthly raster `LARSE/GEDI/GEDI04_A_002_MONTHLY` (25 m) | yes | ≈ 30 s | 326 valid co-loc pixels² | n/a (raster grid fixes co-location) | **10** disturbed co-loc pixels | yes (per-pixel) | yes | no |
| earthaccess CMR granule listing (`GEDI_L4A_AGB_Density_V2_1_2056`) | yes | 4 s | n/a (granule-level) | n/a | n/a | no (granule-level) | yes | no |
| `gedidb` fallback | not attempted | — | — | — | — | yes | yes | yes |

Footnotes:
1. The 9 min is dominated by the two `withinDistance` self-joins (≈5 min for 25 m, ≈4 min for 10 m), each over the disturbance-prefiltered set of 3,140 pre × 1,413 post shots. The granule-listing and quality/disturbance steps together take ≈75 s. Without the disturbance prefilter the join exceeds GEE memory limits.
2. The "326 valid co-located pixels" is much larger than 18 shot pairs because many shots in either window can land on the same 25 m monthly raster cell, so the raster overcounts unique events; conversely, two genuinely co-located shots can fall on different adjacent 25 m cells, undercounting. The shot-level number is the real one.

Pair count is identical at 10 m and 25 m (18 vs 18). Within this AOI every 25 m pair is also a 10 m pair — i.e. the matched GEDI repeat orbits are sub-footprint coincident, not loosely co-located. This is the ISS-resonance signal the proposal narrative depends on.

### CMR vs GEE consistency check

Granule counts intersecting AOI in each window agree across access paths:

| Source | Pre-window granules | Post-window granules |
|---|---|---|
| GEE `LARSE/GEDI/GEDI04_A_002_INDEX` | 54 | 35 |
| earthaccess CMR (`GEDI_L4A_AGB_Density_V2_1_2056`) | 54 | 35 |

Identical counts confirm we are interrogating the same underlying granule set; the only difference is what each path lets you do *with* the granules.

### PALSAR-2 SAR coverage check

| Collection | Pre-window images | Post-window images |
|---|---|---|
| `JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR` (primary) | **41** | **29** |
| `JAXA/ALOS/PALSAR/YEARLY/SAR` (fallback annual mosaic) | 1 | 0 |

Sample of ScanSAR scene metadata (5 pre-window scenes):

| BeamID | AntennaPointing | IncAngleNearRange | IncAngleFarRange |
|---|---|---|---|
| W2 | Right | 25.86° | 48.83° |
| W2 | Right | 25.83° | 48.73° |
| W2 | Right | 25.89° | 48.85° |
| W2 | Right | 25.85° | 48.83° |
| W2 | Right | 25.82° | 48.73° |

All sampled scenes share beam (`W2`), look direction, and a near-identical incidence-angle range — so the ScanSAR time series over this AOI is geometrically uniform, which removes most of the "ScanSAR is hard to work with" risk for this specific use case. The annual mosaic is unusable for 2022–2023 over this AOI; ScanSAR is the only viable choice.

## Findings

1. **The proposed figure is feasible.** A single 0.5° AOI in southern Pará yields 18 GEDI L4A pre/post shot pairs that simultaneously satisfy: ≤25 m horizontal separation, both shots passing `l4_quality_flag` and `degrade_flag`, and Hansen-detected forest loss between visits. ScanSAR has dense, uniform pre/post coverage over the same AOI.

2. **Strict-overlap binding.** All 18 pairs at 25 m are also pairs at 10 m. Within active-disturbance Amazon, the available L4A repeat geometry already produces sub-footprint overlap; lenient thresholds buy nothing here. The figure can use the strictest interpretation without losing sample size.

3. **The shot-level GEE path is unambiguously the right tool.** It returns shot points (not rasterized cells), supports server-side `withinDistance` joins, applies per-shot quality flags, and runs from a normal Python environment with no HDF5 downloads. Latency on a 0.5° AOI is acceptable; scale-up requires AOI tiling.

4. **Disturbance prefilter is a hard requirement.** Joining ~14k pre × ~12k post raw shots blows GEE's user-memory cap. Sampling Hansen `lossyear` at the shots first reduces the join to ~3.1k × ~1.4k disturbed shots, which completes.

5. **`earthaccess` is fine for listing but not for shot-level work in this context.** The CMR path returns granule footprints in seconds and is consistent with the GEE INDEX, but per-shot quality filtering requires streaming HDF5 from S3 with `h5py` + `s3fs` and decoding GEDI's beam-group structure. The GEE asset already does that decoding. `earthaccess` becomes useful only if (a) running off-GEE on CHPC/Daskhub, or (b) needing per-shot variables that the GEE asset omits (full L4A schema).

6. **`gedidb` is not needed.** Per the plan, it was reserved as a fallback; the primary path succeeded, so no install attempt was made.

7. **Monthly raster is fine for a quick sanity-check, not for the figure.** It found 10 disturbed co-located 25 m pixels in the same AOI; this disagrees with the 18 shot pairs because rasterization aliases multiple shots onto the same cell and breaks the 1:1 pre→post correspondence. Useful for AOI prescreening, not for biomass-change calibration.

## Recommended workflow for the proposal figure

1. Pick a continental Amazon AOI; tile into ≈0.5° × 0.5° boxes to bound GEE join cost.
2. For each tile, run the workflow in the notebook to obtain a GeoDataFrame of (pre_shot, post_shot, distance, lossyear) rows.
3. Concatenate all tiles client-side.
4. For each pair, sample ScanSAR backscatter (`HH`, `HV` bands) in the closest pre and post acquisitions. Stratify by `BeamID` to avoid mixing geometries (in the test AOI everything was W2, but the full Amazon will mix W1–W5).
5. Plot ΔAGBD (post − pre) vs SAR change metrics (ΔHV in dB, RFDI, etc.).

## Known limitations

- **GEE memory caps** on `withinDistance` joins force the disturbance prefilter and AOI tiling. Continental-scale joins without tiling are not viable.
- **GEDI ~10 m geolocation σ** means even a 10 m pair samples slightly different ground, but in this AOI the 10 m and 25 m sets are identical, so the figure does not have to choose.
- **Monthly raster overcounts and undercounts** vs the shot-level set; treat its numbers as scoping, not measurement.
- **PALSAR-2 ScanSAR uniformity is AOI-dependent**; in the test AOI everything was W2 right-look, but other Amazon locations will have a mix and will need stratification.
- **Hansen `lossyear` is annual and binary**; sub-annual disturbance timing and partial canopy loss are not captured. Acceptable for a proposal figure but a real calibration will want a finer disturbance product (e.g. MapBiomas Alerta) for sub-annual resolution.
- **CMR short_name has a numeric collection ID suffix.** Use `GEDI_L4A_AGB_Density_V2_1_2056`, not `GEDI_L4A_AGB_DENSITY_V2_1` — the latter returns 0 silently.

## Open questions

1. How many pairs does the workflow return scaled to the full Brazilian Amazon? At ≈ 1 pair / 170 km² of active-arc AOI, the order-of-magnitude estimate is in the low 10⁴, but actual counts depend strongly on disturbance density and overlap density per ecoregion.
2. Does the strict 10 m == 25 m parity hold outside the deforestation arc? Areas with less repeat-orbit density may need the lenient threshold.
3. Should the figure also intersect MapBiomas Alerta sub-annual events, so that pairs straddle a *known-month* disturbance? This would reduce the ΔAGBD timing ambiguity at the cost of fewer pairs.
4. Is there a CHPC/off-GEE need? If yes, the `earthaccess` path becomes mandatory and per-shot decoding code must be written; otherwise GEE alone suffices.

## Artefacts

- [notebook.ipynb](notebook.ipynb) — runnable record of every number in this report.
- [results_summary.csv](results_summary.csv) — method comparison table.
- [sar_coverage.json](sar_coverage.json) — PALSAR-2 ScanSAR + annual coverage details.

## References

- Dubayah, R. et al. *GEDI L4A Footprint Level Aboveground Biomass Density, Version 2.1.* ORNL DAAC. <https://daac.ornl.gov/GEDI/guides/GEDI_L4A_AGB_Density_V2_1.html>.
- LARSE GEE GEDI assets: `LARSE/GEDI/GEDI04_A_002`, `LARSE/GEDI/GEDI04_A_002_INDEX`, `LARSE/GEDI/GEDI04_A_002_MONTHLY`.
- Hansen, M. C. et al. *Global Forest Change 2000–2024.* `UMD/hansen/global_forest_change_2024_v1_12`.
- JAXA ALOS PALSAR-2 ScanSAR Level 2.2: `JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR`.
- `gedidb` (not attempted): <https://gedidb.readthedocs.io>.
