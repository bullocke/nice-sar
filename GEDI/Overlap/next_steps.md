# Next steps and risks

This document is forward-looking. All four production phases (GEDI pair
discovery, SAR sampling, figure generation) are complete on the 26° × 14°
Brazilian Amazon AOI. The figure is in `outputs/figures/`. This file
captures the risks worth flagging for the proposal narrative and the
follow-on work to upgrade from PoC to publication-quality analysis.

## Status

| Phase | Script | Status | Output |
|---|---|---|---|
| GEDI pre/post pair discovery | `scale_pairs.py` | Done (partial AOI: 203 / 1508 tiles) | `outputs/pairs.parquet` (1126 rows) |
| PALSAR-2 ScanSAR sampling | `sample_sar.py` | Done | `outputs/pairs_with_sar.parquet` (609 with both pre+post SAR) |
| Filter, fit, plot | `make_figure.py` | Done | `outputs/figures/delta_hv_vs_delta_agbd.{png,pdf}` |

## Result summary

- 1126 GEDI L4A pre/post pairs over Hansen-disturbed pixels (≤25 m); 233
  survive quality filters. Pixelwise OLS R² = 0.006 (ΔHV), 0.015 (ΔHH),
  0.003 (ΔRFDI). Binned medians do show the expected gradient
  (~80 Mg ha⁻¹ across the ΔHV range; ~0.8 dB ΔHV between stable and
  > 50 Mg ha⁻¹ loss bins) — the signal is present but small per-pixel.
- 54 % of pairs (609 / 1126) had matching pre and post ScanSAR scenes
  within ±90 d. Loss is mostly to single-pol (HH-only) scenes that were
  filtered server-side.

## Risks and decisions to revisit

### PALSAR-2 ScanSAR coverage

Feasibility check on the 0.5° Pará AOI returned 41 pre and 29 post ScanSAR
scenes — comfortable. Coverage on the larger Amazon-frontier AOI is expected
to be variable, especially in 2022–2023.

- If `sar_log.csv` shows <30 % of pairs match a SAR scene within ±90 days on
  both sides, expand the window to ±180 days and document the relaxation in
  `outputs/figures/decisions_log.md`.
- Annual mosaic (`JAXA/ALOS/PALSAR/YEARLY/SAR`) is a fallback; the feasibility
  AOI returned 1 pre and 0 post images, so this is unlikely to help on its own
  but may bridge gaps for tiles with missing post-disturbance ScanSAR.

### Beam-mode mixing across the AOI

PALSAR-2 ScanSAR mixes beams (W1, W2, W3, W4) and pointings (Right, Left). The
feasibility sample showed all five inspected scenes were W2 / Right at
~26°–49° incidence — likely the dominant mode but not exclusive. Mixed beams
introduce incidence-angle and resolution variability that will widen the
ΔHV(dB) noise floor.

- Default for the headline figure: `--require-same-pointing` ON, dropping
  pre/post pairs that switched antenna pointing.
- Supplementary panel without this filter as a robustness check.
- If beam-mode stratification reveals strong mode-dependent slopes, report
  separately in the caption.

### GEDI geolocation uncertainty

GEDI shot geolocation σ ≈ 10 m. The 25 m pairing threshold partly absorbs
this; the 12.5 m circular buffer for SAR sampling is the recommended
compromise. ΔAGBD between two ostensibly-coincident shots is therefore noisy
at the per-pair level. Mitigation: use n large enough (target ≥ 200) for the
regression to be robust; report n in the figure annotation.

### Hansen `lossyear` is annual

Within-year disturbance timing is unresolved. A pair where the post shot
predates the SAR scene's view of the cleared canopy will still satisfy the
Hansen mask but will under-represent the SAR change.

- Mitigation for the proposal: state explicitly in the caption.
- Future improvement (post-proposal): refine with MapBiomas Alerta or the
  GFC near-real-time alerts to constrain disturbance to within 30 days of
  the GEDI post shot.

### Faraday rotation at L-band HV

HV is most sensitive to ionospheric Faraday rotation. For the proposal figure
this is acceptable noise; flag in the caption. Multi-temporal analysis with
NISAR will benefit from per-acquisition ionospheric correction, which is
provided in NISAR L1 products.

### Sample selection

Default post-hoc filters in `make_figure.py`:

- `sensitivity ≥ 0.95` (both shots)
- `pre_AGBD ≥ 50 Mg/ha` — restricts the figure to "biomass loss from forest"
- Full-power beams (configurable)
- Same antenna pointing pre vs post

If the headline R² < 0.05 or n < 50 with default filters, relax in
`make_figure.py` and record alternatives in
`outputs/figures/decisions_log.md`.

## Proposal-narrative caveats to retain

- **NISAR will improve on this baseline**: higher revisit, dual-pol routinely,
  and ionospheric correction. The figure is a PALSAR-2 stand-in, not a
  NISAR simulation. State this explicitly.
- **Annual disturbance grain**: NISAR + MapBiomas Alerta will resolve sub-
  annual change; the PoC figure cannot.
- **Single AOI**: the figure characterizes the Brazilian Amazon
  deforestation arc. Generalization to closed-canopy boreal or savanna
  systems is an open question.

## Follow-on (not for this PoC)

1. Promote the three scripts into a small `nice_sar.gedi_overlap` subpackage if
   the workflow is reused for the actual NISAR analysis.
2. Replace PALSAR-2 with NISAR L2 GCOV once data starts flowing — the GEDI
   side and the figure code are unchanged; only `sample_sar.py` becomes
   `sample_nisar.py`.
3. Add a coherence-change panel (NISAR-only; PALSAR-2 has no routine
   coherence over the Amazon).
4. Add MapBiomas Alerta to refine disturbance timing.
