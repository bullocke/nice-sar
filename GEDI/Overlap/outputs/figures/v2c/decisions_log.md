# v2c decisions log

- input: GEDI/Overlap/outputs/pairs_with_sar_v2c.parquet
- args: min_sensitivity=0.95, min_pre_agbd=50.0, full_power_beams=True, require_same_pointing=True, require_same_beam=False, mad_k=3.0
- v2c cleaning (applied upstream by clean_pairs_v2c.py): A1 Hansen lossyear homogeneous within 30 m of pre footprint (min == max); B2 post_sar_date.year > 2000 + hansen_loss_max.
- start: n=321
- complete pre/post SAR + AGBD: n=321
- sensitivity >= 0.95 (both): n=321
- pre_agbd >= 50 Mg/ha: n=277
- full-power beams (both): n=152
- same antenna pointing: n=152
- MAD trim on delta_hv_db (3 * MAD=1.588): n=141