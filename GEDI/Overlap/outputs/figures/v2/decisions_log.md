# v2 decisions log

- input: GEDI/Overlap/outputs/pairs_with_sar_v2.parquet
- args: min_sensitivity=0.95, min_pre_agbd=50.0, full_power_beams=True, require_same_pointing=True, require_same_beam=False, mad_k=3.0
- v2 invariant (by construction in sample_sar_v2.py): pre_PassDirection == post_PassDirection AND pre_RSP_Path_Number == post_RSP_Path_Number
- start: n=1126
- complete pre/post SAR + AGBD: n=781
- sensitivity >= 0.95 (both): n=781
- pre_agbd >= 50 Mg/ha: n=631
- full-power beams (both): n=354
- same antenna pointing: n=354
- MAD trim on delta_hv_db (3 * MAD=1.402): n=329