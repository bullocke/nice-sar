# Decisions log

- input: GEDI/Overlap/outputs/pairs_with_sar.parquet
- args: min_sensitivity=0.95, min_pre_agbd=50.0, full_power_beams=True, require_same_pointing=True, mad_k=3.0
- start: n=1126
- complete pre/post SAR + AGBD: n=609
- sensitivity >= 0.95 (both): n=609
- pre_agbd >= 50 Mg/ha: n=477
- full-power beams (both): n=254
- same antenna pointing: n=254
- MAD trim on delta_hv_db (3 * MAD=1.493): n=233

## Fit results
- - delta_hv_db: slope=+7.39 (SE 6.07), intercept=-123, R²=0.006, p=2.24e-01, n=233
- - delta_hh_db: slope=+8.16 (SE 4.38), intercept=-130, R²=0.015, p=6.35e-02, n=233
- - delta_rfdi: slope=+45 (SE 51.3), intercept=-132, R²=0.003, p=3.82e-01, n=233
