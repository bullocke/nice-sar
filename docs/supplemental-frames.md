# Supplemental PROVISIONAL Frames

PROVISIONAL (calibrated) NISAR products are forward-processed for acquisitions from **2026-06-17** onward. The [availability overview](https://nisar-docs.asf.alaska.edu/availability-overview/) also says the science team selected "a limited set of NISAR frames" to be processed further back in time for time-series validation, and that these products go into the same PROVISIONAL collections.

The overview does not list those frames, so this page identifies them from the archive. Any PROVISIONAL granule acquired before 2026-06-17 must come from this supplemental processing. Listing those granules and grouping them by track, direction, and frame gives the set of supplemental frames.

!!! info "Snapshot"
    Generated 2026-09-23 from NASA CMR. The frames are **already processed and available**; they are not scheduled for future release. More frames may still be added, so rerun `scripts/list_supplemental_frames.py` to refresh the list.

## Summary

As of 2026-09-23 there are **301 frames** with at least two acquisition dates before 2026-06-17 (5,556 GCOV and 5,100 GUNW granules). They span **2025-10-29 to 2026-06-16**. The median frame has 17 GCOV dates, close to one per 12-day cycle. Most series begin in November 2025, a few start between January and March 2026, and some have gaps. Every granule has CRID `P05023`, except 47 GCOV/GUNW granules with `X05026`, which also belong to the PROVISIONAL collections.

A further 69 frames have only a single GCOV date, 2026-06-16, the day before forward processing began. They are excluded here because they are an edge effect of forward processing, not time series.

Frames are labeled with the nearest Natural Earth country to the frame center. Coastal and polar frames can be assigned to a neighboring country.

| Continent | Frames | GCOV granules | GUNW granules | Countries (frames) |
|---|--:|--:|--:|---|
| North America | 112 | 1997 | 1781 | United States of America (95), Canada (11), Mexico (5), Cuba (1) |
| Antarctica | 78 | 1694 | 1520 | Antarctica (78) |
| South America | 49 | 813 | 809 | Chile (11), Venezuela (11), Brazil (8), Bolivia (7), Peru (6), Argentina (4), Colombia (1), Paraguay (1) |
| Asia | 31 | 516 | 474 | India (20), Japan (4), China (2), South Korea (1), Myanmar (1), Taiwan (1), Afghanistan (1), Pakistan (1) |
| Europe | 13 | 251 | 230 | Iceland (8), Germany (2), Spain (1), France (1), Hungary (1) |
| Africa | 9 | 144 | 145 | South Africa (3), Eritrea (2), Ethiopia (2), Nigeria (1), Dem. Rep. Congo (1) |
| Oceania | 9 | 141 | 141 | Australia (5), New Zealand (4) |
| **Total** | **301** | **5556** | **5100** | |

The selection concentrates on established calibration and validation regions. These include Antarctica and the polar ice sheets, the western United States and Alaska, the Andes and Atacama, India (a joint NASA-ISRO focus), Iceland, and New Zealand. Only a handful of frames fall in tropical forest.

## Tropical frames

The table lists frames centered between 23.5°S and 23.5°N. **GCOV pol** is the polarization token of the first GCOV granule: `DHDH` is dual-pol HH/HV, `SHSH` is single-pol HH, and `QPDH` is quad-pol. Some frames alternate modes between cycles, so check each granule before building an HV time series.

| Country | Track | Dir | Frame | Center (lat, lon) | Dates | GCOV | GUNW | GCOV pol |
|---|--:|:-:|--:|---|---|--:|--:|---|
| Dem. Rep. Congo | 173 | A | 002 | 0.85, 19.34 | 2025-11-10 to 2026-03-22 | 12 | 12 | DHDH |
| Eritrea | 172 | A | 009 | 14.79, 41.09 | 2025-11-10 to 2026-06-14 | 17 | 17 | DHDH |
| Eritrea | 165 | D | 081 | 14.87, 40.70 | 2025-11-09 to 2026-06-13 | 19 | 19 | SHSH |
| Ethiopia | 172 | A | 008 | 12.80, 41.59 | 2025-11-10 to 2026-06-14 | 17 | 17 | DHDH |
| Ethiopia | 165 | D | 082 | 12.88, 40.21 | 2025-11-09 to 2026-06-13 | 19 | 19 | SHSH |
| Nigeria | 087 | A | 004 | 4.87, 6.01 | 2025-11-04 to 2026-06-08 | 17 | 17 | DHDH |
| South Africa | 042 | A | 166 | -23.07, 30.56 | 2025-11-25 to 2026-06-05 | 17 | 18 | DHDH |
| India | 149 | D | 081 | 14.90, 80.23 | 2026-01-19 to 2026-06-12 | 11 | 11 | QPDH |
| India | 055 | A | 010 | 16.04, 82.39 | 2026-01-01 to 2026-06-06 | 27 | 26 | DVDV |
| India | 149 | D | 080 | 16.89, 80.74 | 2026-03-08 to 2026-05-31 | 6 | 6 | QPDH |
| India | 156 | A | 011 | 18.78, 79.60 | 2026-03-09 to 2026-06-01 | 6 | 6 | DHDH |
| India | 098 | A | 012 | 19.95, 87.60 | 2026-03-04 to 2026-05-27 | 14 | 14 | DVDV |
| India | 113 | A | 013 | 22.73, 72.25 | 2025-11-18 to 2026-06-10 | 18 | 16 | DHDH |
| India | 156 | A | 013 | 22.75, 78.51 | 2026-03-09 to 2026-06-13 | 7 | 7 | DHDH |
| India | 135 | D | 077 | 22.76, 71.97 | 2025-11-19 to 2026-06-11 | 17 | 17 | QPDH |
| Mexico | 019 | A | 011 | 18.78, -99.36 | 2025-10-30 to 2026-06-15 | 20 | 20 | DHDH |
| Mexico | 113 | D | 079 | 18.84, -99.74 | 2025-11-06 to 2026-06-10 | 18 | 18 | DHDH |
| Mexico | 019 | A | 012 | 20.77, -99.90 | 2025-10-30 to 2026-06-15 | 20 | 20 | DHDH |
| Mexico | 113 | D | 078 | 20.83, -99.21 | 2025-11-06 to 2026-06-10 | 18 | 18 | DHDH |
| United States of America | 151 | A | 011 | 18.18, -155.42 | 2025-11-08 to 2026-06-12 | 34 | 17 | NADV |
| United States of America | 072 | D | 079 | 19.29, -155.84 | 2025-11-03 to 2026-06-07 | 35 | 17 | DHDH |
| United States of America | 151 | A | 012 | 20.73, -156.07 | 2025-11-08 to 2026-06-12 | 17 | 17 | DHDH |
| United States of America | 072 | D | 078 | 21.44, -155.20 | 2025-11-03 to 2026-06-07 | 35 | 17 | NADV |
| Australia | 110 | A | 171 | -13.11, 130.50 | 2026-03-05 to 2026-05-28 | 7 | 7 | DHDH |
| Australia | 017 | D | 095 | -12.96, 130.26 | 2026-02-27 to 2026-05-22 | 7 | 7 | DHDH |
| Bolivia | 118 | A | 167 | -21.07, -67.67 | 2025-11-06 to 2026-06-10 | 16 | 16 | DHDH |
| Bolivia | 169 | D | 099 | -21.01, -66.93 | 2025-11-09 to 2026-06-13 | 18 | 18 | DHDH |
| Bolivia | 118 | A | 168 | -19.08, -68.08 | 2025-11-06 to 2026-06-10 | 16 | 16 | DHDH |
| Bolivia | 097 | D | 098 | -19.04, -68.62 | 2025-11-04 to 2026-06-08 | 18 | 18 | SHSH |
| Bolivia | 118 | A | 169 | -17.09, -68.48 | 2025-11-06 to 2026-06-10 | 16 | 16 | DHDH |
| Bolivia | 097 | D | 097 | -17.03, -68.27 | 2025-11-04 to 2026-06-08 | 18 | 18 | SHSH |
| Bolivia | 118 | A | 170 | -15.09, -68.88 | 2025-11-06 to 2026-06-10 | 16 | 16 | DHDH |
| Brazil | 160 | A | 174 | -7.13, -39.28 | 2025-11-09 to 2026-06-13 | 18 | 18 | DHDH |
| Brazil | 088 | A | 174 | -7.11, -41.38 | 2025-11-16 to 2026-06-08 | 17 | 17 | DHDH |
| Brazil | 046 | A | 175 | -5.75, -72.86 | 2025-11-01 to 2026-06-05 | 17 | 17 | DHDH |
| Brazil | 089 | A | 176 | -3.13, -67.20 | 2025-11-04 to 2026-06-08 | 19 | 17 | DHDH |
| Brazil | 025 | D | 090 | -3.07, -67.45 | 2025-10-30 to 2026-06-15 | 28 | 27 | SHSH |
| Brazil | 154 | D | 089 | -0.95, -48.29 | 2026-03-08 to 2026-06-12 | 11 | 10 | DHDH |
| Brazil | 104 | A | 002 | 0.83, -57.66 | 2026-01-04 to 2026-05-16 | 8 | 8 | DHDH |
| Brazil | 039 | D | 088 | 0.94, -56.16 | 2025-12-30 to 2026-05-23 | 13 | 13 | DHDH |
| Chile | 046 | A | 166 | -23.10, -69.38 | 2025-11-01 to 2026-06-05 | 17 | 17 | DHDH |
| Chile | 118 | A | 166 | -23.07, -67.25 | 2025-11-06 to 2026-06-10 | 16 | 16 | DHDH |
| Chile | 097 | D | 100 | -23.02, -69.44 | 2025-11-04 to 2026-06-08 | 17 | 17 | DHDH |
| Chile | 169 | D | 100 | -23.00, -67.34 | 2025-11-09 to 2026-06-13 | 18 | 18 | DHDH |
| Chile | 097 | D | 099 | -21.12, -69.04 | 2025-11-04 to 2026-06-08 | 17 | 17 | DHDH |
| Chile | 046 | A | 167 | -21.07, -69.79 | 2025-11-01 to 2026-06-05 | 17 | 17 | DHDH |
| Chile | 046 | A | 168 | -19.08, -70.20 | 2025-11-01 to 2026-06-05 | 17 | 17 | DHDH |
| Chile | 025 | D | 098 | -19.03, -70.77 | 2025-10-30 to 2026-06-15 | 18 | 18 | SHSH |
| Colombia | 083 | D | 088 | 0.94, -74.92 | 2025-11-03 to 2026-06-07 | 17 | 16 | SHSH |
| Paraguay | 161 | A | 167 | -21.02, -61.44 | 2025-11-21 to 2026-06-13 | 17 | 17 | DHDH |
| Peru | 046 | A | 169 | -17.09, -70.62 | 2025-11-01 to 2026-06-05 | 17 | 17 | DHDH |
| Peru | 025 | D | 097 | -17.04, -70.35 | 2025-10-30 to 2026-06-15 | 19 | 19 | SHSH |
| Peru | 046 | A | 170 | -15.06, -71.03 | 2025-11-01 to 2026-06-05 | 17 | 17 | DHDH |
| Peru | 147 | A | 175 | -5.05, -75.09 | 2025-12-02 to 2026-06-12 | 15 | 15 | DHDH |
| Peru | 083 | D | 091 | -5.04, -76.22 | 2025-11-03 to 2026-06-07 | 17 | 17 | SHSH |
| Peru | 155 | D | 091 | -5.03, -74.12 | 2025-11-20 to 2026-06-12 | 16 | 16 | SHSH |
| Venezuela | 054 | D | 084 | 8.78, -69.00 | 2025-11-01 to 2026-06-05 | 19 | 19 | DHDH |
| Venezuela | 162 | A | 006 | 8.84, -67.73 | 2025-11-09 to 2026-06-13 | 18 | 18 | DHDH |
| Venezuela | 061 | A | 006 | 8.86, -65.66 | 2025-11-02 to 2026-06-06 | 13 | 13 | DHDH |
| Venezuela | 025 | D | 084 | 8.88, -64.82 | 2025-10-30 to 2026-06-15 | 19 | 19 | DHDH |
| Venezuela | 126 | D | 084 | 8.91, -66.89 | 2025-11-06 to 2026-06-10 | 17 | 17 | DHDH |
| Venezuela | 133 | A | 007 | 10.84, -64.05 | 2025-11-07 to 2026-06-11 | 18 | 18 | DHDH |
| Venezuela | 162 | A | 007 | 10.84, -68.21 | 2025-11-09 to 2026-06-13 | 18 | 18 | DHDH |
| Venezuela | 061 | A | 007 | 10.85, -66.13 | 2025-11-02 to 2026-06-06 | 12 | 12 | DHDH |
| Venezuela | 025 | D | 083 | 10.88, -64.34 | 2025-10-30 to 2026-06-15 | 19 | 19 | DHDH |
| Venezuela | 054 | D | 083 | 10.89, -68.50 | 2025-11-01 to 2026-06-05 | 19 | 19 | DHDH |
| Venezuela | 126 | D | 083 | 10.91, -66.41 | 2025-11-30 to 2026-06-10 | 16 | 17 | DHDH |

### Amazon basin

Frames in or near the Amazon forest, by approximate location:

- **Central Amazon, Amazonas state (Brazil):** 025 D 090 (28 GCOV dates, the densest series in the region) and 089 A 176 overlap near 3°S, 67°W.
- **Western Amazon, Brazil/Peru border and Loreto (Peru):** 046 A 175, 147 A 175, 083 D 091, and 155 D 091, around 5-6°S, 72-76°W.
- **Colombian Amazon:** 083 D 088, near 0.9°N, 74.9°W.
- **Eastern and northern Pará (Brazil):** 154 D 089 near Belém (starts March 2026), and 104 A 002 and 039 D 088 north of the Amazon River near 0.9°N, 56-58°W.

None of these frames fall in the main "arc of deforestation" (Rondônia, northern Mato Grosso, southern Pará). For those areas PROVISIONAL data start on 2026-06-17, and BETA data cover a few dates between October 2025 and January 2026 (see [Getting Started](getting-started.md#data-maturity-provisional-vs-beta) on mixing maturities).

## Querying a supplemental frame

```python
from nice_sar.search import search_nisar, summarize_results

# Pre-June 2026 PROVISIONAL time series for the central Amazon frame
results = search_nisar(
    "GCOV", track=25, frame=90, direction="D", end="2026-06-16", max_results=100
)
for s in summarize_results(results):
    print(s.start, s.crid, s.full_frame, s.granule_id)
```

```bash
nice-sar search --product GUNW --track 25 --frame 90 --direction D --end 2026-06-16
```

## Data file and regeneration

The full table, including every continent, GUNW rows, footprint bounds, and an example granule ID per frame, is written by [`scripts/list_supplemental_frames.py`](https://github.com/bullocke/nice-sar/blob/main/scripts/list_supplemental_frames.py) to `docs/data/supplemental_provisional_frames.csv` (no Earthdata login is needed):

```bash
python scripts/list_supplemental_frames.py --products GCOV GUNW
```
