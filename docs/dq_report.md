# Meter data-quality report: UCI ElectricityLoadDiagrams20112014 (15-minute raw)

| Check | Result |
|---|---|
| Meters | 370 |
| Timestamps per meter | 140256 |
| Period | 2011-01-01 00:15:00 to 2015-01-01 00:00:00 |
| Frequency | 15min |
| Missing timestamps (calendar gaps) | 0 |
| Duplicate timestamps | 0 |
| Duplicates with conflicting values | 0 |
| Meters with NaN readings | 0 |
| Meters with negative readings | 0 |
| Meters with leading zeros (not yet commissioned) | 212 |
| Meters that never report non-zero | 0 |
| Meters with stuck runs (>= 96 equal) | 6 |
| Meters with zero runs after activation (>= 96) | 38 |

## Clock-change days (01:00-02:00 block vs neighbouring hours)

| Timestamp | Kind | Share of meters at exactly 0 | Median ratio to neighbours |
|---|---|---|---|
| 2011-03-27 01:00:00 | spring | 0.02 | 0.95 |
| 2011-10-30 01:00:00 | autumn | 0.00 | 0.96 |
| 2012-03-25 01:00:00 | spring | 0.63 | 0.00 |
| 2012-10-28 01:00:00 | autumn | 0.01 | 1.94 |
| 2013-03-31 01:00:00 | spring | 0.61 | 0.00 |
| 2013-10-27 01:00:00 | autumn | 0.00 | 1.94 |
| 2014-03-30 01:00:00 | spring | 0.68 | 0.00 |
| 2014-10-26 01:00:00 | autumn | 0.00 | 1.94 |

## Ten meters with the longest stuck / zero runs

| Meter | Longest stuck run | Longest zero run | NaN steps | Leading zeros |
|---|---|---|---|---|
| MT_288 | 985 | 576 | 0 | 0 |
| MT_003 | 344 | 96 | 0 | 35040 |
| MT_179 | 325 | 96 | 0 | 104064 |
| MT_289 | 148 | 0 | 0 | 102048 |
| MT_141 | 115 | 4 | 0 | 35040 |
| MT_292 | 96 | 0 | 0 | 0 |
| MT_305 | 95 | 0 | 0 | 103008 |
| MT_066 | 92 | 76 | 0 | 35040 |
| MT_109 | 85 | 4 | 0 | 109824 |
| MT_150 | 73 | 4 | 0 | 35040 |
