# Data-quality report: Elia total load, 15-minute, 2023-2026

| Check | Result |
|---|---|
| Rows (raw / after de-duplication) | 131424 / 131424 |
| Period | 2023-01-01 00:00 to 2026-09-30 23:45 UTC |
| Last measured quarter-hour | 2026-09-30 23:45 UTC |
| Duplicate timestamps | 0 |
| Missing timestamps on the 15-minute grid | 0 |
| Quarter-hours with no measured load (UTC) | 3: 2023-08-10 09:00, 2023-08-11 09:00, 2023-09-11 09:00 |
| Non-positive load readings | 0 |
| Readings inside runs of 4 or more identical values | 0 |
| Day-ahead p10-p90 band violations (p10 above forecast or p90 below it) | 0 |
| Measured load below / above Elia's day-ahead p10 / p90 | 12.7% / 12.0% |
| Robust sigma of (measured - most recent forecast) | 200 MW |
| Quarter-hours more than 6 sigma from the most recent forecast | 279 |

## Largest departures from Elia's most recent forecast

Consecutive flagged quarter-hours (gaps of up to two) are grouped into one episode. The data alone cannot say whether an episode is a metering or upscaling artefact or a real demand event that the forecast missed.

| Start (UTC) | End (UTC) | Quarter-hours | Largest departure (MW) |
|---|---|---|---|
| 2023-02-05 16:00 | 2023-02-05 18:45 | 12 | 2654 |
| 2024-05-22 06:00 | 2024-05-22 15:00 | 33 | 2061 |
| 2024-04-02 03:00 | 2024-04-02 04:15 | 6 | 1955 |
| 2025-03-25 09:15 | 2025-03-25 09:30 | 2 | 1835 |
| 2024-04-02 16:45 | 2024-04-02 19:45 | 13 | 1784 |
| 2025-12-03 12:00 | 2025-12-03 19:30 | 26 | 1773 |
| 2025-11-05 11:15 | 2025-11-05 15:15 | 15 | 1765 |
| 2025-11-06 11:00 | 2025-11-06 12:45 | 8 | 1592 |
