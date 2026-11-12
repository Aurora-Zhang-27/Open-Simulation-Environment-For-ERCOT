# RTCB-Gym baselines: HB_HOUSTON, two-settlement across days

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0.5 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `ea27b7861e6210514264b51dfed49ab1cd7168ff4b5b693dcbd130dee9750c00`
- day-ahead prices: `6885d4eb2c38d404fd604ce9a4ef3417834071602e382ee0fa353a7e7a7b7933`

| Split | Episodes | Strategy | Days | Revenue (USD) | Capture rate (95% interval) | Cycles/day | Clipped actions |
|---|---|---|---:|---:|---:|---:|---:|
| validation | 7-day episodes | Perfect foresight (DA + RT) | 49 | 707,388 | 100.0% | 1.40 | 0 |
| validation | 7-day episodes | Perfect foresight (RT only) | 49 | 326,662 | 46.2% (40.5 to 50.1) | 1.40 | 0 |
| validation | 7-day episodes | DA schedule, delivered | 49 | 221,706 | 31.3% (21.1 to 40.3) | 1.12 | 0 |
| validation | 7-day episodes | DA schedule + MPC on DA prices | 49 | 261,473 | 37.0% (25.2 to 47.9) | 1.04 | 0 |
| validation | 7-day episodes | MPC on DA prices | 49 | 183,700 | 26.0% (19.0 to 30.8) | 1.04 | 0 |
| validation | 7-day episodes | Forecast + MPC | 49 | 176,703 | 25.0% (13.2 to 32.9) | 1.35 | 0 |
| validation | 1-day episodes | Perfect foresight (DA + RT) | 49 | 674,845 | 100.0% | 1.53 | 0 |
| validation | 1-day episodes | Perfect foresight (RT only) | 49 | 294,120 | 43.6% (37.6 to 47.8) | 1.53 | 0 |
| validation | 1-day episodes | DA schedule, delivered | 49 | 221,706 | 32.9% (22.1 to 42.4) | 1.12 | 0 |
| validation | 1-day episodes | DA schedule + MPC on DA prices | 49 | 225,985 | 33.5% (21.0 to 44.9) | 1.14 | 0 |
| validation | 1-day episodes | MPC on DA prices | 49 | 148,213 | 22.0% (14.0 to 27.3) | 1.14 | 0 |
| validation | 1-day episodes | Forecast + MPC | 49 | 134,843 | 20.0% (7.7 to 28.4) | 1.50 | 0 |
| test | 7-day episodes | Perfect foresight (DA + RT) | 56 | 1,343,416 | 100.0% | 1.79 | 0 |
| test | 7-day episodes | Perfect foresight (RT only) | 56 | 646,053 | 48.1% (44.4 to 51.3) | 1.79 | 0 |
| test | 7-day episodes | DA schedule, delivered | 56 | 308,559 | 23.0% (16.9 to 30.0) | 1.39 | 0 |
| test | 7-day episodes | DA schedule + MPC on DA prices | 56 | 381,330 | 28.4% (21.6 to 35.2) | 1.18 | 0 |
| test | 7-day episodes | MPC on DA prices | 56 | 331,489 | 24.7% (18.0 to 30.5) | 1.18 | 0 |
| test | 7-day episodes | Forecast + MPC | 56 | 268,194 | 20.0% (9.6 to 29.0) | 1.79 | 0 |
| test | 1-day episodes | Perfect foresight (DA + RT) | 56 | 1,308,692 | 100.0% | 1.88 | 0 |
| test | 1-day episodes | Perfect foresight (RT only) | 56 | 611,329 | 46.7% (42.9 to 50.1) | 1.88 | 0 |
| test | 1-day episodes | DA schedule, delivered | 56 | 308,559 | 23.6% (17.2 to 30.8) | 1.39 | 0 |
| test | 1-day episodes | DA schedule + MPC on DA prices | 56 | 343,447 | 26.2% (19.8 to 33.1) | 1.39 | 0 |
| test | 1-day episodes | MPC on DA prices | 56 | 293,607 | 22.4% (15.3 to 29.0) | 1.39 | 0 |
| test | 1-day episodes | Forecast + MPC | 56 | 229,139 | 17.5% (7.2 to 27.0) | 1.90 | 0 |

Every episode starts at SoC 0.5 and must end at or above 0.5. In 7-day episodes the SoC carries over midnight; one-day episodes cover the same operating days. Capture rate is against perfect foresight (DA + RT) of the same episode length. Each day's day-ahead bid is checked against what the battery could deliver from the starting SoC.

Data provided by [Grid Status](https://www.gridstatus.io).
