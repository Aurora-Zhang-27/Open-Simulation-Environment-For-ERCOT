# RTCB-Gym baselines: HB_HOUSTON, day-ahead + real time (two-settlement)

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `ea27b7861e6210514264b51dfed49ab1cd7168ff4b5b693dcbd130dee9750c00`
- day-ahead prices: `6885d4eb2c38d404fd604ce9a4ef3417834071602e382ee0fa353a7e7a7b7933`

| Split | Strategy | Days | Revenue (USD) | DA settlement | RT settlement | Capture rate (95% interval) | Without top 5 days | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Perfect foresight (DA + RT) | 53 | 942,484 | -89,301 | 1,090,417 | 100.0% | 100.0% | 1.38 | 0 |
| validation | Perfect foresight (RT only) | 53 | 438,237 | 0 | 496,869 | 46.5% (42.2 to 49.5) | 43.4% | 1.38 | 0 |
| validation | Perfect foresight (DA only, delivered) | 53 | 345,114 | 388,089 | 0 | 36.6% (27.5 to 48.2) | 42.3% | 1.01 | 0 |
| validation | DA schedule, delivered | 53 | 312,778 | 355,753 | 0 | 33.2% (24.6 to 43.8) | 37.2% | 1.01 | 0 |
| validation | DA schedule + MPC on DA prices | 53 | 308,118 | 355,753 | -4,660 | 32.7% (23.9 to 44.4) | 39.1% | 1.01 | 0 |
| validation | MPC on DA prices | 53 | 250,008 | 0 | 292,983 | 26.5% (21.1 to 31.2) | 25.6% | 1.01 | 0 |
| validation | Forecast + MPC | 53 | 190,990 | 0 | 248,812 | 20.3% (9.5 to 31.8) | 21.5% | 1.36 | 0 |
| test | Perfect foresight (DA + RT) | 59 | 1,561,622 | -223,372 | 1,867,204 | 100.0% | 100.0% | 1.74 | 0 |
| test | Perfect foresight (RT only) | 59 | 753,780 | 0 | 835,989 | 48.3% (44.6 to 51.5) | 45.3% | 1.74 | 0 |
| test | Perfect foresight (DA only, delivered) | 59 | 455,131 | 510,464 | 0 | 29.1% (22.9 to 37.0) | 37.7% | 1.17 | 0 |
| test | DA schedule, delivered | 59 | 400,927 | 455,824 | 0 | 25.7% (20.0 to 32.8) | 33.1% | 1.16 | 0 |
| test | DA schedule + MPC on DA prices | 59 | 413,581 | 455,824 | 13,090 | 26.5% (20.6 to 33.4) | 33.9% | 1.17 | 0 |
| test | MPC on DA prices | 59 | 395,033 | 0 | 450,366 | 25.3% (19.0 to 30.6) | 21.7% | 1.17 | 0 |
| test | Forecast + MPC | 59 | 309,317 | 0 | 392,084 | 19.8% (11.3 to 27.2) | 15.8% | 1.75 | 0 |

Revenue = DA settlement + RT settlement - degradation. The day-ahead bid for each day is made at 10:00 local time the day before. "Clipped actions" counts real-time intervals, plus day-ahead hours clipped to what the battery could deliver.

Data provided by [Grid Status](https://www.gridstatus.io).
