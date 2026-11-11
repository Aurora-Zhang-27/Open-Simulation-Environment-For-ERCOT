# RTCB-Gym baselines: HB_HOUSTON, battery duration (two-settlement)

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `ea27b7861e6210514264b51dfed49ab1cd7168ff4b5b693dcbd130dee9750c00`
- day-ahead prices: `6885d4eb2c38d404fd604ce9a4ef3417834071602e382ee0fa353a7e7a7b7933`

| Split | Duration | Strategy | Revenue (USD) | USD per MW-day | Capture rate | Cycles/day |
|---|---|---|---:|---:|---:|---:|
| validation | 1 h | Perfect foresight (DA + RT) | 610,705 | 115 | 100.0% | 1.76 |
| validation | 1 h | Perfect foresight (RT only) | 283,416 | 53 | 46.4% | 1.76 |
| validation | 1 h | DA schedule, delivered | 162,013 | 31 | 26.5% | 1.02 |
| validation | 1 h | DA schedule + MPC on DA prices | 163,030 | 31 | 26.7% | 1.02 |
| validation | 1 h | MPC on DA prices | 119,074 | 22 | 19.5% | 1.02 |
| validation | 1 h | Forecast + MPC | 90,510 | 17 | 14.8% | 1.73 |
| test | 1 h | Perfect foresight (DA + RT) | 997,248 | 169 | 100.0% | 2.28 |
| test | 1 h | Perfect foresight (RT only) | 473,771 | 80 | 47.5% | 2.28 |
| test | 1 h | DA schedule, delivered | 209,500 | 36 | 21.0% | 1.23 |
| test | 1 h | DA schedule + MPC on DA prices | 229,562 | 39 | 23.0% | 1.25 |
| test | 1 h | MPC on DA prices | 207,601 | 35 | 20.8% | 1.25 |
| test | 1 h | Forecast + MPC | 136,719 | 23 | 13.7% | 2.28 |
| validation | 2 h | Perfect foresight (DA + RT) | 942,484 | 178 | 100.0% | 1.38 |
| validation | 2 h | Perfect foresight (RT only) | 438,237 | 83 | 46.5% | 1.38 |
| validation | 2 h | DA schedule, delivered | 312,778 | 59 | 33.2% | 1.01 |
| validation | 2 h | DA schedule + MPC on DA prices | 308,118 | 58 | 32.7% | 1.01 |
| validation | 2 h | MPC on DA prices | 250,008 | 47 | 26.5% | 1.01 |
| validation | 2 h | Forecast + MPC | 190,990 | 36 | 20.3% | 1.36 |
| test | 2 h | Perfect foresight (DA + RT) | 1,561,622 | 265 | 100.0% | 1.74 |
| test | 2 h | Perfect foresight (RT only) | 753,780 | 128 | 48.3% | 1.74 |
| test | 2 h | DA schedule, delivered | 400,927 | 68 | 25.7% | 1.16 |
| test | 2 h | DA schedule + MPC on DA prices | 413,581 | 70 | 26.5% | 1.17 |
| test | 2 h | MPC on DA prices | 395,033 | 67 | 25.3% | 1.17 |
| test | 2 h | Forecast + MPC | 309,317 | 52 | 19.8% | 1.75 |
| validation | 4 h | Perfect foresight (DA + RT) | 1,271,153 | 240 | 100.0% | 1.11 |
| validation | 4 h | Perfect foresight (RT only) | 607,051 | 115 | 47.8% | 1.11 |
| validation | 4 h | DA schedule, delivered | 519,811 | 98 | 40.9% | 0.99 |
| validation | 4 h | DA schedule + MPC on DA prices | 540,327 | 102 | 42.5% | 0.99 |
| validation | 4 h | MPC on DA prices | 441,413 | 83 | 34.7% | 0.99 |
| validation | 4 h | Forecast + MPC | 356,111 | 67 | 28.0% | 1.10 |
| test | 4 h | Perfect foresight (DA + RT) | 2,169,449 | 368 | 100.0% | 1.31 |
| test | 4 h | Perfect foresight (RT only) | 1,082,689 | 184 | 49.9% | 1.31 |
| test | 4 h | DA schedule, delivered | 709,581 | 120 | 32.7% | 1.08 |
| test | 4 h | DA schedule + MPC on DA prices | 738,959 | 125 | 34.1% | 1.08 |
| test | 4 h | MPC on DA prices | 742,985 | 126 | 34.2% | 1.08 |
| test | 4 h | Forecast + MPC | 585,403 | 99 | 27.0% | 1.31 |

The same 100 MW with other energy capacities; everything else as in the two-settlement table. Capture rate is against perfect foresight (DA + RT) of the same battery.

Data provided by [Grid Status](https://www.gridstatus.io).
