# RTCB-Gym baselines: HB_WEST, battery duration (two-settlement)

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0`
- day-ahead prices: `ee0fd9974f117b07e957b114bd973f2a7890356e02040221b690fae5f344cfe7`

| Split | Duration | Strategy | Revenue (USD) | USD per MW-day | Capture rate | Cycles/day |
|---|---|---|---:|---:|---:|---:|
| validation | 1 h | Perfect foresight (DA + RT) | 702,667 | 133 | 100.0% | 1.71 |
| validation | 1 h | Perfect foresight (RT only) | 335,876 | 63 | 47.8% | 1.71 |
| validation | 1 h | DA schedule, delivered | 207,155 | 39 | 29.5% | 1.02 |
| validation | 1 h | DA schedule + MPC on DA prices | 209,814 | 40 | 29.9% | 1.02 |
| validation | 1 h | MPC on DA prices | 175,673 | 33 | 25.0% | 1.02 |
| validation | 1 h | Forecast + MPC | 135,357 | 26 | 19.3% | 1.69 |
| test | 1 h | Perfect foresight (DA + RT) | 1,147,337 | 194 | 100.0% | 2.22 |
| test | 1 h | Perfect foresight (RT only) | 571,365 | 97 | 49.8% | 2.22 |
| test | 1 h | DA schedule, delivered | 274,425 | 47 | 23.9% | 1.30 |
| test | 1 h | DA schedule + MPC on DA prices | 284,380 | 48 | 24.8% | 1.34 |
| test | 1 h | MPC on DA prices | 266,500 | 45 | 23.2% | 1.34 |
| test | 1 h | Forecast + MPC | 197,066 | 33 | 17.2% | 2.18 |
| validation | 2 h | Perfect foresight (DA + RT) | 1,120,828 | 211 | 100.0% | 1.39 |
| validation | 2 h | Perfect foresight (RT only) | 538,640 | 102 | 48.1% | 1.39 |
| validation | 2 h | DA schedule, delivered | 403,959 | 76 | 36.0% | 1.01 |
| validation | 2 h | DA schedule + MPC on DA prices | 390,586 | 74 | 34.8% | 1.01 |
| validation | 2 h | MPC on DA prices | 338,807 | 64 | 30.2% | 1.01 |
| validation | 2 h | Forecast + MPC | 267,207 | 50 | 23.8% | 1.38 |
| test | 2 h | Perfect foresight (DA + RT) | 1,837,999 | 312 | 100.0% | 1.75 |
| test | 2 h | Perfect foresight (RT only) | 929,058 | 157 | 50.5% | 1.75 |
| test | 2 h | DA schedule, delivered | 534,222 | 91 | 29.1% | 1.25 |
| test | 2 h | DA schedule + MPC on DA prices | 572,101 | 97 | 31.1% | 1.29 |
| test | 2 h | MPC on DA prices | 555,361 | 94 | 30.2% | 1.29 |
| test | 2 h | Forecast + MPC | 435,338 | 74 | 23.7% | 1.73 |
| validation | 4 h | Perfect foresight (DA + RT) | 1,565,631 | 295 | 100.0% | 1.15 |
| validation | 4 h | Perfect foresight (RT only) | 778,001 | 147 | 49.7% | 1.15 |
| validation | 4 h | DA schedule, delivered | 689,348 | 130 | 44.0% | 1.00 |
| validation | 4 h | DA schedule + MPC on DA prices | 692,047 | 131 | 44.2% | 1.00 |
| validation | 4 h | MPC on DA prices | 602,416 | 114 | 38.5% | 1.00 |
| validation | 4 h | Forecast + MPC | 503,072 | 95 | 32.1% | 1.14 |
| test | 4 h | Perfect foresight (DA + RT) | 2,605,333 | 442 | 100.0% | 1.36 |
| test | 4 h | Perfect foresight (RT only) | 1,359,447 | 230 | 52.2% | 1.36 |
| test | 4 h | DA schedule, delivered | 938,279 | 159 | 36.0% | 1.16 |
| test | 4 h | DA schedule + MPC on DA prices | 1,003,337 | 170 | 38.5% | 1.17 |
| test | 4 h | MPC on DA prices | 993,724 | 168 | 38.1% | 1.17 |
| test | 4 h | Forecast + MPC | 801,598 | 136 | 30.8% | 1.35 |

The same 100 MW with other energy capacities; everything else as in the two-settlement table. Capture rate is against perfect foresight (DA + RT) of the same battery.

Data provided by [Grid Status](https://www.gridstatus.io).
