# RTCB-Gym baselines: HB_WEST, two-settlement across days

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0.5 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0`
- day-ahead prices: `ee0fd9974f117b07e957b114bd973f2a7890356e02040221b690fae5f344cfe7`

| Split | Episodes | Strategy | Days | Revenue (USD) | Capture rate (95% interval) | Cycles/day | Clipped actions |
|---|---|---|---:|---:|---:|---:|---:|
| validation | 7-day episodes | Perfect foresight (DA + RT) | 49 | 875,095 | 100.0% | 1.42 | 0 |
| validation | 7-day episodes | Perfect foresight (RT only) | 49 | 424,415 | 48.5% (43.3 to 52.0) | 1.42 | 0 |
| validation | 7-day episodes | DA schedule, delivered | 49 | 301,353 | 34.4% (25.6 to 42.6) | 1.36 | 0 |
| validation | 7-day episodes | DA schedule + MPC on DA prices | 49 | 331,586 | 37.9% (27.3 to 47.9) | 1.06 | 0 |
| validation | 7-day episodes | MPC on DA prices | 49 | 264,962 | 30.3% (24.0 to 34.6) | 1.06 | 0 |
| validation | 7-day episodes | Forecast + MPC | 49 | 246,581 | 28.2% (18.3 to 35.6) | 1.41 | 0 |
| validation | 1-day episodes | Perfect foresight (DA + RT) | 49 | 841,541 | 100.0% | 1.65 | 0 |
| validation | 1-day episodes | Perfect foresight (RT only) | 49 | 390,862 | 46.4% (41.3 to 50.1) | 1.65 | 0 |
| validation | 1-day episodes | DA schedule, delivered | 49 | 301,353 | 35.8% (26.6 to 44.3) | 1.36 | 0 |
| validation | 1-day episodes | DA schedule + MPC on DA prices | 49 | 291,731 | 34.7% (23.7 to 45.1) | 1.38 | 0 |
| validation | 1-day episodes | MPC on DA prices | 49 | 225,107 | 26.7% (20.7 to 30.7) | 1.38 | 0 |
| validation | 1-day episodes | Forecast + MPC | 49 | 204,423 | 24.3% (13.5 to 32.2) | 1.61 | 0 |
| test | 7-day episodes | Perfect foresight (DA + RT) | 56 | 1,566,401 | 100.0% | 1.75 | 0 |
| test | 7-day episodes | Perfect foresight (RT only) | 56 | 801,853 | 51.2% (47.3 to 54.7) | 1.75 | 0 |
| test | 7-day episodes | DA schedule, delivered | 56 | 447,502 | 28.6% (22.3 to 36.2) | 1.54 | 0 |
| test | 7-day episodes | DA schedule + MPC on DA prices | 56 | 522,391 | 33.3% (26.1 to 41.4) | 1.31 | 0 |
| test | 7-day episodes | MPC on DA prices | 56 | 469,144 | 30.0% (22.2 to 37.2) | 1.31 | 0 |
| test | 7-day episodes | Forecast + MPC | 56 | 361,327 | 23.1% (12.9 to 33.0) | 1.75 | 0 |
| test | 1-day episodes | Perfect foresight (DA + RT) | 56 | 1,532,895 | 100.0% | 1.88 | 0 |
| test | 1-day episodes | Perfect foresight (RT only) | 56 | 768,347 | 50.1% (46.2 to 53.9) | 1.88 | 0 |
| test | 1-day episodes | DA schedule, delivered | 56 | 447,502 | 29.2% (23.0 to 37.1) | 1.54 | 0 |
| test | 1-day episodes | DA schedule + MPC on DA prices | 56 | 485,334 | 31.7% (24.2 to 40.5) | 1.55 | 0 |
| test | 1-day episodes | MPC on DA prices | 56 | 432,088 | 28.2% (20.3 to 35.9) | 1.55 | 0 |
| test | 1-day episodes | Forecast + MPC | 56 | 339,629 | 22.2% (12.5 to 31.7) | 1.89 | 0 |

Every episode starts at SoC 0.5 and must end at or above 0.5. In 7-day episodes the SoC carries over midnight; one-day episodes cover the same operating days. Capture rate is against perfect foresight (DA + RT) of the same episode length. Each day's day-ahead bid is checked against what the battery could deliver from the starting SoC.

Data provided by [Grid Status](https://www.gridstatus.io).
