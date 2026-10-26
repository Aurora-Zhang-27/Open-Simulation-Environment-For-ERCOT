# RTCB-Gym baselines: HB_WEST, day-ahead + real time (two-settlement)

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each day.

SHA-256 of the cleaned data:

- real-time prices: `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0`
- day-ahead prices: `ee0fd9974f117b07e957b114bd973f2a7890356e02040221b690fae5f344cfe7`

| Split | Strategy | Days | Revenue (USD) | DA settlement | RT settlement | Capture rate | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | Perfect foresight (DA + RT) | 53 | 1,120,828 | -53,614 | 1,233,498 | 100.0% | 1.39 | 0 |
| validation | Perfect foresight (RT only) | 53 | 538,640 | 0 | 597,697 | 48.1% | 1.39 | 0 |
| validation | Perfect foresight (DA only, delivered) | 53 | 431,923 | 474,899 | 0 | 38.5% | 1.01 | 0 |
| validation | DA schedule, delivered | 53 | 403,959 | 446,934 | 0 | 36.0% | 1.01 | 0 |
| validation | DA schedule + MPC on DA prices | 53 | 390,586 | 446,934 | -13,373 | 34.8% | 1.01 | 0 |
| validation | MPC on DA prices | 53 | 338,807 | 0 | 381,783 | 30.2% | 1.01 | 0 |
| validation | Forecast + MPC | 53 | 267,207 | 0 | 325,893 | 23.8% | 1.38 | 0 |
| test | Perfect foresight (DA + RT) | 59 | 1,837,999 | -205,987 | 2,126,931 | 100.0% | 1.75 | 0 |
| test | Perfect foresight (RT only) | 59 | 929,058 | 0 | 1,012,003 | 50.5% | 1.75 | 0 |
| test | Perfect foresight (DA only, delivered) | 59 | 607,175 | 668,061 | 0 | 33.0% | 1.29 | 0 |
| test | DA schedule, delivered | 59 | 534,222 | 593,628 | 0 | 29.1% | 1.25 | 0 |
| test | DA schedule + MPC on DA prices | 59 | 572,101 | 593,628 | 39,359 | 31.1% | 1.29 | 0 |
| test | MPC on DA prices | 59 | 555,361 | 0 | 616,247 | 30.2% | 1.29 | 0 |
| test | Forecast + MPC | 59 | 435,338 | 0 | 517,371 | 23.7% | 1.73 | 0 |

Revenue = DA settlement + RT settlement - degradation. The day-ahead bid for each day is made at 10:00 local time the day before. "Clipped actions" counts real-time intervals, plus day-ahead hours clipped to what the battery could deliver.

Data provided by [Grid Status](https://www.gridstatus.io).
