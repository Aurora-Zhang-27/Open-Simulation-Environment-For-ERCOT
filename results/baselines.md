# RTCB-Gym baselines: HB_WEST, real-time energy

Operating days 2025-12-05 .. 2026-10-24, data SHA-256 `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0`.
Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost.

| Split | Strategy | Days | Revenue (USD) | Capture rate | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|
| validation | Perfect foresight | 53 | 538,640 | 100.0% | 1.39 | 0 |
| validation | Threshold rule | 53 | 122,064 | 22.7% | 0.46 | 0 |
| validation | Forecast + MPC | 53 | 267,207 | 49.6% | 1.38 | 0 |
| test | Perfect foresight | 59 | 929,058 | 100.0% | 1.75 | 0 |
| test | Threshold rule | 59 | 133,541 | 14.4% | 0.38 | 0 |
| test | Forecast + MPC | 59 | 435,338 | 46.9% | 1.73 | 0 |

Data provided by [Grid Status](https://www.gridstatus.io).
