# RTCB-Gym baselines: HB_HOUSTON, real-time energy

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each day.

SHA-256 of the cleaned data:

- real-time prices: `ea27b7861e6210514264b51dfed49ab1cd7168ff4b5b693dcbd130dee9750c00`

| Split | Strategy | Days | Revenue (USD) | Capture rate | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|
| validation | Perfect foresight | 53 | 438,237 | 100.0% | 1.38 | 0 |
| validation | Threshold rule | 53 | 27,263 | 6.2% | 0.29 | 0 |
| validation | Rolling threshold (30 d) | 53 | 41,742 | 9.5% | 0.88 | 0 |
| validation | Forecast + MPC | 53 | 190,990 | 43.6% | 1.36 | 0 |
| test | Perfect foresight | 59 | 753,780 | 100.0% | 1.74 | 0 |
| test | Threshold rule | 59 | 50,807 | 6.7% | 0.24 | 0 |
| test | Rolling threshold (30 d) | 59 | 76,332 | 10.1% | 0.67 | 0 |
| test | Forecast + MPC | 59 | 309,317 | 41.0% | 1.75 | 0 |

Threshold rule: charge at or below $17.00/MWh, discharge at or above $27.73/MWh (training-split quantiles). Rolling threshold: quantiles 0.2 / 0.6 of the previous 30 days, refitted every day.

Data provided by [Grid Status](https://www.gridstatus.io).
