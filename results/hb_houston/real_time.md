# RTCB-Gym baselines: HB_HOUSTON, real-time energy

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `ea27b7861e6210514264b51dfed49ab1cd7168ff4b5b693dcbd130dee9750c00`

| Split | Strategy | Days | Revenue (USD) | Capture rate (95% interval) | Without top 5 days | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | Perfect foresight | 53 | 438,237 | 100.0% | 100.0% | 1.38 | 0 |
| validation | Threshold rule | 53 | 27,263 | 6.2% (3.4 to 10.7) | 10.5% | 0.29 | 0 |
| validation | Rolling threshold (30 d) | 53 | 41,742 | 9.5% (5.4 to 16.8) | 14.8% | 0.88 | 0 |
| validation | Forecast + MPC | 53 | 190,990 | 43.6% (21.0 to 64.9) | 49.6% | 1.36 | 0 |
| validation | Level-corrected forecast + MPC | 53 | 244,014 | 55.7% (44.2 to 68.0) | 53.3% | 1.38 | 0 |
| validation | Rolling threshold offer (30 d) | 53 | 70,906 | 16.2% (10.6 to 25.3) | 23.3% | 0.82 | 0 |
| validation | Offer MPC | 53 | 142,183 | 32.4% (13.9 to 44.2) | 18.6% | 1.14 | 0 |
| test | Perfect foresight | 59 | 753,780 | 100.0% | 100.0% | 1.74 | 0 |
| test | Threshold rule | 59 | 50,807 | 6.7% (2.3 to 14.7) | 10.8% | 0.24 | 0 |
| test | Rolling threshold (30 d) | 59 | 76,332 | 10.1% (6.1 to 15.3) | 14.9% | 0.67 | 0 |
| test | Forecast + MPC | 59 | 309,317 | 41.0% (25.1 to 55.3) | 34.9% | 1.75 | 0 |
| test | Level-corrected forecast + MPC | 59 | 306,257 | 40.6% (28.4 to 53.9) | 40.4% | 1.67 | 0 |
| test | Rolling threshold offer (30 d) | 59 | 71,204 | 9.4% (4.7 to 14.3) | 13.0% | 0.59 | 0 |
| test | Offer MPC | 59 | 159,441 | 21.2% (10.7 to 32.8) | 17.5% | 1.34 | 0 |

Threshold rule: charge at or below $17.00/MWh, discharge at or above $27.73/MWh (training-split quantiles). Rolling threshold: quantiles 0.2 / 0.6 of the previous 30 days, refitted every day. Level-corrected forecast: last 2 intervals, decay 0.9 (lowest training-split forecast error).

The two offer strategies submit an offer curve instead of a power (`action_mode="offer"`): the market dispatches them against the interval's own price. Rolling threshold offer: quantiles 0.2 / 0.7. The bound is the same: a curve cannot do more than the best power.

Capture rate: revenue / perfect foresight, with a 95% interval from a block bootstrap over days, and again without the five days on which perfect foresight earns most.

Data provided by [Grid Status](https://www.gridstatus.io).
