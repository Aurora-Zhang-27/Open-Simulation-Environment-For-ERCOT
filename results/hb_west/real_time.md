# RTCB-Gym baselines: HB_WEST, real-time energy

Operating days 2025-12-05 .. 2026-10-24. Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost, SoC 0 at the start of each episode.

SHA-256 of the cleaned data:

- real-time prices: `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0`

| Split | Strategy | Days | Revenue (USD) | Capture rate (95% interval) | Without top 5 days | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | Perfect foresight | 53 | 538,640 | 100.0% | 100.0% | 1.39 | 0 |
| validation | Threshold rule | 53 | 122,064 | 22.7% (13.5 to 37.7) | 37.0% | 0.46 | 0 |
| validation | Rolling threshold (30 d) | 53 | 117,801 | 21.9% (15.0 to 31.7) | 34.6% | 0.83 | 0 |
| validation | Forecast + MPC | 53 | 267,207 | 49.6% (29.7 to 68.1) | 55.9% | 1.38 | 0 |
| validation | Level-corrected forecast + MPC | 53 | 321,573 | 59.7% (47.0 to 71.7) | 59.1% | 1.43 | 0 |
| validation | Rolling threshold offer (30 d) | 53 | 102,033 | 18.9% (13.1 to 27.4) | 29.9% | 0.83 | 0 |
| validation | Offer MPC | 53 | 188,659 | 35.0% (22.8 to 45.7) | 30.1% | 1.00 | 0 |
| test | Perfect foresight | 59 | 929,058 | 100.0% | 100.0% | 1.75 | 0 |
| test | Threshold rule | 59 | 133,541 | 14.4% (6.6 to 25.9) | 21.4% | 0.38 | 0 |
| test | Rolling threshold (30 d) | 59 | 208,272 | 22.4% (12.1 to 33.1) | 28.0% | 0.81 | 0 |
| test | Forecast + MPC | 59 | 435,338 | 46.9% (33.1 to 58.9) | 39.3% | 1.73 | 0 |
| test | Level-corrected forecast + MPC | 59 | 417,356 | 44.9% (30.1 to 60.5) | 45.5% | 1.69 | 0 |
| test | Rolling threshold offer (30 d) | 59 | 173,471 | 18.7% (9.8 to 27.7) | 23.7% | 0.81 | 0 |
| test | Offer MPC | 59 | 225,515 | 24.3% (10.7 to 37.9) | 16.9% | 1.12 | 0 |

Threshold rule: charge at or below $15.65/MWh, discharge at or above $30.91/MWh (training-split quantiles). Rolling threshold: quantiles 0.3 / 0.6 of the previous 30 days, refitted every day. Level-corrected forecast: last 2 intervals, decay 0.95 (lowest training-split forecast error).

The two offer strategies submit an offer curve instead of a power (`action_mode="offer"`): the market dispatches them against the interval's own price. Rolling threshold offer: quantiles 0.3 / 0.6. The bound is the same: a curve cannot do more than the best power.

Capture rate: revenue / perfect foresight, with a 95% interval from a block bootstrap over days, and again without the five days on which perfect foresight earns most.

Data provided by [Grid Status](https://www.gridstatus.io).
