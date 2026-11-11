# RTCB-Gym baselines: HB_WEST, five-minute energy + ECRS

Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost; every day starts at SoC 0.5 and ends at or above 0.5. Five-minute dispatch, 15-minute settlement, ECRS duration requirement 1 h, day-ahead bids at 10:00 the day before.

ECRS follows the reduced model of `rtcb_gym/ancillary.py`: reserve from the discharge side only, full awards at the clearing price, no deployment. Not ERCOT's complete ancillary-service settlement.

Settlement price check: the 15-minute mean of the five-minute energy price is within $0.01 of the official settlement point price in 99.991% of 11,127 intervals (largest difference $0.81).

| Split | Strategy | Days | Revenue (USD) | of which ECRS | Capture rate (95% interval) | Cycles/day | ECRS award, mean MW | Clipped: energy / ECRS / DA h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | Perfect foresight (DA + RT, energy + ECRS) | 51 | 1,071,466 | 52,008 | 100.0% | 1.63 | 45.9 | 0 / 0 / 0 |
| validation | Perfect foresight (RT only, energy + ECRS) | 51 | 505,359 | 24,229 | 47.2% (42.4% to 50.5%) | 1.63 | 45.9 | 0 / 0 / 0 |
| validation | Perfect foresight (DA + RT, energy only) | 51 | 1,040,185 | 0 | 97.1% (95.3% to 98.1%) | 1.67 | 0.0 | 0 / 0 / 0 |
| validation | Perfect foresight (RT only, energy only) | 51 | 488,767 | 0 | 45.6% (41.0% to 49.1%) | 1.67 | 0.0 | 0 / 0 / 0 |
| validation | DA schedule, energy + ECRS | 51 | 355,887 | 37,132 | 33.2% (25.5% to 42.2%) | 1.18 | 54.8 | 0 / 0 / 0 |
| validation | ECRS MPC on DA prices | 51 | 268,866 | 15,990 | 25.1% (19.1% to 30.7%) | 1.19 | 57.1 | 0 / 0 / 0 |
| validation | Energy MPC on DA prices | 51 | 260,601 | 0 | 24.3% (18.6% to 29.4%) | 1.39 | 0.0 | 0 / 0 / 0 |
| validation | ECRS MPC | 51 | -27,716 | 5,825 | -2.6% (-8.6% to 4.0%) | 1.14 | 6.2 | 0 / 0 / 0 |
| validation | Energy MPC | 51 | -29,026 | 0 | -2.7% (-8.8% to 4.2%) | 1.15 | 0.0 | 0 / 0 / 0 |
| test | Perfect foresight (DA + RT, energy + ECRS) | 57 | 1,753,128 | 147,665 | 100.0% | 1.83 | 43.9 | 0 / 0 / 0 |
| test | Perfect foresight (RT only, energy + ECRS) | 57 | 860,755 | 53,276 | 49.1% (45.8% to 52.0%) | 1.83 | 43.9 | 0 / 0 / 0 |
| test | Perfect foresight (DA + RT, energy only) | 57 | 1,673,366 | 0 | 95.5% (93.3% to 96.9%) | 1.88 | 0.0 | 0 / 0 / 0 |
| test | Perfect foresight (RT only, energy only) | 57 | 826,615 | 0 | 47.2% (43.7% to 50.2%) | 1.88 | 0.0 | 0 / 0 / 0 |
| test | DA schedule, energy + ECRS | 57 | 488,803 | 125,646 | 27.9% (21.5% to 36.4%) | 1.24 | 54.7 | 0 / 0 / 0 |
| test | ECRS MPC on DA prices | 57 | 468,183 | 68,335 | 26.7% (21.8% to 31.7%) | 1.28 | 58.7 | 0 / 0 / 0 |
| test | Energy MPC on DA prices | 57 | 460,110 | 0 | 26.2% (20.3% to 31.9%) | 1.54 | 0.0 | 0 / 0 / 0 |
| test | ECRS MPC | 57 | -19,245 | 10,111 | -1.1% (-4.1% to 2.5%) | 1.36 | 9.4 | 0 / 0 / 0 |
| test | Energy MPC | 57 | -29,562 | 0 | -1.7% (-4.7% to 1.8%) | 1.37 | 0.0 | 0 / 0 / 0 |

Revenue includes ECRS. Capture rate = revenue / perfect foresight (DA + RT, energy + ECRS) on the same days; the interval is a block bootstrap over days. ECRS award = real-time award averaged over all intervals.

Data provided by [Grid Status](https://www.gridstatus.io).
