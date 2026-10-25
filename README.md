# RTCB-Gym

*The first open battery-storage market environment for the post-RTC+B ERCOT market.*

ERCOT launched **Real-Time Co-optimization + Batteries (RTC+B)** on Operating
Day **2025-12-05**. RTCB-Gym replays real ERCOT real-time prices from that day
on and scores any battery bidding strategy (rule, optimizer, RL agent, LLM
agent) under the same rules.

> **Assumptions to keep in mind**
> - **Price-taker.** The battery never moves the price. This is historical replay.
> - **Post-RTC+B data only:** operating days 2025-12-05 to 2026-10-24 by default.
>   Earlier data comes from a different market design, and the code refuses it.
> - **One trading hub's price** (default `HB_WEST`) is the price the battery
>   gets. A real battery settles at its own resource node.
> - **Energy only, real-time only.** No day-ahead market, no ancillary services, no fees.

## Status

v0.1 is in progress. The simulator, baselines and tests are done. **The results
table is not filled in yet**: it waits for the first real data download (see
[Get the data](#get-the-data)).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"
pytest
```

## Get the data

Prices come from the hosted [GridStatus API](https://www.gridstatus.io)
(dataset `ercot_spp_real_time_15_min`, built from ERCOT report NP6-905-CD).
You need a free API key:

```bash
export GRIDSTATUS_API_KEY=...   # your key
python scripts/download_data.py
```

This downloads one calendar month at a time, caches the raw responses under
`data/raw/` (so later runs work offline), cleans them and writes
`data/processed/ercot_real_time_15_min_hb_west.parquet` and a gap report.

Cleaning rules:
- Times are stored in UTC, plus a `local_time` column in US/Central.
- An interval belongs to the operating day of its local start time. DST days
  have 92 or 100 intervals.
- Exact duplicate rows are dropped. Duplicates with different prices stop the
  run.
- **Gaps are reported, never filled.** Days with gaps are left out of episodes.

## Reproduce the results table

```bash
python scripts/run_baselines.py
```

It prints a markdown table and writes `results/baselines.md` and
`results/baselines.json`.

## Results

*Not generated yet: waiting for the first data download.*

| Split | Strategy | Days | Revenue (USD) | Capture rate | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|
| validation / test | Perfect foresight, Threshold rule, Forecast + MPC | — | — | — | — | — |

Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost.

## Usage

```python
import numpy as np
from rtcb_gym import BatteryParams, RTCBEnv
from rtcb_gym.data import read_prices

prices = read_prices("data/processed/ercot_real_time_15_min_hb_west.parquet")
env = RTCBEnv(prices, BatteryParams(), days=None)  # all usable days

obs, info = env.reset(seed=0)
terminated = False
while not terminated:
    action = np.array([0.0])  # in [-1, 1]; + discharge (sell), - charge (buy)
    obs, reward, terminated, truncated, info = env.step(action)
```

## Environment

| | |
|---|---|
| Episode | one operating day (96 intervals; 92 or 100 on DST days); `episode_days` makes it longer |
| Action | `Box(-1, 1, (1,))`, times `power_mw`. **Positive = discharge (sell), negative = charge (buy).** |
| Observation | SoC; time of day (sin, cos); day of week (one-hot); last 96 settled RT prices. Layout in `env.obs_layout` |
| Reward | net revenue of the interval in USD |
| `info` | requested vs. actual power, clipping and its reason, revenue breakdown, SoC, steps remaining |

**No look-ahead.** Before dispatching interval *t*, the agent sees prices of
intervals *t−96 … t−1* only. Tests fail if a future price leaks into the
observation.

Each episode starts with the battery empty (`soc_init: 0`), and there is no
end-of-day SoC target, so no strategy gets free stored energy.

## Battery and settlement

- 15-minute intervals, `dt = 0.25 h`.
- Round-trip efficiency is split evenly: √rte on charge and √rte on discharge.
- Requests beyond the power or SoC limits are clipped. The clipped power is
  settled, and the clip is recorded.
- `revenue = actual_power_mw × dt × rt_price − |actual_power_mw| × dt × degradation_cost`.
  Charging at a negative price earns money.

## Baselines

1. **Perfect foresight**: solves each day with the true prices (linear
   program; a MILP when needed to forbid charging and discharging at the same
   time). This is the upper bound.
2. **Threshold rule**: charge when the last settled price is at or below a
   low quantile, discharge at or above a high quantile. Quantiles come from the
   training split only.
3. **Forecast + MPC**: at each interval, forecast the rest of the day
   (persistence: same interval 24 hours earlier), solve the LP, apply the first
   action. Better forecasters plug in here.

## Evaluation

- Time-based split, never random (`configs/default.yaml`): train 2025-12-05 to
  2026-06-30, validation 2026-07-01 to 2026-08-24, **test 2026-08-25 to
  2026-10-24** (most recent ~2 months).
- Metrics: total revenue, **capture rate** (revenue ÷ perfect-foresight
  revenue on the same days), equivalent full cycles per day (stored energy
  discharged ÷ capacity), number of clipped actions.

Switching hub or dates means editing `configs/default.yaml` only.

## Layout

```
configs/default.yaml        hub, dates, battery, splits
rtcb_gym/data/fetch.py      download (gridstatusio) with raw cache
rtcb_gym/data/clean.py      timezone, DST, duplicates, gap report
rtcb_gym/data/store.py      parquet schema
rtcb_gym/battery.py         physical model
rtcb_gym/settlement.py      money
rtcb_gym/env.py             Gymnasium environment
rtcb_gym/evaluation.py      runs and metrics
rtcb_gym/baselines/         perfect foresight, threshold rule, forecast + MPC
scripts/                    download_data.py, run_baselines.py
tests/                      unit tests (synthetic prices only)
```

## Roadmap

| Version | Content |
|---|---|
| **v0.1** | real-time energy, three baselines, results table |
| v0.2 | day-ahead market and two-settlement |
| v0.5 | real-time ancillary services co-optimized with energy |
| v1.0 | stress-scenario generator, validated against held-out data |
| v1.x | better forecasters, risk guardrails |
| v2.0 | LLM agent with tools, scored in the same environment |

## License

MIT
