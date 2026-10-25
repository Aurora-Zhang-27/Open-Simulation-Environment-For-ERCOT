# RTCB-Gym

*An open battery-storage environment built for the post-RTC+B ERCOT market.*

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

v0.1: the first real-data run is complete.

- **Done:** simulator, settlement, environment, three baselines and the data
  pipeline. All 97 unit tests pass. They use synthetic prices only.
- **Real data downloaded once** (2026-10-25, gridstatusio 0.16.1): HB_WEST,
  operating days 2025-12-05 to 2026-10-24. The column mapping matches the
  live API response. The [cleaning report](#cleaning-report-first-download)
  found 4 missing intervals, no duplicates and no NaN prices. The
  [data checksum](#data-checksum) is published and set in
  `configs/default.yaml`. The [results table](#results) is filled in.
- **Not checked yet:**
  - The results come from one download on one machine. Nobody has
    reproduced the hash with an independent download yet.
  - The most recent days were downloaded one day after they happened. ERCOT
    can still correct real-time prices, so a later download may hash
    differently.
  - GridStatus's terms for redistribution. The data is still not committed.
  - Only HB_WEST has been run.

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

**The data is not committed to this repository.** GridStatus is a hosted
service that needs a key, and its terms have not been checked for
redistribution. Download with your own key instead, then check the data
against the checksum below.

Cleaning rules:
- Times are stored in UTC, plus a `local_time` column in US/Central.
- An interval belongs to the operating day of its local start time. DST days
  have 92 or 100 intervals.
- Exact duplicate rows are dropped. Duplicates with different prices stop the
  run.
- **Gaps are reported, never filled.** Days with gaps are left out of
  episodes, and so is the day after each one (its price history is not
  contiguous).

### Data checksum

`download_data.py` prints a SHA-256 of the cleaned data. The hash is computed
over a canonical text form of the rows (UTC times, location, market, price),
not over the parquet bytes, so it does not depend on library versions. If
your hash matches the one below, you have exactly the data behind the
results table.

| Hub | Operating days | SHA-256 of cleaned data |
|---|---|---|
| HB_WEST | 2025-12-05 .. 2026-10-24 | `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0` |

This value is already in `data.expected_sha256` in `configs/default.yaml`,
so `download_data.py` fails on a mismatch. A mismatch can also mean the
source revised some prices after publication.

### Cleaning report (first download)

Downloaded 2026-10-25 with gridstatusio 0.16.1.

| | |
|---|---:|
| Raw rows | 31,096 |
| Intervals expected | 31,100 |
| Intervals present | 31,096 |
| Missing intervals (not filled) | 4 |
| Exact duplicates | 0 |
| Conflicting duplicates | 0 |
| NaN prices | 0 |
| Other locations / outside date range | 0 / 0 |

- Each missing interval is one 15-minute interval on a different day:
  2026-01-08 15:30, 2026-06-04 15:30, 2026-07-09 15:30 and 2026-08-27 10:30
  (US/Central start time). Re-querying a narrow window around each gap also
  returned no row, so the gaps are in the source data, not a paging problem.
- The DST day 2026-03-08 has 92 intervals, as expected.
- Prices range from −$31.62 to $1,522.50/MWh, median $24.50/MWh.

## Reproduce the results table

```bash
python scripts/run_baselines.py
```

It prints the data hash and a markdown table, and writes
`results/baselines.md` and `results/baselines.json`.

## Results

HB_WEST, data SHA-256 `a588daed…80009d0` (full value in the
[checksum table](#data-checksum)). Also in `results/baselines.md`, with full
numbers in `results/baselines.json`.

| Split | Strategy | Days | Revenue (USD) | Capture rate | Cycles/day | Clipped actions |
|---|---|---:|---:|---:|---:|---:|
| validation | Perfect foresight | 53 | 538,640 | 100.0% | 1.39 | 0 |
| validation | Threshold rule | 53 | 122,064 | 22.7% | 0.46 | 0 |
| validation | Forecast + MPC | 53 | 267,207 | 49.6% | 1.38 | 0 |
| test | Perfect foresight | 59 | 929,058 | 100.0% | 1.75 | 0 |
| test | Threshold rule | 59 | 133,541 | 14.4% | 0.38 | 0 |
| test | Forecast + MPC | 59 | 435,338 | 46.9% | 1.73 | 0 |

Battery: 100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost.
Every day starts empty, with no end target.

- **Days left out:** each day with a missing interval, and the day after it
  (its 96-interval price history is not contiguous). Validation leaves out
  2026-07-09 and 2026-07-10, test leaves out 2026-08-27 and 2026-08-28.
- **Threshold rule:** fitted on the training split. It charges at or below
  $15.65/MWh and discharges at or above $30.91/MWh (quantiles 0.4 / 0.7).
- **Perfect foresight:** replaying its schedule in the environment earned
  the optimizer's objective (within 1e-6 relative) on both splits.

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

### SoC at the start and end of an episode

Two settings in the `battery` section of the config:

| Setting | Meaning | Default |
|---|---|---|
| `soc_init` | SoC at the start of every episode | 0 |
| `soc_terminal_min` | lowest SoC allowed at the end of the episode | 0 |

The default starts every day empty with no end target, so no strategy gets
free stored energy. The cost is that cross-day arbitrage (charge tonight,
sell tomorrow morning) is not possible. For that, use the setting common in
the literature: `soc_init: 0.5`, `soc_terminal_min: 0.5`.

The end target is enforced step by step. Before each interval the env checks
that the target can still be reached by charging at full power for the
remaining intervals; if not, it limits discharge or forces charging, and
counts the step as clipped (`clip_reason: "soc_terminal"`). So every
trajectory the env allows ends at or above the target, which is exactly the
constraint set the perfect-foresight program solves. Perfect foresight stays
an exact upper bound with either setting.

## Battery and settlement

- 15-minute intervals, `dt = 0.25 h`.
- Round-trip efficiency is split evenly: √rte on charge and √rte on discharge.
- Requests beyond the power or SoC limits are clipped. The clipped power is
  settled, and the clip is recorded.
- `revenue = actual_power_mw × dt × rt_price − |actual_power_mw| × dt × degradation_cost`.
  Charging at a negative price earns money.

## Baselines

1. **Perfect foresight**: solves each day with the true prices. This is the
   upper bound and the denominator of the capture rate (see below for why it
   sometimes needs integer programming).
2. **Threshold rule**: charge when the last settled price is at or below a
   low quantile, discharge at or above a high quantile. Quantiles come from the
   training split only.
3. **Forecast + MPC**: at each interval, forecast the rest of the day
   (persistence: same interval 24 hours earlier), solve the LP, apply the first
   action. Better forecasters plug in here.

### Why the upper bound switches to integer programming at negative prices

The usual linear program has separate charge and discharge variables, and
nothing stops both from being positive in the same interval. At very
negative prices that pays: charging and discharging at once burns energy
through the efficiency losses, so the battery takes in more paid-for energy
than it could store. A real battery has one net power, and so does this
environment's action, so that schedule is physically impossible.

Left in, it would make the "upper bound" too high and push down every
strategy's capture rate. With 85% round-trip efficiency and $2/MWh
degradation it pays below about −$25/MWh, and ERCOT real-time prices can
fall that low.
So the solver checks the LP solution, and if any interval charges and
discharges at once, it re-solves as a mixed-integer program with one binary
per interval (charge or discharge, not both). Both run on HiGHS and are
deterministic. A test replays the schedule in the environment and checks
that it earns exactly the optimizer's objective.

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

## Related work

General storage and power-system RL environments already exist. RTCB-Gym
does not replace them. It narrows the scope to one market after one rule
change: ERCOT after RTC+B, with settlement-level accounting, a tested
no-look-ahead guarantee and an exact upper bound.

- [energy-py](https://github.com/ADGEfficiency/energy-py): RL framework with
  a Gymnasium battery-arbitrage environment on historical electricity prices,
  integrated with Stable Baselines 3.
- [Grid2Op](https://github.com/rte-france/Grid2Op) (RTE France): grid
  operation and topology control, used for the L2RPN competitions.
- [CityLearn](https://github.com/intelligent-environments-lab/CityLearn):
  multi-agent building demand response and storage control.
- [PowerGridworld](https://github.com/NREL/PowerGridworld) (NREL): framework
  for building multi-agent power-system Gym environments.

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
