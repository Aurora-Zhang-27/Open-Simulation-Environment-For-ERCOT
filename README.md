# RTCB-Gym

*An open battery-storage environment built for the post-RTC+B ERCOT market.*

ERCOT launched **Real-Time Co-optimization + Batteries (RTC+B)** on Operating
Day **2025-12-05**. RTCB-Gym replays real ERCOT prices from that day on and
scores any battery bidding strategy (rule, optimizer, RL agent, LLM agent)
under the same rules: in the real-time market alone, or in the day-ahead and
real-time markets together (two-settlement).

> **Assumptions to keep in mind**
> - **Price-taker.** The battery never moves the price. This is historical replay.
> - **Post-RTC+B data only:** operating days 2025-12-05 to 2026-10-24 by default.
>   Earlier data comes from a different market design, and the code refuses it.
> - **One trading hub's price** (`HB_WEST` by default, `HB_HOUSTON` also
>   configured) is the price the battery gets. A real battery settles at its
>   own resource node.
> - **Energy only.** Real time, optionally with the day-ahead market. No
>   ancillary services, no fees.

## Start without an API key or GPU

The current development target is a reusable CPU tool: install it, run a
strategy, and inspect reproducible outputs. Training an agent is optional.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m rtcb_gym.demo --out results/demo
```

This **synthetic** example runs idle, a rule using delayed historical prices,
and a perfect-foresight optimizer in the experimental energy + ECRS model.
It writes `trace.csv`, `summary.json`, `manifest.json`, input prices and
`SHA256SUMS`. The oracle is replayed through the same execution model and
checked against its objective. This example makes no market-data requests.
It uses cash-flow proxies; it does not validate official AS settlement.

See [the quickstart](docs/quickstart.md) for outputs, real-data requirements
and current limitations, and [the product blueprint](BLUEPRINT.md) for release
criteria. The new development commands must be included in a release before
users can obtain them from the public main branch.

## Status

**v0.2.0**: day-ahead market and two-settlement, a rolling threshold
baseline, and results for two hubs.

- **Done:** simulator, settlement (real-time only and two-settlement),
  Gymnasium environment, nine baselines and the data pipeline. All 151 unit
  tests pass. They use synthetic prices only.
- **Data:** real-time and day-ahead prices for HB_WEST and HB_HOUSTON,
  downloaded 2026-10-25/26 with gridstatusio 0.16.1. The column mapping
  matches the live API responses. All four [checksums](#data-checksum) are
  published and set in the configs. HB_WEST real-time prices were downloaded
  a second time from scratch a day later and gave the same hash.
- **Data terms:** GridStatus's terms of use claim ownership of the data, so
  it is not committed here, and this README credits Grid Status as the terms
  require.
- **Not checked yet:**
  - Nobody has reproduced the hashes on another machine.
  - ERCOT can still correct recent prices, so a download weeks from now may
    hash differently.
  - The [4 missing real-time intervals](#missing-intervals) have not been
    checked against ERCOT's own archive (it is not reachable from where this
    was built).

## Results

Validation 2026-07-01 .. 2026-08-24, test 2026-08-25 .. 2026-10-24. Battery:
100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost; every
day starts empty, with no end target. Each cell is revenue (USD) and capture
rate. Full tables with settlement breakdown, cycles and clipped actions:
[`results/hb_west/`](results/hb_west) and [`results/hb_houston/`](results/hb_houston).
Data provided by [Grid Status](https://www.gridstatus.io).

**Real-time energy only.** Capture rate = revenue / perfect foresight on the same days.

| Strategy | HB_WEST validation | HB_WEST test | HB_HOUSTON validation | HB_HOUSTON test |
|---|---:|---:|---:|---:|
| Perfect foresight | 539k (100.0%) | 929k (100.0%) | 438k (100.0%) | 754k (100.0%) |
| Threshold rule | 122k (22.7%) | 134k (14.4%) | 27k (6.2%) | 51k (6.7%) |
| Rolling threshold (30 d) | 118k (21.9%) | 208k (22.4%) | 42k (9.5%) | 76k (10.1%) |
| Forecast + MPC | 267k (49.6%) | 435k (46.9%) | 191k (43.6%) | 309k (41.0%) |

**Day-ahead + real time (two-settlement).** Capture rate = revenue / perfect foresight (DA + RT).

| Strategy | HB_WEST validation | HB_WEST test | HB_HOUSTON validation | HB_HOUSTON test |
|---|---:|---:|---:|---:|
| Perfect foresight (DA + RT) | 1,121k (100.0%) | 1,838k (100.0%) | 942k (100.0%) | 1,562k (100.0%) |
| Perfect foresight (RT only) | 539k (48.1%) | 929k (50.5%) | 438k (46.5%) | 754k (48.3%) |
| Perfect foresight (DA only, delivered) | 432k (38.5%) | 607k (33.0%) | 345k (36.6%) | 455k (29.1%) |
| DA schedule, delivered | 404k (36.0%) | 534k (29.1%) | 313k (33.2%) | 401k (25.7%) |
| DA schedule + MPC on DA prices | 391k (34.8%) | 572k (31.1%) | 308k (32.7%) | 414k (26.5%) |
| MPC on DA prices | 339k (30.2%) | 555k (30.2%) | 250k (26.5%) | 395k (25.3%) |
| Forecast + MPC | 267k (23.8%) | 435k (23.7%) | 191k (20.3%) | 309k (19.8%) |

Days scored: 53 (validation) and 59 (test) at both hubs. Clipped actions are
0 for every strategy, by design (all of them stay inside the battery limits;
[tests](tests/test_baselines.py) check that the counter does count). Every
perfect-foresight schedule, replayed in the environment, earns its
optimizer's objective (within 1e-6 relative).

### What the numbers show

- **The day-ahead market pays even with a naive forecast.** Bidding
  yesterday's day-ahead shape and delivering it earns 23–64% more than
  Forecast + MPC, the best strategy here that uses real-time prices only, at
  both hubs and on both splits.
- **Persistence is already close to perfect for the day-ahead schedule.**
  It reaches 88–94% of the day-ahead-only bound. Day-ahead prices are smooth
  and their daily shape repeats.
- **Day-ahead prices are a better real-time forecast than yesterday's real
  time.** The same rolling LP earns 27–31% more with the day's day-ahead
  prices as its forecast (MPC on DA prices) than with persistence
  (Forecast + MPC).
- **Re-dispatching around the bid is a small, mixed gain.** Bidding the
  same schedule but dispatching with MPC on DA prices in real time changes
  revenue by −3% to +7% against delivering the bid. The day-ahead settlement
  is identical either way; the only difference is which dispatch earns more
  at real-time prices, the day-old schedule or the MPC, and neither wins
  reliably.
- **The DA + RT bound is 2.0–2.2 times the RT-only bound.** It includes
  perfect knowledge of the day-ahead/real-time spread on a battery-sized
  financial position. Capture rates in the two-settlement table are measured
  against it, so compare strategies by revenue too, or against the two other
  bounds.
- **Static thresholds go stale.** At HB_WEST the threshold rule (fitted on
  December–June prices) drops from 22.7% to 14.4%; test prices sit higher
  (median $32.14/MWh against $21.40 in training). Refitted every day on the
  last 30 days, the same rule holds 21.9% and 22.4%. As a diagnostic only
  (it looks ahead), refitting on each split's own prices gives about 35%.
- **The threshold rule does not transfer to HB_HOUSTON** (6–10%). Houston
  has far fewer cheap intervals: the 10th percentile of real-time prices is
  $13–21/MWh, against −$2 to $14 at HB_WEST. The static charge threshold
  ($17.00) sits below the 10th percentile of validation and test prices, so
  the rule rarely charges (0.24–0.29 cycles/day). Even the look-ahead refit
  reaches only 25% and 17%. Forecast + MPC still gets 41–44%.
- **Test pays more than validation** (perfect foresight about $157 against
  $102 per MW per day at HB_WEST). The five late-August test days average
  $33.7k/day; September and October still average $14.1k/day, against
  $10.2k/day on validation, where July is the quietest month.

## Install

An experimental five-minute **energy + ECRS reservation gate** is available
under `rtcb_gym.experimental.ecrs`; see [its model and validation status](docs/ecrs_gate.md).
It is a conservative discharge-side-only research model with no AS deployment
or market clearing. Its LMP/MCPC cash-flow proxies are separate from the v0.2
energy-settlement results above.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,dev]"
pytest
```

## Get the data

Prices come from the hosted [GridStatus API](https://www.gridstatus.io):

| Market | Dataset | ERCOT report |
|---|---|---|
| Real time, 15 minutes | `ercot_spp_real_time_15_min` | NP6-905-CD |
| Day-ahead, hourly | `ercot_spp_day_ahead_hourly` | NP4-190-CD |

You need a free API key:

```bash
export GRIDSTATUS_API_KEY=...   # your key
python scripts/download_data.py                               # HB_WEST
python scripts/download_data.py --config configs/hb_houston.yaml
```

This downloads one calendar month at a time, caches the raw responses under
`data/raw/` (so later runs work offline), cleans them and writes one parquet
file and one gap report per market under `data/processed/`. Without a
`day_ahead` section in the config, only real-time prices are downloaded.

**Data provided by [Grid Status](https://www.gridstatus.io).** Their terms
of use require this credit wherever their data is used, so keep it if you
publish results built on this data.

**The data is not committed to this repository.** GridStatus's terms of use
claim ownership of the data and treat it as their trade secret, so it is not
redistributed here. Download with your own key instead, then check the data
against the checksums below.

Cleaning rules:
- Times are stored in UTC, plus a `local_time` column in US/Central.
- An interval belongs to the operating day of its local start time. DST days
  have 92 or 100 real-time intervals and 23 or 25 day-ahead hours.
- Exact duplicate rows are dropped. Duplicates with different prices stop the
  run.
- **Gaps are reported, never filled.** Days with gaps are left out of
  episodes, and so is the day after each one (its price history is not
  contiguous).

### Data checksum

`download_data.py` prints a SHA-256 of each cleaned data set. The hash is
computed over a canonical text form of the rows (UTC times, location,
market, price), not over the parquet bytes, so it does not depend on library
versions. If your hashes match, you have exactly the data behind the results.

| Hub | Market | Operating days | SHA-256 of cleaned data |
|---|---|---|---|
| HB_WEST | real time | 2025-12-05 .. 2026-10-24 | `a588daed478fc7dbba60aa144e264fb4ce49c92845188214dea43596a80009d0` |
| HB_WEST | day-ahead | 2025-12-05 .. 2026-10-24 | `ee0fd9974f117b07e957b114bd973f2a7890356e02040221b690fae5f344cfe7` |
| HB_HOUSTON | real time | 2025-12-05 .. 2026-10-24 | `ea27b7861e6210514264b51dfed49ab1cd7168ff4b5b693dcbd130dee9750c00` |
| HB_HOUSTON | day-ahead | 2025-12-05 .. 2026-10-24 | `6885d4eb2c38d404fd604ce9a4ef3417834071602e382ee0fa353a7e7a7b7933` |

These values are set as `expected_sha256` in the configs, so
`download_data.py` fails on a mismatch. A mismatch can also mean the source
revised some prices after publication.

### Cleaning report

| | Real time (each hub) | Day-ahead (each hub) |
|---|---:|---:|
| Intervals expected | 31,100 | 7,775 |
| Intervals present | 31,096 | 7,775 |
| Missing (not filled) | 4 | 0 |
| Exact / conflicting duplicates | 0 / 0 | 0 / 0 |
| NaN prices | 0 | 0 |

| Hub | Market | Min | Median | Max (USD/MWh) |
|---|---|---:|---:|---:|
| HB_WEST | real time | −31.62 | 24.5 | 1,522.50 |
| HB_WEST | day-ahead | −19.40 | 24.8 | 2,000.17 |
| HB_HOUSTON | real time | −251.00 | 26.8 | 1,170.38 |
| HB_HOUSTON | day-ahead | −0.13 | 28.2 | 1,704.92 |

The DST day 2026-03-08 has 92 real-time intervals and 23 day-ahead hours, as
expected.

### Missing intervals

The same four 15-minute real-time intervals are missing at both hubs:
2026-01-08 15:30, 2026-06-04 15:30, 2026-07-09 15:30 and 2026-08-27 10:30
(US/Central start time). Checked against other GridStatus data sets:

- At each of them **every settlement point is missing** (0 rows, against
  about 1,100 in the intervals before and after). The whole 15-minute report
  is absent, not one price.
- **The market ran:** `ercot_lmp_by_settlement_point` has 2 to 4 SCED runs
  with LMPs at HB_WEST inside each of these intervals.
- `ercot_spp_dart_15_min` is missing the same intervals, and
  `ercot_spp_real_time_price_corrections` has no corrections for them.
- Re-querying a narrow window around each gap returns the same result, so it
  is not a paging problem.

So the prices existed, but the report for these intervals is missing from
GridStatus's copy (or ERCOT did not post it). Three of the four start at
15:30 local time. They are reported, not filled or reconstructed.

## Reproduce the results

```bash
python scripts/run_baselines.py                               # HB_WEST
python scripts/run_baselines.py --config configs/hb_houston.yaml
python scripts/summarize_results.py                           # the tables above
```

Each run prints the data hashes and the tables, and writes
`results/<hub>/real_time.{md,json}` and `results/<hub>/two_settlement.{md,json}`.
It takes about five minutes per hub.

## Usage

Real time only (the v0.1 setting):

```python
import numpy as np
from rtcb_gym import BatteryParams, RTCBEnv
from rtcb_gym.data import read_prices

rt = read_prices("data/processed/ercot_real_time_15_min_hb_west.parquet")
env = RTCBEnv(rt, BatteryParams())

obs, info = env.reset(seed=0)
terminated = False
while not terminated:
    action = np.array([0.0])  # in [-1, 1]; + discharge (sell), - charge (buy)
    obs, reward, terminated, truncated, info = env.step(action)
```

Day-ahead + real time:

```python
da = read_prices("data/processed/ercot_day_ahead_hourly_hb_west.parquet")
env = RTCBEnv(rt, BatteryParams(), da_prices=da)

day = env.days[0]
ctx = env.day_ahead_context(day)   # what is known at 10:00 the day before
bid = np.zeros(ctx.n_hours)        # hourly MW: + sell, - buy
obs, info = env.reset(options={"day": day, "da_schedule": bid})
info["da_award_mw"]                # the bid after clipping to what the battery can deliver
# ... then step through the day in real time as above
```

Scoring a strategy the way the results tables do:

```python
from rtcb_gym.baselines import DayAheadSchedule
from rtcb_gym.evaluation import run_policy

result = run_policy(env, DayAheadSchedule(env.battery_params, env.obs_layout))
result.revenue_usd, result.da_revenue_usd, result.rt_revenue_usd
```

A strategy is a [`Policy`](rtcb_gym/baselines/base.py): `day_ahead(context)`
returns the hourly bid (or None for no day-ahead position), `reset` starts
the day, `act` returns the real-time action.

## Environment

| | |
|---|---|
| Episode | one operating day (96 intervals; 92 or 100 on DST days); `episode_days` makes real-time-only episodes longer |
| Action | `Box(-1, 1, (1,))`, times `power_mw`. **Positive = discharge (sell), negative = charge (buy).** |
| Observation | SoC; time of day (sin, cos); day of week (one-hot); last 96 settled RT prices. Two-settlement adds the day's DA prices, awards and a mask for the next 96 intervals. Layout in `env.obs_layout` |
| Reward | net revenue of the interval in USD |
| `info` | requested vs. actual power, clipping and its reason, revenue breakdown (DA, RT, degradation), SoC, steps remaining |

**No look-ahead.** Before dispatching interval *t*, the agent sees real-time
prices of intervals *t−96 … t−1* only. Tests fail if a future price leaks
into the observation or into the day-ahead context.

### Two-settlement episodes

Pass `da_prices` to get them. For operating day D (Central Prevailing Time):

| When | What happens |
|---|---|
| D−1, 10:00 | Day-ahead offers are due: the agent bids an hourly schedule for D (24 hours; 23 or 25 on DST days). It sees real-time prices of intervals that ended by 10:00 and day-ahead prices of the days before D (`env.day_ahead_context(D)`), never D's own. |
| D−1, afternoon | Results are published: the bid, clipped to what the battery could deliver from `soc_init`, is awarded in full at the day-ahead price. D's day-ahead prices become visible. |
| D, every 15 minutes | Real-time dispatch as before. The interval settles the award at the day-ahead price and the deviation from it at the real-time price. Degradation is charged on delivered power only. |

The bid comes from `reset(options={"da_schedule": ...})`, else from the
`day_ahead_policy` passed to the env, else it is zero. A zero position gives
exactly the real-time-only results, so the two settings compare directly.

**Why the bound stays exact.** Settling the deviation at the real-time
price splits each day's revenue into two independent parts:

```
sum over hours h:      award_h × (DA price_h − mean RT price in h) × 1 h
sum over intervals t:  power_t × RT price_t × 0.25 h − degradation
```

The best real-time dispatch therefore does not depend on the award, and the
award is a bet on the day-ahead/real-time spread. Perfect foresight solves
both parts exactly, so it is still a true upper bound. Tests check the split
on a full day with random bids and actions, and that no strategy beats the
bound.

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
constraint set the perfect-foresight program solves. Day-ahead bids are
clipped against the same target. Perfect foresight stays an exact upper
bound with either setting.

## Battery and settlement

- 15-minute real-time intervals (`dt = 0.25 h`), hourly day-ahead intervals.
- Round-trip efficiency is split evenly: √rte on charge and √rte on discharge.
- Requests beyond the power or SoC limits are clipped. The clipped power is
  settled, and the clip is recorded.
- Real time only: `revenue = power × dt × RT price − |power| × dt × degradation_cost`.
- Two-settlement: `revenue = award × dt × DA price + (power − award) × dt × RT price − |power| × dt × degradation_cost`.
- Charging at a negative price earns money.

## Baselines

Real time only:

1. **Perfect foresight**: solves each day with the true prices. The upper
   bound and the denominator of the capture rate (see below for why it
   sometimes needs integer programming).
2. **Threshold rule**: charge when the last settled price is at or below a
   low quantile, discharge at or above a high quantile. Quantiles come from
   the training split only.
3. **Rolling threshold (30 d)**: the same rule, with thresholds refitted
   every day on the previous 30 days of settled prices. The quantile pair is
   chosen on the training split.
4. **Forecast + MPC**: at each interval, forecast the rest of the day
   (persistence: same interval 24 hours earlier), solve the LP, apply the
   first action. Better forecasters plug in here.

Two-settlement adds:

5. **Perfect foresight (DA + RT)**: the exact bound, solving the spread part
   and the real-time part with the true prices.
6. **Perfect foresight (DA only, delivered)**: the schedule on the true
   day-ahead prices, delivered exactly. The bound for day-ahead-only
   strategies.
7. **DA schedule, delivered**: forecast tomorrow's day-ahead prices with
   today's (by clock hour), solve the schedule, bid it, deliver it.
8. **MPC on DA prices**: no day-ahead position; the rolling LP uses the
   day's day-ahead prices as its real-time forecast.
9. **DA schedule + MPC on DA prices**: 7 for the bid, 8 in real time.

Perfect foresight (RT only) and Forecast + MPC also appear in the
two-settlement table, with no day-ahead position.

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
fall that low. So the solver checks the LP solution, and if any interval
charges and discharges at once, it re-solves as a mixed-integer program with
one binary per interval (charge or discharge, not both). Both run on HiGHS
and are deterministic. Tests replay the schedules in the environment and
check that they earn exactly the optimizer's objective.

## Evaluation

- Time-based split, never random (`configs/default.yaml`): train 2025-12-05 to
  2026-06-30, validation 2026-07-01 to 2026-08-24, **test 2026-08-25 to
  2026-10-24** (most recent ~2 months).
- Metrics: total revenue (and its day-ahead and real-time settlement),
  **capture rate** (revenue ÷ perfect-foresight revenue on the same days),
  equivalent full cycles per day (stored energy discharged ÷ capacity),
  number of clipped actions (and clipped day-ahead hours).

Switching hub or dates means editing a config file only. A config can start
from another one and change a few keys (`extends: default.yaml`, see
[`configs/hb_houston.yaml`](configs/hb_houston.yaml)).

## Layout

```
configs/default.yaml          HB_WEST: dates, battery, splits, baselines, day-ahead
configs/hb_houston.yaml       the same at HB_HOUSTON (extends default.yaml)
rtcb_gym/data/fetch.py        download (gridstatusio) with raw cache
rtcb_gym/data/clean.py        timezone, DST, duplicates, gap report
rtcb_gym/data/store.py        parquet schema, checksum
rtcb_gym/data/sced.py         SCED cache and time-weighted input-price reconstruction
rtcb_gym/demo.py              offline CPU example using synthetic prices
rtcb_gym/experimental/ecrs.py experimental energy + ECRS reservation model
rtcb_gym/battery.py           physical model
rtcb_gym/settlement.py        money: real-time only and two-settlement
rtcb_gym/day_ahead.py         day-ahead timeline, context, bid clipping, persistence forecast
rtcb_gym/env.py               Gymnasium environment
rtcb_gym/evaluation.py        runs and metrics
rtcb_gym/baselines/           perfect foresight, threshold rules, MPC, day-ahead strategies
scripts/                      download_data.py, run_baselines.py, summarize_results.py
results/<hub>/                real_time.{md,json}, two_settlement.{md,json}
tests/                        unit tests (synthetic prices only)
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

The next release is a CPU tool that another user can clone, install and run
with versioned real data. [The blueprint](BLUEPRINT.md) defines its acceptance
criteria; positive strategy returns are not a release requirement.

| Priority | Deliverable |
|---|---|
| 1 | Offline installation example and reproducible output artifacts |
| 2 | Audited real-data import, provenance, statuses and an executable tutorial |
| 3 | Complete the scoped energy + ECRS constraints, timing and settlement ledger |
| 4 | Custom-policy tutorial, independent reproduction and release verification |
| Later | Other AS products, each with its own rule and accounting validation |

Training agents, large forecasting models and scenario generation are deferred.

## License

MIT
