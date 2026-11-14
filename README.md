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
> - **Energy, optionally with ECRS.** Real time, optionally with the day-ahead
>   market. ECRS is a reduced model (reserve from the discharge side, full
>   awards, no deployment unless you supply one). No other ancillary
>   services, no fees.

## Start without an API key or GPU

Install it, run a strategy, and inspect reproducible outputs. Training an
agent is optional.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m rtcb_gym.demo --out results/demo
```

This **synthetic** example runs idle, a rule using delayed historical prices,
and a perfect-foresight optimizer in the five-minute energy + ECRS gate model.
It writes `trace.csv`, `summary.json`, `manifest.json`, input prices and
`SHA256SUMS`. The oracle is replayed through the same execution model and
checked against its objective. This example makes no market-data requests.
It uses cash-flow proxies; it does not validate official AS settlement.

See [the quickstart](docs/quickstart.md) for outputs, real-data requirements
and current limitations.

## Score your own agent

```bash
python -m rtcb_gym.evaluate --list                                   # the five tasks
python -m rtcb_gym.evaluate --task da-rt-energy \
    --agent examples/agents/threshold.py --source synthetic          # no data needed
```

An agent is a small class built from a task spec (the rules, the layouts
and the training prices) and scored against perfect foresight into a JSON
scorecard: revenue, capture rate with its interval, invalid and clipped
actions, time per step. Rules, optimisers, learning agents and language
models go through the same door; `rtcb_gym.text.TextAgent` gives a
language model the task as text and parses its JSON replies, without
tying the library to any provider. [Guide](docs/benchmark.md).

Where the reference agents stand (HB_WEST, test split, capture rate with
its 95% interval; [all scorecards](results/benchmark/hb_west/LEADERBOARD.md)):

| Task | Best reference agent | Capture rate |
|---|---|---:|
| `rt-energy` | persistence MPC | 46.9% (33.1 to 58.9) |
| `rt-energy-offer` | persistence MPC, as price-taking offers | 46.9% (33.1 to 58.9) |
| `da-rt-energy` | day-ahead schedule, delivered | 29.1% (22.9 to 37.0) |
| `da-rt-energy-7day` | day-ahead schedule, delivered | 28.6% (22.3 to 36.2) |
| `da-rt-energy-ecrs-5min` | day-ahead schedule, delivered | 25.0% (19.3 to 32.6) |

These agents are simple on purpose: on every task with a day-ahead market
more than two thirds of the bound is unclaimed. The scripted "text agent"
in the examples is a few lines of Python standing in for a language
model; no language model has been scored yet.

## Status

**v0.5.0**: a benchmark layer on top of the environment: five named
tasks, one way to plug in any agent, a text interface for language models,
synthetic prices for development without data, and scorecards
([guide](docs/benchmark.md)).

**v0.4.0**: one environment for everything. `RTCBEnv` now also does
five-minute dispatch with 15-minute settlement, ECRS (real time and
day-ahead, with an optional deployment input), offer-curve actions,
day-ahead bids with limit prices, and two-settlement episodes of several
days. All of it is optional; the default is the v0.2 environment, and its
results are unchanged. v0.3.0 added the offline CPU example and the
five-minute SCED price export; v0.2.0 the day-ahead market.

- **Done:** simulator, settlement, Gymnasium environment
  (`RTCBGym-v0`), sixteen baselines, the data pipeline, and results on
  real data for every setting. All unit tests pass. They use synthetic
  prices only.
- **Data:** 15-minute real-time and hourly day-ahead prices for HB_WEST
  and HB_HOUSTON (2025-12-05 to 2026-10-24); five-minute SCED prices and
  ECRS prices for HB_WEST (July to October 2026, plus two winter weeks).
  All four [checksums](#data-checksum) of the 15-minute and day-ahead data
  were reproduced on 2026-11-11 by a fresh download on another machine,
  16 days after the first.
- **Data terms:** GridStatus's terms of use claim ownership of the data, so
  it is not committed here, and this README credits Grid Status as the terms
  require.
- **Not checked yet:**
  - The ECRS settlement follows RTC+B's two-settlement logic but has not
    been compared term by term with ERCOT's Section 6.7 formulas, and the
    reserve rule is a conservative subset of SCED's
    ([details](docs/ecrs_gate.md)).
  - No result uses ECRS deployment: there is no deployment data here.
  - ERCOT can still correct recent prices, so a later download may hash
    differently.
  - The [4 missing real-time intervals](#missing-intervals) have not been
    checked against ERCOT's own archive.

## Results

Validation 2026-07-01 .. 2026-08-24, test 2026-08-25 .. 2026-10-24. Battery:
100 MW / 200 MWh, 85% round-trip efficiency, $2/MWh degradation cost; every
day starts empty, with no end target. Each cell is revenue (USD) and capture
rate. Full tables with settlement breakdown, cycles, clipped actions and
the interval of every capture rate:
[`results/hb_west/`](results/hb_west) and [`results/hb_houston/`](results/hb_houston).
Data provided by [Grid Status](https://www.gridstatus.io).

All of the v0.2 numbers below were reproduced to the dollar by v0.4 on a
fresh download (2026-11-11); the rows and sections added since are marked.

**Real-time energy only.** Capture rate = revenue / perfect foresight on the same days.

| Strategy | HB_WEST validation | HB_WEST test | HB_HOUSTON validation | HB_HOUSTON test |
|---|---:|---:|---:|---:|
| Perfect foresight | 539k (100.0%) | 929k (100.0%) | 438k (100.0%) | 754k (100.0%) |
| Threshold rule | 122k (22.7%) | 134k (14.4%) | 27k (6.2%) | 51k (6.7%) |
| Rolling threshold (30 d) | 118k (21.9%) | 208k (22.4%) | 42k (9.5%) | 76k (10.1%) |
| Forecast + MPC | 267k (49.6%) | 435k (46.9%) | 191k (43.6%) | 309k (41.0%) |
| Level-corrected forecast + MPC | 322k (59.7%) | 417k (44.9%) | 244k (55.7%) | 306k (40.6%) |
| Rolling threshold offer (30 d) | 102k (18.9%) | 173k (18.7%) | 71k (16.2%) | 71k (9.4%) |
| Offer MPC | 189k (35.0%) | 226k (24.3%) | 142k (32.4%) | 159k (21.2%) |

The last two submit an offer curve instead of a power (see
[Options](#options)); the bound is the same.

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

### How sure the numbers are

New in v0.4. Each split has 53 or 59 days, and a few days carry much of the
revenue: on the HB_WEST test split, perfect foresight earns 36% of its
revenue on its five best days. The full tables therefore give every
capture rate a 95% interval (block bootstrap over days) and the capture
rate without those five days.

- **The intervals are wide.** Forecast + MPC on the HB_WEST test split is
  46.9%, with an interval of 33% to 59%. Most strategies have intervals 15
  to 35 points wide.
- **Neighbouring strategies cannot be ranked.** The intervals of most
  adjacent rows overlap, including Forecast + MPC against the day-ahead
  strategies and the three day-ahead strategies among themselves. Read a
  difference of a few points as not established.
- The intervals only describe which days happened to be in the split. They
  say nothing about a season that is not in the data.

### What the new strategies show

- **Reacting to the current price through a fixed offer does not help
  these strategies; it hurts.** Offer MPC earns 21–35%, against 41–50% for
  the same plan executed as a power; the threshold rule as an offer is
  lower at HB_WEST (19% against 22%), mixed at HB_HOUSTON. An offer at a
  fixed price trades the moment the price crosses it, which on an evening
  ramp is too early. The environment can express offers; a strategy that
  uses them well is still missing.
- **A slightly better forecast helps on one split and not the other.**
  Correcting persistence for the last half hour's level raises capture from
  50% to 60% (HB_WEST) and 44% to 56% (HB_HOUSTON) on validation, and
  changes nothing on test (45% against 47%, 41% against 41%). The two
  intervals overlap almost entirely on test.

### Carrying energy over midnight

New in v0.4: two-settlement with every episode starting and ending half
full, as 7-day episodes (SoC carries over) and as one-day episodes on the
same 49 validation and 56 test days
([`cross_day.md`](results/hb_west/cross_day.md)).

| Revenue, 1-day → 7-day episodes | HB_WEST validation | HB_WEST test | HB_HOUSTON validation | HB_HOUSTON test |
|---|---:|---:|---:|---:|
| Perfect foresight (DA + RT) | 842k → 875k (+4.0%) | 1,533k → 1,566k (+2.2%) | 675k → 707k (+4.8%) | 1,309k → 1,343k (+2.7%) |
| Perfect foresight (RT only) | 391k → 424k (+8.6%) | 768k → 802k (+4.4%) | 294k → 327k (+11.1%) | 611k → 646k (+5.7%) |
| DA schedule + MPC on DA prices | 292k → 332k (+13.7%) | 485k → 522k (+7.6%) | 226k → 261k (+15.7%) | 343k → 381k (+11.0%) |
| MPC on DA prices | 225k → 265k (+17.7%) | 432k → 469k (+8.6%) | 148k → 184k (+23.9%) | 294k → 331k (+12.9%) |
| Forecast + MPC | 204k → 247k (+20.6%) | 340k → 361k (+6.4%) | 135k → 177k (+31.0%) | 229k → 268k (+17.0%) |

- **Forcing the battery back to half full every midnight costs the bound
  2–5%** (4–11% of the real-time part). The day-ahead part is the same in
  both, because each day's award is checked from the same starting SoC.
- **The MPC strategies gain more than the bound** (6–31%): a rolling plan
  that must hit a target at midnight spends the evening meeting it.
- A delivered day-ahead schedule gains nothing: it is planned one day at a
  time.

### Battery duration

New in v0.4: the two-settlement table for 100 MW with 100, 200 and 400 MWh
([`sensitivity.md`](results/hb_west/sensitivity.md)). HB_WEST test split,
USD per MW per day:

| Strategy | 1 h | 2 h | 4 h |
|---|---:|---:|---:|
| Perfect foresight (DA + RT) | 194 | 312 | 442 |
| Perfect foresight (RT only) | 97 | 157 | 230 |
| DA schedule + MPC on DA prices | 48 | 97 | 170 |
| Forecast + MPC | 33 | 74 | 136 |

The second and fourth hours of storage each add less than the one before
under perfect foresight (1.6 times the revenue from 1 h to 2 h, 1.4 times
from 2 h to 4 h). The causal strategies capture more of the bound with
longer duration (Forecast + MPC: 17%, 24%, 31%): with more energy, timing
errors cost less.

### Five-minute energy + ECRS

New in v0.4: five-minute dispatch, 15-minute settlement, ECRS, both markets
for both products, HB_WEST, every day starting and ending half full
([`energy_ecrs.md`](results/hb_west/energy_ecrs.md); 51 validation and 57
test days). Revenue in USD, the part from ECRS in brackets, and capture
rate against the first row.

| Strategy | Validation | Test |
|---|---:|---:|
| Perfect foresight (DA + RT, energy + ECRS) | 1,071k (52k), 100.0% | 1,753k (148k), 100.0% |
| Perfect foresight (RT only, energy + ECRS) | 505k (24k), 47.2% | 861k (53k), 49.1% |
| Perfect foresight (DA + RT, energy only) | 1,040k, 97.1% | 1,673k, 95.5% |
| Perfect foresight (RT only, energy only) | 489k, 45.6% | 827k, 47.2% |
| DA schedule, energy + ECRS | 356k (37k), 33.2% | 489k (126k), 27.9% |
| ECRS MPC on DA prices | 269k (16k), 25.1% | 468k (68k), 26.7% |
| Energy MPC on DA prices | 261k, 24.3% | 460k, 26.2% |
| ECRS MPC (persistence, 1 h) | −28k (6k), −2.6% | −19k (10k), −1.1% |
| Energy MPC (persistence, 1 h) | −29k, −2.7% | −30k, −1.7% |

- **ECRS adds 3–5% to what this battery could earn**: 3.0% and 4.8% with
  both markets, 3.4% and 4.1% in real time alone. This is under a
  conservative rule (reserve from the discharge side only, held for one
  hour, never deployed), so it is not an estimate for a real battery.
- **A delivered day-ahead schedule gets a real share of its revenue from
  ECRS** (10% on validation, 26% on test), because an hourly schedule
  leaves the battery idle and charged for many hours.
- **Adding ECRS to the real-time MPC is worth 2–3%**, well inside the
  intervals of the two rows (269k against 261k, 468k against 460k).
- **The one-hour persistence MPC of the ECRS gate loses money here.** A
  likely reason (not tested): it trades on five-minute price moves that
  15-minute settlement averages away, and pays degradation for it. It is
  kept as the link to the gate, not as a baseline to beat.
- The best causal strategy is at 28–33% of the bound. Closing that gap is
  the open problem in this setting.

The ECRS numbers follow a reduced model, not ERCOT's complete
ancillary-service settlement: see [the model and its limits](docs/ecrs_gate.md).

## Install

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

### Five-minute prices and ECRS

The five-minute settings need SCED data, downloaded separately because it
is larger (about 9,000 rows per series and month; the free Grid Status plan
allows 500,000 rows a month):

```bash
python scripts/download_5min.py --start 2026-07-01 --end 2026-10-24
```

| Series | Dataset |
|---|---|
| Energy, per SCED run | `ercot_lmp_by_settlement_point` (LMP) + `ercot_lmp_with_adders_by_settlement_point` (reliability deployment adder) |
| ECRS, per SCED run | `ercot_mcpc_sced` |
| ECRS, day-ahead hourly | `ercot_as_prices` |

SCED does not run on a fixed clock, so each five-minute price is the
time-weighted price in effect during the interval
([`rtcb_gym/data/sced.py`](rtcb_gym/data/sced.py)). An interval priced by
a SCED run that lacks a value is an estimate; days containing one are left
out of episodes, like days with gaps. For HB_WEST, July to October, that
is 13 of 33,408 intervals on 4 days (one of them the first interval of
the download). The 15-minute mean of the five-minute
energy price is within $0.01 of the official settlement point price in
99.99% of 11,127 intervals.

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
python scripts/summarize_results.py                           # the first two tables above
python scripts/run_market.py                                  # five-minute energy + ECRS (HB_WEST)
```

`run_baselines.py` prints the data hashes and the tables, and writes
`real_time`, `two_settlement`, `cross_day` and `sensitivity` (`.md` and
`.json`) under `results/<hub>/`. It takes about 25 minutes per hub on four
cores. `run_market.py` needs the five-minute data and writes
`results/<hub>/energy_ecrs.{md,json}`; it solves an integer program per
planning step and takes about an hour.

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

Five-minute dispatch with ECRS, both markets (needs the five-minute data,
see [below](#five-minute-prices-and-ecrs)):

```python
from rtcb_gym import DayAheadBid

env = RTCBEnv(
    rt5, BatteryParams(soc_init=0.5, soc_terminal_min=0.5),
    dt_hours=1 / 12, settlement_minutes=15, history_len=288,
    ecrs_prices=ecrs, da_prices=da, da_ecrs_prices=da_ecrs,
)
obs, info = env.reset(options={"day": day, "da_schedule": DayAheadBid(energy_mw=bid, ecrs_mw=reserve)})
obs, reward, terminated, truncated, info = env.step(np.array([0.2, 0.5]))  # energy, ECRS: shares of power_mw
```

For a learning agent, `ScaledObservation(env)` rescales the price blocks
of the observation to a bounded range, and `gymnasium.make("RTCBGym-v0",
prices=rt, ...)` builds the same environment by id.

## Environment

| | |
|---|---|
| Episode | one operating day (96 intervals; 92 or 100 on DST days); `episode_days` makes episodes longer |
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

### Options

Everything above is the default. Each option below is off unless you ask
for it, and each keeps perfect foresight an exact bound (tests replay the
bound and check that random play never beats it).

| Option | What it does |
|---|---|
| `dt_hours=1/12`, `settlement_minutes=15` | Five-minute dispatch. Each step settles at the mean price of its 15-minute interval, as ERCOT settles energy. |
| `ecrs_prices` | Adds ECRS. The action gains a second value in [0, 1], the reserve requested as a share of `power_mw`. It is clipped to what is left after the energy dispatch: none while charging, discharge + reserve within rated power, and enough stored energy to sustain it for one hour. Paid the real-time ECRS price. [Model and limits](docs/ecrs_gate.md). |
| `da_ecrs_prices` | A day-ahead ECRS award (`DayAheadBid(ecrs_mw=...)`), paid the day-ahead price; the real-time deviation from it settles at the real-time ECRS price. |
| `ecrs_deployment` | A share of the award deployed in each interval: delivered on top of the dispatch, paid the energy price, taken from the battery. An input you supply; no result here uses it. |
| `action_mode="offer"` | The action is an offer curve, `[sell share, sell price, buy share, buy price]`. The battery sells if the interval's price turns out at or above the sell price, buys if it is at or below the buy price, else idles. The agent still never sees the price in advance; the market applies the curve to it. |
| `DayAheadBid(energy_mw, energy_limit_price)` | Day-ahead bids with limit prices, one or several segments an hour: a sell segment clears if the day-ahead price is at or above its limit, a buy segment if it is at or below. What clears is paid the day-ahead price. |
| `episode_days > 1` with `da_prices` | Two-settlement episodes of several days: the SoC carries over midnight, each day has its own bid, and day D+1's day-ahead results enter the observation at 13:30 on D. |
| `obs_lag_steps` | Prices become visible that many steps late. |

The offer curve and the limit prices clear against historical prices: the
battery is still a price-taker and never moves them.

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

Added in v0.4:

10. **Level-corrected forecast + MPC**: persistence, shifted by how far the
    last intervals settled from the same intervals yesterday, fading with
    the horizon. Two parameters, fitted on training-split forecast error.
11. **Rolling threshold offer**: the rolling thresholds as the prices of an
    offer curve.
12. **Offer MPC**: the rolling LP's marginal buy and sell prices as an offer
    curve.
13. **Perfect foresight (energy + ECRS)**: the exact bound for any
    combination of the environment's options, with or without the
    day-ahead market and with or without ECRS.
14. **ECRS MPC**: a rolling integer program on a forecast of energy and
    ECRS prices (persistence, or the day's day-ahead prices).
15. **Energy MPC**: the same without ECRS, in the same environment.
16. **DA schedule, energy + ECRS**: yesterday's day-ahead prices of both
    products as the forecast, a joint hourly schedule, bid and delivered.

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
- Uncertainty: a 95% interval for revenue and capture rate from a circular
  block bootstrap over days (blocks of 5 days), and the capture rate without
  the bound's five best days (`rtcb_gym.evaluation.bootstrap_ci`).

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
rtcb_gym/data/sced.py         SCED cache, five-minute prices in effect, day-ahead ECRS prices
rtcb_gym/demo.py              offline CPU example using synthetic prices
rtcb_gym/battery.py           physical model
rtcb_gym/ancillary.py         ECRS reservation rule, feasible step, MILP
rtcb_gym/settlement.py        money: real-time only and two-settlement
rtcb_gym/day_ahead.py         day-ahead timeline, context, bids, clearing and clipping
rtcb_gym/env.py               Gymnasium environment (all options)
rtcb_gym/wrappers.py          observation scaling for learning agents
rtcb_gym/benchmark.py         tasks, task spec, evaluate, scorecard, leaderboard
rtcb_gym/evaluate.py          command line: python -m rtcb_gym.evaluate
rtcb_gym/text.py              text interface for language-model agents
rtcb_gym/data/synthetic.py    seeded synthetic prices (not ERCOT data)
examples/agents/              a threshold agent and a scripted text agent
rtcb_gym/evaluation.py        runs, metrics, bootstrap intervals
rtcb_gym/baselines/           perfect foresight, threshold rules, MPC, day-ahead, offers, ECRS
rtcb_gym/experimental/ecrs.py the continuous-panel model of the ECRS gate
scripts/                      download_data.py, download_5min.py, run_baselines.py, run_market.py,
                              run_ecrs_gate.py, export_5min.py, summarize_results.py
results/<hub>/                real_time, two_settlement, cross_day, sensitivity, energy_ecrs (.md, .json)
results/ecrs_gate/            ECRS duration gate: protocols, data checks, summaries
results/benchmark/<hub>/      scorecards of the reference agents, LEADERBOARD.md
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

| Priority | Deliverable |
|---|---|
| 1 | ERCOT's ancillary-service settlement (Section 6.7) checked term by term against the ECRS ledger here |
| 2 | Deployment from data instead of an input; charging-side reserve (curtailing charge as upward reserve) |
| 3 | Stronger agents on the benchmark: a real forecaster with MPC, a trained RL policy, a language model |
| 4 | Independent reproduction on another machine; resource-node prices |
| Later | Other AS products, each with its own rule and accounting validation |

## License

MIT
