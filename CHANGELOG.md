# Changelog

## 0.5.0 (2026-11-15)

A benchmark layer, so any agent can be scored the same way
([guide](docs/benchmark.md)).

- Five named tasks (`rt-energy`, `rt-energy-offer`, `da-rt-energy`,
  `da-rt-energy-7day`, `da-rt-energy-ecrs-5min`), each fixing market
  setting, battery, episode length, splits and bound.
- One agent contract: an agent is built from a `TaskSpec` (rules, layouts,
  training-split prices) and never receives the environment.
- `evaluate()` and `python -m rtcb_gym.evaluate`: JSON scorecards with
  revenue, capture rate and its interval, invalid and clipped actions,
  time per step, task and data hashes; a leaderboard from scorecards. A
  failing agent is scored, not crashed.
- `rtcb_gym.text.TextAgent`: a provider-agnostic text interface for
  language-model agents.
- Seeded synthetic prices (`--source synthetic`) for developing agents
  without market data.
- Reference agents scored on real data (`results/benchmark/hb_west/`).

## 0.4.0 (2026-11-11)

One environment for energy and ECRS; real-data ECRS results.

- `RTCBEnv` options, all off by default: five-minute dispatch with
  15-minute settlement, ECRS (real time, day-ahead, deployment input),
  offer-curve actions, day-ahead bids with limit prices, two-settlement
  episodes of several days, delayed price visibility.
- An exact perfect-foresight bound for every combination.
- The ECRS duration gate rerun on Grid Status SCED data
  ([results](docs/ecrs_gate.md#results-on-grid-status-sced-data)).
- New results: offer-curve strategies, cross-day episodes, battery
  duration, five-minute energy + ECRS; bootstrap intervals on every
  capture rate.
- `RTCBGym-v0` registration, `ScaledObservation`, Python 3.13 in CI.
- All v0.2 results reproduced to the dollar on a fresh download.

## 0.3.0

Offline CPU example and five-minute SCED price export.

## 0.2.0

Day-ahead market and two-settlement; rolling threshold baseline; results
for HB_WEST and HB_HOUSTON.

## 0.1.0

Real-time environment, settlement, baselines and data pipeline.
