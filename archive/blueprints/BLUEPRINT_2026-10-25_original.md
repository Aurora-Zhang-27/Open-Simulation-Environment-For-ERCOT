# RTCB-Gym — Blueprint

An open simulation environment for battery storage bidding in ERCOT after **Real-Time Co-optimization + Batteries (RTC+B)**, which went live on Operating Day **2025-12-05**.

Positioning (README first line): *the first open battery-storage market environment for the post-RTC+B ERCOT market.*

This file is the plan. Build in the order below; do not start a later milestone before the earlier one is tested and released.

---

## 1. Goal and long-term direction

- **Short term:** a small, correct, reproducible simulator that any strategy (rule, optimization, RL, LLM agent) can plug into and be scored on the same terms. Release v0.1 on GitHub fast.
- **Long term:** the environment becomes a benchmark, then the home of a vertical grid agent that bids storage, explains its decisions, and adapts to market rules.

Layers of realism (built in this order):

1. **Historical replay** — real prices, battery assumed price-taker. Nearly no sim-to-real gap.
2. **Scenario generation** — synthetic stress scenarios (cold snaps, price spikes, renewable ramps), validated against held-out real data.
3. **Price impact** — large storage fleets move the clearing price. Last and hardest.

---

## 2. Scope of v0

| Item | v0 choice |
|---|---|
| Market | ERCOT, real-time only |
| Period | 2025-12-05 onward (post-RTC+B only; never mix with earlier data) |
| Location | One trading hub, configurable (default `HB_WEST`; `HB_HOUSTON` as alternative) |
| Products | Energy only. Ancillary services come in v0.5 |
| Resolution | 15-minute real-time Settlement Point Prices |
| Price impact | None (price-taker) |
| Default battery | 100 MW / 200 MWh, 85% round-trip efficiency |

---

## 3. Repository layout

```
rtcb-gym/
├── README.md
├── LICENSE                  # MIT or Apache-2.0
├── pyproject.toml
├── configs/
│   └── default.yaml         # hub, dates, battery params, splits
├── rtcb_gym/
│   ├── data/
│   │   ├── fetch.py         # download raw prices
│   │   ├── clean.py         # DST, gaps, duplicates, timezone
│   │   └── store.py         # parquet read/write, one schema
│   ├── battery.py           # physical model
│   ├── settlement.py        # money
│   ├── env.py               # Gymnasium environment
│   └── baselines/
│       ├── perfect_foresight.py
│       ├── threshold_rule.py
│       └── forecast_mpc.py
├── scripts/
│   ├── download_data.py
│   └── run_baselines.py     # prints the results table
└── tests/
    ├── test_battery.py
    ├── test_settlement.py
    └── test_env.py
```

---

## 4. Module specs

### 4.1 Data layer
- Source: the `gridstatus` Python library first. ERCOT's official Public API (requires a free account and key) is the fallback.
- **Verify report formats and column names against the actual data. Do not assume.**
- Clean: convert to a single timezone (store UTC, plus a `local_time` column in US/Central). Handle DST transition days (23 or 25 hours). Report gaps; never silently fill them.
- Schema (parquet): `interval_start_utc, interval_end_utc, location, market, price_usd_per_mwh`.
- Cache raw downloads so the repo works offline after the first run.

### 4.2 Battery model (`battery.py`)
- Parameters: `power_mw`, `energy_mwh`, `rte` (round-trip efficiency, split as √rte on charge and discharge), `soc_min`, `soc_max`, `soc_init`, `degradation_cost_usd_per_mwh` (throughput-based, default small, e.g. 2).
- `step(power_mw, dt_hours) -> (actual_power_mw, new_soc, info)`
  - Sign convention: positive = discharge (sell), negative = charge (buy). Document this everywhere.
  - Clip to power limits and SoC limits. Return the clipped action and record the clipping in `info`.
- Pure and deterministic. No prices inside.

### 4.3 Settlement (`settlement.py`)
- v0 (real-time only): `revenue = actual_power_mw × dt_hours × rt_price`, minus degradation cost on throughput.
- Charging at negative prices earns money; this must work correctly.
- v0.2 adds day-ahead two-settlement: `DA_award × DA_price + (RT_actual − DA_award) × RT_price`.
- **Tests with hand-calculated examples are mandatory**: positive price, negative price, clipped action, zero action, efficiency loss over a full cycle.

### 4.4 Environment (`env.py`)
- Gymnasium API: `reset(seed)`, `step(action)`.
- Episode: one operating day by default (96 intervals); configurable.
- Action: continuous, normalized to [−1, 1] and scaled to `power_mw`.
- Observation (v0): SoC, time-of-day (sin/cos), day-of-week, last N real-time prices (default N = 96).
- **No look-ahead:** at interval t, only prices from intervals that have already settled are visible. Add a test that fails if future prices leak.
- Reward: settlement revenue for the interval.
- `info`: clipping, revenue breakdown, SoC.

### 4.5 Baselines
1. **Perfect foresight** — linear program per day over the true prices (scipy `linprog` or `cvxpy`). Upper bound.
2. **Threshold rule** — charge below a low quantile, discharge above a high quantile; quantiles fitted on the training split only.
3. **Forecast + MPC** — rolling-horizon LP using a simple forecast (same-hour-yesterday persistence in v0). This is where better forecasters plug in later.

### 4.6 Evaluation
- Time-based split, never random: train / validation / test, with the most recent ~2 months as test.
- Metrics: total revenue (USD), **capture rate** (revenue ÷ perfect-foresight revenue), equivalent full cycles per day, number of clipped actions.
- `scripts/run_baselines.py` prints one markdown table for the README.

---

## 5. Milestones

| Version | Content | Done when |
|---|---|---|
| **v0.1** | Data layer, battery, RT settlement, env, 3 baselines, README with results table | Fresh clone → one command installs and reproduces the table; all tests pass. **Release to GitHub.** |
| v0.2 | Day-ahead market + two-settlement | DA+RT baselines in the table |
| v0.5 | Real-time ancillary services (RegUp, RegDown, RRS, ECRS, Non-Spin) co-optimized with energy | Verify product list and SoC-duration rules against current ERCOT protocols first |
| v1.0 | Scenario generator (stress scenarios), with statistical validation vs held-out data: tail distribution, autocorrelation, spike frequency | Validation report in docs |
| v1.x | Better forecasters (reuse TFT pipeline from earlier carbon forecasting work); risk guardrails | Beat forecast+MPC baseline on test split |
| v2.0 | LLM agent with tools: forecaster, optimizer, market-rule retrieval, risk check | Agent scored in the same env against all baselines |
| later | NYISO, PJM via config; price impact | — |

---

## 6. Rules for the build (for Claude Code)

- Build only the current milestone. No speculative features.
- Every module testable on its own; no hidden global state.
- All randomness seeded; results reproducible bit for bit.
- Configuration in YAML; switching hub or date range touches config only.
- When unsure about an ERCOT rule or data format, check the source data or ERCOT documentation and note the source in a comment. Do not guess.
- Keep the README honest: state the price-taker assumption and the post-RTC+B date range prominently.
