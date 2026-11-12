# Energy + ECRS research gate

Status: experimental. The reservation model below started as a separate
five-minute environment (`rtcb_gym.experimental.ecrs`, still used by the
gate runner). Since v0.4 the same rule is an option of the main
environment (`RTCBEnv(ecrs_prices=...)`, see the README), which adds
15-minute settlement, a day-ahead ECRS award and an optional deployment
input. The rule, the feasible step and the MILP are shared code
(`rtcb_gym/ancillary.py`); a test steps both environments with the same
actions and checks that they agree.

[Results on real SCED data](#results-on-grid-status-sced-data) are at the
end.

## What is established and what is assumed

ERCOT [NPRR1282](https://www.ercot.com/mktrules/issues/NPRR1282), effective
2025-12-05, sets the ECRS duration requirement to **one hour**. The current
[Section 6](https://www.ercot.com/files/docs/2024/06/28/06-082826_Nodal.docx),
section 6.5.7.3(1), describes approximately five-minute joint energy/AS
dispatch that considers ESR SoC and operating limits. The official document
was downloaded and read, not inferred from a third-party repository.

The following model is a deliberately conservative **subset**, not a complete
translation of SCED. One hour is a verified rule parameter. The precise
inventory combination, post-dispatch reserve timing, no charging-side reserve,
full award and no deployment assumptions below are research choices. Complete
SCED, telemetry, qualification and settlement conformance is still open.

The comparator's 0.25-hour parameter is labelled a **sensitivity model**.
It is not the historical pre-RTC+B ECRS rule. In particular, this experiment
does not equate a 0.25-to-1-hour correction with the entire market redesign.

## Rule sources

The rule documents read for this model (SHA-256 of the downloaded file):

| Document | SHA-256 |
|---|---|
| [NPRR1282](https://www.ercot.com/mktrules/issues/NPRR1282) | web page, not hashed |
| [Nodal Protocols Section 6](https://www.ercot.com/files/docs/2024/06/28/06-082826_Nodal.docx), Adjustment Period and Real-Time Operations | `00180491327f146a48a9ae2fcbdefd7d322ca5273b89eec891d02ae77a43bdae` |
| [Nodal Protocols Section 9](https://www.ercot.com/files/docs/2023/06/01/09-080126_Nodal.docx), Settlement and Billing | `031ddf86ef8c876a350e35b9dfbe42066bca03c00fb357f73dd5f9bccde5fb92` |

## Mathematical task

State is stored energy e_t, in storage-side MWh. Actions are grid-side charge
c_t >= 0, discharge d_t >= 0 and ECRS reserve a_t >= 0, all MW. Positive net
power p_t = d_t - c_t discharges. dt = 1/12 hour. Efficiency is split as in
the existing battery model: eta_c = eta_d = sqrt(round-trip efficiency).

```
e_(t+1) = e_t + eta_c * c_t * dt - d_t * dt / eta_d
E_min <= e_(t+1) <= E_max
c_t <= P * (1 - u_t)
d_t + a_t <= P * u_t,             u_t in {0, 1}
duration * a_t / eta_d <= e_(t+1) - E_min
e_T >= E * soc_terminal_min
```

This reserves a full duration of energy after the base dispatch, prohibits
simultaneous charge/discharge, and allows reserve only in discharge/idle mode.
Actual ERCOT upward flexibility may include charging curtailment; this gate
does not model it. Do not describe these conservative choices as full SCED
equivalence or use the resulting oracle as an unrestricted ERCOT upper bound.

There is **no AS deployment**. A reserve award does not consume energy along
the realized base trajectory. The reserve inequality tests contingent
deliverability in this reduced model, not a multi-step deployment policy.
Deployment, activation uncertainty, response/ramp constraints, qualification,
price-quantity bids, partial awards and market-price formation are out of scope.

Cash-flow **proxy**, per interval:

```
energy_proxy = p_t * energy_LMP_t * dt
ecrs_proxy   = a_t * ECRS_MCPC_t * dt
degradation = abs(p_t) * degradation_cost * dt
net_proxy   = energy_proxy + ecrs_proxy - degradation
```

Five-minute LMP multiplication is not ERCOT's complete energy settlement.
The official Section 6 also includes 15-minute settlement prices and AS
imbalance calculations; these are not implemented here. No DA positions,
fees, uplift or resource-node/hub basis adjustments are added in this gate.

## Information and execution timeline

1. At interval start, the policy sees SoC, clock features and past price rows
   whose `available_at_utc <= decision_time_utc`.
2. A tape row cannot become observable before its interval end. A mask marks
   missing initial history; observed zeros and padded values are distinguished.
3. The action requests normalized `[energy/P, ECRS/P]` in `[-1,1] x [0,1]`.
4. The battery first clips energy to power, SoC and reachable terminal limits;
   the environment then clips ECRS to remaining power and reserve inventory.
5. Only executed energy and awarded reserve enter the cash-flow proxy.
6. The outcome is returned after execution. SoC carries continuously across
   the complete panel. There are no midnight resets or terminal liquidation.

The exploratory runner assumes price publication at interval end + five
minutes. This is a **sensitivity assumption**, not a measured ERCOT release
latency. Actual SCED runs can be irregular; the current tape intentionally
requires a regular nominal five-minute grid and rejects gaps. A later raw-data
adapter must preserve and explain execution timestamps and actual durations.

## Rule-to-code-to-test map

| Requirement or assumption | Implementation | Meaningful check |
|---|---|---|
| One-hour duration parameter | `ECRSRules()` | inventory/power boundary test; alternate duration labelled sensitivity |
| Battery limits and terminal reachability | existing `battery_step` | forced terminal charging, random trajectories |
| Discharge-side reserve subset | `ecrs_capacity_mw` / MILP mode constraints | no reserve while charging, no reserve at empty inventory |
| Post-dispatch reserve inventory | `reservation_step` and MILP energy inequality | six randomized oracle replays with alternate durations and terminal targets |
| No charge/discharge energy burning | MILP binary mode | full battery, negative-price one-step counterexample |
| Same reward for executed actions and oracle | `reservation_revenue` / MILP objective | objective/replay equality and no clipped oracle actions |
| Causal price visibility | `FiveMinuteTape`, `_observation` | future and unpublished-row perturbations leave observation unchanged |
| Gap/duplicate/NaN rejection | `FiveMinuteTape` | malformed tape cases are rejected |

The rule map is an audit aid, not a regulatory compliance certificate.
The oracle is optimal for the mathematical subset above within the solver's
floating-point tolerances, not a theorem about actual ERCOT revenues.

## Run

Install the ordinary package development dependencies, then:

```bash
pytest
python scripts/run_ecrs_gate.py --input-dir data/raw/third_party_hybridbid
```

The runner expects `energy_prices/*.parquet` with `rt_lmp` and
`as_prices/*.parquet` with `rt_mcpc_ecrs`, on exactly matching timezone-aware
datetime indices. It audits every file and rejects missing values in both
the panel and its preceding history. It never interpolates, fills prices,
or silently uses the nearest timestamp.

Each run writes a protocol lock **before outcomes**, input hashes, quality
checks, full per-interval trajectories and summaries. Only the protocol,
quality checks and summaries are committed: input prices can be recovered
from the trajectories, so those stay local (git-ignored) and are rebuilt by
rerunning the script on the same input files. The default panel is
2026-02-02 through 2026-02-04 Central Time. Battery: 100 MW / 200 MWh,
85% round-trip efficiency, $2/MWh throughput cost, initial and terminal
SoC 50%. Policies use a one-hour MPC horizon and past-day price persistence,
corrected for the observation delay. No forecast parameters are fitted.
MPC plans often tie (several steps with the same forecast price), and
different solver versions return different tied plans. The MPC planner
therefore adds 0.001 USD per MWh of charge, discharge and ECRS, highest in
the first step, so ties resolve to the plan that acts least and latest
(`tie_break_usd_per_mwh` in `solve_reservation`). Oracles use no tie-break
and remain exact.

All causal policies execute in the same one-hour model, including the policy
that plans with a 0.25-hour assumption. Its inadmissible reserve requests are
recorded and clipped before revenue is computed. The two oracles use their
respective feasible sets and are reported separately.

The public mirror comes from HybridBid commit
`68c1a62398e53ed4785c4dd67996ed547c4da34b`. It is third-party processed data,
not independently checked ERCOT raw data. Source files preserve some missing
values; the original HybridBid evaluation fills missing prices with zero.
This runner does not. Cell-level imputation lineage is unavailable.

## Extensions in the main environment

These are in `RTCBEnv` only, not in the gate model above.

**Day-ahead ECRS award.** With `da_ecrs_prices`, the day-ahead bid can
carry an hourly ECRS offer. It is clipped, together with the energy
schedule, to what the reservation rule allows from `soc_init` (the same
step function with a one-hour step), and settles the way energy does:

```
ecrs_da = award_h * DA ECRS price_h * dt
ecrs_rt = (a_t - award_h) * RT ECRS price_t * dt
```

A battery that holds less reserve in real time than it sold day-ahead buys
the difference back at the real-time price. This follows the two-settlement
logic of RTC+B's ancillary-service imbalance; the exact ERCOT formula
(Section 6.7) has not been checked term by term.

**Deployment.** `ecrs_deployment` is a share in [0, 1] per interval: that
share of the award is delivered on top of the base dispatch, paid the
energy price, charged degradation, and taken from the battery. An award
must then also leave the end-of-episode SoC target reachable if deployed
in full. The share is an input: this repository has no ERCOT deployment
data, and no result here uses it. Tests check that the bound stays exact
and the SoC target holds under random deployment.

**15-minute settlement.** With five-minute dispatch and
`settlement_minutes=15`, energy and ECRS of each step settle at the mean
price of the step's 15-minute interval. For HB_WEST, July to October 2026,
that mean is within $0.01 of the official settlement point price in 99.99%
of 11,127 intervals (largest difference $0.81).

## Decision gate

Internal software checks can establish reduced-model consistency. They do
not establish actual market fidelity. The public-mirror panels are exploratory,
not untouched confirmatory tests. Published benchmark profit tables have not
been reproduced by these runs.

Before expanding to five AS products or promoting the environment as a new
market benchmark, obtain independently traceable five-minute data and release
times, resolve the complete joint inventory constraints, and freeze a new
held-out evaluation protocol. Measure infeasible commitments, revenue/risk
changes, ranking sensitivity, and concentration in individual extreme days.

If oracle gaps are tiny and corrected models do not change meaningful
research decisions, preserve the small environment and audit result instead
of claiming that a large new simulation framework is necessary.

## Results on Grid Status SCED data

2026-11-11. The gate was rerun on SCED rows from the Grid Status API
(`--source gridstatus`): energy price = SCED LMP + Real-Time Reliability
Deployment Price Adder at HB_WEST, ECRS price = real-time MCPC, both
time-weighted to five minutes. Model, battery, policies and parameters
are those of the pilots; nothing was tuned. The test-split run is the
held-out one: its protocol was written before any outcome was computed,
after the two pilot windows and before the validation split. (Summary
statistics of the prices in that range, such as their median, had been
looked at during data checks; no policy had been run on them.)

A panel runs only if every five-minute row of it and of its one-day
history is a published price in effect. Nine of 17,569 rows in the test
range are not (a SCED run without an adder or MCPC value); they remove
four of twenty panels. Two of eighteen validation panels are removed the
same way.

| Window (Central) | Panels run | 0.25h plan clipped | Oracle revenue, 1h vs 0.25h | ECRS share of 1h oracle |
|---|---:|---:|---:|---:|
| 2026-02-02 .. 02-04 | 1 | 240 / 864 (27.8%) | −0.19% | 0.5% |
| 2026-01-24 .. 01-26 | 1 | 199 / 864 (23.0%) | −0.16% | 0.6% |
| Validation, 2026-07-02 .. 08-24 | 16 of 18 | 3,019 / 13,824 (21.8%) | −2.34% (panels: −0.40 to −6.80, median −1.86) | 3.6% |
| Test, 2026-08-25 .. 10-23 | 16 of 20 | 3,639 / 13,824 (26.3%) | −2.73% (panels: −0.19 to −7.73, median −2.23) | 3.8% |

| Net revenue proxy (USD) | February | January stress | Validation | Test |
|---|---:|---:|---:|---:|
| Energy MPC | 3,285 | −19,056 | 34,417 | 14,494 |
| Energy + ECRS MPC, 1h plan | 3,309 | −18,477 | 35,381 | 18,091 |
| 0.25h plan, executed under 1h | 3,216 | −19,948 | 29,985 | 21,265 |
| 1h oracle | 25,678 | 281,214 | 517,466 | 643,205 |
| 0.25h oracle | 25,727 | 281,652 | 529,850 | 661,288 |
| ECRS part of the 1h / 0.25h oracle | 139 / 222 | 1,820 / 2,762 | 18,649 / 38,117 | 24,592 / 54,404 |

What this shows, in this reduced model:

- **The duration rule matters more than the mirror pilots suggested.** On
  the held-out test split the one-hour rule lowers the oracle by 2.7%
  (0.02% and 0.41% on the mirror). A 0.25-hour model would count about
  twice the ECRS revenue the one-hour rule allows (54k against 25k).
- **About a quarter of a 0.25-hour planner's reserve requests cannot be
  held** (22–28% of intervals in all four windows).
- **Planning with the right duration does not reliably earn more.** The
  1h planner beats the 0.25h planner on validation and loses on test (9 of
  16 test panels, 8 of 16 validation panels). Both use a one-hour horizon
  and a persistence forecast and capture under 7% of the oracle, so this
  comparison is dominated by forecast quality, as in the pilots.
- **ECRS is a small part of this battery's value here**: 3.6–3.8% of
  oracle revenue in summer and autumn, under 1% in the winter windows.
  Real-time ECRS prices are near zero most of the time (median $0.05 per
  MW per hour, mean $1.07, July to October) with rare spikes.
- Adding ECRS to the MPC raised its revenue in 12 and 13 of 16 panels.

Limits: one hub, one battery, a conservative discharge-only reserve model
with no deployment, cash-flow proxies, and a publication delay that is
still an assumption (SCED timestamps show each run starting about 20
seconds into its interval; when its prices are posted was not measured).
The January and February windows use a different hub and energy price
definition than the mirror pilots, so those rows are not a like-for-like
replication.

Reproduce (needs the SCED cache, see `scripts/download_5min.py`):

```bash
python scripts/download_5min.py --start 2026-01-20 --end 2026-02-05
python scripts/download_5min.py --start 2026-07-01 --end 2026-10-24
python scripts/run_ecrs_gate.py --source gridstatus --start 2026-08-25 --end 2026-10-23 \
    --panel-days 3 --out-dir results/ecrs_gate/real_test_holdout
```

Data provided by [Grid Status](https://www.gridstatus.io).
