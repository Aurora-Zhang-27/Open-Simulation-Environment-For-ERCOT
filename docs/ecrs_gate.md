# Energy + ECRS research gate

Status: experimental, local, unreleased. This adds a small testable model to
the v0.2.0 source at commit `7586e31095666d7c8e5cc59c7a88b47e73e5c058`.
The existing energy-only and day-ahead environments are unchanged.

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
