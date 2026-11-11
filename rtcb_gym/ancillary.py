"""Energy + ECRS reservation: the rule, one feasible step, and the MILP.

A conservative, discharge-side-only, price-taking subset of ERCOT's
real-time co-optimisation (see docs/ecrs_gate.md for sources and limits):

    e_(t+1) = e_t + eta_c * c_t * dt - (d_t + k_t * a_t) * dt / eta_d
    E_min <= e_(t+1) <= E_max
    c_t <= P * (1 - u_t)
    d_t + a_t <= P * u_t,                      u_t in {0, 1}
    duration * a_t / eta_d <= e_base_(t+1) - E_min
    e_T >= E * soc_end_min

with c, d = grid-side charge and discharge MW, a = ECRS award MW, e = stored
MWh and e_base_(t+1) the stored energy after the base dispatch, before any
deployment. ECRS is awarded in full at the clearing price.

Deployment (optional)
    ``k_t`` in [0, 1] is the share of the award ERCOT deploys in interval t,
    an exogenous input (default: none, k = 0). Deployed MW are delivered on
    top of the base dispatch, paid at the energy price and charged
    degradation. With deployment modelled, an award must also leave the
    end-of-episode target reachable if it is deployed in full for the
    interval:

        a_t * dt / eta_d <= e_base_(t+1) - floor_(t+1)

    where floor is :func:`rtcb_gym.battery.terminal_floor`. Without a
    deployment input this last constraint is not applied.

Used by :class:`rtcb_gym.env.RTCBEnv` (any step length) and by the
five-minute gate model in :mod:`rtcb_gym.experimental.ecrs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from rtcb_gym.battery import BatteryParams, battery_step

FIVE_MINUTES_HOURS = 5 / 60
TOL_MW = 1e-4


@dataclass(frozen=True)
class ECRSRules:
    """One-hour requirement; other durations are labelled sensitivity models.

    Shared power and post-dispatch reserve are deliberate conservative model
    choices. They have NOT been certified as the full SCED feasible region.
    """
    duration_hours: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_hours) or self.duration_hours <= 0:
            raise ValueError("duration_hours must be finite and positive")

    @property
    def rule_id(self) -> str:
        prefix = "ecrs" if self.duration_hours == 1 else "sensitivity"
        return f"{prefix}-{self.duration_hours:g}h-discharge-reservation-v1"


def _floor(params: BatteryParams, end_min: float, steps_after: int, dt_hours: float) -> float:
    """Lowest SoC from which ``end_min`` is reachable in ``steps_after`` steps."""
    gain = params.power_mw * dt_hours * params.eta_charge / params.energy_mwh
    return max(params.soc_min, end_min - steps_after * gain)


def ecrs_capacity_mw(
    params: BatteryParams, rules: ECRSRules, power_mw: float, soc_after: float,
    *, deployable_floor: float | None = None, dt_hours: float = FIVE_MINUTES_HOURS,
) -> float:
    """Available upward reserve after base dispatch, in this reduced model.

    No ECRS while charging. Discharge + ECRS <= rated power. Remaining
    stored energy must sustain the award for the configured full duration.
    ``deployable_floor`` (SoC; only with deployment modelled): a full
    deployment for ``dt_hours`` must not take the SoC below it.
    """
    if power_mw < -1e-8:
        return 0.0
    available = max(0.0, (soc_after - params.soc_min) * params.energy_mwh)
    cap = min(
        params.power_mw - max(power_mw, 0.0),
        available * params.eta_discharge / rules.duration_hours,
    )
    if deployable_floor is not None:
        room = max(0.0, (soc_after - deployable_floor) * params.energy_mwh)
        cap = min(cap, room * params.eta_discharge / dt_hours)
    return max(0.0, cap)


def reservation_step(
    params: BatteryParams, rules: ECRSRules, soc: float,
    requested_mw: np.ndarray, steps_left: int,
    *, dt_hours: float = FIVE_MINUTES_HOURS, deployment: float | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Execute energy first, then clip the ECRS request to remaining capacity.

    Returns ``([base power MW, ECRS award MW], next SoC, info)``. With
    ``deployment`` (share of the award deployed this interval), the deployed
    MW are delivered on top of the base power and the next SoC reflects them;
    ``info["delivered_mw"]`` is base + deployed.
    """
    request = np.asarray(requested_mw, dtype=float)
    if request.shape != (2,) or not np.isfinite(request).all():
        raise ValueError("action must contain two finite numbers: energy MW, ECRS MW")
    if deployment is not None and not 0.0 <= deployment <= 1.0:
        raise ValueError("deployment must be a share in [0, 1]")
    power, next_soc, info = battery_step(params, soc, request[0], dt_hours, steps_left=steps_left)
    floor = None
    if deployment is not None:
        floor = _floor(params, params.soc_terminal_min, steps_left - 1, dt_hours)
    cap = ecrs_capacity_mw(params, rules, power, next_soc, deployable_floor=floor, dt_hours=dt_hours)
    award = float(np.clip(request[1], 0.0, cap))
    deployed = award * deployment if deployment else 0.0
    if deployed:
        stored = deployed * dt_hours / params.eta_discharge
        next_soc = max(next_soc - stored / params.energy_mwh, params.soc_min)
        info["storage_discharge_mwh"] += stored
        info["soc_after"] = next_soc
    actual = np.array([power, award])
    info.update(
        requested_mw=request.copy(), actual_mw=actual.copy(),
        ecrs_capacity_mw=cap, deployed_mw=deployed, delivered_mw=power + deployed,
        energy_clipped=bool(abs(request[0] - power) > TOL_MW),
        ecrs_clipped=bool(abs(request[1] - award) > TOL_MW),
    )
    return actual, next_soc, info


def reservation_revenue(
    actual_mw: np.ndarray, prices: np.ndarray, params: BatteryParams,
    *, dt_hours: float = FIVE_MINUTES_HOURS, deployed_mw: float = 0.0,
) -> dict[str, float]:
    """LMP/MCPC revenue proxy; degradation applies only to delivered energy."""
    x, p = np.asarray(actual_mw, dtype=float), np.asarray(prices, dtype=float)
    if x.shape != (2,) or p.shape != (2,) or not np.isfinite(np.r_[x, p]).all():
        raise ValueError("actual_mw and prices must each have two finite values")
    delivered = x[0] + deployed_mw
    energy = float(delivered * p[0] * dt_hours)
    ecrs = float(x[1] * p[1] * dt_hours)
    deg = float(abs(delivered) * params.degradation_cost_usd_per_mwh * dt_hours)
    return dict(energy_proxy_usd=energy, ecrs_proxy_usd=ecrs,
                degradation_usd=deg, net_proxy_usd=energy + ecrs - deg)


@dataclass(frozen=True)
class ReservationPlan:
    actions_mw: np.ndarray
    soc_after: np.ndarray
    net_proxy_usd: float
    mip_gap: float


def solve_reservation(
    prices: np.ndarray, params: BatteryParams, rules: ECRSRules, soc0: float,
    *, soc_end_min: float | None = None, time_limit: float = 60.,
    tie_break_usd_per_mwh: float = 0., dt_hours: float = FIVE_MINUTES_HOURS,
    deployment: np.ndarray | None = None,
) -> ReservationPlan:
    """MILP oracle/forecast planner for exactly the reduced environment model.

    Variables: charge, discharge, ECRS, stored energy, binary discharge mode.
    Nonzero ECRS excludes charging; discharge + ECRS <= P. The reserve
    inventory inequality uses energy AFTER base dispatch, as does the env.
    ``prices`` has one row per step: energy price (USD/MWh) and ECRS price
    (USD/MW per hour). ``deployment`` (share of the award deployed in each
    step, known to the planner) switches on the deployment model of the
    module docstring. Returns only a proven optimal solve (within solver
    numerical tolerance).

    Ties: many plans can earn the same (e.g. several zero-price steps to
    charge in, or free reserve), and which one a solver returns can change
    between solver versions. ``tie_break_usd_per_mwh`` > 0 adds a small cost
    per MWh of charge, discharge and ECRS: the full amount in the first step,
    falling linearly to 1/T of it in the last. Equal-profit plans then
    resolve to the one that acts least and latest. It can also give up at
    most that amount per MWh of activity, so leave it at 0 for an exact
    oracle. ``net_proxy_usd`` is always the plan's unpenalized value.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 2 or prices.shape[1] != 2 or not len(prices) or not np.isfinite(prices).all():
        raise ValueError("prices must be finite, nonempty, shape (T, 2)")
    if not np.isfinite(soc0) or not params.soc_min <= soc0 <= params.soc_max:
        raise ValueError("soc0 outside battery limits")
    end_min = params.soc_terminal_min if soc_end_min is None else float(soc_end_min)
    if not np.isfinite(end_min) or not 0 <= end_min <= params.soc_max:
        raise ValueError("invalid terminal SoC")
    if not dt_hours > 0:
        raise ValueError("dt_hours must be > 0")
    T, P, E, dt = len(prices), params.power_mw, params.energy_mwh, dt_hours
    k = None
    if deployment is not None:
        k = np.asarray(deployment, dtype=float)
        if k.shape != (T,) or not np.isfinite(k).all() or k.min() < 0 or k.max() > 1:
            raise ValueError("deployment must be T shares in [0, 1]")
    deg = params.degradation_cost_usd_per_mwh
    c, d, a, e, u = 0, T, 2*T, 3*T, 4*T+1
    N = 5*T+1
    cost = np.zeros(N)
    cost[c:c+T] = (prices[:, 0] + deg) * dt
    cost[d:d+T] = -(prices[:, 0] - deg) * dt
    cost[a:a+T] = -prices[:, 1] * dt
    if k is not None:  # deployed MW earn the energy price and wear the battery
        cost[a:a+T] -= k * (prices[:, 0] - deg) * dt
    lower, upper = np.zeros(N), np.full(N, P)
    lower[e:e+T+1], upper[e:e+T+1] = params.soc_min*E, params.soc_max*E
    lower[e] = upper[e] = soc0*E
    lower[e+T] = max(params.soc_min, end_min)*E
    upper[u:] = 1.
    integ = np.zeros(N); integ[u:] = 1.
    if not np.isfinite(tie_break_usd_per_mwh) or tie_break_usd_per_mwh < 0:
        raise ValueError("tie_break_usd_per_mwh must be finite and nonnegative")
    objective = cost.copy()
    later = (T - np.arange(T)) / T  # 1 in the first step down to 1/T in the last
    for start in (c, d, a):
        objective[start:start+T] += tie_break_usd_per_mwh * later * dt
    rows, cols, values, lo, hi = [], [], [], [], []
    def add(entries, lb, ub):
        row = len(lo)
        for col, value in entries:
            rows.append(row); cols.append(col); values.append(value)
        lo.append(lb); hi.append(ub)
    for t in range(T):
        balance = [(e+t+1, 1.), (e+t, -1.), (c+t, -params.eta_charge*dt), (d+t, dt/params.eta_discharge)]
        if k is not None and k[t] > 0:
            balance.append((a+t, k[t]*dt/params.eta_discharge))
        add(balance, 0., 0.)
        add([(c+t, 1.), (u+t, P)], -np.inf, P)
        add([(d+t, 1.), (a+t, 1.), (u+t, -P)], -np.inf, 0.)
        if k is None:
            add([(a+t, rules.duration_hours/params.eta_discharge), (e+t+1, -1.)],
                -np.inf, -params.soc_min*E)
        else:
            # e_base_(t+1) = e_(t+1) + k_t * a_t * dt / eta_d
            back = k[t]*dt/params.eta_discharge
            add([(a+t, rules.duration_hours/params.eta_discharge - back), (e+t+1, -1.)],
                -np.inf, -params.soc_min*E)
            floor = _floor(params, end_min, T-t-1, dt)*E
            add([(a+t, dt/params.eta_discharge - back), (e+t+1, -1.)], -np.inf, -floor)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lo), N)).tocsc()
    result = milp(objective, integrality=integ, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, lo, hi),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.})
    if result.status != 0:
        raise RuntimeError(f"reservation MILP not optimal: {result.message}")
    x = result.x
    return ReservationPlan(
        actions_mw=np.column_stack([x[d:d+T] - x[c:c+T], x[a:a+T]]),
        soc_after=x[e+1:e+T+1]/E, net_proxy_usd=float(-(cost @ x)),
        mip_gap=float(result.mip_gap),
    )
