"""Deterministic battery dispatch optimisation over a known price vector.

Solves exactly the problem the environment scores:

    max  sum_t  price_t * (d_t - c_t) * dt  -  deg * (d_t + c_t) * dt
    s.t. s_t = s_{t-1} + (eta_c * c_t - d_t / eta_d) * dt
         E * soc_min <= s_t <= E * soc_max
         s_T >= E * soc_end_min
         0 <= c_t, d_t <= P

with c = charge MW, d = discharge MW, s = stored MWh. By default
``soc_end_min`` is ``params.soc_terminal_min``, the environment's end target.

The LP relaxation may charge and discharge in the same interval, which can
pay at very negative prices (it burns energy through losses). A single
battery cannot do that, so if it happens we re-solve as a MILP with one
binary per interval. Both use HiGHS via SciPy and are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from rtcb_gym.battery import BatteryParams

SIMULTANEOUS_TOL_MW = 1e-6


@dataclass(frozen=True)
class DispatchPlan:
    power_mw: np.ndarray  # + discharge, - charge
    soc: np.ndarray  # SoC fraction at the end of each interval
    revenue_usd: float  # objective value (net of degradation)
    used_milp: bool


def solve_dispatch(
    prices: np.ndarray,
    params: BatteryParams,
    soc0: float,
    dt_hours: float,
    soc_end_min: float | None = None,
) -> DispatchPlan:
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("prices must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices must be finite")
    T = p.size
    E, P, deg = params.energy_mwh, params.power_mw, params.degradation_cost_usd_per_mwh
    eta_c, eta_d = params.eta_charge, params.eta_discharge
    s0 = min(max(soc0, params.soc_min), params.soc_max) * E

    # x = [c_0..c_{T-1}, d_0..d_{T-1}]; minimise -revenue.
    cost = np.concatenate([(p + deg) * dt_hours, -(p - deg) * dt_hours])
    L = np.tril(np.ones((T, T)))
    cum = np.hstack([eta_c * dt_hours * L, -(dt_hours / eta_d) * L])  # s_t - s0
    a_ub = np.vstack([cum, -cum])
    b_ub = np.concatenate([np.full(T, params.soc_max * E - s0), np.full(T, s0 - params.soc_min * E)])
    end_min = params.soc_terminal_min if soc_end_min is None else soc_end_min
    if end_min > params.soc_min:
        a_ub = np.vstack([a_ub, -cum[-1:]])  # s_T - s0 >= E * end_min - s0
        b_ub = np.append(b_ub, s0 - end_min * E)

    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=[(0.0, P)] * (2 * T), method="highs")
    if res.status != 0:
        raise RuntimeError(f"dispatch LP failed: {res.message}")
    x = res.x
    used_milp = False
    if np.any(np.minimum(x[:T], x[T:]) > SIMULTANEOUS_TOL_MW):
        x, used_milp = _solve_milp(cost, a_ub, b_ub, T, P), True
    c, d = x[:T], x[T:]
    power = np.clip(d - c, -P, P)
    soc = (s0 + cum @ x) / E
    revenue = float(-(cost @ x))
    return DispatchPlan(power_mw=power, soc=soc, revenue_usd=revenue, used_milp=used_milp)


def _solve_milp(cost: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray, T: int, P: float) -> np.ndarray:
    """Same problem plus binaries u_t: c_t <= P u_t, d_t <= P (1 - u_t)."""
    z = np.zeros((T, T))
    eye = np.eye(T)
    full_cost = np.concatenate([cost, np.zeros(T)])
    a_soc = np.hstack([a_ub, np.zeros((a_ub.shape[0], T))])
    a_charge = np.hstack([eye, z, -P * eye])  # c - P u <= 0
    a_dis = np.hstack([z, eye, P * eye])  # d + P u <= P
    constraints = [
        LinearConstraint(a_soc, -np.inf, b_ub),
        LinearConstraint(a_charge, -np.inf, 0.0),
        LinearConstraint(a_dis, -np.inf, P),
    ]
    lb = np.zeros(3 * T)
    ub = np.concatenate([np.full(2 * T, P), np.ones(T)])
    integrality = np.concatenate([np.zeros(2 * T), np.ones(T)])
    res = milp(full_cost, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
    if res.status != 0:
        raise RuntimeError(f"dispatch MILP failed: {res.message}")
    return res.x[: 2 * T]
