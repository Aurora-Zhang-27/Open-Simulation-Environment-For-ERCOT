"""Five-minute energy + ECRS reservation research model.

This is a conservative, discharge-side-only, price-taking subset. ECRS is
awarded in full and never deployed. Five-minute energy LMP cash flows are
proxies, not ERCOT's complete 15-minute settlement. See docs/ecrs_gate.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from rtcb_gym.battery import BatteryParams, battery_step, terminal_floor

DT_HOURS = 5 / 60
STEP = pd.Timedelta(minutes=5)
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


def ecrs_capacity_mw(
    params: BatteryParams, rules: ECRSRules, power_mw: float, soc_after: float,
) -> float:
    """Available upward reserve after base dispatch, in this reduced model.

    No ECRS while charging. Discharge + ECRS <= rated power. Remaining
    stored energy must sustain the award for the configured full duration.
    """
    if power_mw < -1e-8:
        return 0.0
    available = max(0.0, (soc_after - params.soc_min) * params.energy_mwh)
    return max(0.0, min(
        params.power_mw - max(power_mw, 0.0),
        available * params.eta_discharge / rules.duration_hours,
    ))


def reservation_step(
    params: BatteryParams, rules: ECRSRules, soc: float,
    requested_mw: np.ndarray, steps_left: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Execute energy first, then clip the ECRS request to remaining capacity."""
    request = np.asarray(requested_mw, dtype=float)
    if request.shape != (2,) or not np.isfinite(request).all():
        raise ValueError("action must contain two finite numbers: energy MW, ECRS MW")
    power, next_soc, info = battery_step(
        params, soc, request[0], DT_HOURS, steps_left=steps_left,
    )
    cap = ecrs_capacity_mw(params, rules, power, next_soc)
    award = float(np.clip(request[1], 0.0, cap))
    actual = np.array([power, award])
    info.update(
        requested_mw=request.copy(), actual_mw=actual.copy(),
        ecrs_capacity_mw=cap,
        energy_clipped=bool(abs(request[0] - power) > TOL_MW),
        ecrs_clipped=bool(abs(request[1] - award) > TOL_MW),
    )
    return actual, next_soc, info


def reservation_revenue(
    actual_mw: np.ndarray, prices: np.ndarray, params: BatteryParams,
) -> dict[str, float]:
    """LMP/MCPC revenue proxy; degradation applies only to executed energy."""
    x, p = np.asarray(actual_mw, dtype=float), np.asarray(prices, dtype=float)
    if x.shape != (2,) or p.shape != (2,) or not np.isfinite(np.r_[x, p]).all():
        raise ValueError("actual_mw and prices must each have two finite values")
    energy = float(x[0] * p[0] * DT_HOURS)
    ecrs = float(x[1] * p[1] * DT_HOURS)
    deg = float(abs(x[0]) * params.degradation_cost_usd_per_mwh * DT_HOURS)
    return dict(energy_proxy_usd=energy, ecrs_proxy_usd=ecrs,
                degradation_usd=deg, net_proxy_usd=energy + ecrs - deg)


class FiveMinuteTape:
    """Strict, gap-free price tape with explicit observation availability times.

    Required columns: interval_start_utc, available_at_utc,
    energy_price_usd_per_mwh, ecrs_price_usd_per_mw_h. No filling or resampling.
    available_at must be at or after the interval end. If publication times
    were assumed, callers must record that assumption in provenance.
    """
    def __init__(self, frame: pd.DataFrame, provenance: dict[str, Any]) -> None:
        columns = ["interval_start_utc", "available_at_utc",
                   "energy_price_usd_per_mwh", "ecrs_price_usd_per_mw_h"]
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"tape requires columns {columns}")
        for col in columns[:2]:
            if not isinstance(frame[col].dtype, pd.DatetimeTZDtype):
                raise ValueError(f"{col} must be timezone-aware")
        starts = pd.DatetimeIndex(frame[columns[0]]).tz_convert("UTC").as_unit("ns")
        available = pd.DatetimeIndex(frame[columns[1]]).tz_convert("UTC").as_unit("ns")
        if len(starts) < 2 or starts.hasnans or available.hasnans:
            raise ValueError("tape needs at least two valid timestamps")
        if np.any(np.diff(starts.asi8) != STEP.value):
            raise ValueError("tape must be sorted, unique and exactly five minutes; gaps are errors")
        if starts[0].tz_convert("US/Central").date() < pd.Timestamp("2025-12-05").date():
            raise ValueError("tape must be post-RTC+B")
        if np.any(available.asi8 < starts.asi8 + STEP.value):
            raise ValueError("prices cannot be observable before interval end")
        if np.any(np.diff(available.asi8) < 0):
            raise ValueError("availability timestamps must be nondecreasing")
        prices = frame[columns[2:]].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(prices).all():
            raise ValueError("prices must be finite; missing data must be reported, never filled")
        if not provenance.get("price_basis"):
            raise ValueError("provenance must declare price_basis")
        prices.setflags(write=False)
        self.starts, self.available_at, self.prices = starts, available, prices
        self.provenance = dict(provenance)


class ECRSReservationEnv(gym.Env):
    """One continuous panel, normalized [energy, ECRS] actions.

    Positive energy discharges. ECRS is nonnegative. History contains only
    completed intervals whose available_at <= decision_time. Padding is
    zero with a mask; future outcomes are never returned before execution.
    """
    metadata = {"render_modes": []}

    def __init__(self, tape: FiveMinuteTape, battery: BatteryParams,
                 rules: ECRSRules | None = None, *, start: int = 0,
                 stop: int | None = None, history_len: int = 288) -> None:
        super().__init__()
        self.tape, self.battery = tape, battery
        self.rules = rules or ECRSRules()
        self.start, self.stop = int(start), len(tape.starts) if stop is None else int(stop)
        if not 0 <= self.start < self.stop <= len(tape.starts):
            raise ValueError("invalid panel bounds")
        if history_len < 1:
            raise ValueError("history_len must be positive")
        self.history_len = int(history_len)
        n = self.stop - self.start
        if battery.soc_init < terminal_floor(battery, n, DT_HOURS) - 1e-12:
            raise ValueError("terminal SoC cannot be reached")
        self.action_space = gym.spaces.Box(
            low=np.array([-1., 0.]), high=np.ones(2), dtype=np.float64,
        )
        self.observation_space = gym.spaces.Dict({
            "soc": gym.spaces.Box(0., 1., (1,), dtype=np.float64),
            "prices": gym.spaces.Box(-np.inf, np.inf, (self.history_len, 2), dtype=np.float64),
            "price_mask": gym.spaces.Box(0., 1., (self.history_len,), dtype=np.float64),
            "price_age_steps": gym.spaces.Box(0., np.inf, (1,), dtype=np.float64),
            "time_of_day": gym.spaces.Box(-1., 1., (2,), dtype=np.float64),
        })
        self._t: int | None = None
        self._soc = battery.soc_init

    def _observation(self) -> dict[str, np.ndarray]:
        t = self._t
        assert t is not None
        decision = self.tape.starts[t] if t < len(self.tape.starts) else self.tape.starts[-1] + STEP
        # Availability time and interval-end validation together prevent t leakage.
        known = int(np.searchsorted(self.tape.available_at.asi8, decision.value, side="right"))
        known = min(known, t)
        n = min(known, self.history_len)
        history, mask = np.zeros((self.history_len, 2)), np.zeros(self.history_len)
        if n:
            history[-n:] = self.tape.prices[known - n:known]
            mask[-n:] = 1.
        local = decision.tz_convert("US/Central")
        angle = 2 * np.pi * (local.hour * 60 + local.minute) / 1440
        return {"soc": np.array([self._soc]), "prices": history,
                "price_mask": mask,
                "price_age_steps": np.array([float((decision.value - self.tape.starts[known-1].value) / STEP.value) if known else 0.]),
                "time_of_day": np.array([np.sin(angle), np.cos(angle)])}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("panel is configured at construction; reset options are unsupported")
        self._t, self._soc = self.start, self.battery.soc_init
        info = {"steps_remaining": self.stop - self.start,
                "decision_time_utc": self.tape.starts[self.start],
                "rule_id": self.rules.rule_id,
                "as_deployment": "none", "price_basis": self.tape.provenance["price_basis"]}
        return self._observation(), info

    def step(self, action):
        if self._t is None or self._t >= self.stop:
            raise RuntimeError("reset required before stepping")
        a = np.asarray(action, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("normalized action must have shape (2,) and finite values")
        t = self._t
        actual, self._soc, info = reservation_step(
            self.battery, self.rules, self._soc, a * self.battery.power_mw, self.stop - t,
        )
        ledger = reservation_revenue(actual, self.tape.prices[t], self.battery)
        self._t += 1
        info.update(ledger, soc=self._soc, steps_remaining=self.stop - self._t,
                    settled_interval_start_utc=self.tape.starts[t],
                    settled_prices=self.tape.prices[t].copy())
        return self._observation(), ledger["net_proxy_usd"], self._t >= self.stop, False, info


@dataclass(frozen=True)
class ReservationPlan:
    actions_mw: np.ndarray
    soc_after: np.ndarray
    net_proxy_usd: float
    mip_gap: float


def solve_reservation(
    prices: np.ndarray, params: BatteryParams, rules: ECRSRules, soc0: float,
    *, soc_end_min: float | None = None, time_limit: float = 60.,
) -> ReservationPlan:
    """MILP oracle/forecast planner for exactly the reduced environment model.

    Variables: charge, discharge, ECRS, stored energy, binary discharge mode.
    Nonzero ECRS excludes charging; discharge + ECRS <= P. The reserve
    inventory inequality uses energy AFTER base dispatch, as does the env.
    Returns only a proven optimal solve (within solver numerical tolerance).
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 2 or prices.shape[1] != 2 or not len(prices) or not np.isfinite(prices).all():
        raise ValueError("prices must be finite, nonempty, shape (T, 2)")
    if not np.isfinite(soc0) or not params.soc_min <= soc0 <= params.soc_max:
        raise ValueError("soc0 outside battery limits")
    end_min = params.soc_terminal_min if soc_end_min is None else float(soc_end_min)
    if not np.isfinite(end_min) or not 0 <= end_min <= params.soc_max:
        raise ValueError("invalid terminal SoC")
    T, P, E = len(prices), params.power_mw, params.energy_mwh
    c, d, a, e, u = 0, T, 2*T, 3*T, 4*T+1
    N = 5*T+1
    cost = np.zeros(N)
    cost[c:c+T] = (prices[:, 0] + params.degradation_cost_usd_per_mwh) * DT_HOURS
    cost[d:d+T] = -(prices[:, 0] - params.degradation_cost_usd_per_mwh) * DT_HOURS
    cost[a:a+T] = -prices[:, 1] * DT_HOURS
    lower, upper = np.zeros(N), np.full(N, P)
    lower[e:e+T+1], upper[e:e+T+1] = params.soc_min*E, params.soc_max*E
    lower[e] = upper[e] = soc0*E
    lower[e+T] = max(params.soc_min, end_min)*E
    upper[u:] = 1.
    integ = np.zeros(N); integ[u:] = 1.
    rows, cols, values, lo, hi = [], [], [], [], []
    def add(entries, lb, ub):
        row = len(lo)
        for col, value in entries:
            rows.append(row); cols.append(col); values.append(value)
        lo.append(lb); hi.append(ub)
    for t in range(T):
        add([(e+t+1, 1.), (e+t, -1.), (c+t, -params.eta_charge*DT_HOURS),
             (d+t, DT_HOURS/params.eta_discharge)], 0., 0.)
        add([(c+t, 1.), (u+t, P)], -np.inf, P)
        add([(d+t, 1.), (a+t, 1.), (u+t, -P)], -np.inf, 0.)
        add([(a+t, rules.duration_hours/params.eta_discharge), (e+t+1, -1.)],
            -np.inf, -params.soc_min*E)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lo), N)).tocsc()
    result = milp(cost, integrality=integ, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, lo, hi),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.})
    if result.status != 0:
        raise RuntimeError(f"reservation MILP not optimal: {result.message}")
    x = result.x
    return ReservationPlan(
        actions_mw=np.column_stack([x[d:d+T] - x[c:c+T], x[a:a+T]]),
        soc_after=x[e+1:e+T+1]/E, net_proxy_usd=float(-result.fun),
        mip_gap=float(result.mip_gap),
    )
