"""Five-minute energy + ECRS reservation research model.

This is a conservative, discharge-side-only, price-taking subset. ECRS is
awarded in full and never deployed. Five-minute energy LMP cash flows are
proxies, not ERCOT's complete 15-minute settlement. See docs/ecrs_gate.md.

The reservation rule, its feasibility step and the MILP live in
:mod:`rtcb_gym.ancillary`, shared with :class:`rtcb_gym.env.RTCBEnv`; this
module keeps the strict five-minute tape and the continuous-panel
environment used by the ECRS gate (scripts/run_ecrs_gate.py).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from rtcb_gym.ancillary import (  # noqa: F401  (re-exported)
    TOL_MW, ECRSRules, ReservationPlan, ecrs_capacity_mw, reservation_revenue,
    reservation_step, solve_reservation,
)
from rtcb_gym.battery import BatteryParams, terminal_floor

DT_HOURS = 5 / 60
STEP = pd.Timedelta(minutes=5)


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
