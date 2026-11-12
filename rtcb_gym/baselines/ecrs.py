"""Baselines for environments with ECRS (``RTCBEnv(ecrs_prices=...)``).

Perfect foresight (energy + ECRS)
    The upper bound, for any combination of the environment's options. In
    real time it solves the reservation MILP (:mod:`rtcb_gym.ancillary`) on
    the true settlement prices and deployment of the whole episode. With a
    day-ahead market the revenue splits as for energy alone:

        sum_h energy award_h * (DA price_h - mean RT price in h)
      + sum_h ECRS award_h   * (DA ECRS price_h - mean RT ECRS price in h)
      + the real-time part, which does not depend on the awards

    so the awards are solved separately, on the true spreads, as an hourly
    MILP over exactly the set of awards the environment accepts.

ECRS MPC
    Rolling MILP on a forecast of both prices: persistence (the same
    interval one day earlier) over a short horizon, or, in a two-settlement
    environment, the day's day-ahead prices to the end of what is published
    (``forecast="day_ahead"``). ``energy_only=True`` plans with a zero ECRS
    price: the same strategy without the product.

DA schedule, energy + ECRS
    Forecast tomorrow's day-ahead energy and ECRS prices with today's,
    solve the hourly joint schedule, bid it, and in real time deliver the
    energy award and offer the ECRS award.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rtcb_gym.ancillary import solve_reservation
from rtcb_gym.baselines.base import Policy, env_action
from rtcb_gym.baselines.optimize import solve_dispatch
from rtcb_gym.battery import terminal_floor
from rtcb_gym.data.clean import DA_INTERVAL
from rtcb_gym.day_ahead import DA_DT_HOURS, DayAheadBid, DayAheadContext, day_hours_utc, persistence_da_forecast

MPC_TIE_BREAK_USD_PER_MWH = 1e-3


class PerfectForesightMarket(Policy):
    """Privileged: reads the true prices of ``env`` itself.

    ``day_ahead=False`` takes no day-ahead position and ``use_ecrs=False``
    never offers ECRS: the bounds for strategies without that market or
    product, in the same environment.
    """

    name = "Perfect foresight"

    def __init__(self, env: Any, time_limit: float = 600.0, day_ahead: bool = True, use_ecrs: bool = True) -> None:
        self.env = env
        self.time_limit = time_limit
        self.use_da = day_ahead and env.two_settlement
        self.use_ecrs = use_ecrs and env.ecrs
        self._plan = np.zeros((0, 2))
        self._k = 0
        self.planned_revenue_usd = 0.0  # real-time objectives
        self.planned_da_usd = 0.0  # day-ahead (spread) objectives

    def day_ahead(self, context: DayAheadContext) -> DayAheadBid | None:
        env = self.env
        if not self.use_da:
            return None
        i0, i1 = env._day_rows[context.operating_day]
        hours = day_hours_utc(context.operating_day, env.tz)
        hour_of_row = (env._ts_ns[i0:i1] - hours.asi8[0]) // DA_INTERVAL.value
        first = np.searchsorted(hour_of_row, np.arange(len(hours)))
        per_hour = np.bincount(hour_of_row, minlength=len(hours))

        def spread(da_rt: np.ndarray, settle: np.ndarray) -> np.ndarray:
            return da_rt[i0:i1][first] - np.bincount(hour_of_row, settle[i0:i1], len(hours)) / per_hour

        financial = replace(env.battery_params, degradation_cost_usd_per_mwh=0.0)
        energy = spread(env._da_price_rt, env._settle)
        if not (env.da_ecrs and self.use_ecrs):
            plan = solve_dispatch(energy, financial, context.soc_init, DA_DT_HOURS)
            self.planned_da_usd += plan.revenue_usd
            return DayAheadBid(energy_mw=plan.power_mw)
        both = np.column_stack([energy, spread(env._da_ecrs_rt, env._ecrs_settle)])
        plan = solve_reservation(both, financial, env.ecrs_rules, context.soc_init,
                                 dt_hours=DA_DT_HOURS, time_limit=self.time_limit)
        self.planned_da_usd += plan.net_proxy_usd
        return DayAheadBid(energy_mw=plan.actions_mw[:, 0], ecrs_mw=np.maximum(plan.actions_mw[:, 1], 0.0))

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        env = self.env
        t0, n = env._t, int(info["steps_remaining"])
        rows = slice(t0, t0 + n)
        if self.use_ecrs:
            prices = np.column_stack([env._settle[rows], env._ecrs_settle[rows]])
            deployment = None if env._deploy is None else env._deploy[rows]
            plan = solve_reservation(prices, env.battery_params, env.ecrs_rules, info["soc"],
                                     dt_hours=env.dt_hours, deployment=deployment, time_limit=self.time_limit)
            self._plan = plan.actions_mw
            self.planned_revenue_usd += plan.net_proxy_usd
        else:
            plan = solve_dispatch(env._settle[rows], env.battery_params, info["soc"], env.dt_hours)
            self._plan = np.column_stack([plan.power_mw, np.zeros(n)])
            self.planned_revenue_usd += plan.revenue_usd
        self._k = 0

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        energy, ecrs = self._plan[self._k] / self.env.battery_params.power_mw
        self._k += 1
        return env_action(self.env, energy, ecrs)


class ECRSMPC(Policy):
    """Rolling MILP on a price forecast, re-solved every ``replan_every``
    steps (the plan is followed in between).

    ``forecast="persistence"`` needs ``history_len`` of at least one day of
    steps; ``forecast="day_ahead"`` needs a two-settlement environment and
    plans as far as day-ahead prices are published, at most ``horizon``.
    """

    def __init__(self, env: Any, horizon: int = 12, energy_only: bool = False, forecast: str = "persistence",
                 replan_every: int = 1, tie_break_usd_per_mwh: float = MPC_TIE_BREAK_USD_PER_MWH,
                 time_limit: float = 60.0) -> None:
        if forecast not in ("persistence", "day_ahead"):
            raise ValueError("forecast must be 'persistence' or 'day_ahead'")
        self.env = env
        self.horizon = int(horizon)
        self.energy_only = energy_only
        self.source = forecast
        self.replan_every = int(replan_every)
        self.tie_break = tie_break_usd_per_mwh
        self.time_limit = time_limit
        self.name = ("Energy MPC" if energy_only else "ECRS MPC") + (" on DA prices" if forecast == "day_ahead" else "")
        per_day = env._steps_per_day
        if self.replan_every < 1 or self.horizon < self.replan_every:
            raise ValueError("need 1 <= replan_every <= horizon")
        if forecast == "day_ahead":
            if not env.two_settlement or (not energy_only and not env.da_ecrs):
                raise ValueError("forecast='day_ahead' needs day-ahead prices (and day-ahead ECRS prices)")
        elif env.history_len < per_day or self.horizon > per_day - env.obs_lag_steps:
            raise ValueError("persistence needs history_len >= one day and horizon <= one day - obs_lag_steps")
        # persistence forecast of step t+j: the price at t+j-1 day, inside the visible window
        self._first = env.history_len + env.obs_lag_steps - per_day
        self._plan = np.zeros((0, 2))
        self._k = 0

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        self._plan, self._k = np.zeros((0, 2)), 0

    def forecast(self, obs: np.ndarray, n: int) -> np.ndarray:
        """Prices of steps t .. t+m-1 (m <= n), from the observation only."""
        lay = self.env.obs_layout
        if self.source == "day_ahead":
            n = min(n, int(obs[lay["da_ahead_mask"]].sum()))
            energy = obs[lay["da_price_ahead"]][:n]
            ecrs = np.zeros(n) if self.energy_only else obs[lay["da_ecrs_price_ahead"]][:n]
        else:
            rows = slice(self._first, self._first + n)
            energy = obs[lay["prices"]][rows]
            ecrs = np.zeros(n) if self.energy_only else obs[lay["ecrs_prices"]][rows]
        return np.column_stack([energy, ecrs])

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        env, p = self.env, self.env.battery_params
        if self._k >= min(self.replan_every, len(self._plan)):
            steps_left = int(info["steps_remaining"])
            forecast = self.forecast(obs, min(self.horizon, steps_left))
            floor = terminal_floor(p, steps_left - len(forecast), env.dt_hours)
            self._plan = solve_reservation(
                forecast, p, env.ecrs_rules, float(obs[env.obs_layout["soc"]][0]), soc_end_min=floor,
                tie_break_usd_per_mwh=self.tie_break, dt_hours=env.dt_hours, time_limit=self.time_limit,
            ).actions_mw
            self._k = 0
        energy, ecrs = self._plan[self._k] / p.power_mw
        self._k += 1
        return env_action(env, energy, 0.0 if self.energy_only else ecrs)


class DayAheadEnergyECRS(Policy):
    name = "DA schedule, energy + ECRS"

    def __init__(self, env: Any) -> None:
        if not env.da_ecrs:
            raise ValueError("DayAheadEnergyECRS needs day-ahead ECRS prices")
        self.env = env
        self.n_days_without_forecast = 0

    def day_ahead(self, context: DayAheadContext) -> DayAheadBid | None:
        try:
            energy = persistence_da_forecast(context)
            ecrs = persistence_da_forecast(replace(context, da_history=context.da_ecrs_history))
        except ValueError:  # no day-ahead prices for the day before
            self.n_days_without_forecast += 1
            return None
        plan = solve_reservation(np.column_stack([energy, ecrs]), context.battery, self.env.ecrs_rules,
                                 context.soc_init, dt_hours=DA_DT_HOURS,
                                 tie_break_usd_per_mwh=MPC_TIE_BREAK_USD_PER_MWH)
        return DayAheadBid(energy_mw=plan.actions_mw[:, 0], ecrs_mw=np.maximum(plan.actions_mw[:, 1], 0.0))

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        lay = self.env.obs_layout
        return env_action(self.env, obs[lay["da_award_ahead"]][0], obs[lay["da_ecrs_award_ahead"]][0])
