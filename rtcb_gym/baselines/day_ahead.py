"""Baselines for two-settlement episodes (day-ahead + real time).

Perfect foresight (DA + RT)
    The upper bound. Revenue splits into a day-ahead part, award times the
    day-ahead / real-time spread, and a real-time part that does not depend
    on the award (see :mod:`rtcb_gym.day_ahead`). It solves both exactly:
    the award on the true spread (no degradation: the award is a financial
    position), the dispatch on the true real-time prices.

Perfect foresight (DA only, delivered)
    Solves the battery schedule on the true day-ahead prices and delivers
    the award exactly. The upper bound for strategies that only use the
    day-ahead market.

DA schedule, delivered
    Forecast tomorrow's day-ahead prices with today's (persistence), solve
    the battery schedule on that forecast, bid it, and deliver exactly the
    award in real time. The classic way a battery uses the day-ahead market.

MPC on DA prices
    No day-ahead position. In real time, run the rolling LP with the day's
    day-ahead prices as the forecast of real-time prices (they are known by
    then). Compare with Forecast + MPC, which uses yesterday's real-time
    prices.

DA schedule + MPC on DA prices
    The DA schedule above, then the MPC above in real time. Its revenue is
    the MPC's plus award times (day-ahead - real-time price).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from rtcb_gym.baselines.base import Policy
from rtcb_gym.baselines.forecast_mpc import ForecastMPC
from rtcb_gym.baselines.optimize import solve_dispatch
from rtcb_gym.baselines.perfect_foresight import PerfectForesight
from rtcb_gym.battery import BatteryParams
from rtcb_gym.data.clean import INTERVAL
from rtcb_gym.day_ahead import DA_DT_HOURS, DayAheadContext, persistence_da_forecast

DAForecaster = Callable[[DayAheadContext], np.ndarray]


class PerfectForesightTwoSettlement(PerfectForesight):
    name = "Perfect foresight (DA + RT)"

    def __init__(
        self, prices: pd.DataFrame, da_prices: pd.DataFrame, params: BatteryParams, dt_hours: float = 0.25
    ) -> None:
        super().__init__(prices, params, dt_hours)
        da = da_prices.sort_values("interval_start_utc")
        self._da = pd.Series(
            da["price_usd_per_mwh"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(da["interval_start_utc"]).as_unit("ns"),
        )
        self._rt = pd.Series(self._price, index=pd.DatetimeIndex(self._ts_ns, tz="UTC"))
        self.planned_da_usd = 0.0  # sum of day-ahead LP objectives

    def day_ahead(self, context: DayAheadContext) -> np.ndarray:
        hours = context.hour_start_utc.as_unit("ns")
        da = self._da.reindex(hours).to_numpy()
        per_hour = int(pd.Timedelta(hours=1) / INTERVAL)
        rt = self._rt.reindex(pd.DatetimeIndex(
            (hours.asi8[:, None] + INTERVAL.value * np.arange(per_hour)).ravel(), tz="UTC"
        )).to_numpy().reshape(len(hours), per_hour)
        if np.isnan(da).any() or np.isnan(rt).any():
            raise ValueError(f"true prices for {context.operating_day} not found")
        spread = da - rt.mean(axis=1)
        financial = replace(self.params, degradation_cost_usd_per_mwh=0.0)
        plan = solve_dispatch(spread, financial, context.soc_init, DA_DT_HOURS)
        self.planned_da_usd += plan.revenue_usd
        return plan.power_mw


class DayAheadSchedule(Policy):
    name = "DA schedule, delivered"

    def __init__(
        self, params: BatteryParams, obs_layout: dict[str, slice], forecaster: DAForecaster = persistence_da_forecast
    ) -> None:
        self.params = params
        self.layout = obs_layout
        self.forecaster = forecaster
        self.n_days_without_forecast = 0

    def plan(self, context: DayAheadContext) -> np.ndarray:
        try:
            forecast = self.forecaster(context)
        except ValueError:  # e.g. no day-ahead prices for the day before
            self.n_days_without_forecast += 1
            return np.zeros(context.n_hours)
        return solve_dispatch(forecast, context.battery, context.soc_init, DA_DT_HOURS).power_mw

    def day_ahead(self, context: DayAheadContext) -> np.ndarray:
        return self.plan(context)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        # deliver the award of the current interval (stored as award / power_mw)
        return np.array([obs[self.layout["da_award_ahead"]][0]])


class PerfectForesightDayAhead(DayAheadSchedule):
    name = "Perfect foresight (DA only, delivered)"

    def __init__(self, da_prices: pd.DataFrame, params: BatteryParams, obs_layout: dict[str, slice]) -> None:
        da = da_prices.sort_values("interval_start_utc")
        true = pd.Series(
            da["price_usd_per_mwh"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(da["interval_start_utc"]).as_unit("ns"),
        )

        def true_prices(context: DayAheadContext) -> np.ndarray:  # privileged: the day's own prices
            values = true.reindex(context.hour_start_utc).to_numpy()
            if np.isnan(values).any():
                raise ValueError(f"true day-ahead prices for {context.operating_day} not found")
            return values

        super().__init__(params, obs_layout, forecaster=true_prices)
        self.planned_revenue_usd = 0.0

    def plan(self, context: DayAheadContext) -> np.ndarray:
        plan = solve_dispatch(self.forecaster(context), context.battery, context.soc_init, DA_DT_HOURS)
        self.planned_revenue_usd += plan.revenue_usd
        return plan.power_mw


class DayAheadForecastMPC(ForecastMPC):
    name = "MPC on DA prices"

    def forecast(self, obs: np.ndarray, horizon: int) -> np.ndarray:
        ahead = obs[self.layout["da_price_ahead"]]
        if obs[self.layout["da_ahead_mask"]][:horizon].min() < 1.0:
            raise ValueError("horizon runs past the day-ahead prices in the observation")
        return np.asarray(ahead[:horizon], dtype=np.float64)


class DayAheadPlusMPC(Policy):
    name = "DA schedule + MPC on DA prices"

    def __init__(self, params: BatteryParams, obs_layout: dict[str, slice], dt_hours: float = 0.25) -> None:
        self.schedule = DayAheadSchedule(params, obs_layout)
        self.mpc = DayAheadForecastMPC(params, obs_layout, dt_hours)

    def day_ahead(self, context: DayAheadContext) -> np.ndarray:
        return self.schedule.plan(context)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return self.mpc.act(obs, info)
