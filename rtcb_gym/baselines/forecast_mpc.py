"""Forecast + model predictive control (rolling-horizon LP).

At every interval: forecast prices from now to the end of the horizon, solve
the dispatch LP from the current SoC (with the episode's end-SoC target),
apply only the first action, repeat.

The forecaster only sees the observation, so it cannot look ahead. Better
forecasters plug in through the ``forecaster`` argument:

    forecaster(past_prices: np.ndarray, horizon: int) -> np.ndarray

v0 forecaster: persistence, same interval 24 hours (96 intervals) earlier.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rtcb_gym.baselines.base import Policy
from rtcb_gym.baselines.optimize import solve_dispatch
from rtcb_gym.battery import BatteryParams, terminal_floor

Forecaster = Callable[[np.ndarray, int], np.ndarray]
LAG_24H = 96  # 15-minute intervals in 24 hours


def persistence_24h(past_prices: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast for interval t+j is the price at t+j-96, j = 0..horizon-1.

    ``past_prices`` are the settled prices t-N .. t-1 (oldest first), so
    t+j-96 is visible only for j < 96.
    """
    n = len(past_prices)
    if n < LAG_24H:
        raise ValueError(f"persistence_24h needs at least {LAG_24H} past prices, got {n}")
    if not 1 <= horizon <= LAG_24H:
        raise ValueError(f"horizon must be in [1, {LAG_24H}], got {horizon}")
    return np.asarray(past_prices[n - LAG_24H : n - LAG_24H + horizon], dtype=np.float64)


class ForecastMPC(Policy):
    name = "Forecast + MPC"

    def __init__(
        self,
        params: BatteryParams,
        obs_layout: dict[str, slice],
        dt_hours: float = 0.25,
        forecaster: Forecaster = persistence_24h,
        max_horizon: int = LAG_24H,
    ) -> None:
        self.params = params
        self.layout = obs_layout
        self.dt_hours = dt_hours
        self.forecaster = forecaster
        self.max_horizon = max_horizon

    def forecast(self, obs: np.ndarray, horizon: int) -> np.ndarray:
        """Prices for intervals t .. t+horizon-1, from the observation only."""
        return self.forecaster(obs[self.layout["prices"]], horizon)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        steps_left = int(info["steps_remaining"])
        horizon = min(steps_left, self.max_horizon)
        forecast = self.forecast(obs, horizon)
        soc = float(obs[self.layout["soc"]][0])
        # If the horizon stops before the episode ends, only require the SoC
        # from which the end target is still reachable.
        end_min = terminal_floor(self.params, steps_left - horizon, self.dt_hours)
        plan = solve_dispatch(forecast, self.params, soc, self.dt_hours, soc_end_min=end_min)
        a = plan.power_mw[0] / self.params.power_mw
        return np.array([min(max(a, -1.0), 1.0)])
