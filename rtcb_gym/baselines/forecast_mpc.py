"""Forecast + model predictive control (rolling-horizon LP).

At every interval: forecast prices from now to the end of the horizon, solve
the dispatch LP from the current SoC (with the episode's end-SoC target),
apply only the first action, repeat.

The forecaster only sees the observation, so it cannot look ahead. Better
forecasters plug in through the ``forecaster`` argument:

    forecaster(past_prices: np.ndarray, horizon: int) -> np.ndarray

v0 forecaster: persistence, same interval 24 hours (96 intervals) earlier.

:class:`LevelCorrectedMPC` adds one thing to persistence: if the last hour
settled above (below) the same hour yesterday, the forecast for the next
intervals is raised (lowered) by that difference, fading with the horizon.
Its two parameters are fitted on the training split only
(:func:`fit_level_correction`).
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

    def horizon(self, obs: np.ndarray, steps_left: int) -> int:
        """How many intervals ahead to plan."""
        return min(steps_left, self.max_horizon)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        steps_left = int(info["steps_remaining"])
        horizon = self.horizon(obs, steps_left)
        forecast = self.forecast(obs, horizon)
        soc = float(obs[self.layout["soc"]][0])
        # If the horizon stops before the episode ends, only require the SoC
        # from which the end target is still reachable.
        end_min = terminal_floor(self.params, steps_left - horizon, self.dt_hours)
        plan = solve_dispatch(forecast, self.params, soc, self.dt_hours, soc_end_min=end_min)
        a = plan.power_mw[0] / self.params.power_mw
        return np.array([min(max(a, -1.0), 1.0)])


class LevelCorrectedMPC(ForecastMPC):
    """Persistence plus the recent level error, fading with the horizon:

        forecast(t+j) = price(t+j-96) + rho**(j+1) * (mean of the last k prices
                                                      - mean of the same k intervals yesterday)

    The policy keeps the prices it has seen since the start of the episode
    (all from observations), because the 96-price window no longer holds
    yesterday's counterpart of the last hour.
    """

    name = "Level-corrected forecast + MPC"

    def __init__(self, params: BatteryParams, obs_layout: dict[str, slice], dt_hours: float = 0.25,
                 k: int = 4, rho: float = 0.9, max_horizon: int = LAG_24H) -> None:
        super().__init__(params, obs_layout, dt_hours, max_horizon=max_horizon)
        if k < 1 or not 0 <= rho <= 1:
            raise ValueError("need k >= 1 and 0 <= rho <= 1")
        self.k, self.rho = int(k), float(rho)
        self._seen: list[float] = []
        self._fresh = True

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        self._seen = list(obs[self.layout["prices"]])
        self._fresh = True

    def forecast(self, obs: np.ndarray, horizon: int) -> np.ndarray:
        base = persistence_24h(obs[self.layout["prices"]], horizon)
        return base + level_correction(np.asarray(self._seen), self.k, self.rho, horizon)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        if not self._fresh:  # one more interval has settled since the last call
            self._seen.append(float(obs[self.layout["prices"]][-1]))
        self._fresh = False
        return super().act(obs, info)


def level_correction(seen: np.ndarray, k: int, rho: float, horizon: int) -> np.ndarray:
    """``rho**(j+1) * (mean(last k) - mean(the same k one day earlier))``,
    j = 0..horizon-1; zero until ``seen`` holds 96 + k prices."""
    if len(seen) < LAG_24H + k:
        return np.zeros(horizon)
    delta = seen[-k:].mean() - seen[-k - LAG_24H : -LAG_24H].mean()
    return delta * rho ** np.arange(1, horizon + 1)


def fit_level_correction(
    train_env: Any, ks: tuple[int, ...] = (2, 4, 8), rhos: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98)
) -> tuple[int, float, list[dict[str, float]]]:
    """Pick (k, rho) with the lowest mean absolute forecast error on the
    training env's days, forecasting from every fourth interval to the end
    of the day, as the MPC does. ``rho = 0`` is plain persistence."""
    price = train_env._price
    table = []
    for k in ks:
        for rho in rhos:
            err = n = 0.0
            for day in train_env.days:
                i0, i1 = train_env._day_rows[day]
                for j in range(k, i1 - i0, 4):
                    t = i0 + j
                    h = min(i1 - t, LAG_24H)
                    forecast = price[t - LAG_24H : t - LAG_24H + h] + level_correction(price[t - LAG_24H - k : t], k, rho, h)
                    err += np.abs(forecast - price[t : t + h]).sum()
                    n += h
            table.append({"k": k, "rho": rho, "train_mae_usd_per_mwh": err / n})
    best = min(table, key=lambda r: r["train_mae_usd_per_mwh"])
    return int(best["k"]), float(best["rho"]), table
