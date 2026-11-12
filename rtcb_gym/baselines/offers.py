"""Baselines for the offer-curve action (``RTCBEnv(action_mode="offer")``).

With a power action the agent decides before it knows the interval's price.
With an offer curve it names prices, and the market dispatches it against
the price that actually clears, so a strategy can react to the current
interval without seeing it in advance.

Rolling threshold offer
    The rolling threshold rule's two thresholds, used as offer prices: sell
    as much as the battery can at or above the high threshold, buy as much
    as it can at or below the low threshold.

Offer MPC
    Solve the rolling LP on the forecast as Forecast + MPC does. Offer to
    sell at the lowest forecast price at which the plan sells, and bid to
    buy at the highest forecast price at which it buys: the plan's marginal
    prices. If the interval clears beyond them, the battery trades now.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rtcb_gym.baselines.forecast_mpc import ForecastMPC
from rtcb_gym.baselines.optimize import solve_dispatch
from rtcb_gym.baselines.threshold_rule import RollingThresholdRule
from rtcb_gym.battery import BatteryParams, feasible_power_range, terminal_floor
from rtcb_gym.env import OFFER_CAP, OFFER_FLOOR

PLAN_TOL_MW = 1e-6


def offer(params: BatteryParams, lo: float, hi: float, sell_price: float, buy_price: float) -> np.ndarray:
    """An offer curve for the feasible power range ``[lo, hi]`` MW: sell
    ``hi`` at or above ``sell_price``, buy ``-lo`` at or below ``buy_price``.
    If the end-of-episode target forces charging (``hi < 0``), buy that at
    any price."""
    p = params.power_mw
    if hi < 0:
        return np.array([0.0, OFFER_CAP, -hi / p, OFFER_CAP])
    sell_price = min(max(sell_price, OFFER_FLOOR + 0.01), OFFER_CAP)
    buy_price = min(max(buy_price, OFFER_FLOOR), sell_price, OFFER_CAP - 0.01)
    return np.array([hi / p, sell_price, -lo / p, buy_price])


class RollingThresholdOffer(RollingThresholdRule):
    """:class:`RollingThresholdRule` with its thresholds as offer prices."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = f"Rolling threshold offer ({self.window_days} d)"

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        soc = float(obs[self.layout["soc"]][0])
        lo, hi = feasible_power_range(self.params, soc, self.dt_hours, int(info["steps_remaining"]))
        low, high = self._rule.low_price, self._rule.high_price
        if not np.isfinite(low):  # not enough history: idle
            lo, hi = min(max(0.0, lo), hi), min(max(0.0, lo), hi)
            low, high = OFFER_FLOOR, OFFER_CAP
        return offer(self.params, lo, hi, high, low)


class OfferMPC(ForecastMPC):
    name = "Offer MPC"

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        steps_left = int(info["steps_remaining"])
        horizon = self.horizon(obs, steps_left)
        forecast = self.forecast(obs, horizon)
        soc = float(obs[self.layout["soc"]][0])
        end_min = terminal_floor(self.params, steps_left - horizon, self.dt_hours)
        plan = solve_dispatch(forecast, self.params, soc, self.dt_hours, soc_end_min=end_min).power_mw
        lo, hi = feasible_power_range(self.params, soc, self.dt_hours, steps_left)
        sells, buys = forecast[plan > PLAN_TOL_MW], forecast[plan < -PLAN_TOL_MW]
        if not sells.size:  # the plan never sells: do not sell now either
            hi = min(hi, max(0.0, lo)) if hi >= 0 else hi
        if not buys.size:
            lo = max(lo, min(0.0, hi))
        return offer(self.params, lo, hi,
                     sells.min() if sells.size else OFFER_CAP, buys.max() if buys.size else OFFER_FLOOR)
