"""Threshold rule.

Signal: the last settled real-time price (the newest price in the
observation). Charge at the largest feasible power when the signal is at or
below a low threshold, discharge at the largest feasible power when it is at
or above a high threshold, otherwise stay idle. Requests always stay within
what the battery and the end-of-episode SoC target allow.

Thresholds are quantiles of training-split prices. The quantile pair is
chosen by training-split revenue only (:func:`fit_threshold_rule`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rtcb_gym.baselines.base import Policy
from rtcb_gym.battery import BatteryParams, feasible_power_range


class ThresholdRule(Policy):
    name = "Threshold rule"

    def __init__(
        self,
        low_price: float,
        high_price: float,
        params: BatteryParams,
        obs_layout: dict[str, slice],
        dt_hours: float = 0.25,
    ) -> None:
        if low_price > high_price:
            raise ValueError("low_price must be <= high_price")
        self.low_price = float(low_price)
        self.high_price = float(high_price)
        self.params = params
        self.layout = obs_layout
        self.dt_hours = dt_hours

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        signal = obs[self.layout["prices"]][-1]
        soc = float(obs[self.layout["soc"]][0])
        steps_left = int(info["steps_remaining"])
        lo, hi = feasible_power_range(self.params, soc, self.dt_hours, steps_left)
        if signal <= self.low_price:
            power = lo
        elif signal >= self.high_price:
            power = hi
        else:
            power = min(max(0.0, lo), hi)  # idle, unless the end-SoC target forces charging
        return np.array([power / self.params.power_mw])


def fit_threshold_rule(
    train_env,
    train_prices: np.ndarray,
    low_quantiles: Sequence[float],
    high_quantiles: Sequence[float],
) -> tuple[ThresholdRule, list[dict[str, float]]]:
    """Pick the quantile pair with the highest revenue on the training split.

    ``train_prices`` must contain training-split prices only. Returns the
    fitted rule and the full search table.
    """
    from rtcb_gym.evaluation import run_policy

    prices = np.asarray(train_prices, dtype=np.float64)
    table = []
    best = None
    for ql in low_quantiles:
        for qh in high_quantiles:
            if not ql < qh:
                continue
            low, high = np.quantile(prices, [ql, qh])
            rule = ThresholdRule(low, high, train_env.battery_params, train_env.obs_layout, train_env.dt_hours)
            revenue = run_policy(train_env, rule).revenue_usd
            table.append({"low_q": ql, "high_q": qh, "low_price": low, "high_price": high, "train_revenue_usd": revenue})
            if best is None or revenue > best[0]:
                best = (revenue, rule)
    if best is None:
        raise ValueError("no valid (low_q < high_q) quantile pair")
    return best[1], table
