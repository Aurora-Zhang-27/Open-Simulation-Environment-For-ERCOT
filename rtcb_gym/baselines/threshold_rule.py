"""Threshold rule.

Signal: the last settled real-time price (the newest price in the
observation). Charge at the largest feasible power when the signal is at or
below a low threshold, discharge at the largest feasible power when it is at
or above a high threshold, otherwise stay idle. Requests always stay within
what the battery and the end-of-episode SoC target allow.

Thresholds are quantiles of training-split prices. The quantile pair is
chosen by training-split revenue only (:func:`fit_threshold_rule`).

:class:`RollingThresholdRule` is the same rule with thresholds refitted
before every episode on the prices of the previous ``window_days`` days, so
it follows a shifting price distribution. Its quantile pair is also chosen
on the training split only (:func:`fit_rolling_threshold_rule`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from rtcb_gym.baselines.base import Policy
from rtcb_gym.battery import BatteryParams, feasible_power_range
from rtcb_gym.data.clean import ERCOT_TZ, operating_day_bounds_utc, to_date


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


class RollingThresholdRule(Policy):
    """Threshold rule refitted before every episode on recent prices.

    At the start of each episode it takes the settled real-time prices of
    the ``window_days`` operating days before the episode's first day and
    sets the thresholds to their ``low_q`` and ``high_q`` quantiles. Within
    the episode it acts exactly like :class:`ThresholdRule`.

    Only prices that settled before the episode starts are read (from the
    rule's own copy of the price table), so there is no look-ahead. If the
    window holds fewer than ``min_window_days`` operating days of prices,
    which can only happen at the very start of the data, the rule stays
    idle for that episode.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        low_q: float,
        high_q: float,
        params: BatteryParams,
        obs_layout: dict[str, slice],
        dt_hours: float = 0.25,
        window_days: int = 30,
        min_window_days: int = 7,
        tz: str = ERCOT_TZ,
    ) -> None:
        if not 0 <= low_q < high_q <= 1:
            raise ValueError(f"need 0 <= low_q < high_q <= 1, got {low_q}, {high_q}")
        if not 1 <= min_window_days <= window_days:
            raise ValueError("need 1 <= min_window_days <= window_days")
        df = prices.sort_values("interval_start_utc")
        ts = pd.DatetimeIndex(df["interval_start_utc"]).as_unit("ns")
        self._ts_ns = ts.asi8
        self._price = df["price_usd_per_mwh"].to_numpy(dtype=np.float64)
        self._op_day = np.array(ts.tz_convert(tz).date)
        self.low_q, self.high_q = float(low_q), float(high_q)
        self.params, self.layout, self.dt_hours = params, obs_layout, dt_hours
        self.window_days, self.min_window_days, self.tz = int(window_days), int(min_window_days), tz
        self.name = f"Rolling threshold ({self.window_days} d)"
        self.history: list[dict[str, Any]] = []  # thresholds used, one entry per episode
        self._rule = ThresholdRule(-np.inf, np.inf, params, obs_layout, dt_hours)

    def window_prices(self, day: dt.date) -> np.ndarray:
        """Settled prices of the ``window_days`` operating days before ``day``."""
        lo, _ = operating_day_bounds_utc(day - dt.timedelta(days=self.window_days), self.tz)
        hi, _ = operating_day_bounds_utc(day, self.tz)
        i0 = int(np.searchsorted(self._ts_ns, lo.value, side="left"))
        i1 = int(np.searchsorted(self._ts_ns, hi.value, side="left"))
        n_days = len(set(self._op_day[i0:i1]))
        return self._price[i0:i1] if n_days >= self.min_window_days else self._price[:0]

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        day = to_date(info["operating_day"])
        window = self.window_prices(day)
        if window.size:
            low, high = np.quantile(window, [self.low_q, self.high_q])
        else:  # not enough history: stay idle
            low, high = -np.inf, np.inf
        self._rule = ThresholdRule(low, high, self.params, self.layout, self.dt_hours)
        self.history.append({"day": str(day), "low_price": float(low), "high_price": float(high)})

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return self._rule.act(obs, info)


def fit_rolling_threshold_rule(
    train_env,
    prices: pd.DataFrame,
    low_quantiles: Sequence[float],
    high_quantiles: Sequence[float],
    window_days: int = 30,
    min_window_days: int = 7,
) -> tuple[RollingThresholdRule, list[dict[str, float]]]:
    """Pick the quantile pair with the highest revenue on the training split.

    Each candidate runs on ``train_env``; its thresholds still come only from
    the prices before each training day. Returns a fresh rule with the best
    pair and the full search table.
    """
    from rtcb_gym.evaluation import run_policy

    def make(ql: float, qh: float) -> RollingThresholdRule:
        return RollingThresholdRule(
            prices, ql, qh, train_env.battery_params, train_env.obs_layout, train_env.dt_hours,
            window_days=window_days, min_window_days=min_window_days, tz=train_env.tz,
        )

    table = []
    best = None
    for ql in low_quantiles:
        for qh in high_quantiles:
            if not ql < qh:
                continue
            revenue = run_policy(train_env, make(ql, qh)).revenue_usd
            table.append({"low_q": ql, "high_q": qh, "train_revenue_usd": revenue})
            if best is None or revenue > best[0]:
                best = (revenue, ql, qh)
    if best is None:
        raise ValueError("no valid (low_q < high_q) quantile pair")
    return make(best[1], best[2]), table
