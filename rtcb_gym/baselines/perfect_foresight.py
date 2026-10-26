"""Perfect foresight: the upper bound.

At the start of each episode it reads the TRUE prices of the whole episode
(privileged access, from its own copy of the price table, not from the env)
and solves the dispatch problem exactly. No real strategy can beat it in the
same environment, so it is the denominator of the capture rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rtcb_gym.baselines.base import Policy
from rtcb_gym.baselines.optimize import solve_dispatch
from rtcb_gym.battery import BatteryParams


class PerfectForesight(Policy):
    name = "Perfect foresight"

    def __init__(self, prices: pd.DataFrame, params: BatteryParams, dt_hours: float = 0.25) -> None:
        df = prices.sort_values("interval_start_utc")
        self._ts_ns = pd.DatetimeIndex(df["interval_start_utc"]).as_unit("ns").asi8
        self._price = df["price_usd_per_mwh"].to_numpy(dtype=np.float64)
        self.params = params
        self.dt_hours = dt_hours
        self._plan = np.zeros(0)
        self._k = 0
        self.planned_revenue_usd = 0.0  # sum of real-time LP objectives, for checking the env replay

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        start_ns = pd.Timestamp(info["interval_start_utc"]).as_unit("ns").value
        i0 = int(np.searchsorted(self._ts_ns, start_ns))
        n = int(info["steps_remaining"])
        if i0 >= len(self._ts_ns) or self._ts_ns[i0] != start_ns or i0 + n > len(self._price):
            raise ValueError("episode prices not found in the perfect-foresight price table")
        plan = solve_dispatch(self._price[i0 : i0 + n], self.params, info["soc"], self.dt_hours)
        self._plan = plan.power_mw
        self._k = 0
        self.planned_revenue_usd += plan.revenue_usd

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        a = self._plan[self._k] / self.params.power_mw
        self._k += 1
        return np.array([min(max(a, -1.0), 1.0)])
