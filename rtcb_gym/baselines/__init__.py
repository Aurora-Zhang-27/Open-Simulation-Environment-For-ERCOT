"""Baseline strategies."""

from rtcb_gym.baselines.base import Policy
from rtcb_gym.baselines.forecast_mpc import ForecastMPC, persistence_24h
from rtcb_gym.baselines.optimize import DispatchPlan, solve_dispatch
from rtcb_gym.baselines.perfect_foresight import PerfectForesight
from rtcb_gym.baselines.threshold_rule import (
    RollingThresholdRule,
    ThresholdRule,
    fit_rolling_threshold_rule,
    fit_threshold_rule,
)

__all__ = [
    "DispatchPlan",
    "ForecastMPC",
    "PerfectForesight",
    "Policy",
    "RollingThresholdRule",
    "ThresholdRule",
    "fit_rolling_threshold_rule",
    "fit_threshold_rule",
    "persistence_24h",
    "solve_dispatch",
]
