"""Baseline strategies."""

from rtcb_gym.baselines.base import Policy, env_action
from rtcb_gym.baselines.day_ahead import (
    DayAheadForecastMPC,
    DayAheadPlusMPC,
    DayAheadSchedule,
    PerfectForesightDayAhead,
    PerfectForesightTwoSettlement,
)
from rtcb_gym.baselines.ecrs import DayAheadEnergyECRS, ECRSMPC, PerfectForesightMarket
from rtcb_gym.baselines.forecast_mpc import ForecastMPC, LevelCorrectedMPC, fit_level_correction, persistence_24h
from rtcb_gym.baselines.offers import OfferMPC, RollingThresholdOffer
from rtcb_gym.baselines.optimize import DispatchPlan, solve_dispatch
from rtcb_gym.baselines.perfect_foresight import PerfectForesight
from rtcb_gym.baselines.threshold_rule import (
    RollingThresholdRule,
    ThresholdRule,
    fit_rolling_threshold_rule,
    fit_threshold_rule,
)

__all__ = [
    "DayAheadEnergyECRS",
    "DayAheadForecastMPC",
    "DayAheadPlusMPC",
    "DayAheadSchedule",
    "DispatchPlan",
    "ECRSMPC",
    "ForecastMPC",
    "LevelCorrectedMPC",
    "OfferMPC",
    "PerfectForesight",
    "PerfectForesightDayAhead",
    "PerfectForesightMarket",
    "PerfectForesightTwoSettlement",
    "Policy",
    "RollingThresholdOffer",
    "RollingThresholdRule",
    "ThresholdRule",
    "fit_rolling_threshold_rule",
    "env_action",
    "fit_threshold_rule",
    "fit_level_correction",
    "persistence_24h",
    "solve_dispatch",
]
