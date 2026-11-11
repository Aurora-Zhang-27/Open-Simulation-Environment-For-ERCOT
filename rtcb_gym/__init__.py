"""RTCB-Gym: battery-storage bidding environment for post-RTC+B ERCOT,
real-time only or day-ahead + real time (two-settlement), energy alone or
with ECRS.

Sign convention everywhere: positive power = discharge (sell),
negative power = charge (buy).

Also available as ``gymnasium.make("RTCBGym-v0", prices=..., ...)``, with the
arguments of :class:`rtcb_gym.env.RTCBEnv`.
"""

import gymnasium as _gym

from rtcb_gym.ancillary import ECRSRules
from rtcb_gym.battery import Battery, BatteryParams, battery_step
from rtcb_gym.day_ahead import DayAheadBid, DayAheadContext
from rtcb_gym.env import RTCBEnv
from rtcb_gym.settlement import Settlement, settle_real_time, settle_two_settlement
from rtcb_gym.wrappers import ScaledObservation

__version__ = "0.4.0"

ENV_ID = "RTCBGym-v0"
if ENV_ID not in _gym.registry:
    _gym.register(id=ENV_ID, entry_point="rtcb_gym.env:RTCBEnv")

__all__ = [
    "Battery",
    "BatteryParams",
    "DayAheadBid",
    "DayAheadContext",
    "ECRSRules",
    "ENV_ID",
    "RTCBEnv",
    "ScaledObservation",
    "Settlement",
    "battery_step",
    "settle_real_time",
    "settle_two_settlement",
]
