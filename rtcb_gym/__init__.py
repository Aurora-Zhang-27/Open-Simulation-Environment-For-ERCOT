"""RTCB-Gym: battery-storage bidding environment for post-RTC+B ERCOT,
real-time only or day-ahead + real time (two-settlement).

Sign convention everywhere: positive power = discharge (sell),
negative power = charge (buy).
"""

from rtcb_gym.battery import Battery, BatteryParams, battery_step
from rtcb_gym.day_ahead import DayAheadContext
from rtcb_gym.env import RTCBEnv
from rtcb_gym.settlement import Settlement, settle_real_time, settle_two_settlement

__version__ = "0.2.0"

__all__ = [
    "Battery",
    "BatteryParams",
    "DayAheadContext",
    "RTCBEnv",
    "Settlement",
    "battery_step",
    "settle_real_time",
    "settle_two_settlement",
]
