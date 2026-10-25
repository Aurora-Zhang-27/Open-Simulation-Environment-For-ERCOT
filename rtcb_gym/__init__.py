"""RTCB-Gym: battery-storage bidding environment for post-RTC+B ERCOT.

Sign convention everywhere: positive power = discharge (sell),
negative power = charge (buy).
"""

from rtcb_gym.battery import Battery, BatteryParams, battery_step
from rtcb_gym.env import RTCBEnv
from rtcb_gym.settlement import Settlement, settle_real_time

__version__ = "0.1.0"

__all__ = [
    "Battery",
    "BatteryParams",
    "RTCBEnv",
    "Settlement",
    "battery_step",
    "settle_real_time",
]
