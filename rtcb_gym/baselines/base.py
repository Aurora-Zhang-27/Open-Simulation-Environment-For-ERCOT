"""Common interface for strategies scored in :class:`rtcb_gym.env.RTCBEnv`.

Any strategy (rule, optimiser, RL agent, LLM agent) plugs in the same way:
``reset`` is called with the first observation and info of an episode, then
``act`` once per interval with the latest observation and info.

In a two-settlement environment, ``day_ahead`` is called first (once per
operating day of the episode), with what is known at the day-ahead decision
(:class:`rtcb_gym.day_ahead.DayAheadContext`), and returns the hourly
schedule in MW, a :class:`rtcb_gym.day_ahead.DayAheadBid` (limit prices, an
ECRS offer), or None for no day-ahead position.

The shape of the real-time action depends on the environment
(``env.action_layout``): a power, or an offer curve, plus an ECRS request
if the environment has ECRS. :func:`env_action` builds it from a power.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def env_action(env: Any, energy: float, ecrs: float = 0.0) -> np.ndarray:
    """The action that asks ``env`` for ``energy`` (share of power_mw in
    [-1, 1], + discharge) and ``ecrs`` (share in [0, 1]) at any price.

    In an offer-curve environment this is a price-taking curve: sell at the
    offer floor or buy at the offer cap, which always clear.
    """
    from rtcb_gym.env import OFFER_CAP, OFFER_FLOOR

    energy = min(max(float(energy), -1.0), 1.0)
    lay = env.action_layout
    a = np.zeros(len(lay))
    if "energy" in lay:
        a[lay["energy"]] = energy
    else:
        a[lay["sell_price"]] = a[lay["buy_price"]] = OFFER_FLOOR
        if energy >= 0:
            a[lay["sell"]] = energy
        else:
            a[lay["sell_price"]] = a[lay["buy_price"]] = OFFER_CAP
            a[lay["buy"]] = -energy
    if "ecrs" in lay:
        a[lay["ecrs"]] = min(max(float(ecrs), 0.0), 1.0)
    return a


class Policy:
    name = "policy"

    def day_ahead(self, context: Any) -> np.ndarray | None:
        """Hourly day-ahead schedule in MW (+ sell, - buy), or a DayAheadBid. Default: none."""
        return None

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Start of an episode. Default: nothing to do."""

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Return the action for this interval (see ``env.action_layout``);
        in the default environment one number in [-1, 1]."""
        raise NotImplementedError
