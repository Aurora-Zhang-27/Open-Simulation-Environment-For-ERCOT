"""Common interface for strategies scored in :class:`rtcb_gym.env.RTCBEnv`.

Any strategy (rule, optimiser, RL agent, LLM agent) plugs in the same way:
``reset`` is called with the first observation and info of an episode, then
``act`` once per interval with the latest observation and info.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    name = "policy"

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Start of an episode. Default: nothing to do."""

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Return a normalised action in [-1, 1], shape (1,)."""
        raise NotImplementedError
