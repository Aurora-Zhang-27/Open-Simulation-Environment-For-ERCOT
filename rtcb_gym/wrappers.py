"""Observation scaling for learning agents.

:class:`rtcb_gym.env.RTCBEnv` returns prices in USD/MWh, unbounded, which
is what rules and optimisers want and what most neural networks do not.
:class:`ScaledObservation` leaves the environment and its rewards alone and
only rescales the price blocks of the observation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

PRICE_BLOCKS = ("prices", "da_price_ahead", "ecrs_prices", "da_ecrs_price_ahead")
MAX_ABS_PRICE = 100_000.0  # far beyond any ERCOT price; only sets the declared bounds


class ScaledObservation(gym.ObservationWrapper):
    """Price blocks become ``sign(p) * log1p(|p| / price_scale)``, float32.

    The map is monotone and keeps the sign, is close to ``p / price_scale``
    for everyday prices and compresses spikes (with the default scale of
    25 USD/MWh: 25 -> 0.69, 1,000 -> 3.7, 5,000 -> 5.3, -250 -> -2.4).
    Every other block (SoC, clock, awards, masks) is already in [-1, 1].
    The observation space becomes a bounded float32 Box; ``obs_layout`` is
    unchanged. Policies written for raw prices (the baselines) need the
    unwrapped environment.
    """

    def __init__(self, env: gym.Env, price_scale: float = 25.0) -> None:
        super().__init__(env)
        if not price_scale > 0:
            raise ValueError("price_scale must be > 0")
        self.price_scale = float(price_scale)
        layout = env.unwrapped.obs_layout
        self._blocks = [layout[name] for name in PRICE_BLOCKS if name in layout]
        bound = float(np.log1p(MAX_ABS_PRICE / self.price_scale))
        low = np.asarray(env.observation_space.low, dtype=np.float32).copy()
        high = np.asarray(env.observation_space.high, dtype=np.float32).copy()
        for block in self._blocks:
            low[block], high[block] = -bound, bound
        self._low, self._high = low, high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        out = np.asarray(observation, dtype=np.float64).copy()
        for block in self._blocks:
            p = out[block]
            out[block] = np.sign(p) * np.log1p(np.abs(p) / self.price_scale)
        return np.clip(out, self._low, self._high).astype(np.float32)
