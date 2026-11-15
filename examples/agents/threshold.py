"""A complete agent in thirty lines: buy low, sell high, thresholds from the
training split.

    python -m rtcb_gym.evaluate --task rt-energy --agent examples/agents/threshold.py --source synthetic

It uses only what the task spec gives it (the rules and the training
prices) and what it observes, and it works on every task because it builds
its action with ``spec.action``.
"""

import numpy as np

from rtcb_gym.baselines import Policy


class Threshold(Policy):
    name = "example: threshold"

    def __init__(self, spec):
        self.spec = spec
        train = spec.train["rt5" if "rt5" in spec.train else "rt"]["price_usd_per_mwh"]
        self.low, self.high = np.quantile(train, [0.3, 0.7])

    def act(self, obs, info):
        last_price = obs[self.spec.obs_layout["prices"]][-1]
        if last_price <= self.low:
            return self.spec.action(-1.0)  # charge at full power
        if last_price >= self.high:
            return self.spec.action(1.0)  # discharge at full power
        return self.spec.action(0.0)


def make_agent(spec):
    return Threshold(spec)
