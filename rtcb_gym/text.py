"""A text interface, so a language model can be scored like any other agent.

:class:`TextAgent` turns each observation into a short JSON message, hands
it to a function you supply (``complete(prompt) -> reply``: a call to any
model, local or hosted) and parses the reply into an action. This module
makes no network calls and knows no provider.

    from rtcb_gym.benchmark import evaluate
    from rtcb_gym.text import TextAgent

    def complete(prompt: str) -> str:
        return my_model(prompt)       # must return JSON, e.g. {"energy": -0.5}

    card = evaluate(lambda spec: TextAgent(spec, complete, decide_every=4), "rt-energy")

The prompt holds the task rules (:meth:`rtcb_gym.benchmark.TaskSpec.describe`)
and the current state; the model sees exactly what a numeric agent sees,
no more. A reply that cannot be parsed counts as an invalid action and
the battery idles. ``decide_every`` repeats a decision for that many
intervals, which keeps the number of model calls manageable (96 or 288
intervals a day).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from rtcb_gym.baselines.base import Policy
from rtcb_gym.day_ahead import DayAheadBid

JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _round(x: Any, digits: int = 2) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(x).ravel()]


def render_observation(spec: Any, obs: np.ndarray, info: dict[str, Any], recent: int = 16) -> dict[str, Any]:
    """The observation as a small dictionary: everything the agent may know
    at this interval, with the last ``recent`` prices in full and the rest
    of the price history summarised."""
    lay = spec.obs_layout
    prices = obs[lay["prices"]]
    out: dict[str, Any] = {
        "interval_start_utc": str(info.get("interval_start_utc", "")),
        "intervals_left_in_episode": int(info["steps_remaining"]),
        "interval_minutes": round(spec.dt_hours * 60),
        "soc": round(float(obs[lay["soc"]][0]), 4),
        "day_of_week": int(np.argmax(obs[lay["day_of_week"]])),
        "recent_prices_usd_per_mwh": _round(prices[-recent:]),
        "price_history": {"intervals": len(prices), "min": round(float(prices.min()), 2),
                          "median": round(float(np.median(prices)), 2), "max": round(float(prices.max()), 2)},
    }
    if spec.two_settlement:
        m = int(obs[lay["da_ahead_mask"]].sum())
        per_hour = round(1 / spec.dt_hours)
        out["day_ahead_price_next_hours"] = _round(obs[lay["da_price_ahead"]][:m:per_hour])
        out["day_ahead_award_mw_next_hours"] = _round(
            obs[lay["da_award_ahead"]][:m:per_hour] * spec.battery.power_mw)
    if spec.ecrs:
        out["recent_ecrs_prices_usd_per_mw_h"] = _round(obs[lay["ecrs_prices"]][-recent:])
    if spec.da_ecrs:
        m = int(obs[lay["da_ahead_mask"]].sum())
        per_hour = round(1 / spec.dt_hours)
        out["day_ahead_ecrs_price_next_hours"] = _round(obs[lay["da_ecrs_price_ahead"]][:m:per_hour])
        out["day_ahead_ecrs_award_mw_next_hours"] = _round(
            obs[lay["da_ecrs_award_ahead"]][:m:per_hour] * spec.battery.power_mw)
    return out


def render_day_ahead(context: Any, spec: Any) -> dict[str, Any]:
    """What is known at the day-ahead decision, as a small dictionary."""
    out = {
        "operating_day": str(context.operating_day),
        "hours_to_bid": context.n_hours,
        "previous_day_ahead_prices": _round(context.da_history.to_numpy()[-48:]),
        "recent_real_time_prices_hourly_mean": _round(
            context.rt_history.to_numpy()[-(round(1 / spec.dt_hours) * 24):]
            .reshape(-1, round(1 / spec.dt_hours)).mean(axis=1)) if len(context.rt_history) >= round(1 / spec.dt_hours) * 24 else [],
        "soc_at_start_of_day": context.soc_init,
    }
    if context.da_ecrs_history is not None:
        out["previous_day_ahead_ecrs_prices"] = _round(context.da_ecrs_history.to_numpy()[-48:])
    return out


def parse_json(reply: str) -> dict[str, Any]:
    """The first JSON object in a reply (models often wrap it in prose)."""
    match = JSON_OBJECT.search(reply)
    if not match:
        raise ValueError(f"no JSON object in reply: {reply[:120]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("reply must be a JSON object")
    return data


def parse_action(spec: Any, reply: str) -> np.ndarray:
    """A reply such as ``{"energy": 0.5, "ecrs": 0.2}`` (or the offer-curve
    keys) as the task's action. Missing keys are zero; raises ValueError on
    anything that is not a number."""
    data = parse_json(reply)
    unknown = set(data) - set(spec.action_layout)
    if unknown:
        raise ValueError(f"unknown action keys {sorted(unknown)}; expected {sorted(spec.action_layout)}")
    a = np.zeros(len(spec.action_layout))
    for key, k in spec.action_layout.items():
        a[k] = float(data.get(key, 0.0))
    if not np.all(np.isfinite(a)):
        raise ValueError("action values must be finite")
    return a


def parse_bid(spec: Any, reply: str, n_hours: int) -> DayAheadBid:
    data = parse_json(reply)
    energy = np.asarray(data.get("energy_mw", np.zeros(n_hours)), dtype=np.float64)
    ecrs = data.get("ecrs_mw")
    if energy.shape != (n_hours,):
        raise ValueError(f"energy_mw needs {n_hours} numbers")
    return DayAheadBid(energy_mw=energy, ecrs_mw=None if ecrs is None or not spec.da_ecrs
                       else np.asarray(ecrs, dtype=np.float64))


class TextAgent(Policy):
    """An agent that decides by exchanging text with ``complete``.

    A reply that does not parse raises, which the benchmark counts as an
    invalid action (the battery idles for that interval and the previous
    decision is dropped).
    """

    def __init__(self, spec: Any, complete: Callable[[str], str], decide_every: int = 1,
                 bid_day_ahead: bool = True, name: str = "text agent") -> None:
        if decide_every < 1:
            raise ValueError("decide_every must be >= 1")
        self.spec, self.complete = spec, complete
        self.decide_every, self.bid_day_ahead = int(decide_every), bid_day_ahead
        self.name = name
        self.rules = spec.describe()
        self._held: np.ndarray | None = None
        self._age = 0
        self.calls = 0

    def prompt(self, state: dict[str, Any], ask: str) -> str:
        return f"{self.rules}\n\nCurrent state:\n{json.dumps(state)}\n\n{ask}"

    def day_ahead(self, context: Any) -> DayAheadBid | None:
        if not (self.spec.two_settlement and self.bid_day_ahead):
            return None
        self.calls += 1
        reply = self.complete(self.prompt(
            render_day_ahead(context, self.spec),
            "Reply with one JSON object holding your day-ahead schedule and nothing else."))
        return parse_bid(self.spec, reply, context.n_hours)

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        self._held, self._age = None, 0

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        if self._held is None or self._age >= self.decide_every:
            self._held, self._age = None, 0
            self.calls += 1
            reply = self.complete(self.prompt(
                render_observation(self.spec, obs, info),
                "Reply with one JSON object holding your action for this interval and nothing else."))
            self._held = parse_action(self.spec, reply)
        self._age += 1
        return self._held
