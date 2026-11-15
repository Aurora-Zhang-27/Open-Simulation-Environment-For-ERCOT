"""Score an agent on a benchmark task from the command line.

    python -m rtcb_gym.evaluate --list
    python -m rtcb_gym.evaluate --task rt-energy --agent examples/agents/threshold.py \\
        --source synthetic --out scorecard.json
    python -m rtcb_gym.evaluate --task da-rt-energy --agent reference:da-schedule --split validation
    python -m rtcb_gym.evaluate --leaderboard results/benchmark/*.json

``--agent`` is ``FILE.py`` or ``FILE.py:FUNCTION``: a function taking the
task spec and returning the agent (default name ``make_agent``), or
``reference:NAME`` for a built-in reference agent. ``--source synthetic``
needs no market data. The test split is only scored when asked for by name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rtcb_gym.baselines.base import Policy
from rtcb_gym.baselines.day_ahead import DayAheadSchedule
from rtcb_gym.baselines.forecast_mpc import ForecastMPC
from rtcb_gym.benchmark import TASKS, TaskSpec, evaluate, leaderboard
from rtcb_gym.config import load_config
from rtcb_gym.evaluation import clean_nan


class Idle(Policy):
    name = "idle"

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return self.spec.action(0.0)


class _AsTaskAction(Policy):
    """A baseline that returns a power, in whatever action the task uses."""

    def __init__(self, spec: TaskSpec, inner: Policy, name: str, day_ahead: bool = True) -> None:
        self.spec, self.inner, self.name, self.use_da = spec, inner, name, day_ahead

    def day_ahead(self, context: Any) -> Any:
        return self.inner.day_ahead(context) if self.use_da else None

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        self.inner.reset(obs, info)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return self.spec.action(float(np.asarray(self.inner.act(obs, info)).ravel()[0]))


def _persistence_mpc(spec: TaskSpec) -> Policy:
    if abs(spec.dt_hours - 0.25) > 1e-9:
        raise ValueError("reference:persistence-mpc is defined for 15-minute tasks")
    return _AsTaskAction(spec, ForecastMPC(spec.battery, spec.obs_layout, spec.dt_hours), "persistence MPC")


def _da_schedule(spec: TaskSpec) -> Policy:
    if not spec.two_settlement:
        raise ValueError("reference:da-schedule needs a task with a day-ahead market")
    return _AsTaskAction(spec, DayAheadSchedule(spec.battery, spec.obs_layout), "day-ahead schedule, delivered")


REFERENCE: dict[str, Callable[[TaskSpec], Policy]] = {
    "idle": Idle, "persistence-mpc": _persistence_mpc, "da-schedule": _da_schedule,
}


def load_agent(target: str) -> Callable[[TaskSpec], Policy]:
    if target.startswith("reference:"):
        name = target.split(":", 1)[1]
        if name not in REFERENCE:
            raise SystemExit(f"unknown reference agent {name!r}; choose from {sorted(REFERENCE)}")
        return REFERENCE[name]
    path, _, function = target.partition(":")
    file = Path(path)
    if not file.exists():
        raise SystemExit(f"agent file not found: {file}")
    module_spec = importlib.util.spec_from_file_location(f"rtcb_agent_{file.stem}", file)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    factory = getattr(module, function or "make_agent", None)
    if not callable(factory):
        raise SystemExit(f"{file} has no function {function or 'make_agent'}(spec)")
    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the tasks and reference agents")
    ap.add_argument("--leaderboard", nargs="+", metavar="CARD.json", help="print a table of scorecards")
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--agent", help="FILE.py[:FUNCTION] or reference:NAME")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--source", default="real", choices=["real", "synthetic"])
    ap.add_argument("--config", default=None, help="YAML config (real data: hub, dates, battery)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the synthetic prices")
    ap.add_argument("--limit-days", type=int, default=None, help="score only the first days (a quick check)")
    ap.add_argument("--out", default=None, help="write the scorecard here")
    args = ap.parse_args(argv)

    if args.list:
        for task in TASKS.values():
            print(f"{task.name}\n    {task.summary}")
        print("\nreference agents: " + ", ".join(f"reference:{k}" for k in REFERENCE))
        return 0
    if args.leaderboard:
        print(leaderboard([json.loads(Path(p).read_text()) for p in args.leaderboard]))
        return 0
    if not args.task or not args.agent:
        ap.error("--task and --agent are required (or --list / --leaderboard)")
    card = evaluate(load_agent(args.agent), args.task, split=args.split, source=args.source,
                    config=load_config(args.config), seed=args.seed, limit_days=args.limit_days)
    lo, hi = card["capture_ci95"]
    print(f"{card['agent']} on {card['task']} ({card['split']}, {card['data']['source']} data, {card['days']} days)")
    print(f"  revenue        {card['revenue_usd']:,.0f} USD   (perfect foresight {card['bound_revenue_usd']:,.0f})")
    print(f"  capture rate   {card['capture_rate']:.1%}   (95% interval {lo:.1%} to {hi:.1%})")
    print(f"  invalid        {card['invalid_actions']} actions, {card['invalid_day_ahead_bids']} day-ahead bids")
    print(f"  clipped        {card['clipped_actions']} actions")
    for err in card["first_errors"]:
        print(f"  error          {err}")
    if card["data"]["source"] == "synthetic":
        print("  synthetic prices: this checks that the agent runs, it is not a result")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(clean_nan(card), indent=2, default=str) + "\n")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
