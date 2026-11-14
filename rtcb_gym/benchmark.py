"""Named tasks, one way to plug in an agent, and a scorecard.

A *task* fixes everything about an evaluation except the agent: market
setting, battery, episode length, splits and the bound it is scored
against. An *agent* is built from a :class:`TaskSpec` (never from the
environment, which holds the future prices) and is scored by
:func:`evaluate`, which returns a scorecard.

    from rtcb_gym.benchmark import evaluate

    def make_agent(spec):
        return MyAgent(spec)          # any rtcb_gym.baselines.Policy

    card = evaluate(make_agent, "da-rt-energy", split="validation", source="synthetic")

``source="synthetic"`` needs no market data (see
:mod:`rtcb_gym.data.synthetic`); ``source="real"`` reads the cleaned ERCOT
data named in the config. Command line: ``python -m rtcb_gym.evaluate``.

What an agent may use
    The :class:`TaskSpec`: the rules, the layouts of observation and
    action, and the prices of the *training* split. Then, step by step,
    the observation and ``info`` the environment returns. Nothing else:
    the scorecard is only meaningful if the agent never reads validation
    or test prices by another route.

Scoring
    Revenue, and capture rate against perfect foresight in the same task
    on the same days (an exact bound). An action that raises, has the
    wrong shape or is not finite is replaced by "do nothing" and counted
    (``invalid_actions``); out-of-range and infeasible requests are clipped
    by the environment and counted (``clipped_actions``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import rtcb_gym
from rtcb_gym.ancillary import ECRSRules
from rtcb_gym.baselines.base import Policy, env_action
from rtcb_gym.baselines.ecrs import PerfectForesightMarket
from rtcb_gym.battery import BatteryParams
from rtcb_gym.config import load_config, resolve_path
from rtcb_gym.data.clean import ERCOT_TZ
from rtcb_gym.day_ahead import clear_bid
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import bootstrap_ci, days_between, parse_splits, run_policy

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Task:
    name: str
    summary: str
    needs: tuple[str, ...]  # price series: rt, rt5, da, ecrs, da_ecrs
    env: dict[str, Any] = field(default_factory=dict)  # RTCBEnv arguments
    battery: dict[str, float] = field(default_factory=dict)  # overrides of the config's battery
    episode_days: int = 1


HALF_FULL = {"soc_init": 0.5, "soc_terminal_min": 0.5}

TASKS: dict[str, Task] = {t.name: t for t in (
    Task("rt-energy", "Real-time energy, 15-minute intervals, one day at a time from an empty battery. "
         "Action: power.", ("rt",)),
    Task("rt-energy-offer", "The same market, but the action is an offer curve the market applies to "
         "the interval's own price.", ("rt",), env={"action_mode": "offer"}),
    Task("da-rt-energy", "Day-ahead bid at 10:00 the day before, then real-time dispatch; deviations "
         "from the award settle at the real-time price.", ("rt", "da")),
    Task("da-rt-energy-7day", "Two-settlement over 7 consecutive days: the battery starts and ends half "
         "full and carries energy over midnight.", ("rt", "da"), battery=HALF_FULL, episode_days=7),
    Task("da-rt-energy-ecrs-5min", "Five-minute dispatch with 15-minute settlement, energy and ECRS, "
         "day-ahead and real time for both. Starts and ends half full every day.",
         ("rt5", "da", "ecrs", "da_ecrs"), battery=HALF_FULL,
         env={"dt_hours": 1 / 12, "settlement_minutes": 15, "history_len": 288}),
)}


@dataclass(frozen=True)
class TaskSpec:
    """Everything an agent is given before the first observation."""

    task: str
    summary: str
    battery: BatteryParams
    dt_hours: float
    history_len: int
    episode_days: int
    obs_layout: dict[str, slice]
    action_layout: dict[str, int]
    action_low: np.ndarray
    action_high: np.ndarray
    two_settlement: bool
    ecrs: bool
    da_ecrs: bool
    ecrs_duration_hours: float
    tz: str
    train: dict[str, Any]  # the task's price series, training split only

    def action(self, energy: float, ecrs: float = 0.0) -> np.ndarray:
        """The task's action for a power (share of power_mw, + discharge)
        and an ECRS share, whatever the action layout."""
        return env_action(self, energy, ecrs)

    def describe(self) -> str:
        """The rules in plain language, for agents that read text."""
        b = self.battery
        minutes = round(self.dt_hours * 60)
        lines = [
            f"Task {self.task}: {self.summary}",
            f"You operate a battery: {b.power_mw:g} MW power, {b.energy_mwh:g} MWh energy, "
            f"{b.rte:.0%} round-trip efficiency, ${b.degradation_cost_usd_per_mwh:g}/MWh cost on every MWh "
            f"charged or discharged. State of charge (SoC) is a fraction between {b.soc_min:g} and {b.soc_max:g}.",
            f"Each episode is {self.episode_days} operating day(s) of {minutes}-minute intervals. It starts at "
            f"SoC {b.soc_init:g} and must end at or above {b.soc_terminal_min:g}; the market enforces this.",
            "You are a price-taker: your actions never move prices. Reward is revenue in USD. Positive power "
            "discharges (sells), negative charges (buys). Charging at a negative price earns money.",
            "Before each interval you see past prices only, never the price of the interval you act in.",
        ]
        if "energy" in self.action_layout:
            lines.append('Real-time action: {"energy": x} with x in [-1, 1], a share of rated power.')
        else:
            lines.append(
                'Real-time action: an offer curve {"sell": s, "sell_price": p1, "buy": b, "buy_price": p2}. '
                "You sell share s of rated power if the interval's price turns out >= p1, buy share b if it "
                "is <= p2, otherwise do nothing. Shares in [0, 1], prices in USD/MWh with p2 <= p1.")
        if self.ecrs:
            lines.append(
                'Add "ecrs": y with y in [0, 1] to hold that share of rated power as ECRS reserve, paid the '
                f"ECRS price per MW per hour. Reserve needs stored energy for {self.ecrs_duration_hours:g} hour, "
                "cannot be held while charging, and discharge + reserve cannot exceed rated power; what does "
                "not fit is cut.")
        if self.two_settlement:
            lines.append(
                "Day-ahead: at 10:00 the day before each operating day you submit an hourly schedule in MW "
                '({"energy_mw": [24 numbers]}, + sell, - buy), awarded at the day-ahead price. In real time, '
                "the difference between what you deliver and the award settles at the real-time price.")
        if self.da_ecrs:
            lines.append('You may add "ecrs_mw": [24 numbers >= 0], an hourly day-ahead ECRS offer.')
        return "\n".join(lines)


def load_series(task: Task, source: str = "real", config: dict | None = None, seed: int = 0
                ) -> tuple[dict[str, Any], dict[str, tuple[dt.date, dt.date]], dict[str, Any]]:
    """The task's price series, the splits and a description of the data."""
    if source == "synthetic":
        from rtcb_gym.data.synthetic import SYNTHETIC_SPLITS, synthetic_prices

        data = synthetic_prices(seed)
        return ({k: data[k] for k in task.needs}, parse_splits(SYNTHETIC_SPLITS),
                {"source": "synthetic", "seed": seed, "note": "not ERCOT data; not a result"})
    if source != "real":
        raise ValueError("source must be 'real' or 'synthetic'")
    from rtcb_gym.data import sced
    from rtcb_gym.data.clean import operating_day_bounds_utc
    from rtcb_gym.data.store import data_sha256, processed_path, read_prices

    cfg = config or load_config()
    m, d = cfg["market"], cfg["data"]
    processed, raw = resolve_path(d["processed_dir"]), resolve_path(d["raw_dir"])
    splits = parse_splits(cfg["splits"])
    out: dict[str, Any] = {}
    info: dict[str, Any] = {"source": "real", "location": m["location"], "sha256": {},
                            "credit": "Data provided by Grid Status (https://www.gridstatus.io)"}
    if "rt" in task.needs:
        out["rt"] = read_prices(processed_path(processed, m["location"], m["market"]))
        info["sha256"]["rt"] = data_sha256(out["rt"])
    if "da" in task.needs:
        out["da"] = read_prices(processed_path(processed, m["location"], cfg["day_ahead"]["market"]))
        info["sha256"]["da"] = data_sha256(out["da"])
    if "rt5" in task.needs or "ecrs" in task.needs:
        cached = sced.read_sced_cache(raw, sced.SCED_LMP_DATASET, m["location"])
        stamps = pd.to_datetime(cached["sced_timestamp_utc"], utc=True)
        lo = max(stamps.min().ceil("5min"), operating_day_bounds_utc(splits["train"][0], m["timezone"])[0])
        hi = min(stamps.max().floor("5min"), operating_day_bounds_utc(splits["test"][1], m["timezone"])[1])
        tape = sced.real_time_tape(
            cached, sced.read_sced_cache(raw, sced.ADDERS_DATASET, m["location"]),
            sced.read_sced_cache(raw, sced.SCED_MCPC_DATASET, "ECRS"), lo, hi)
        tape = tape[tape["status"].isin(sced.OBSERVED_STATUSES)]
        ts = pd.DatetimeIndex(tape["interval_start_utc"])
        out["rt5"] = pd.DataFrame({"interval_start_utc": ts, "location": m["location"],
                                   "price_usd_per_mwh": tape["energy_price_usd_per_mwh"].to_numpy()})
        out["ecrs"] = pd.Series(tape["ecrs_price_usd_per_mw_h"].to_numpy(), index=ts)
        info["five_minute_rows"] = len(tape)
    if "da_ecrs" in task.needs:
        out["da_ecrs"] = sced.day_ahead_ecrs_prices(sced.read_sced_cache(raw, sced.DA_AS_DATASET, "DAM"))
    return {k: out[k] for k in task.needs}, splits, info


def _in_days(x: Any, start: dt.date, end: dt.date, tz: str) -> Any:
    """Rows of a price table or series whose local operating day is in start..end."""
    index = pd.DatetimeIndex(x["interval_start_utc"] if isinstance(x, pd.DataFrame) else x.index)
    day = np.array(index.tz_convert(tz).date)
    keep = (day >= start) & (day <= end)
    return x[keep].copy()


def make_env(task: str | Task, split: str = "train", source: str = "real", config: dict | None = None,
             seed: int = 0, sample: str = "random", **overrides: Any) -> RTCBEnv:
    """The task's environment on one split, e.g. to train a learning agent
    on ``split="train"``. Agents under evaluation never receive it."""
    env, _, _ = _build(task, split, source, config, seed, sample, overrides)
    return env


def _build(task, split, source, config, seed, sample, overrides=None):
    task = TASKS[task] if isinstance(task, str) else task
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    cfg = config or load_config()
    series, splits, data_info = load_series(task, source, cfg, seed)
    tz = cfg["market"]["timezone"] if source == "real" else ERCOT_TZ
    battery = BatteryParams.from_dict({**cfg["battery"], **task.battery})
    da_cfg = cfg.get("day_ahead") or {}
    kwargs: dict[str, Any] = dict(
        days=days_between(*splits[split]), episode_days=task.episode_days, tz=tz, sample=sample,
        da_prices=series.get("da"), ecrs_prices=series.get("ecrs"), da_ecrs_prices=series.get("da_ecrs"),
        da_decision_time=str(da_cfg.get("decision_time", "10:00")),
        da_history_days=int(da_cfg.get("history_days", 7)),
    )
    if "ecrs" in series:
        kwargs["ecrs_rules"] = ECRSRules(float((cfg.get("energy_ecrs") or {}).get("duration_hours", 1.0)))
    kwargs.update(task.env)
    kwargs.update(overrides or {})
    env = RTCBEnv(series["rt5"] if "rt5" in series else series["rt"], battery, **kwargs)
    train = {k: _in_days(v, *splits["train"], tz) for k, v in series.items()}
    spec = TaskSpec(
        task=task.name, summary=task.summary, battery=battery, dt_hours=env.dt_hours,
        history_len=env.history_len, episode_days=env.episode_days, obs_layout=dict(env.obs_layout),
        action_layout=dict(env.action_layout), action_low=env._a_low.copy(), action_high=env._a_high.copy(),
        two_settlement=env.two_settlement, ecrs=env.ecrs, da_ecrs=env.da_ecrs,
        ecrs_duration_hours=env.ecrs_rules.duration_hours, tz=tz, train=train,
    )
    return env, spec, data_info


def task_spec(task: str | Task, source: str = "real", config: dict | None = None, seed: int = 0) -> TaskSpec:
    """The spec an agent would receive, without running anything."""
    return _build(task, "train", source, config, seed, "sequential")[1]


class _Guarded(Policy):
    """Runs the agent; an invalid action becomes "do nothing" and is counted."""

    def __init__(self, agent: Policy, spec: TaskSpec) -> None:
        self.agent, self.spec = agent, spec
        self.name = getattr(agent, "name", type(agent).__name__)
        self.invalid_actions = self.invalid_bids = self.steps = 0
        self.seconds = 0.0
        self.errors: list[str] = []

    def _note(self, where: str, err: Exception | str) -> None:
        if len(self.errors) < 5:
            self.errors.append(f"{where}: {err!r}" if isinstance(err, Exception) else f"{where}: {err}")

    def day_ahead(self, context: Any) -> Any:
        t = time.perf_counter()
        try:
            bid = self.agent.day_ahead(context)
            # the same checks the market makes: shape, finite, ECRS >= 0
            zeros = np.zeros(context.n_hours)
            clear_bid(bid, zeros, zeros if self.spec.da_ecrs else None)
        except Exception as err:  # noqa: BLE001 - an agent's failure is a score, not a crash
            self.invalid_bids += 1
            self._note("day_ahead", err)
            bid = None
        self.seconds += time.perf_counter() - t
        return bid

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        self.agent.reset(obs, info)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        t = time.perf_counter()
        self.steps += 1
        try:
            a = np.asarray(self.agent.act(obs.copy(), dict(info)), dtype=np.float64).reshape(-1)
            if a.shape != self.spec.action_low.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"action {a!r} is not {len(self.spec.action_low)} finite numbers")
        except Exception as err:  # noqa: BLE001
            self.invalid_actions += 1
            self._note("act", err)
            a = self.spec.action(0.0)
        self.seconds += time.perf_counter() - t
        return a


def evaluate(make_agent: Callable[[TaskSpec], Policy], task: str | Task, split: str = "validation",
             source: str = "real", config: dict | None = None, seed: int = 0,
             limit_days: int | None = None) -> dict[str, Any]:
    """Score an agent on a task. Returns the scorecard (JSON-serialisable).

    ``make_agent(spec)`` builds the agent from the :class:`TaskSpec`. With
    ``limit_days`` only the first episodes are run (a quick check; the
    scorecard says so).
    """
    env, spec, data_info = _build(task, split, source, config, seed, "sequential")
    n = env.episode_days
    starts, last = [], None
    for day in env.days:  # consecutive, non-overlapping episodes
        if last is None or (day - last).days >= n:
            starts.append(day)
            last = day
    if limit_days:
        starts = starts[: max(1, limit_days // n)]
    guarded = _Guarded(make_agent(spec), spec)
    result = run_policy(env, guarded, starts)
    bound_policy = PerfectForesightMarket(env)
    bound = run_policy(env, bound_policy, starts)
    planned = bound_policy.planned_revenue_usd + bound_policy.planned_da_usd
    capture = result.revenue_usd / bound.revenue_usd if bound.revenue_usd > 0 else float("nan")
    card = {
        "benchmark": "rtcb-gym", "version": rtcb_gym.__version__,
        "task": spec.task, "task_sha256": task_hash(TASKS[spec.task], spec.battery),
        "split": split, "data": data_info, "partial": bool(limit_days),
        "agent": guarded.name,
        "days": result.n_days, "episodes": len(starts), "steps": result.n_steps,
        "revenue_usd": result.revenue_usd,
        "revenue_ci95_usd": list(bootstrap_ci(result.daily_revenue_usd)),
        "bound_revenue_usd": bound.revenue_usd,
        "bound_replay_error_usd": abs(bound.revenue_usd - planned),
        "capture_rate": capture,
        "capture_ci95": list(bootstrap_ci(result.daily_revenue_usd, bound.daily_revenue_usd)),
        "ecrs_revenue_usd": result.ecrs_revenue_usd,
        "da_revenue_usd": result.da_revenue_usd + result.ecrs_da_revenue_usd,
        "cycles_per_day": result.efc_per_day(spec.battery.energy_mwh),
        "clipped_actions": result.n_clipped, "clipped_ecrs_requests": result.n_ecrs_clipped,
        "clipped_day_ahead_hours": result.n_da_clipped_hours,
        "invalid_actions": guarded.invalid_actions, "invalid_day_ahead_bids": guarded.invalid_bids,
        "first_errors": guarded.errors,
        "agent_seconds_per_step": guarded.seconds / max(guarded.steps, 1),
        "daily_revenue_usd": result.daily_revenue_usd,
        "excluded_days": env.excluded_days,
    }
    return card


def task_hash(task: Task, battery: BatteryParams) -> str:
    """Changes whenever the task's definition or battery does."""
    payload = json.dumps({"task": asdict(task), "battery": battery.to_dict()}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def leaderboard(cards: list[dict[str, Any]]) -> str:
    """A markdown table of scorecards, grouped by task and split, best first."""
    lines = ["| Task | Split | Data | Agent | Days | Revenue (USD) | Capture rate (95% interval) | "
             "Invalid / clipped actions | Seconds per step |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    order = {name: i for i, name in enumerate(TASKS)}
    for c in sorted(cards, key=lambda c: (order.get(c["task"], 99), c["split"], c["data"]["source"], -c["revenue_usd"])):
        lo, hi = c["capture_ci95"]
        partial = " (partial)" if c.get("partial") else ""
        lines.append(
            f"| {c['task']} | {c['split']}{partial} | {c['data']['source']} | {c['agent']} | {c['days']} | "
            f"{round(c['revenue_usd']) + 0:,} | {c['capture_rate']:.1%} ({lo * 100:.1f} to {hi * 100:.1f}) | "
            f"{c['invalid_actions'] + c['invalid_day_ahead_bids']} / {c['clipped_actions']} | "
            f"{c['agent_seconds_per_step']:.4f} |")
    return "\n".join(lines)
