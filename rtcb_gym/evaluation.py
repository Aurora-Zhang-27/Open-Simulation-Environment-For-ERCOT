"""Run strategies through the environment and compute the metrics.

Metrics
    revenue_usd     total net revenue (energy revenue - degradation cost)
    capture_rate    revenue / perfect-foresight revenue on the same days
    efc_per_day     equivalent full cycles per day: storage-side energy
                    discharged / energy_mwh / days
    n_clipped       intervals where the requested action was clipped

Two-settlement runs also report the day-ahead settlement (award x DA price)
and the real-time settlement (deviation x RT price) separately, and the
number of day-ahead hours that had to be clipped to what the battery could
deliver.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from rtcb_gym.baselines.base import Policy
from rtcb_gym.data.clean import ERCOT_TZ, check_post_rtcb, to_date
from rtcb_gym.env import RTCBEnv

SPLIT_ORDER = ("train", "validation", "test")


@dataclass
class RunResult:
    policy: str
    n_days: int = 0
    n_steps: int = 0
    revenue_usd: float = 0.0
    energy_revenue_usd: float = 0.0
    degradation_cost_usd: float = 0.0
    storage_discharge_mwh: float = 0.0
    n_clipped: int = 0
    da_revenue_usd: float = 0.0  # day-ahead settlement
    rt_revenue_usd: float = 0.0  # real-time settlement (all of energy revenue without a DA position)
    n_da_clipped_hours: int = 0
    daily_revenue_usd: list[float] = field(default_factory=list)  # one entry per operating day
    # environments with ECRS only
    ecrs_revenue_usd: float = 0.0  # included in revenue_usd
    ecrs_da_revenue_usd: float = 0.0
    ecrs_rt_revenue_usd: float = 0.0
    ecrs_mw_hours: float = 0.0  # real-time ECRS award, MW x hours
    n_ecrs_clipped: int = 0

    def efc_per_day(self, energy_mwh: float) -> float:
        return self.storage_discharge_mwh / energy_mwh / self.n_days if self.n_days else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_policy(env: RTCBEnv, policy: Policy, days: list[dt.date] | None = None) -> RunResult:
    """Run ``policy`` on each episode start day in order (default: all env days).

    In a two-settlement env the policy's ``day_ahead`` is asked first, once
    per operating day of the episode, with the context of that day's
    day-ahead decision. With ``episode_days > 1`` pass start days that do not
    overlap (e.g. ``env.days[::env.episode_days]``).
    """
    result = RunResult(policy=policy.name)
    for day in days if days is not None else env.days:
        options: dict[str, Any] = {"day": day}
        span = env.episode_operating_days(day)
        if env.two_settlement:
            bids = [policy.day_ahead(env.day_ahead_context(d)) for d in span]
            if any(b is not None for b in bids):
                options["da_schedule"] = bids
        obs, info = env.reset(options=options)
        result.n_da_clipped_hours += int(info.get("da_clipped_hours", 0))
        policy.reset(obs, info)
        rewards = []
        terminated = False
        while not terminated:
            obs, reward, terminated, _, info = env.step(policy.act(obs, info))
            rewards.append(reward)
            result.energy_revenue_usd += info["energy_revenue_usd"]
            result.da_revenue_usd += info["da_revenue_usd"]
            result.rt_revenue_usd += info["rt_revenue_usd"]
            result.degradation_cost_usd += info["degradation_cost_usd"]
            result.storage_discharge_mwh += info["storage_discharge_mwh"]
            result.n_clipped += int(info["clipped"])
            if env.ecrs:
                result.ecrs_revenue_usd += info["ecrs_revenue_usd"]
                result.ecrs_da_revenue_usd += info["ecrs_da_revenue_usd"]
                result.ecrs_rt_revenue_usd += info["ecrs_rt_revenue_usd"]
                result.ecrs_mw_hours += info["ecrs_award_mw"] * env.dt_hours
                result.n_ecrs_clipped += int(info["ecrs_clipped"])
        result.n_steps += len(rewards)
        result.revenue_usd += float(np.sum(rewards))
        k = 0
        for d in span:  # revenue of each operating day of the episode
            i0, i1 = env._day_rows[d]
            result.daily_revenue_usd.append(float(np.sum(rewards[k : k + i1 - i0])))
            k += i1 - i0
        result.n_days += len(span)
    return result


def bootstrap_ci(
    daily: list[float] | np.ndarray,
    daily_upper: list[float] | np.ndarray | None = None,
    block_days: int = 5,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for a total over days, by circular block bootstrap.

    Days are resampled in blocks of ``block_days`` consecutive days (prices
    of neighbouring days are related). With ``daily_upper`` (the bound's
    revenue on the same days) the interval is for the capture rate, the two
    series resampled together. It describes how much the number depends on
    which days happened to be in the split, nothing else: with about 55 days
    it is wide, and it says nothing about days unlike any in the split.
    """
    x = np.asarray(daily, dtype=np.float64)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n, size=(n_boot, blocks))
    idx = (starts[:, :, None] + np.arange(block_days)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    total = x[idx].sum(axis=1)
    if daily_upper is not None:
        upper = np.asarray(daily_upper, dtype=np.float64)[idx].sum(axis=1)
        total = np.where(upper > 0, total / np.where(upper > 0, upper, 1.0), np.nan)
    lo, hi = np.nanquantile(total, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def top_days_share(daily: list[float] | np.ndarray, k: int = 5) -> float:
    """Share of total revenue earned on the ``k`` best days."""
    x = np.sort(np.asarray(daily, dtype=np.float64))[::-1]
    total = x.sum()
    return float(x[:k].sum() / total) if total > 0 else float("nan")


def parse_splits(splits_cfg: dict[str, list[str]]) -> dict[str, tuple[dt.date, dt.date]]:
    """Validate the time-based split: ordered, non-overlapping, post-RTC+B."""
    out = {}
    for name in SPLIT_ORDER:
        if name not in splits_cfg:
            raise ValueError(f"splits is missing {name!r}")
        start, end = (to_date(x) for x in splits_cfg[name])
        check_post_rtcb(start)
        if end < start:
            raise ValueError(f"split {name}: end {end} before start {start}")
        out[name] = (start, end)
    for a, b in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        if not out[a][1] < out[b][0]:
            raise ValueError(f"split {a} must end before {b} starts")
    return out


def days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]


def prices_in_days(
    prices: pd.DataFrame, start: dt.date, end: dt.date, tz: str = ERCOT_TZ
) -> np.ndarray:
    """Prices of intervals whose local operating day is in start..end."""
    op_day = prices["interval_start_utc"].dt.tz_convert(tz).dt.date
    mask = (op_day >= start) & (op_day <= end)
    return prices.loc[mask, "price_usd_per_mwh"].to_numpy(dtype=np.float64)


def metrics_row(split: str, result: RunResult, upper: RunResult, energy_mwh: float) -> dict[str, Any]:
    if result.n_days != upper.n_days:
        raise ValueError("result and perfect-foresight run cover different days")
    capture = result.revenue_usd / upper.revenue_usd if upper.revenue_usd > 0 else float("nan")
    same_days = len(result.daily_revenue_usd) == len(upper.daily_revenue_usd)
    without_top = np.argsort(upper.daily_revenue_usd)[: max(len(upper.daily_revenue_usd) - 5, 0)]
    rest_upper = float(np.sum(np.asarray(upper.daily_revenue_usd)[without_top]))
    rest = float(np.sum(np.asarray(result.daily_revenue_usd)[without_top])) if same_days else float("nan")
    return {
        "bound": upper.policy,
        "revenue_ci95_usd": bootstrap_ci(result.daily_revenue_usd),
        "capture_ci95": bootstrap_ci(result.daily_revenue_usd, upper.daily_revenue_usd) if same_days
        else (float("nan"), float("nan")),
        "top5_days_share": top_days_share(result.daily_revenue_usd),
        # capture rate without the bound's five best days: is the ranking carried by a few days?
        "capture_rate_without_top5_days": rest / rest_upper if rest_upper > 0 else float("nan"),
        "ecrs_revenue_usd": result.ecrs_revenue_usd,
        "split": split,
        "policy": result.policy,
        "days": result.n_days,
        "revenue_usd": result.revenue_usd,
        "capture_rate": capture,
        "efc_per_day": result.efc_per_day(energy_mwh),
        "n_clipped": result.n_clipped,
        "da_revenue_usd": result.da_revenue_usd,
        "rt_revenue_usd": result.rt_revenue_usd,
        "degradation_cost_usd": result.degradation_cost_usd,
        "n_da_clipped_hours": result.n_da_clipped_hours,
    }


def clean_nan(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_nan(v) for v in x]
    return x


def check_replay(split: str, label: str, replayed: float, planned: float) -> None:
    if abs(replayed - planned) > 1e-6 * max(1.0, abs(replayed)):
        print(f"WARNING {split} {label}: env replay {replayed:.2f} != optimizer objective {planned:.2f}")


def _usd(x: float) -> str:
    """Whole dollars with thousands separators; round-off never shows as -0."""
    return f"{round(x) + 0:,}"


def _capture(r: dict[str, Any]) -> str:
    """Capture rate with its bootstrap interval, e.g. ``46.9% (41.0 to 52.3)``."""
    lo, hi = r.get("capture_ci95", (float("nan"), float("nan")))
    text = f"{r['capture_rate']:.1%}"
    return text if math.isnan(lo) or r["policy"] == r.get("bound") else f"{text} ({lo * 100:.1f} to {hi * 100:.1f})"


def markdown_table(rows: list[dict[str, Any]], two_settlement: bool = False) -> str:
    """Results table. ``two_settlement`` adds the day-ahead and real-time
    settlement columns (net revenue = DA + RT - degradation). The capture
    rate carries its 95% interval over days (:func:`bootstrap_ci`); the next
    column is the capture rate without the bound's five best days."""
    def rest(r: dict[str, Any]) -> str:
        x = r.get("capture_rate_without_top5_days", float("nan"))
        return "" if x is None or math.isnan(x) else f"{x:.1%}"

    if not two_settlement:
        header = ("| Split | Strategy | Days | Revenue (USD) | Capture rate (95% interval) | "
                  "Without top 5 days | Cycles/day | Clipped actions |")
        lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {r['split']} | {r['policy']} | {r['days']} | {_usd(r['revenue_usd'])} | "
                f"{_capture(r)} | {rest(r)} | {r['efc_per_day']:.2f} | {r['n_clipped']} |"
            )
        return "\n".join(lines)
    header = (
        "| Split | Strategy | Days | Revenue (USD) | DA settlement | RT settlement | "
        "Capture rate (95% interval) | Without top 5 days | Cycles/day | Clipped actions |"
    )
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        clipped = f"{r['n_clipped']}" + (f" + {r['n_da_clipped_hours']} DA h" if r["n_da_clipped_hours"] else "")
        lines.append(
            f"| {r['split']} | {r['policy']} | {r['days']} | {_usd(r['revenue_usd'])} | "
            f"{_usd(r['da_revenue_usd'])} | {_usd(r['rt_revenue_usd'])} | "
            f"{_capture(r)} | {rest(r)} | {r['efc_per_day']:.2f} | {clipped} |"
        )
    return "\n".join(lines)
