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
    daily_revenue_usd: list[float] = field(default_factory=list)

    def efc_per_day(self, energy_mwh: float) -> float:
        return self.storage_discharge_mwh / energy_mwh / self.n_days if self.n_days else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_policy(env: RTCBEnv, policy: Policy, days: list[dt.date] | None = None) -> RunResult:
    """Run ``policy`` on each episode start day in order (default: all env days).

    In a two-settlement env the policy's ``day_ahead`` is asked first, with
    the context of that day's day-ahead decision.
    """
    result = RunResult(policy=policy.name)
    for day in days if days is not None else env.days:
        options: dict[str, Any] = {"day": day}
        if env.two_settlement:
            schedule = policy.day_ahead(env.day_ahead_context(day))
            if schedule is not None:
                options["da_schedule"] = schedule
        obs, info = env.reset(options=options)
        result.n_da_clipped_hours += int(info.get("da_clipped_hours", 0))
        policy.reset(obs, info)
        day_revenue = 0.0
        terminated = False
        while not terminated:
            obs, reward, terminated, _, info = env.step(policy.act(obs, info))
            day_revenue += reward
            result.n_steps += 1
            result.energy_revenue_usd += info["energy_revenue_usd"]
            result.da_revenue_usd += info["da_revenue_usd"]
            result.rt_revenue_usd += info["rt_revenue_usd"]
            result.degradation_cost_usd += info["degradation_cost_usd"]
            result.storage_discharge_mwh += info["storage_discharge_mwh"]
            result.n_clipped += int(info["clipped"])
        result.revenue_usd += day_revenue
        result.daily_revenue_usd.append(day_revenue)
        result.n_days += 1
    return result


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
    return {
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


def _usd(x: float) -> str:
    """Whole dollars with thousands separators; round-off never shows as -0."""
    return f"{round(x) + 0:,}"


def markdown_table(rows: list[dict[str, Any]], two_settlement: bool = False) -> str:
    """Results table. ``two_settlement`` adds the day-ahead and real-time
    settlement columns (net revenue = DA + RT - degradation)."""
    if not two_settlement:
        header = "| Split | Strategy | Days | Revenue (USD) | Capture rate | Cycles/day | Clipped actions |"
        lines = [header, "|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {r['split']} | {r['policy']} | {r['days']} | {_usd(r['revenue_usd'])} | "
                f"{r['capture_rate']:.1%} | {r['efc_per_day']:.2f} | {r['n_clipped']} |"
            )
        return "\n".join(lines)
    header = (
        "| Split | Strategy | Days | Revenue (USD) | DA settlement | RT settlement | "
        "Capture rate | Cycles/day | Clipped actions |"
    )
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        clipped = f"{r['n_clipped']}" + (f" + {r['n_da_clipped_hours']} DA h" if r["n_da_clipped_hours"] else "")
        lines.append(
            f"| {r['split']} | {r['policy']} | {r['days']} | {_usd(r['revenue_usd'])} | "
            f"{_usd(r['da_revenue_usd'])} | {_usd(r['rt_revenue_usd'])} | "
            f"{r['capture_rate']:.1%} | {r['efc_per_day']:.2f} | {clipped} |"
        )
    return "\n".join(lines)
