"""Day-ahead market: what the agent knows when it bids, and what it may bid.

Timeline for operating day D (ERCOT, Central Prevailing Time)::

    D-1 10:00    DAM offers due        -> the day-ahead decision is made here
    D-1 ~13:30   DAM results published -> D's day-ahead prices and awards known
    D   00:00    real time starts      -> 15-minute dispatch, as in v0.1

Decision
    An hourly schedule in MW for the hours of D (24; 23 or 25 on DST days).
    Positive = sell, negative = buy, as everywhere in RTCB-Gym. The battery
    is a price-taker, so the whole schedule is awarded at the day-ahead
    price.

What may be bid
    A schedule the battery could deliver: :func:`feasible_day_ahead` runs it
    through the battery model hour by hour from ``soc_init`` (with the
    end-of-day target) and clips what is not feasible. The award is still a
    financial position: in real time the battery may deviate from it, and
    the deviation settles at the real-time price
    (:func:`rtcb_gym.settlement.settle_two_settlement`).

What is known at the decision (:class:`DayAheadContext`)
    Real-time prices of intervals that ended by the decision time, and
    day-ahead prices of the days before D. Never D's own day-ahead prices,
    and never a real-time price after the decision time.

Consequence of price-taking with deviations at the RT price: the revenue
of an episode splits into

    sum_h award_h * (DA_h - mean RT price in hour h) * 1 h      (day-ahead)
  + sum_t power_t * RT_t * dt - degradation                    (real time)

so the best real-time dispatch does not depend on the award, and the award
is a bet on the day-ahead / real-time spread. Perfect foresight therefore
solves two independent problems and stays an exact upper bound.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rtcb_gym.battery import BatteryParams, battery_step
from rtcb_gym.data.clean import DA_INTERVAL, ERCOT_TZ, INTERVAL, operating_day_bounds_utc, to_date

DA_DT_HOURS = 1.0
DA_AHEAD = 96  # 15-minute intervals of day-ahead prices/awards in the observation


@dataclass(frozen=True)
class DayAheadContext:
    """Everything a day-ahead strategy may use for operating day D."""

    operating_day: dt.date
    decision_time_utc: pd.Timestamp
    hour_start_utc: pd.DatetimeIndex  # the hours of D to bid for
    da_history: pd.Series  # day-ahead prices of the days before D, by hour start (UTC)
    rt_history: pd.Series  # real-time prices settled by the decision time, by interval start (UTC)
    soc_init: float
    battery: BatteryParams
    tz: str = ERCOT_TZ

    @property
    def n_hours(self) -> int:
        return len(self.hour_start_utc)


def decision_time_utc(day: str | dt.date, decision_time: str = "10:00", tz: str = ERCOT_TZ) -> pd.Timestamp:
    """The day-ahead decision time for operating day ``day``: ``decision_time``
    (local, ``HH:MM``) on the day before."""
    hh, mm = (int(x) for x in decision_time.split(":"))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"decision_time must be HH:MM, got {decision_time!r}")
    before = to_date(day) - dt.timedelta(days=1)
    local = pd.Timestamp(dt.datetime.combine(before, dt.time(hh, mm))).tz_localize(tz)
    return local.tz_convert("UTC")


def day_hours_utc(day: str | dt.date, tz: str = ERCOT_TZ) -> pd.DatetimeIndex:
    lo, hi = operating_day_bounds_utc(day, tz)
    return pd.date_range(lo, hi, freq=DA_INTERVAL, inclusive="left").as_unit("ns")


def build_context(
    day: str | dt.date,
    rt: pd.Series,
    da: pd.Series,
    battery: BatteryParams,
    decision_time: str = "10:00",
    history_days: int = 7,
    tz: str = ERCOT_TZ,
) -> DayAheadContext:
    """Cut the information available at the decision time out of full price
    series (``rt`` and ``da``: prices indexed by interval start, UTC, sorted)."""
    if history_days < 1:
        raise ValueError("history_days must be >= 1")
    d = to_date(day)
    decision = decision_time_utc(d, decision_time, tz)
    rt_lo = decision - pd.Timedelta(days=history_days)
    # an interval is settled once it has ended
    rt_known = rt[(rt.index >= rt_lo) & (rt.index + INTERVAL <= decision)]
    da_lo, _ = operating_day_bounds_utc(d - dt.timedelta(days=history_days), tz)
    day_start, _ = operating_day_bounds_utc(d, tz)
    da_known = da[(da.index >= da_lo) & (da.index < day_start)]
    return DayAheadContext(
        operating_day=d,
        decision_time_utc=decision,
        hour_start_utc=day_hours_utc(d, tz),
        da_history=da_known.copy(),
        rt_history=rt_known.copy(),
        soc_init=battery.soc_init,
        battery=battery,
        tz=tz,
    )


def feasible_day_ahead(
    schedule_mw: np.ndarray, battery: BatteryParams, soc0: float
) -> tuple[np.ndarray, int, np.ndarray]:
    """Clip an hourly schedule to what the battery could deliver.

    Runs the schedule through :func:`rtcb_gym.battery.battery_step` with
    1-hour steps from ``soc0``, with the end-of-day target. Returns
    ``(award_mw, n_clipped_hours, soc_after_each_hour)``. A feasible
    schedule held constant over each hour is also feasible at 15-minute
    resolution, so the battery can always deliver its award exactly.
    """
    x = np.asarray(schedule_mw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("day-ahead schedule must be finite")
    n = len(x)
    award = np.empty(n)
    soc_path = np.empty(n)
    soc = float(soc0)
    clipped = 0
    for h in range(n):
        award[h], soc, info = battery_step(battery, soc, x[h], DA_DT_HOURS, steps_left=n - h)
        soc_path[h] = soc
        clipped += int(info["clipped"])
    return award, clipped, soc_path


def persistence_da_forecast(context: DayAheadContext) -> np.ndarray:
    """Forecast D's day-ahead prices with D-1's, matched by local clock hour.

    On the fall-back day the repeated hour gets the same forecast twice; if
    D-1 had a repeated hour, its two prices are averaged; an hour missing
    on D-1 (spring forward) takes the previous hour's price. Raises if D-1
    has fewer than 23 day-ahead prices.
    """
    prev = context.operating_day - dt.timedelta(days=1)
    hist = context.da_history
    local = hist.index.tz_convert(context.tz)
    on_prev = np.array(local.date) == prev
    if on_prev.sum() < 23:
        raise ValueError(f"no complete day-ahead prices for {prev}")
    by_hour = pd.Series(hist.to_numpy()[on_prev], index=local.hour[on_prev]).groupby(level=0).mean()
    by_hour = by_hour.reindex(range(24)).ffill().bfill()
    hours = context.hour_start_utc.tz_convert(context.tz).hour
    return by_hour.loc[hours].to_numpy(dtype=np.float64)
