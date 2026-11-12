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

from rtcb_gym.ancillary import ECRSRules, reservation_step
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
    # Only in environments with ECRS: real-time ECRS prices settled by the
    # decision time, and day-ahead ECRS prices of the days before D.
    ecrs_history: pd.Series | None = None
    da_ecrs_history: pd.Series | None = None

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


@dataclass(frozen=True)
class DayAheadBid:
    """A day-ahead bid with optional limit prices and an optional ECRS offer.

    ``energy_mw``
        Hourly MW, shape (n_hours,) or (n_hours, K) for K segments per hour.
        Positive = offer to sell, negative = bid to buy.
    ``energy_limit_price``
        Same shape, USD/MWh. A sell segment clears if the day-ahead price is
        at or above its limit, a buy segment if it is at or below. None =
        every segment clears (a quantity-only schedule, as a plain array).
    ``ecrs_mw``, ``ecrs_limit_price``
        Hourly ECRS offer in MW (>= 0) and the lowest day-ahead ECRS price
        (USD/MW per hour) at which it clears. Needs an environment with
        day-ahead ECRS prices.

    The battery is a price-taker: whatever clears is paid the day-ahead
    price, not the limit price, and clearing never moves the price.
    """

    energy_mw: np.ndarray | None = None
    energy_limit_price: np.ndarray | None = None
    ecrs_mw: np.ndarray | None = None
    ecrs_limit_price: np.ndarray | None = None


def clear_bid(
    bid: DayAheadBid | np.ndarray | None,
    da_price: np.ndarray,
    da_ecrs_price: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clear a bid against the day's day-ahead prices.

    Returns hourly ``(offered_mw, cleared_mw, ecrs_offered_mw,
    ecrs_cleared_mw)``; offered is the net of all segments. Raises
    ValueError on a wrong shape, non-finite quantities or a negative ECRS
    offer.
    """
    n = len(da_price)
    if bid is None:
        bid = DayAheadBid()
    elif not isinstance(bid, DayAheadBid):
        bid = DayAheadBid(energy_mw=bid)
    zeros = np.zeros(n)
    offered = cleared = zeros
    if bid.energy_mw is not None:
        q = np.asarray(bid.energy_mw, dtype=np.float64)
        if q.ndim == 1 or (q.ndim == 2 and bid.energy_limit_price is None and 1 in q.shape):
            q = q.reshape(-1, 1)
        if q.ndim != 2 or q.shape[0] != n:
            raise ValueError(f"day-ahead schedule needs {n} hourly values, got {q.size}")
        if not np.all(np.isfinite(q)):
            raise ValueError("day-ahead schedule must be finite")
        offered = q.sum(axis=1)
        if bid.energy_limit_price is None:
            cleared = offered
        else:
            lim = np.asarray(bid.energy_limit_price, dtype=np.float64).reshape(q.shape)
            if np.isnan(lim).any():
                raise ValueError("day-ahead limit prices must not be NaN")
            price = np.asarray(da_price, dtype=np.float64)[:, None]
            hit = ((q > 0) & (price >= lim)) | ((q < 0) & (price <= lim))
            cleared = np.where(hit, q, 0.0).sum(axis=1)
    ecrs_offered = ecrs_cleared = zeros
    if bid.ecrs_mw is not None:
        if da_ecrs_price is None:
            raise ValueError("an ECRS day-ahead offer needs an environment with day-ahead ECRS prices")
        a = np.asarray(bid.ecrs_mw, dtype=np.float64).reshape(-1)
        if a.shape != (n,) or not np.all(np.isfinite(a)) or a.min() < 0:
            raise ValueError(f"day-ahead ECRS offer needs {n} finite hourly values >= 0")
        ecrs_offered = ecrs_cleared = a
        if bid.ecrs_limit_price is not None:
            lim = np.asarray(bid.ecrs_limit_price, dtype=np.float64).reshape(-1)
            if lim.shape != (n,) or np.isnan(lim).any():
                raise ValueError(f"day-ahead ECRS limit prices need {n} hourly values")
            ecrs_cleared = np.where(np.asarray(da_ecrs_price) >= lim, a, 0.0)
    return offered, cleared, ecrs_offered, ecrs_cleared


def build_context(
    day: str | dt.date,
    rt: pd.Series,
    da: pd.Series,
    battery: BatteryParams,
    decision_time: str = "10:00",
    history_days: int = 7,
    tz: str = ERCOT_TZ,
    interval: pd.Timedelta = INTERVAL,
    ecrs: pd.Series | None = None,
    da_ecrs: pd.Series | None = None,
) -> DayAheadContext:
    """Cut the information available at the decision time out of full price
    series (``rt`` and ``da``: prices indexed by interval start, UTC, sorted)."""
    if history_days < 1:
        raise ValueError("history_days must be >= 1")
    d = to_date(day)
    decision = decision_time_utc(d, decision_time, tz)
    rt_lo = decision - pd.Timedelta(days=history_days)
    # an interval is settled once it has ended
    rt_known = rt[(rt.index >= rt_lo) & (rt.index + interval <= decision)]
    da_lo, _ = operating_day_bounds_utc(d - dt.timedelta(days=history_days), tz)
    day_start, _ = operating_day_bounds_utc(d, tz)
    da_known = da[(da.index >= da_lo) & (da.index < day_start)]
    ecrs_known = da_ecrs_known = None
    if ecrs is not None:
        ecrs_known = ecrs[(ecrs.index >= rt_lo) & (ecrs.index + interval <= decision)].copy()
    if da_ecrs is not None:
        da_ecrs_known = da_ecrs[(da_ecrs.index >= da_lo) & (da_ecrs.index < day_start)].copy()
    return DayAheadContext(
        operating_day=d,
        decision_time_utc=decision,
        hour_start_utc=day_hours_utc(d, tz),
        da_history=da_known.copy(),
        rt_history=rt_known.copy(),
        soc_init=battery.soc_init,
        battery=battery,
        tz=tz,
        ecrs_history=ecrs_known,
        da_ecrs_history=da_ecrs_known,
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


def feasible_day_ahead_with_ecrs(
    schedule_mw: np.ndarray, ecrs_mw: np.ndarray, battery: BatteryParams, rules: ECRSRules, soc0: float
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """:func:`feasible_day_ahead` for an energy schedule plus an hourly ECRS
    offer: each hour runs through :func:`rtcb_gym.ancillary.reservation_step`
    with a 1-hour step (energy first, then ECRS clipped to what is left).
    Returns ``(award_mw, ecrs_award_mw, n_clipped_hours, soc_after_each_hour)``;
    an hour counts as clipped if either part was. With a zero ECRS offer the
    energy award is exactly :func:`feasible_day_ahead`'s."""
    x = np.asarray(schedule_mw, dtype=np.float64).reshape(-1)
    a = np.asarray(ecrs_mw, dtype=np.float64).reshape(-1)
    if x.shape != a.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(a)):
        raise ValueError("day-ahead energy and ECRS schedules must be finite and the same length")
    n = len(x)
    award, ecrs_award, soc_path = np.empty(n), np.empty(n), np.empty(n)
    soc, clipped = float(soc0), 0
    for h in range(n):
        actual, soc, info = reservation_step(
            battery, rules, soc, np.array([x[h], a[h]]), n - h, dt_hours=DA_DT_HOURS
        )
        award[h], ecrs_award[h], soc_path[h] = actual[0], actual[1], soc
        clipped += int(info["clipped"] or info["ecrs_clipped"])
    return award, ecrs_award, clipped, soc_path


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
