"""Gymnasium environment for one battery in the ERCOT real-time market,
optionally with the day-ahead market (two-settlement).

Historical replay, price-taker: the battery's actions never move the price.

Episode
    ``episode_days`` consecutive operating days (default 1). A normal day has
    96 intervals; DST days have 92 or 100. Only complete days whose preceding
    ``history_len`` intervals are also present can start or be part of an
    episode; the rest are listed in ``env.excluded_days``.

Action
    Box [-1, 1], shape (1,). Scaled by ``power_mw``. Positive = discharge
    (sell), negative = charge (buy). Values outside [-1, 1] are clipped and
    counted as clipped actions.

Observation (float64, layout in ``env.obs_layout``)
    soc            1   state of charge, fraction of energy_mwh
    time_of_day    2   sin, cos of the interval's local start time
    day_of_week    7   one-hot, Monday first, local operating day
    prices         N   last N settled RT prices (USD/MWh), oldest first

No look-ahead: before dispatching interval t the agent sees prices of
intervals t-N .. t-1 only. The price of interval t is revealed (in ``info``
and in the next observation) after the action is settled.

Reward
    Net settlement revenue of the interval in USD (energy revenue minus
    degradation cost), see :mod:`rtcb_gym.settlement`.

SoC at the episode edges
    Each episode starts at ``soc_init`` and must end at or above
    ``soc_terminal_min`` (both in :class:`~rtcb_gym.battery.BatteryParams`;
    default 0 and 0). The end target is enforced step by step, so near the
    end the env may limit discharge or force charging; such steps are counted
    as clipped with reason ``soc_terminal``.

Two-settlement (pass ``da_prices``)
    Before each operating day of the episode, the day-ahead decision for
    that day is taken (see :mod:`rtcb_gym.day_ahead` for the timeline
    and what is known): an hourly schedule in MW, from
    ``reset(options={"da_schedule": ...})``, else from ``day_ahead_policy``
    (called with :meth:`day_ahead_context`), else zero. It is clipped to
    what the battery could deliver and awarded at the day-ahead price. The
    real-time action works as above; each interval settles its deviation
    from the award at the real-time price
    (:func:`rtcb_gym.settlement.settle_two_settlement`). Days without a
    complete set of day-ahead prices are excluded.

    The observation gains three blocks of ``DA_AHEAD`` (96) values for the
    intervals t .. t+95 of the current operating day (zero beyond its end):
    ``da_price_ahead`` (USD/MWh), ``da_award_ahead`` (award / power_mw) and
    ``da_ahead_mask`` (1 inside the day). All of them are known at t: D's
    day-ahead results are published the afternoon before.

    In an episode of several days, day D+1's day-ahead prices and awards
    enter these blocks when they are published (``da_publish_time`` on D,
    default 13:30 local), not before. Every day's award is checked against
    what the battery could deliver from ``soc_init`` (a fixed position
    limit: the award is financial), so the bound stays exact.

Options beyond the v0.2 setting (all off by default)
    ``dt_hours=1/12``
        Five-minute dispatch. ``prices`` are then five-minute prices, and
        ``settlement_minutes=15`` settles every step at the mean price of
        its 15-minute interval, as ERCOT settles energy.
    ``ecrs_prices``
        Adds ECRS (see :mod:`rtcb_gym.ancillary`): the action gains a second
        value in [0, 1], the ECRS request as a share of ``power_mw``, clipped
        to what is left after the energy dispatch. Paid the real-time ECRS
        price; with ``da_ecrs_prices`` a day-ahead ECRS award is paid the
        day-ahead price and the real-time deviation from it the real-time
        price. ``ecrs_deployment`` (share of the award deployed in each
        interval) makes awards cost energy.
    ``action_mode="offer"``
        The action is an offer curve instead of a power: ``[sell share, sell
        price, buy share, buy price]``. The battery discharges the sell
        share if the interval's price turns out at or above the sell price,
        charges the buy share if it is at or below the buy price, else
        idles. The agent still does not see the price; the market applies
        the curve to it, as SCED does with an offer. Prices are limited to
        ERCOT's offer floor and cap (-250 and 5,000 USD/MWh); a sell price
        at the floor or a buy price at the cap clears at any price.
    ``obs_lag_steps``
        Prices become visible that many steps late.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from rtcb_gym.ancillary import ECRSRules, reservation_step
from rtcb_gym.battery import BatteryParams, battery_step, terminal_floor
from rtcb_gym.data.clean import DA_INTERVAL, ERCOT_TZ, intervals_per_day, to_date
from rtcb_gym.day_ahead import (
    DayAheadBid,
    DayAheadContext,
    build_context,
    clear_bid,
    day_hours_utc,
    decision_time_utc,
    feasible_day_ahead,
    feasible_day_ahead_with_ecrs,
)
from rtcb_gym.settlement import settle_real_time, settle_two_settlement

ACTION_TOL = 1e-6  # float32 round-off allowance on the [-1, 1] action bounds
# ERCOT energy offer floor and system-wide offer cap (USD/MWh): the range of
# the prices in an offer-curve action.
OFFER_FLOOR, OFFER_CAP = -250.0, 5000.0
STEP_MINUTES = {0.25: 15, 1 / 12: 5}


def _time_features(local: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    angle = 2 * np.pi * minutes / 1440.0
    return np.sin(angle), np.cos(angle), local.day_of_week.to_numpy()


def _series(x: pd.DataFrame | pd.Series, what: str) -> pd.Series:
    """A price table (processed schema) or a Series indexed by interval start."""
    if isinstance(x, pd.DataFrame):
        x = pd.Series(x["price_usd_per_mwh"].to_numpy(dtype=np.float64),
                      index=pd.DatetimeIndex(x["interval_start_utc"]))
    idx = pd.DatetimeIndex(x.index)
    if idx.tz is None:
        raise ValueError(f"{what} must be indexed by timezone-aware interval starts")
    out = pd.Series(x.to_numpy(dtype=np.float64), index=idx.tz_convert("UTC").as_unit("ns")).sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"duplicate interval starts in {what}")
    return out


def _align(source: pd.Series, target_ns: np.ndarray) -> np.ndarray:
    """Values of ``source`` at the given interval starts (NaN where absent)."""
    src = source.index.asi8
    idx = np.searchsorted(src, target_ns)
    ok = idx < len(src)
    ok[ok] = src[idx[ok]] == target_ns[ok]
    out = np.full(len(target_ns), np.nan)
    out[ok] = source.to_numpy()[idx[ok]]
    return out


class RTCBEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        battery: BatteryParams | None = None,
        days: Iterable[str | dt.date] | None = None,
        episode_days: int = 1,
        history_len: int = 96,
        dt_hours: float = 0.25,
        tz: str = ERCOT_TZ,
        sample: str = "random",
        da_prices: pd.DataFrame | None = None,
        day_ahead_policy: Callable[[DayAheadContext], Any] | None = None,
        da_decision_time: str = "10:00",
        da_history_days: int = 7,
        da_publish_time: str = "13:30",
        action_mode: str = "power",
        settlement_minutes: int | None = None,
        ecrs_prices: pd.DataFrame | pd.Series | None = None,
        ecrs_rules: ECRSRules | None = None,
        ecrs_deployment: pd.Series | None = None,
        da_ecrs_prices: pd.DataFrame | pd.Series | None = None,
        obs_lag_steps: int = 0,
    ) -> None:
        super().__init__()
        step = next((m for h, m in STEP_MINUTES.items() if abs(dt_hours - h) < 1e-9), None)
        if step is None:
            raise ValueError("dt_hours must be 0.25 (15-minute intervals) or 1/12 (5-minute intervals)")
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        if episode_days < 1:
            raise ValueError("episode_days must be >= 1")
        if sample not in ("random", "sequential"):
            raise ValueError("sample must be 'random' or 'sequential'")
        if action_mode not in ("power", "offer"):
            raise ValueError("action_mode must be 'power' or 'offer'")
        if obs_lag_steps < 0:
            raise ValueError("obs_lag_steps must be >= 0")
        settlement_minutes = step if settlement_minutes is None else int(settlement_minutes)
        if settlement_minutes not in (step, 15):
            raise ValueError(f"settlement_minutes must be {step} (each step) or 15")
        locations = prices["location"].unique()
        if len(locations) != 1:
            raise ValueError(f"prices must hold exactly one location, got {list(locations)}")

        self.battery_params = battery or BatteryParams()
        self.history_len = int(history_len)
        self.episode_days = int(episode_days)
        self.dt_hours = step / 60
        self.tz = tz
        self.sample = sample
        self.location = str(locations[0])
        self.action_mode = action_mode
        self.obs_lag_steps = int(obs_lag_steps)
        self.settlement_minutes = settlement_minutes
        self._interval = pd.Timedelta(minutes=step)
        self._steps_per_day = 24 * 60 // step  # of a normal day: DA look-ahead length

        df = prices.sort_values("interval_start_utc").reset_index(drop=True)
        ts = pd.DatetimeIndex(df["interval_start_utc"]).as_unit("ns")
        if ts.has_duplicates:
            raise ValueError("duplicate interval_start_utc in prices")
        self._ts = ts
        self._ts_ns = ts.asi8
        self._price = df["price_usd_per_mwh"].to_numpy(dtype=np.float64)
        self._settle = self._settlement_price(self._price)
        local = ts.tz_convert(tz)
        self._tod_sin, self._tod_cos, self._dow = _time_features(local)

        self.ecrs = ecrs_prices is not None
        self.ecrs_rules = ecrs_rules or ECRSRules()
        self._deploy: np.ndarray | None = None
        if self.ecrs:
            self._ecrs_series = _series(ecrs_prices, "ecrs_prices")
            self._ecrs_price = _align(self._ecrs_series, self._ts_ns)
            self._ecrs_settle = self._settlement_price(self._ecrs_price)
            if ecrs_deployment is not None:
                self._deploy = _align(_series(ecrs_deployment, "ecrs_deployment"), self._ts_ns)
                known = self._deploy[~np.isnan(self._deploy)]
                if known.size and (known.min() < 0 or known.max() > 1):
                    raise ValueError("ecrs_deployment must be a share in [0, 1]")
        elif ecrs_deployment is not None or da_ecrs_prices is not None:
            raise ValueError("ecrs_deployment and da_ecrs_prices need ecrs_prices")

        self.two_settlement = da_prices is not None
        self.da_ecrs = da_ecrs_prices is not None
        self.day_ahead_policy = day_ahead_policy
        self.da_decision_time = da_decision_time
        self.da_publish_time = da_publish_time
        self.da_history_days = int(da_history_days)
        if self.da_ecrs and not self.two_settlement:
            raise ValueError("da_ecrs_prices needs da_prices (two-settlement mode)")
        if self.two_settlement:
            da_locations = list(da_prices["location"].unique())
            if da_locations != [self.location]:
                raise ValueError(f"da_prices must be for {self.location}, got {da_locations}")
            if da_prices["interval_start_utc"].duplicated().any():
                raise ValueError("duplicate interval_start_utc in da_prices")
            self._da_series = _series(da_prices, "da_prices")
            self._rt_series = pd.Series(self._price, index=ts)
            # day-ahead price of each real-time interval's hour (NaN if missing)
            hour_ns = (self._ts_ns // DA_INTERVAL.value) * DA_INTERVAL.value
            self._da_price_rt = _align(self._da_series, hour_ns)
            if self.da_ecrs:
                self._da_ecrs_series = _series(da_ecrs_prices, "da_ecrs_prices")
                self._da_ecrs_rt = _align(self._da_ecrs_series, hour_ns)
            # when each interval's day-ahead results are published: the day before, local time
            publish = {}
            for d in np.unique(np.array(local.date)):
                publish[d] = decision_time_utc(d, da_publish_time, tz).as_unit("ns").value
            self._da_known_from = np.array([publish[d] for d in local.date], dtype=np.int64)
        self._day_rows: dict[dt.date, tuple[int, int]] = {}
        self._episodes, self.excluded_days = self._find_episodes(local, days)
        if not self._episodes:
            raise ValueError("no usable episode days (check gaps and history_len)")
        self.days: list[dt.date] = sorted(self._episodes)
        shortest = min(end - start for start, end in self._episodes.values())
        p = self.battery_params
        if p.soc_init < terminal_floor(p, shortest, self.dt_hours) - 1e-12:
            raise ValueError(
                f"soc_terminal_min {p.soc_terminal_min} cannot be reached from soc_init "
                f"{p.soc_init} within {shortest} intervals"
            )

        n = self.history_len
        self.obs_layout = {
            "soc": slice(0, 1),
            "time_of_day": slice(1, 3),
            "day_of_week": slice(3, 10),
            "prices": slice(10, 10 + n),
        }
        low = [np.array([0.0, -1.0, -1.0]), np.zeros(7), np.full(n, -np.inf)]
        high = [np.array([1.0, 1.0, 1.0]), np.ones(7), np.full(n, np.inf)]

        def block(name: str, size: int, lo: float, hi: float) -> None:
            k = sum(len(x) for x in low)
            self.obs_layout[name] = slice(k, k + size)
            low.append(np.full(size, lo))
            high.append(np.full(size, hi))

        a = self._steps_per_day
        if self.two_settlement:
            block("da_price_ahead", a, -np.inf, np.inf)
            block("da_award_ahead", a, -1.0, 1.0)
            block("da_ahead_mask", a, 0.0, 1.0)
        if self.ecrs:
            block("ecrs_prices", n, -np.inf, np.inf)
        if self.da_ecrs:
            block("da_ecrs_price_ahead", a, -np.inf, np.inf)
            block("da_ecrs_award_ahead", a, 0.0, 1.0)
        low, high = np.concatenate(low), np.concatenate(high)
        self._obs_size = len(low)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        if action_mode == "power":
            a_low, a_high, self.action_layout = [-1.0], [1.0], {"energy": 0}
        else:
            a_low, a_high = [0.0, OFFER_FLOOR, 0.0, OFFER_FLOOR], [1.0, OFFER_CAP, 1.0, OFFER_CAP]
            self.action_layout = {"sell": 0, "sell_price": 1, "buy": 2, "buy_price": 3}
        if self.ecrs:
            self.action_layout["ecrs"] = len(a_low)
            a_low, a_high = a_low + [0.0], a_high + [1.0]
        self._a_low, self._a_high = np.array(a_low), np.array(a_high)
        self.action_space = gym.spaces.Box(
            low=self._a_low.astype(np.float32), high=self._a_high.astype(np.float32), dtype=np.float32
        )

        self._t: int | None = None
        self._end: int | None = None
        self._soc = self.battery_params.soc_init
        self._episode_start: dt.date | None = None
        self._cursor = 0
        self._ep_first = 0
        self._ep_award = np.zeros(0)  # day-ahead award (MW) of each interval of the episode
        self._ep_ecrs_award = np.zeros(0)  # day-ahead ECRS award (MW), likewise

    # ------------------------------------------------------------------ setup
    def _settlement_price(self, price: np.ndarray) -> np.ndarray:
        """The price each step settles at: its own, or the mean of its
        15-minute interval (NaN if that interval is not complete)."""
        if self.settlement_minutes == self._interval.value // 60_000_000_000:
            return price
        s = pd.Series(price)
        key = self._ts_ns // pd.Timedelta(minutes=self.settlement_minutes).value
        group = s.groupby(key)
        per = self.settlement_minutes * 60_000_000_000 // self._interval.value
        return group.transform("mean").where(group.transform("count") == per).to_numpy()

    def _find_episodes(
        self, local: pd.DatetimeIndex, days: Iterable[str | dt.date] | None
    ) -> tuple[dict[dt.date, tuple[int, int]], dict[str, str]]:
        """Map each usable start day to its [start, end) row range."""
        op_day = np.array(local.date)
        uniq, first, counts = np.unique(op_day, return_index=True, return_counts=True)
        complete: dict[dt.date, tuple[int, int]] = {}
        excluded: dict[str, str] = {}
        allowed = None if days is None else {to_date(d) for d in days}
        for d, i0, cnt in zip(uniq, first, counts):
            if allowed is not None and d not in allowed:
                continue
            expected = intervals_per_day(d, self.tz, self._interval)
            rows = slice(i0, i0 + cnt)
            if cnt != expected:
                excluded[str(d)] = f"incomplete day ({cnt}/{expected} intervals)"
                continue
            if np.isnan(self._settle[rows]).any():
                excluded[str(d)] = "incomplete settlement intervals"
                continue
            if self.two_settlement and np.isnan(self._da_price_rt[rows]).any():
                excluded[str(d)] = "incomplete day-ahead prices"
                continue
            if self.ecrs and (np.isnan(self._ecrs_price[rows]).any() or np.isnan(self._ecrs_settle[rows]).any()):
                excluded[str(d)] = "incomplete ECRS prices"
                continue
            if self.da_ecrs and np.isnan(self._da_ecrs_rt[rows]).any():
                excluded[str(d)] = "incomplete day-ahead ECRS prices"
                continue
            if self._deploy is not None and np.isnan(self._deploy[rows]).any():
                excluded[str(d)] = "incomplete ECRS deployment"
                continue
            complete[d] = (int(i0), int(i0 + cnt))
        if allowed is not None:
            for d in sorted(allowed - set(uniq)):
                excluded[str(d)] = "no data"
        self._day_rows = complete

        n = self.history_len + self.obs_lag_steps
        hist_span = n * self._interval.value
        episodes: dict[dt.date, tuple[int, int]] = {}
        for d, (i0, _) in complete.items():
            if i0 < n or self._ts_ns[i0] - self._ts_ns[i0 - n] != hist_span:
                excluded[str(d)] = f"fewer than {n} contiguous history intervals"
                continue
            if self.ecrs and np.isnan(self._ecrs_price[i0 - n : i0]).any():
                excluded[str(d)] = f"fewer than {n} history intervals of ECRS prices"
                continue
            span = self._span(d)
            if all(s in complete for s in span):
                episodes[d] = (i0, complete[span[-1]][1])
            else:
                excluded[str(d)] = f"not followed by {self.episode_days - 1} complete days"
        return episodes, dict(sorted(excluded.items()))

    def _span(self, start: dt.date) -> list[dt.date]:
        return [start + dt.timedelta(days=k) for k in range(self.episode_days)]

    def episode_operating_days(self, start: str | dt.date) -> list[dt.date]:
        """The operating days of the episode that starts on ``start``."""
        return self._span(to_date(start))

    # -------------------------------------------------------------- gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._cursor = 0
        options = options or {}
        if options.get("day") is not None:
            start = to_date(options["day"])
            if start not in self._episodes:
                reason = self.excluded_days.get(str(start), "not in this env's days")
                raise ValueError(f"cannot start an episode on {start}: {reason}")
        elif self.sample == "sequential":
            start = self.days[self._cursor % len(self.days)]
            self._cursor += 1
        else:
            start = self.days[int(self.np_random.integers(len(self.days)))]

        self._episode_start = start
        self._t, self._end = self._episodes[start]
        self._ep_first = self._t
        self._soc = self.battery_params.soc_init
        info = {
            "operating_day": str(start),
            "interval_start_utc": self._ts[self._t],
            "steps_remaining": self._end - self._t,
            "soc": self._soc,
        }
        if self.two_settlement:
            info.update(self._set_day_ahead(start, options.get("da_schedule")))
        return self._obs(), info

    # ------------------------------------------------------------ day-ahead
    def day_ahead_context(self, day: str | dt.date) -> DayAheadContext:
        """What is known at the day-ahead decision for operating day ``day``."""
        if not self.two_settlement:
            raise RuntimeError("day-ahead context needs da_prices (two-settlement mode)")
        return build_context(
            day, self._rt_series, self._da_series, self.battery_params,
            decision_time=self.da_decision_time, history_days=self.da_history_days, tz=self.tz,
            interval=self._interval,
            ecrs=self._ecrs_series if self.ecrs else None,
            da_ecrs=self._da_ecrs_series if self.da_ecrs else None,
        )

    def _set_day_ahead(self, start: dt.date, schedule: Any) -> dict[str, Any]:
        """Clear and award the day-ahead bid of every day of the episode.

        ``schedule``: one bid (an hourly array or a
        :class:`~rtcb_gym.day_ahead.DayAheadBid`), or a list with one bid per
        operating day. Missing bids come from ``day_ahead_policy``, else are
        zero. A day's bid uses only what its
        :meth:`day_ahead_context` holds, so taking all of an episode's
        decisions here changes nothing about what each one may know.
        """
        span = self._span(start)
        if isinstance(schedule, (list, tuple)) and (
            len(span) > 1 or (len(schedule) == 1 and np.ndim(schedule[0]) > 0)
            or (len(schedule) == 1 and isinstance(schedule[0], DayAheadBid))
        ):
            bids = list(schedule)
        elif schedule is None:
            bids = [None] * len(span)
        else:
            bids = [schedule]
        if len(bids) != len(span):
            raise ValueError(f"da_schedule needs one bid per operating day ({len(span)}), as a list")
        p = self.battery_params
        n_rows = self._end - self._ep_first
        self._ep_award = np.zeros(n_rows)
        self._ep_ecrs_award = np.zeros(n_rows)
        out: dict[str, list] = {k: [] for k in ("hours", "requested", "cleared", "award",
                                                "ecrs_requested", "ecrs_cleared", "ecrs_award")}
        n_clipped = 0
        for day, bid in zip(span, bids):
            hours = day_hours_utc(day, self.tz)
            if bid is None and self.day_ahead_policy is not None:
                bid = self.day_ahead_policy(self.day_ahead_context(day))
            i0, i1 = self._day_rows[day]
            hour_of_row = (self._ts_ns[i0:i1] - hours.asi8[0]) // DA_INTERVAL.value
            first_row = np.searchsorted(hour_of_row, np.arange(len(hours)))
            da_price = self._da_price_rt[i0:i1][first_row]
            da_ecrs_price = self._da_ecrs_rt[i0:i1][first_row] if self.da_ecrs else None
            try:
                requested, cleared, e_req, e_cleared = clear_bid(bid, da_price, da_ecrs_price)
            except ValueError as err:
                if "hourly values" in str(err) and "ECRS" not in str(err):
                    size = np.size(bid.energy_mw if isinstance(bid, DayAheadBid) else bid)
                    raise ValueError(
                        f"da_schedule for {day} needs {len(hours)} hourly values, got {size}"
                    ) from None
                raise
            if self.da_ecrs:
                award, e_award, clipped, _ = feasible_day_ahead_with_ecrs(
                    cleared, e_cleared, p, self.ecrs_rules, p.soc_init
                )
            else:
                award, clipped, _ = feasible_day_ahead(cleared, p, p.soc_init)
                e_award = np.zeros(len(hours))
            k = i0 - self._ep_first
            self._ep_award[k : k + (i1 - i0)] = award[hour_of_row]
            self._ep_ecrs_award[k : k + (i1 - i0)] = e_award[hour_of_row]
            n_clipped += clipped
            for key, value in (("hours", hours), ("requested", requested), ("cleared", cleared),
                               ("award", award), ("ecrs_requested", e_req), ("ecrs_cleared", e_cleared),
                               ("ecrs_award", e_award)):
                out[key].append(value)
        cat = {k: np.concatenate(v) for k, v in out.items() if k != "hours"}
        hours_all = out["hours"][0]
        for h in out["hours"][1:]:
            hours_all = hours_all.append(h)
        info = {
            "da_hour_start_utc": hours_all,
            "da_requested_mw": cat["requested"],
            "da_cleared_mw": cat["cleared"],
            "da_award_mw": cat["award"],
            "da_clipped_hours": n_clipped,
        }
        if self.da_ecrs:
            info.update(da_ecrs_requested_mw=cat["ecrs_requested"], da_ecrs_cleared_mw=cat["ecrs_cleared"],
                        da_ecrs_award_mw=cat["ecrs_award"])
        return info

    def _command(self, a: np.ndarray, price: float) -> tuple[float, float, bool]:
        """Turn an action into (energy MW commanded, ECRS MW requested,
        action out of bounds). An offer curve is applied to ``price``."""
        out = bool(np.any(a < self._a_low - ACTION_TOL) or np.any(a > self._a_high + ACTION_TOL))
        a = np.clip(a, self._a_low, self._a_high)
        p_max = self.battery_params.power_mw
        lay = self.action_layout
        ecrs = float(a[lay["ecrs"]]) * p_max if self.ecrs else 0.0
        if self.action_mode == "power":
            return float(a[0]) * p_max, ecrs, out
        sell_price, buy_price = float(a[lay["sell_price"]]), float(a[lay["buy_price"]])
        if buy_price > sell_price:  # a curve cannot buy above the price it sells at
            buy_price, out = sell_price, True
        # an offer at the floor, or a bid at the cap, takes any price
        if price >= sell_price or sell_price <= OFFER_FLOOR:
            command = float(a[lay["sell"]]) * p_max
        elif price <= buy_price or buy_price >= OFFER_CAP:
            command = -float(a[lay["buy"]]) * p_max
        else:
            command = 0.0
        return command, ecrs, out

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t is None or self._end is None:
            raise RuntimeError("call reset() before step()")
        if self._t >= self._end:
            raise RuntimeError("episode is over; call reset()")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != len(self._a_low) or not np.all(np.isfinite(a)):
            what = "one finite number" if len(self._a_low) == 1 else f"{len(self._a_low)} finite numbers"
            raise ValueError(f"action must be {what}, got {action!r}")
        params = self.battery_params
        t = self._t
        command_mw, ecrs_request, action_clipped = self._command(a, float(self._price[t]))
        # what was asked for: the raw power action (before bounds, as in v0.1),
        # or what the offer curve dispatched at this interval's price
        requested_mw = float(a[0]) * params.power_mw if self.action_mode == "power" else command_mw

        steps_left = self._end - t
        deployed = 0.0
        if self.ecrs:
            share = None if self._deploy is None else float(self._deploy[t])
            actual2, self._soc, binfo = reservation_step(
                params, self.ecrs_rules, self._soc, np.array([command_mw, ecrs_request]), steps_left,
                dt_hours=self.dt_hours, deployment=share,
            )
            actual, ecrs_award, deployed = float(actual2[0]), float(actual2[1]), binfo["deployed_mw"]
        else:
            actual, self._soc, binfo = battery_step(params, self._soc, command_mw, self.dt_hours, steps_left=steps_left)
        delivered = actual + deployed
        price = float(self._settle[t])
        deg = params.degradation_cost_usd_per_mwh
        if self.two_settlement:
            award = float(self._ep_award[t - self._ep_first])
            da_price = float(self._da_price_rt[t])
            s = settle_two_settlement(delivered, award, self.dt_hours, price, da_price, deg)
        else:
            s = settle_real_time(delivered, self.dt_hours, price, deg)
        reward = s.net_revenue_usd

        if binfo["clipped"]:
            reason = binfo["clip_reason"]
        elif action_clipped:
            reason = "action_bounds"
        else:
            reason = None
        self._t = t + 1
        terminated = self._t >= self._end
        info = {
            "operating_day": str(self._episode_start),
            "interval_start_utc": self._ts[t],
            "price_usd_per_mwh": price,
            "requested_power_mw": requested_mw,
            "actual_power_mw": actual,
            "clipped": reason is not None,
            "clip_reason": reason,
            "energy_revenue_usd": s.energy_revenue_usd,
            "da_revenue_usd": s.da_revenue_usd,
            "rt_revenue_usd": s.rt_revenue_usd,
            "degradation_cost_usd": s.degradation_cost_usd,
            "reward_usd": reward,
            "storage_discharge_mwh": binfo["storage_discharge_mwh"],
            "soc": self._soc,
            "steps_remaining": self._end - self._t,
        }
        if self.settlement_minutes != self._interval.value // 60_000_000_000:
            info["dispatch_price_usd_per_mwh"] = float(self._price[t])
        if self.two_settlement:
            info["da_award_mw"] = award
            info["da_price_usd_per_mwh"] = da_price
        if self.ecrs:
            ecrs_price = float(self._ecrs_settle[t])
            da_ecrs_award = float(self._ep_ecrs_award[t - self._ep_first]) if self.da_ecrs else 0.0
            da_ecrs_price = float(self._da_ecrs_rt[t]) if self.da_ecrs else 0.0
            ecrs_da = da_ecrs_award * self.dt_hours * da_ecrs_price
            ecrs_rt = (ecrs_award - da_ecrs_award) * self.dt_hours * ecrs_price
            reward += ecrs_da + ecrs_rt
            info.update(
                reward_usd=reward,
                delivered_power_mw=delivered,
                ecrs_requested_mw=ecrs_request,
                ecrs_award_mw=ecrs_award,
                ecrs_clipped=binfo["ecrs_clipped"],
                ecrs_deployed_mw=deployed,
                ecrs_price_usd_per_mw_h=ecrs_price,
                ecrs_revenue_usd=ecrs_da + ecrs_rt,
                ecrs_da_revenue_usd=ecrs_da,
                ecrs_rt_revenue_usd=ecrs_rt,
            )
            if self.da_ecrs:
                info.update(da_ecrs_award_mw=da_ecrs_award, da_ecrs_price_usd_per_mw_h=da_ecrs_price)
        return self._obs(), float(reward), terminated, False, info

    # -------------------------------------------------------------- helpers
    def _obs(self) -> np.ndarray:
        t = self._t
        assert t is not None
        obs = np.zeros(self._obs_size, dtype=np.float64)
        obs[0] = self._soc
        if t < len(self._ts) and (t == 0 or self._ts_ns[t] - self._ts_ns[t - 1] == self._interval.value):
            sin, cos, dow = self._tod_sin[t], self._tod_cos[t], self._dow[t]
        else:  # terminal observation after the last row or before a gap
            nxt = pd.DatetimeIndex([self._ts[t - 1] + self._interval]).tz_convert(self.tz)
            s, c, w = _time_features(nxt)
            sin, cos, dow = s[0], c[0], w[0]
        obs[1], obs[2] = sin, cos
        obs[3:10] = 0.0
        obs[3 + int(dow)] = 1.0
        # Settled prices only: intervals t-N .. t-1 (earlier by obs_lag_steps).
        # Interval t is not visible.
        lay = self.obs_layout
        seen = t - self.obs_lag_steps
        obs[lay["prices"]] = self._price[seen - self.history_len : seen]
        if self.ecrs:
            obs[lay["ecrs_prices"]] = self._ecrs_price[seen - self.history_len : seen]
        if self.two_settlement:
            # Day-ahead prices and awards of intervals t .. t+ahead-1 of this
            # episode, as far as they are published; zero beyond.
            m = min(self._steps_per_day, max(self._end - t, 0))
            if m:
                now = self._ts_ns[t]
                m = int(np.searchsorted(self._da_known_from[t : t + m], now, side="right"))
            k = t - self._ep_first
            p_max = self.battery_params.power_mw
            obs[lay["da_price_ahead"]][:m] = self._da_price_rt[t : t + m]
            obs[lay["da_award_ahead"]][:m] = self._ep_award[k : k + m] / p_max
            obs[lay["da_ahead_mask"]][:m] = 1.0
            if self.da_ecrs:
                obs[lay["da_ecrs_price_ahead"]][:m] = self._da_ecrs_rt[t : t + m]
                obs[lay["da_ecrs_award_ahead"]][:m] = self._ep_ecrs_award[k : k + m] / p_max
        return obs
