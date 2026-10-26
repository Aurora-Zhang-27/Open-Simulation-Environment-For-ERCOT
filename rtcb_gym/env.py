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
    Episodes are one operating day. Before it starts, the day-ahead decision
    for that day is taken (see :mod:`rtcb_gym.day_ahead` for the timeline
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
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from rtcb_gym.battery import BatteryParams, battery_step, terminal_floor
from rtcb_gym.data.clean import DA_INTERVAL, ERCOT_TZ, INTERVAL, intervals_per_day, to_date
from rtcb_gym.day_ahead import DA_AHEAD, DayAheadContext, build_context, day_hours_utc, feasible_day_ahead
from rtcb_gym.settlement import settle_real_time, settle_two_settlement

ACTION_TOL = 1e-6  # float32 round-off allowance on the [-1, 1] action bounds


def _time_features(local: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    angle = 2 * np.pi * minutes / 1440.0
    return np.sin(angle), np.cos(angle), local.day_of_week.to_numpy()


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
    ) -> None:
        super().__init__()
        if dt_hours != 0.25:
            raise ValueError("v0 supports 15-minute intervals only (dt_hours=0.25)")
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        if episode_days < 1:
            raise ValueError("episode_days must be >= 1")
        if sample not in ("random", "sequential"):
            raise ValueError("sample must be 'random' or 'sequential'")
        locations = prices["location"].unique()
        if len(locations) != 1:
            raise ValueError(f"prices must hold exactly one location, got {list(locations)}")

        self.battery_params = battery or BatteryParams()
        self.history_len = int(history_len)
        self.episode_days = int(episode_days)
        self.dt_hours = float(dt_hours)
        self.tz = tz
        self.sample = sample
        self.location = str(locations[0])

        df = prices.sort_values("interval_start_utc").reset_index(drop=True)
        ts = pd.DatetimeIndex(df["interval_start_utc"]).as_unit("ns")
        if ts.has_duplicates:
            raise ValueError("duplicate interval_start_utc in prices")
        self._ts = ts
        self._ts_ns = ts.asi8
        self._price = df["price_usd_per_mwh"].to_numpy(dtype=np.float64)
        local = ts.tz_convert(tz)
        self._tod_sin, self._tod_cos, self._dow = _time_features(local)

        self.two_settlement = da_prices is not None
        self.day_ahead_policy = day_ahead_policy
        self.da_decision_time = da_decision_time
        self.da_history_days = int(da_history_days)
        if self.two_settlement:
            if self.episode_days != 1:
                raise ValueError("two-settlement episodes are one operating day (episode_days=1)")
            da_locations = list(da_prices["location"].unique())
            if da_locations != [self.location]:
                raise ValueError(f"da_prices must be for {self.location}, got {da_locations}")
            da = da_prices.sort_values("interval_start_utc")
            da_ts = pd.DatetimeIndex(da["interval_start_utc"]).as_unit("ns")
            if da_ts.has_duplicates:
                raise ValueError("duplicate interval_start_utc in da_prices")
            da_price = da["price_usd_per_mwh"].to_numpy(dtype=np.float64)
            self._da_series = pd.Series(da_price, index=da_ts)
            self._rt_series = pd.Series(self._price, index=ts)
            # day-ahead price of each real-time interval's hour (NaN if missing)
            hour_ns = (self._ts_ns // DA_INTERVAL.value) * DA_INTERVAL.value
            idx = np.searchsorted(da_ts.asi8, hour_ns)
            ok = idx < len(da_ts)
            ok[ok] = da_ts.asi8[idx[ok]] == hour_ns[ok]
            self._da_price_rt = np.full(len(ts), np.nan)
            self._da_price_rt[ok] = da_price[idx[ok]]
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
        low = np.concatenate([[0.0], [-1.0, -1.0], np.zeros(7), np.full(n, -np.inf)])
        high = np.concatenate([[1.0], [1.0, 1.0], np.ones(7), np.full(n, np.inf)])
        if self.two_settlement:
            k, a = 10 + n, DA_AHEAD
            self.obs_layout.update(
                da_price_ahead=slice(k, k + a),
                da_award_ahead=slice(k + a, k + 2 * a),
                da_ahead_mask=slice(k + 2 * a, k + 3 * a),
            )
            low = np.concatenate([low, np.full(a, -np.inf), np.full(a, -1.0), np.zeros(a)])
            high = np.concatenate([high, np.full(a, np.inf), np.ones(a), np.ones(a)])
        self._obs_size = len(low)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._t: int | None = None
        self._end: int | None = None
        self._soc = self.battery_params.soc_init
        self._episode_start: dt.date | None = None
        self._cursor = 0
        self._ep_first = 0
        self._ep_award = np.zeros(0)  # day-ahead award (MW) of each interval of the episode

    # ------------------------------------------------------------------ setup
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
            if cnt != intervals_per_day(d, self.tz):
                excluded[str(d)] = f"incomplete day ({cnt}/{intervals_per_day(d, self.tz)} intervals)"
                continue
            if self.two_settlement and np.isnan(self._da_price_rt[i0 : i0 + cnt]).any():
                excluded[str(d)] = "incomplete day-ahead prices"
                continue
            complete[d] = (int(i0), int(i0 + cnt))
        if allowed is not None:
            for d in sorted(allowed - set(uniq)):
                excluded[str(d)] = "no data"

        n = self.history_len
        hist_span = n * INTERVAL.value
        episodes: dict[dt.date, tuple[int, int]] = {}
        for d, (i0, _) in complete.items():
            if i0 < n or self._ts_ns[i0] - self._ts_ns[i0 - n] != hist_span:
                excluded[str(d)] = f"fewer than {n} contiguous history intervals"
                continue
            span = [d + dt.timedelta(days=k) for k in range(self.episode_days)]
            if all(s in complete for s in span):
                episodes[d] = (i0, complete[span[-1]][1])
            else:
                excluded[str(d)] = f"not followed by {self.episode_days - 1} complete days"
        return episodes, dict(sorted(excluded.items()))

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
        )

    def _set_day_ahead(self, day: dt.date, schedule: Any) -> dict[str, Any]:
        hours = day_hours_utc(day, self.tz)
        if schedule is None and self.day_ahead_policy is not None:
            schedule = self.day_ahead_policy(self.day_ahead_context(day))
        if schedule is None:
            schedule = np.zeros(len(hours))
        requested = np.asarray(schedule, dtype=np.float64).reshape(-1)
        if requested.shape != (len(hours),):
            raise ValueError(f"da_schedule for {day} needs {len(hours)} hourly values, got {requested.size}")
        award, n_clipped, _ = feasible_day_ahead(requested, self.battery_params, self.battery_params.soc_init)
        rows = self._ts_ns[self._t : self._end]
        hour_of_row = (rows - hours.asi8[0]) // DA_INTERVAL.value
        self._ep_award = award[hour_of_row]
        return {
            "da_hour_start_utc": hours,
            "da_requested_mw": requested,
            "da_award_mw": award,
            "da_clipped_hours": n_clipped,
        }

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t is None or self._end is None:
            raise RuntimeError("call reset() before step()")
        if self._t >= self._end:
            raise RuntimeError("episode is over; call reset()")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != 1 or not np.isfinite(a[0]):
            raise ValueError(f"action must be one finite number, got {action!r}")
        a = float(a[0])
        params = self.battery_params
        action_clipped = a < -1.0 - ACTION_TOL or a > 1.0 + ACTION_TOL
        command_mw = min(max(a, -1.0), 1.0) * params.power_mw

        t = self._t
        actual, self._soc, binfo = battery_step(
            params, self._soc, command_mw, self.dt_hours, steps_left=self._end - t
        )
        price = float(self._price[t])
        deg = params.degradation_cost_usd_per_mwh
        if self.two_settlement:
            award = float(self._ep_award[t - self._ep_first])
            da_price = float(self._da_price_rt[t])
            s = settle_two_settlement(actual, award, self.dt_hours, price, da_price, deg)
        else:
            s = settle_real_time(actual, self.dt_hours, price, deg)

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
            "requested_power_mw": a * params.power_mw,
            "actual_power_mw": actual,
            "clipped": reason is not None,
            "clip_reason": reason,
            "energy_revenue_usd": s.energy_revenue_usd,
            "da_revenue_usd": s.da_revenue_usd,
            "rt_revenue_usd": s.rt_revenue_usd,
            "degradation_cost_usd": s.degradation_cost_usd,
            "reward_usd": s.net_revenue_usd,
            "storage_discharge_mwh": binfo["storage_discharge_mwh"],
            "soc": self._soc,
            "steps_remaining": self._end - self._t,
        }
        if self.two_settlement:
            info["da_award_mw"] = award
            info["da_price_usd_per_mwh"] = da_price
        return self._obs(), float(s.net_revenue_usd), terminated, False, info

    # -------------------------------------------------------------- helpers
    def _obs(self) -> np.ndarray:
        t = self._t
        assert t is not None
        obs = np.zeros(self._obs_size, dtype=np.float64)
        obs[0] = self._soc
        if t < len(self._ts) and (t == 0 or self._ts_ns[t] - self._ts_ns[t - 1] == INTERVAL.value):
            sin, cos, dow = self._tod_sin[t], self._tod_cos[t], self._dow[t]
        else:  # terminal observation after the last row or before a gap
            nxt = pd.DatetimeIndex([self._ts[t - 1] + INTERVAL]).tz_convert(self.tz)
            s, c, w = _time_features(nxt)
            sin, cos, dow = s[0], c[0], w[0]
        obs[1], obs[2] = sin, cos
        obs[3:10] = 0.0
        obs[3 + int(dow)] = 1.0
        # Settled prices only: intervals t-N .. t-1. Interval t is not visible.
        obs[10 : 10 + self.history_len] = self._price[t - self.history_len : t]
        if self.two_settlement:
            # Day-ahead prices and awards of intervals t .. t+95 of this day
            # (published the day before), zero beyond the end of the day.
            m = min(DA_AHEAD, max(self._end - t, 0))
            k = t - self._ep_first
            lay = self.obs_layout
            obs[lay["da_price_ahead"]][:m] = self._da_price_rt[t : t + m]
            obs[lay["da_award_ahead"]][:m] = self._ep_award[k : k + m] / self.battery_params.power_mw
            obs[lay["da_ahead_mask"]][:m] = 1.0
        return obs
