"""Synthetic prices for trying the benchmark without market data.

Seeded and deterministic. The shapes are loosely ERCOT-like (cheap nights,
an evening peak, occasional spikes and negative prices; ECRS near zero with
rare spikes) but they are NOT ERCOT data and say nothing about the market.
Scores on synthetic tasks check that an agent runs and behaves; they are
not results.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from rtcb_gym.data.clean import DA_INTERVAL, ERCOT_TZ, INTERVAL, operating_day_bounds_utc

SYNTHETIC_START = "2026-01-05"
SYNTHETIC_DAYS = 42
# operating days, inclusive; the first day has no price history and is never scored
SYNTHETIC_SPLITS = {
    "train": ["2026-01-05", "2026-01-25"],
    "validation": ["2026-01-26", "2026-02-04"],
    "test": ["2026-02-05", "2026-02-15"],
}


def _frame(ts: pd.DatetimeIndex, price: np.ndarray, step: pd.Timedelta, market: str, location: str) -> pd.DataFrame:
    return pd.DataFrame({
        "interval_start_utc": ts, "interval_end_utc": ts + step, "local_time": ts.tz_convert(ERCOT_TZ),
        "location": location, "market": market, "price_usd_per_mwh": price.astype("float64"),
    })


def synthetic_prices(seed: int = 0, location: str = "SYNTHETIC_HUB") -> dict[str, pd.DataFrame | pd.Series]:
    """Every series a task can need, on the same days:

    ``rt`` (15-minute), ``rt5`` (five-minute, 15-minute means equal ``rt``),
    ``da`` (hourly), ``ecrs`` (five-minute Series), ``da_ecrs`` (hourly Series).
    """
    rng = np.random.default_rng(seed)
    first = dt.date.fromisoformat(SYNTHETIC_START)
    lo, _ = operating_day_bounds_utc(first)
    _, hi = operating_day_bounds_utc(first + dt.timedelta(days=SYNTHETIC_DAYS - 1))

    ts = pd.date_range(lo, hi, freq=INTERVAL, inclusive="left").as_unit("ns")
    local = ts.tz_convert(ERCOT_TZ)
    hours = (local.hour + local.minute / 60).to_numpy()
    day = np.array((local.normalize() - local.normalize()[0]).days)
    level = 30 + 8 * np.sin(2 * np.pi * np.arange(SYNTHETIC_DAYS) / 11) + rng.normal(0, 4, SYNTHETIC_DAYS)
    shape = 22 * np.sin(2 * np.pi * (hours - 12) / 24) + 14 * np.exp(-((hours - 19) ** 2) / 4)
    rt = level[day] + shape + rng.normal(0, 6, len(ts))
    spikes = rng.choice(len(ts), size=len(ts) // 150, replace=False)
    rt[spikes] = rng.choice([-40.0, 250.0, 600.0], size=len(spikes), p=[0.3, 0.5, 0.2])

    noise = rng.normal(0, 5, (len(ts), 3))
    noise -= noise.mean(axis=1, keepdims=True)
    ts5 = pd.DatetimeIndex((ts.asi8[:, None] + np.arange(3) * pd.Timedelta(minutes=5).value).ravel(), tz="UTC")
    rt5 = (rt[:, None] + noise).ravel()

    th = pd.date_range(lo, hi, freq=DA_INTERVAL, inclusive="left").as_unit("ns")
    local_h = th.tz_convert(ERCOT_TZ)
    hh = (local_h.hour + 0.5).to_numpy()
    day_h = np.array((local_h.normalize() - local_h.normalize()[0]).days)
    da = (level[day_h] + 20 * np.sin(2 * np.pi * (hh - 12) / 24) + 12 * np.exp(-((hh - 19) ** 2) / 4)
          + rng.normal(0, 3, len(th)))

    ecrs = np.abs(rng.normal(0.1, 0.1, len(ts5)))
    hot = rng.choice(len(ts5), size=len(ts5) // 80, replace=False)
    ecrs[hot] = rng.gamma(2.0, 8.0, len(hot))
    da_ecrs = 0.3 + np.abs(rng.normal(0, 0.3, len(th))) + 3 * np.exp(-((hh - 19) ** 2) / 6)

    return {
        "rt": _frame(ts, rt, INTERVAL, "SYNTHETIC_REAL_TIME_15_MIN", location),
        "rt5": _frame(ts5, rt5, pd.Timedelta(minutes=5), "SYNTHETIC_REAL_TIME_5_MIN", location),
        "da": _frame(th, da, DA_INTERVAL, "SYNTHETIC_DAY_AHEAD_HOURLY", location),
        "ecrs": pd.Series(ecrs, index=ts5),
        "da_ecrs": pd.Series(da_ecrs, index=th),
    }
