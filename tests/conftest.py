"""Shared test fixtures.

All prices here are SYNTHETIC test data, built only to exercise the code.
They are not ERCOT data and are never used for reported results.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from rtcb_gym.data.clean import ERCOT_TZ, INTERVAL, operating_day_bounds_utc


def make_prices(
    start: str = "2026-03-01",
    days: int = 10,
    location: str = "HB_WEST",
    seed: int | None = 0,
    periodic: bool = False,
) -> pd.DataFrame:
    """Synthetic 15-minute prices in the processed schema.

    Daily shape: cheap overnight, expensive evening peak. With ``seed`` some
    noise and a few negative and spike prices are added. ``periodic=True``
    gives the same price for every interval start time-of-day (UTC), i.e.
    prices repeat exactly every 96 intervals.
    """
    first = dt.date.fromisoformat(start)
    lo, _ = operating_day_bounds_utc(first)
    _, hi = operating_day_bounds_utc(first + dt.timedelta(days=days - 1))
    ts = pd.date_range(lo, hi, freq=INTERVAL, inclusive="left")
    if periodic:
        hours = (ts.hour + ts.minute / 60).to_numpy()
    else:
        local = ts.tz_convert(ERCOT_TZ)
        hours = (local.hour + local.minute / 60).to_numpy()
    price = 30.0 + 25.0 * np.sin(2 * np.pi * (hours - 12.0) / 24.0)
    if seed is not None and not periodic:
        rng = np.random.default_rng(seed)
        price = price + rng.normal(0.0, 6.0, len(ts))
        spikes = rng.choice(len(ts), size=max(1, len(ts) // 200), replace=False)
        price[spikes] = rng.choice([-60.0, 400.0], size=len(spikes))
    return pd.DataFrame(
        {
            "interval_start_utc": ts,
            "interval_end_utc": ts + INTERVAL,
            "local_time": ts.tz_convert(ERCOT_TZ),
            "location": location,
            "market": "REAL_TIME_15_MIN",
            "price_usd_per_mwh": price.astype("float64"),
        }
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()
