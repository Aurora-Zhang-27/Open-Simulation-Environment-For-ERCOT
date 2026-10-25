"""Download raw ERCOT real-time Settlement Point Prices.

Provider: the hosted GridStatus API (https://www.gridstatus.io) through the
``gridstatusio`` client (``pip install -e ".[data]"``). It needs a free API
key in the ``GRIDSTATUS_API_KEY`` environment variable.

Dataset ``ercot_spp_real_time_15_min``: 15-minute real-time SPPs. The
underlying ERCOT report is NP6-905-CD ("Settlement Point Prices at Resource
Nodes, Hubs and Load Zones"; source: report id 12301 in the open-source
``gridstatus`` library, ``gridstatus/ercot.py``). The column names used below
are checked on every download; if GridStatus changes them, this module raises
instead of guessing.

Raw API responses are cached as parquet, one file per calendar month, under
``<raw_dir>/gridstatusio/<dataset>/<location>/``. Later runs read the cache
and work offline. Pass ``force=True`` to re-download.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from rtcb_gym.data.clean import (
    ERCOT_TZ,
    check_post_rtcb,
    operating_day_bounds_utc,
    to_date,
)
from rtcb_gym.data.store import to_utc_ns

API_KEY_ENV = "GRIDSTATUS_API_KEY"
DEFAULT_DATASET = "ercot_spp_real_time_15_min"
# Columns we rely on in the gridstatusio response.
GRIDSTATUSIO_REQUIRED = ("interval_start_utc", "interval_end_utc", "location", "spp")


def month_chunks(
    start_date: str | dt.date, end_date: str | dt.date
) -> list[tuple[dt.date, dt.date]]:
    """Split operating days start..end (inclusive) into calendar-month chunks."""
    start, end = to_date(start_date), to_date(end_date)
    chunks = []
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        chunk_end = min(end, next_month - dt.timedelta(days=1))
        chunks.append((cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)
    return chunks


def normalize_gridstatusio(raw: pd.DataFrame, market: str) -> pd.DataFrame:
    """Map a gridstatusio SPP response to :data:`rtcb_gym.data.clean.RAW_COLUMNS`."""
    missing = [c for c in GRIDSTATUSIO_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(
            f"gridstatusio response is missing columns {missing}; "
            f"got {list(raw.columns)}. Check the dataset schema before changing code."
        )
    out = pd.DataFrame(
        {
            "interval_start_utc": to_utc_ns(raw["interval_start_utc"]),
            "interval_end_utc": to_utc_ns(raw["interval_end_utc"]),
            "location": raw["location"].astype(str),
            "market": raw["market"].astype(str) if "market" in raw.columns else market,
            "price_usd_per_mwh": pd.to_numeric(raw["spp"]),
        }
    )
    return out


def make_client(api_key: str | None = None) -> Any:
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"No GridStatus API key. Set the {API_KEY_ENV} environment variable "
            "(free key at https://www.gridstatus.io)."
        )
    try:
        from gridstatusio import GridStatusClient
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError('gridstatusio not installed: pip install -e ".[data]"') from e
    return GridStatusClient(api_key=key)


def fetch_rt_spp(
    location: str,
    start_date: str | dt.date,
    end_date: str | dt.date,
    raw_dir: str | Path,
    dataset: str = DEFAULT_DATASET,
    market: str = "REAL_TIME_15_MIN",
    tz: str = ERCOT_TZ,
    client: Any = None,
    sleep_sec: float = 2.0,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Return raw prices for ``location`` over operating days start..end.

    Output columns are :data:`rtcb_gym.data.clean.RAW_COLUMNS`; pass the
    result to :func:`rtcb_gym.data.clean.clean_prices`.
    """
    check_post_rtcb(start_date)
    cache_dir = Path(raw_dir) / "gridstatusio" / dataset / location
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, (c_start, c_end) in enumerate(month_chunks(start_date, end_date)):
        path = cache_dir / f"{c_start}_{c_end}.parquet"
        if path.exists() and not force:
            log(f"[cache] {path}")
            raw = pd.read_parquet(path)
        else:
            if client is None:
                client = make_client()
            lo, _ = operating_day_bounds_utc(c_start, tz)
            _, hi = operating_day_bounds_utc(c_end, tz)
            log(f"[fetch] {dataset} {location} {c_start} .. {c_end}")
            raw = client.get_dataset(
                dataset,
                start=lo.isoformat(),
                end=hi.isoformat(),
                filter_column="location",
                filter_value=location,
                verbose=False,
            )
            raw = pd.DataFrame(raw)
            raw.to_parquet(path, index=False)
            log(f"        {len(raw)} rows -> {path}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        frames.append(normalize_gridstatusio(raw, market))

    if not frames:
        raise RuntimeError(f"no data for {location} {start_date} .. {end_date}")
    return pd.concat(frames, ignore_index=True)
