"""Parquet read/write with one fixed schema for cleaned prices.

Schema (one row per 15-minute interval and location):

    interval_start_utc   datetime, tz=UTC
    interval_end_utc     datetime, tz=UTC
    local_time           interval start in ERCOT local time (US/Central)
    location             settlement point name, e.g. "HB_WEST"
    market               e.g. "REAL_TIME_15_MIN"
    price_usd_per_mwh    float64
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CHECKSUM_FORMAT = "rtcb-gym-prices-v1"

COLUMNS = [
    "interval_start_utc",
    "interval_end_utc",
    "local_time",
    "location",
    "market",
    "price_usd_per_mwh",
]


def to_utc_ns(values) -> pd.Series:
    """Parse to tz-aware UTC timestamps with nanosecond resolution."""
    return pd.Series(pd.to_datetime(values, utc=True)).dt.as_unit("ns")


def _is_utc(series: pd.Series) -> bool:
    dtype = series.dtype
    return isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"


def validate_prices(df: pd.DataFrame) -> None:
    """Raise ValueError if ``df`` does not follow the schema."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; have {list(df.columns)}")
    for col in ("interval_start_utc", "interval_end_utc"):
        if not _is_utc(df[col]):
            raise ValueError(f"{col} must be tz-aware UTC, got {df[col].dtype}")
    if not isinstance(df["local_time"].dtype, pd.DatetimeTZDtype):
        raise ValueError("local_time must be tz-aware")
    if df["price_usd_per_mwh"].dtype != "float64":
        raise ValueError("price_usd_per_mwh must be float64")
    if df["price_usd_per_mwh"].isna().any():
        raise ValueError("price_usd_per_mwh has NaN; gaps must be missing rows")
    for _, g in df.groupby("location", sort=False):
        starts = g["interval_start_utc"]
        if not starts.is_monotonic_increasing or starts.duplicated().any():
            raise ValueError("interval_start_utc must be sorted and unique per location")


def processed_path(processed_dir: str | Path, location: str, market: str) -> Path:
    return Path(processed_dir) / f"ercot_{market.lower()}_{location.lower()}.parquet"


def write_prices(df: pd.DataFrame, path: str | Path) -> Path:
    validate_prices(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(path, index=False, engine="pyarrow")
    return path


def read_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    for col in ("interval_start_utc", "interval_end_utc"):
        df[col] = to_utc_ns(df[col])
    df["local_time"] = df["local_time"].dt.as_unit("ns")
    validate_prices(df)
    return df[COLUMNS].reset_index(drop=True)


def data_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of cleaned prices in a canonical text form.

    The parquet bytes depend on library versions, so the hash is taken over
    text instead: a header line, then one line per row,
    ``start_utc,end_utc,location,market,price``, with times as
    ``YYYY-MM-DDTHH:MM:SSZ`` and prices in Python's shortest round-trip
    float form. Same data -> same hash on any machine.
    """
    validate_prices(df)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    h = hashlib.sha256(f"{CHECKSUM_FORMAT}\n".encode())
    rows = zip(
        df["interval_start_utc"].dt.strftime(fmt),
        df["interval_end_utc"].dt.strftime(fmt),
        df["location"].astype(str),
        df["market"].astype(str),
        df["price_usd_per_mwh"].tolist(),
    )
    for start, end, location, market, price in rows:
        h.update(f"{start},{end},{location},{market},{price!r}\n".encode())
    return h.hexdigest()
