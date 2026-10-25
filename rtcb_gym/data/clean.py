"""Clean raw real-time prices into the schema in :mod:`rtcb_gym.data.store`.

Rules:
- Store UTC. Add ``local_time`` (interval start in US/Central).
- An interval belongs to the operating day of its local start time. DST days
  have 92 (spring forward) or 100 (fall back) 15-minute intervals; working in
  UTC handles both without special cases.
- Exact duplicate rows are dropped and counted. Duplicates with different
  prices raise an error: we do not guess which one is right.
- Gaps (missing intervals, NaN prices) are reported, never filled.
- Data before the RTC+B go-live (Operating Day 2025-12-05) is refused.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

import pandas as pd

from rtcb_gym.data.store import COLUMNS, to_utc_ns

# ERCOT RTC+B go-live: Operating Day 2025-12-05.
RTCB_GO_LIVE = dt.date(2025, 12, 5)
ERCOT_TZ = "US/Central"
INTERVAL = pd.Timedelta(minutes=15)

RAW_COLUMNS = [
    "interval_start_utc",
    "interval_end_utc",
    "location",
    "market",
    "price_usd_per_mwh",
]


def to_date(d: str | dt.date) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


def check_post_rtcb(start_date: str | dt.date) -> dt.date:
    start = to_date(start_date)
    if start < RTCB_GO_LIVE:
        raise ValueError(
            f"start_date {start} is before the RTC+B go-live ({RTCB_GO_LIVE}). "
            "Pre-RTC+B data is a different market design and must not be mixed in."
        )
    return start


def operating_day_bounds_utc(
    day: str | dt.date, tz: str = ERCOT_TZ
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [start, end) of an operating day defined in local time."""
    d = to_date(day)
    start = pd.Timestamp(d).tz_localize(tz)
    end = pd.Timestamp(d + dt.timedelta(days=1)).tz_localize(tz)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def expected_intervals_utc(
    start_date: str | dt.date, end_date: str | dt.date, tz: str = ERCOT_TZ
) -> pd.DatetimeIndex:
    """All 15-minute interval starts (UTC) for operating days start..end inclusive."""
    lo, _ = operating_day_bounds_utc(start_date, tz)
    _, hi = operating_day_bounds_utc(end_date, tz)
    return pd.date_range(lo, hi, freq=INTERVAL, inclusive="left")


def intervals_per_day(day: str | dt.date, tz: str = ERCOT_TZ) -> int:
    lo, hi = operating_day_bounds_utc(day, tz)
    return int((hi - lo) / INTERVAL)


@dataclass
class CleanReport:
    location: str
    start_date: str
    end_date: str
    n_raw_rows: int = 0
    n_other_location: int = 0
    n_outside_range: int = 0
    n_nan_price: int = 0
    n_exact_duplicates: int = 0
    n_expected_intervals: int = 0
    n_present_intervals: int = 0
    missing_intervals_utc: list[str] = field(default_factory=list)
    incomplete_days: list[str] = field(default_factory=list)

    @property
    def n_missing_intervals(self) -> int:
        return len(self.missing_intervals_utc)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_missing_intervals"] = self.n_missing_intervals
        return d

    def summary(self) -> str:
        lines = [
            f"location {self.location}, operating days {self.start_date} .. {self.end_date}",
            f"  raw rows:              {self.n_raw_rows}",
            f"  other locations:       {self.n_other_location} (dropped)",
            f"  outside date range:    {self.n_outside_range} (dropped)",
            f"  NaN prices:            {self.n_nan_price} (dropped, reported as gaps)",
            f"  exact duplicates:      {self.n_exact_duplicates} (dropped)",
            f"  intervals expected:    {self.n_expected_intervals}",
            f"  intervals present:     {self.n_present_intervals}",
            f"  intervals missing:     {self.n_missing_intervals} (NOT filled)",
            f"  incomplete days:       {len(self.incomplete_days)}",
        ]
        if self.incomplete_days:
            shown = ", ".join(self.incomplete_days[:10])
            more = " ..." if len(self.incomplete_days) > 10 else ""
            lines.append(f"    {shown}{more}")
        return "\n".join(lines)


def clean_prices(
    raw: pd.DataFrame,
    location: str,
    start_date: str | dt.date,
    end_date: str | dt.date,
    market: str = "REAL_TIME_15_MIN",
    tz: str = ERCOT_TZ,
) -> tuple[pd.DataFrame, CleanReport]:
    """Return (clean_df, report). ``raw`` must have :data:`RAW_COLUMNS`."""
    start = check_post_rtcb(start_date)
    end = to_date(end_date)
    if end < start:
        raise ValueError(f"end_date {end} is before start_date {start}")

    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data missing columns {missing}; have {list(raw.columns)}")

    report = CleanReport(location=location, start_date=str(start), end_date=str(end))
    report.n_raw_rows = len(raw)

    df = raw[RAW_COLUMNS].copy()
    for col in ("interval_start_utc", "interval_end_utc"):
        df[col] = to_utc_ns(df[col])
    df["price_usd_per_mwh"] = pd.to_numeric(df["price_usd_per_mwh"]).astype("float64")
    df["location"] = df["location"].astype(str)
    df["market"] = df["market"].astype(str)

    is_loc = df["location"] == location
    report.n_other_location = int((~is_loc).sum())
    df = df[is_loc]

    bad_market = sorted(set(df["market"]) - {market})
    if bad_market:
        raise ValueError(f"unexpected market values {bad_market}; expected {market!r}")

    lo, _ = operating_day_bounds_utc(start, tz)
    _, hi = operating_day_bounds_utc(end, tz)
    in_range = (df["interval_start_utc"] >= lo) & (df["interval_start_utc"] < hi)
    report.n_outside_range = int((~in_range).sum())
    df = df[in_range]

    length = df["interval_end_utc"] - df["interval_start_utc"]
    if (length != INTERVAL).any():
        bad = df.loc[length != INTERVAL, "interval_start_utc"].head(5).tolist()
        raise ValueError(f"intervals not 15 minutes long, e.g. {bad}")
    starts = df["interval_start_utc"]
    off_grid = (starts.dt.minute % 15 != 0) | (starts.dt.second != 0)
    if off_grid.any():
        raise ValueError(
            f"interval starts not on a 15-minute grid, e.g. {starts[off_grid].head(5).tolist()}"
        )

    is_nan = df["price_usd_per_mwh"].isna()
    report.n_nan_price = int(is_nan.sum())
    df = df[~is_nan]

    exact_dup = df.duplicated(subset=["interval_start_utc", "price_usd_per_mwh"])
    report.n_exact_duplicates = int(exact_dup.sum())
    df = df[~exact_dup]
    conflict = df["interval_start_utc"].duplicated(keep=False)
    if conflict.any():
        examples = df[conflict].sort_values("interval_start_utc").head(6)
        raise ValueError(
            "duplicate intervals with different prices; check the source data:\n"
            f"{examples.to_string(index=False)}"
        )

    df = df.sort_values("interval_start_utc").reset_index(drop=True)
    df["local_time"] = df["interval_start_utc"].dt.tz_convert(tz)

    expected = expected_intervals_utc(start, end, tz)
    present = pd.DatetimeIndex(df["interval_start_utc"])
    missing_idx = expected.difference(present)
    report.n_expected_intervals = len(expected)
    report.n_present_intervals = len(present)
    report.missing_intervals_utc = [t.isoformat() for t in missing_idx]
    report.incomplete_days = sorted(
        {str(t.tz_convert(tz).date()) for t in missing_idx}
    )
    return df[COLUMNS], report
