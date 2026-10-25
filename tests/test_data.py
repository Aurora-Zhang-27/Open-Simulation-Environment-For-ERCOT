"""Data layer: cleaning rules, parquet schema, fetch caching (no network)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from rtcb_gym.data import clean, fetch, store
from rtcb_gym.data.clean import RAW_COLUMNS, clean_prices, intervals_per_day, operating_day_bounds_utc


def raw_from(prices: pd.DataFrame) -> pd.DataFrame:
    return prices[RAW_COLUMNS].copy()


def test_operating_day_bounds_and_dst_lengths():
    lo, hi = operating_day_bounds_utc("2026-01-15")
    assert lo == pd.Timestamp("2026-01-15 06:00", tz="UTC")  # CST = UTC-6
    assert hi - lo == pd.Timedelta(hours=24)
    lo, _ = operating_day_bounds_utc("2026-07-15")
    assert lo == pd.Timestamp("2026-07-15 05:00", tz="UTC")  # CDT = UTC-5
    assert intervals_per_day("2026-03-08") == 92
    assert intervals_per_day("2026-11-01") == 100
    assert intervals_per_day("2026-03-09") == 96


def test_clean_happy_path_and_local_time():
    p = make_prices(start="2026-03-07", days=3)
    df, report = clean_prices(raw_from(p), "HB_WEST", "2026-03-07", "2026-03-09")
    assert list(df.columns) == store.COLUMNS
    assert report.n_missing_intervals == 0 and report.incomplete_days == []
    assert report.n_expected_intervals == 96 + 92 + 96
    assert str(df["interval_start_utc"].dt.tz) == "UTC"
    assert str(df["local_time"].dt.tz) == "US/Central"
    assert (df["local_time"].dt.date == dt.date(2026, 3, 8)).sum() == 92
    assert df["local_time"].iloc[0] == pd.Timestamp("2026-03-07 00:00", tz="US/Central")


def test_pre_rtcb_start_is_refused():
    p = make_prices(start="2026-03-01", days=2)
    with pytest.raises(ValueError, match="RTC\\+B"):
        clean_prices(raw_from(p), "HB_WEST", "2025-12-04", "2026-03-02")
    with pytest.raises(ValueError, match="RTC\\+B"):
        fetch.fetch_rt_spp("HB_WEST", "2025-11-01", "2025-12-31", raw_dir="unused")


def test_pre_rtcb_rows_are_dropped_not_mixed():
    p = make_prices(start="2025-12-03", days=4)  # 12-03 and 12-04 are pre-RTC+B
    df, report = clean_prices(raw_from(p), "HB_WEST", "2025-12-05", "2025-12-06")
    assert df["local_time"].dt.date.min() == clean.RTCB_GO_LIVE
    assert report.n_outside_range == 2 * 96


def test_other_locations_dropped():
    p = pd.concat([make_prices(days=2), make_prices(days=2, location="HB_HOUSTON")])
    df, report = clean_prices(raw_from(p), "HB_WEST", "2026-03-01", "2026-03-02")
    assert set(df["location"]) == {"HB_WEST"}
    assert report.n_other_location == 2 * 96


def test_exact_duplicates_dropped_and_counted():
    raw = raw_from(make_prices(days=2))
    raw = pd.concat([raw, raw.iloc[:5]])
    df, report = clean_prices(raw, "HB_WEST", "2026-03-01", "2026-03-02")
    assert len(df) == 2 * 96 and report.n_exact_duplicates == 5


def test_conflicting_duplicates_raise():
    raw = raw_from(make_prices(days=2))
    dup = raw.iloc[:1].copy()
    dup["price_usd_per_mwh"] += 1.0
    with pytest.raises(ValueError, match="different prices"):
        clean_prices(pd.concat([raw, dup]), "HB_WEST", "2026-03-01", "2026-03-02")


def test_gaps_are_reported_not_filled():
    raw = raw_from(make_prices(days=3))
    gap = raw.iloc[100:104]
    raw = raw.drop(index=gap.index)
    raw.loc[raw.index[150], "price_usd_per_mwh"] = np.nan
    df, report = clean_prices(raw, "HB_WEST", "2026-03-01", "2026-03-03")
    assert len(df) == 3 * 96 - 5
    assert report.n_missing_intervals == 5 and report.n_nan_price == 1
    assert report.incomplete_days == ["2026-03-02"]
    assert set(gap["interval_start_utc"].map(pd.Timestamp.isoformat)) <= set(report.missing_intervals_utc)
    assert not df["price_usd_per_mwh"].isna().any()


def test_missing_days_reported():
    raw = raw_from(make_prices(days=2))
    df, report = clean_prices(raw, "HB_WEST", "2026-03-01", "2026-03-04")
    assert report.incomplete_days == ["2026-03-03", "2026-03-04"]


def test_bad_interval_length_raises():
    raw = raw_from(make_prices(days=1))
    raw["interval_end_utc"] = raw["interval_start_utc"] + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="15 minutes"):
        clean_prices(raw, "HB_WEST", "2026-03-01", "2026-03-01")


def test_unexpected_market_raises():
    raw = raw_from(make_prices(days=1))
    raw["market"] = "DAY_AHEAD_HOURLY"
    with pytest.raises(ValueError, match="market"):
        clean_prices(raw, "HB_WEST", "2026-03-01", "2026-03-01")


def test_store_round_trip(tmp_path):
    df, _ = clean_prices(raw_from(make_prices(days=2)), "HB_WEST", "2026-03-01", "2026-03-02")
    path = store.write_prices(df, store.processed_path(tmp_path, "HB_WEST", "REAL_TIME_15_MIN"))
    assert path.name == "ercot_real_time_15_min_hb_west.parquet"
    back = store.read_prices(path)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    assert back["price_usd_per_mwh"].tolist() == df["price_usd_per_mwh"].tolist()


def test_store_rejects_bad_frames(tmp_path):
    df, _ = clean_prices(raw_from(make_prices(days=1)), "HB_WEST", "2026-03-01", "2026-03-01")
    with pytest.raises(ValueError):
        store.write_prices(df.drop(columns=["market"]), tmp_path / "x.parquet")
    nan = df.copy()
    nan.loc[0, "price_usd_per_mwh"] = np.nan
    with pytest.raises(ValueError):
        store.write_prices(nan, tmp_path / "x.parquet")
    with pytest.raises(ValueError):
        store.write_prices(df.iloc[::-1], tmp_path / "x.parquet")
    with pytest.raises(FileNotFoundError):
        store.read_prices(tmp_path / "missing.parquet")


def test_month_chunks():
    chunks = fetch.month_chunks("2025-12-05", "2026-02-10")
    assert chunks == [
        (dt.date(2025, 12, 5), dt.date(2025, 12, 31)),
        (dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
        (dt.date(2026, 2, 1), dt.date(2026, 2, 10)),
    ]


def gridstatusio_like(prices: pd.DataFrame) -> pd.DataFrame:
    """Shape of a gridstatusio ercot_spp_real_time_15_min response (expected)."""
    return pd.DataFrame(
        {
            "interval_start_local": prices["local_time"],
            "interval_start_utc": prices["interval_start_utc"],
            "interval_end_local": prices["interval_end_utc"].dt.tz_convert("US/Central"),
            "interval_end_utc": prices["interval_end_utc"],
            "location": prices["location"],
            "location_type": "Trading Hub",
            "market": prices["market"],
            "spp": prices["price_usd_per_mwh"],
        }
    )


class FakeClient:
    def __init__(self, prices):
        self.prices = prices
        self.calls = []

    def get_dataset(self, dataset, start, end, filter_column, filter_value, verbose):
        self.calls.append((dataset, start, end, filter_value))
        p = self.prices
        m = (p["interval_start_utc"] >= pd.Timestamp(start)) & (p["interval_start_utc"] < pd.Timestamp(end))
        m &= p["location"] == filter_value
        return gridstatusio_like(p[m])


def test_fetch_uses_cache_on_second_run(tmp_path):
    p = make_prices(start="2026-01-30", days=4)
    client = FakeClient(p)
    kw = dict(raw_dir=tmp_path, client=client, sleep_sec=0, log=lambda s: None)
    a = fetch.fetch_rt_spp("HB_WEST", "2026-01-30", "2026-02-02", **kw)
    assert len(client.calls) == 2  # January and February chunks
    assert client.calls[0][1] == "2026-01-30T06:00:00+00:00"
    b = fetch.fetch_rt_spp("HB_WEST", "2026-01-30", "2026-02-02", **kw)
    assert len(client.calls) == 2  # served from cache
    pd.testing.assert_frame_equal(a, b)
    df, report = clean_prices(a, "HB_WEST", "2026-01-30", "2026-02-02")
    assert report.n_missing_intervals == 0 and len(df) == 4 * 96
    fetch.fetch_rt_spp("HB_WEST", "2026-01-30", "2026-02-02", force=True, **kw)
    assert len(client.calls) == 4


def test_normalize_rejects_unknown_schema():
    raw = gridstatusio_like(make_prices(days=1)).rename(columns={"spp": "price"})
    with pytest.raises(ValueError, match="missing columns"):
        fetch.normalize_gridstatusio(raw, "REAL_TIME_15_MIN")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv(fetch.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="GRIDSTATUS_API_KEY"):
        fetch.make_client()
