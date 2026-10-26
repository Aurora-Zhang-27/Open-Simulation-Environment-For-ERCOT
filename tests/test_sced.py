"""Five-minute SCED prices: price in effect, gaps, fills (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from rtcb_gym.data import sced

T0 = pd.Timestamp("2026-03-02 06:00", tz="UTC")
S = pd.Timedelta(seconds=1)
M = pd.Timedelta(minutes=1)


def runs(offsets_s, values):
    idx = pd.DatetimeIndex([T0 + k * S for k in offsets_s]).as_unit("ns")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def regular(n=12, lag=20):
    """A run 20 s after every 5-minute mark, price 10, 20, 30, ..."""
    return runs([lag + 300 * k for k in range(n)], [10.0 * (k + 1) for k in range(n)])


def grid(s, n_bins=6, **kw):
    return sced.five_minute_grid(s, T0, T0 + n_bins * 5 * M, **kw)


def test_price_in_effect_is_time_weighted():
    g = grid(regular())
    # bin 0: nothing before the first run at +20 s -> averaged over the known 280 s only
    assert g.loc[0, "status"] == "partial" and g.loc[0, "value"] == pytest.approx(10.0)
    # bin 1: 20 s at 10 (previous run), 280 s at 20
    assert g.loc[1, "status"] == "ok"
    assert g.loc[1, "value"] == pytest.approx((10 * 20 + 20 * 280) / 300)
    assert g.loc[1, "last_sced_value"] == 20.0 and g.loc[1, "sced_runs"] == 1
    assert g["interval_start_utc"].iloc[1] == T0 + 5 * M


def test_rerun_inside_an_interval():
    s = runs([20, 320, 440, 620], [10, 20, 50, 30])  # extra run at 7:20 into the hour
    g = grid(s, n_bins=3)
    assert g.loc[1, "status"] == "rerun" and g.loc[1, "sced_runs"] == 2
    assert g.loc[1, "value"] == pytest.approx((10 * 20 + 20 * 120 + 50 * 160) / 300)
    assert g.loc[1, "last_sced_value"] == 50.0


def test_skipped_run_carries_the_price_in_effect():
    s = runs([20, 320, 920], [10, 20, 40])  # the run near 10:20 is skipped
    g = grid(s, n_bins=4)
    assert g.loc[2, "status"] == "carried" and g.loc[2, "sced_runs"] == 0
    assert g.loc[2, "value"] == 20.0
    assert g.loc[2, "fill_source_sced_utc"] == T0 + 320 * S
    assert np.isnan(g.loc[2, "last_sced_value"])
    assert pd.isna(g.loc[1, "fill_source_sced_utc"])  # only filled intervals name a source


def test_prices_are_not_carried_past_max_carry():
    s = runs([20, 3620], [10, 99])  # an hour without SCED
    g = grid(s, n_bins=13, max_carry=pd.Timedelta(minutes=15))
    assert list(g["status"][1:3]) == ["carried", "carried"]  # 10 and 15 min after the run
    assert (g["status"][3:12] == "missing").all() and g["value"][3:12].isna().all()
    assert g.loc[12, "status"] == "partial"  # 20 s unknown, then the new run
    assert g.loc[12, "value"] == 99.0


def test_a_run_missing_from_one_series_is_imputed():
    clock = runs([20, 320, 620, 920], [0, 0, 0, 0]).index
    s = runs([20, 320, 920], [10, 20, 40])  # this series lacks the run at 620 s
    g = grid(s, n_bins=4, clock=clock)
    assert g.loc[2, "status"] == "imputed" and g.loc[2, "sced_runs"] == 1
    assert g.loc[2, "value"] == 20.0  # last known price
    assert g.loc[2, "fill_source_sced_utc"] == T0 + 320 * S
    assert np.isnan(g.loc[2, "last_sced_value"])
    assert g.loc[3, "status"] == "imputed"  # its first 20 s also used the imputed price
    assert g.loc[3, "value"] == pytest.approx((20 * 20 + 40 * 280) / 300)


def test_run_exactly_on_the_boundary():
    s = runs([0, 300, 600], [10, 20, 30])
    g = grid(s, n_bins=3)
    assert list(g["status"]) == ["ok", "ok", "ok"]
    assert list(g["value"]) == [10.0, 20.0, 30.0]


def test_three_five_minute_values_average_to_the_fifteen_minute_price():
    rng = np.random.default_rng(0)
    offsets = np.sort(rng.choice(np.arange(1, 3600), size=14, replace=False))
    s = runs(offsets, rng.uniform(-20, 200, size=14))
    g = grid(s, n_bins=12)
    avg = sced.fifteen_minute_average(g)
    # direct time-weighted average over 15:00 .. 30:00
    lo, hi = T0 + 15 * M, T0 + 30 * M
    t = list(s.index[(s.index > lo) & (s.index < hi)])
    prev = s[s.index <= lo].iloc[-1]
    times = [lo] + t + [hi]
    vals = [prev] + [s[x] for x in t]
    direct = sum(v * (b - a).total_seconds() for v, a, b in zip(vals, times, times[1:])) / 900
    assert avg.loc[lo] == pytest.approx(direct)
    assert np.isnan(avg.loc[T0])  # its first 5 minutes start before the first run: partial


def test_sced_series_dedupes_and_refuses_conflicts():
    raw = pd.DataFrame({
        "sced_timestamp_utc": [T0 + 20 * S, T0 + 20 * S, T0 + 320 * S, T0 + 620 * S],
        "lmp": [10.0, 10.0, 20.0, None],
    })
    s, info = sced.sced_series(raw, "lmp")
    assert list(s) == [10.0, 20.0]
    assert info == {"rows": 4, "rows_without_price": 1, "exact_duplicates": 1, "sced_runs": 2}
    raw.loc[1, "lmp"] = 11.0
    with pytest.raises(ValueError, match="two different prices"):
        sced.sced_series(raw, "lmp")


def test_utc_month_chunks():
    lo, hi = pd.Timestamp("2025-12-05 06:00", tz="UTC"), pd.Timestamp("2026-02-10 05:00", tz="UTC")
    assert sced.utc_month_chunks(lo, hi) == [
        (lo, pd.Timestamp("2026-01-01", tz="UTC")),
        (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC")),
        (pd.Timestamp("2026-02-01", tz="UTC"), hi),
    ]


class FakeClient:
    def __init__(self, frame):
        self.frame, self.calls = frame, 0

    def get_dataset(self, dataset, start, end, filter_column, filter_value, verbose):
        self.calls += 1
        f = self.frame
        m = (f["sced_timestamp_utc"] >= pd.Timestamp(start)) & (f["sced_timestamp_utc"] < pd.Timestamp(end))
        return f[m & (f[filter_column] == filter_value)]


def test_fetch_sced_caches_and_checks_columns(tmp_path):
    ts = pd.date_range("2026-01-31 23:00", periods=24, freq="5min", tz="UTC") + 20 * S
    frame = pd.DataFrame({"sced_timestamp_utc": ts, "as_type": "ECRS", "mcpc": 1.0, "uncapped_mcpc": None})
    client = FakeClient(frame)
    kw = dict(raw_dir=tmp_path, filter_column="as_type", filter_value="ECRS", client=client, sleep_sec=0, log=lambda m: None)
    lo, hi = pd.Timestamp("2026-01-31 23:00", tz="UTC"), pd.Timestamp("2026-02-01 01:00", tz="UTC")
    a = sced.fetch_sced(sced.SCED_MCPC_DATASET, lo, hi, required=sced.MCPC_REQUIRED, **kw)
    assert len(a) == 24 and client.calls == 2  # January and February chunks
    b = sced.fetch_sced(sced.SCED_MCPC_DATASET, lo, hi, required=sced.MCPC_REQUIRED, **kw)
    assert client.calls == 2 and len(b) == 24  # from the cache
    with pytest.raises(ValueError, match="missing columns"):
        sced.fetch_sced(sced.SCED_MCPC_DATASET, lo, hi, required=("price",), **kw)
