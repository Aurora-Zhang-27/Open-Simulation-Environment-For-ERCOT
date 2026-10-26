"""Five-minute real-time prices from SCED: energy LMPs and AS MCPCs.

ERCOT's Security-Constrained Economic Dispatch (SCED) runs about every five
minutes, but not on a fixed clock: a run can be late, skipped, or followed
by extra runs. Each run publishes an LMP per settlement point
(``ercot_lmp_by_settlement_point``) and, since RTC+B (2025-12-05), a Market
Clearing Price for Capacity per ancillary service (``ercot_mcpc_sced``).

Price in effect
    A run's price takes effect at its SCED timestamp and stays in effect
    until the next run. This is how ERCOT settles: the 15-minute real-time
    price is the time-weighted average of the SCED prices in effect during
    the interval. :func:`five_minute_grid` applies the same weighting to a
    regular 5-minute UTC grid, so the mean of the three 5-minute values of a
    15-minute interval is that interval's time-weighted price.

SCED clock
    The SCED runs that happened: every SCED timestamp seen in any of the
    series exported together (energy LMPs have a row for every run; an AS
    series can lack some). A run missing from one series still happened,
    so its price is unknown, not "not set".

Statuses on the 5-minute grid (most severe first)
    ``missing``   no price known for any part of the interval: left empty
    ``partial``   part of the interval has no known price: averaged over
                  the known part only
    ``imputed``   SCED ran, but this series lacks that run's price; the
                  last known price of the series is used in its place
    ``carried``   no SCED run started in the interval; the previous run's
                  price was still in effect (ERCOT's rule, not an estimate)
    ``rerun``     more than one run started in the interval
    ``ok``        exactly one run started in the interval

A price is never used more than ``max_carry`` after the run that published
it (default 60 minutes); beyond that the time counts as unknown.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from rtcb_gym.data.fetch import make_client

SCED_LMP_DATASET = "ercot_lmp_by_settlement_point"
SCED_MCPC_DATASET = "ercot_mcpc_sced"
LMP_REQUIRED = ("sced_timestamp_utc", "interval_start_utc", "interval_end_utc", "market", "location", "lmp")
MCPC_REQUIRED = ("sced_timestamp_utc", "as_type", "mcpc", "uncapped_mcpc")
FIVE_MIN = pd.Timedelta(minutes=5)
STATUSES = ("ok", "rerun", "carried", "imputed", "partial", "missing")


def utc_month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) at UTC month starts."""
    chunks, cur = [], start
    while cur < end:
        nxt = (cur.tz_convert("UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
               + pd.offsets.MonthBegin(1))
        chunks.append((cur, min(nxt, end)))
        cur = min(nxt, end)
    return chunks


def fetch_sced(
    dataset: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    raw_dir: str | Path,
    filter_column: str,
    filter_value: str,
    required: tuple[str, ...],
    client: Any = None,
    sleep_sec: float = 2.0,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Rows of a SCED dataset with SCED timestamps in [start_utc, end_utc).

    Cached as parquet, one file per UTC month, under
    ``<raw_dir>/gridstatusio/<dataset>/<filter_value>/``. Raises if the
    response lacks a required column.
    """
    cache = Path(raw_dir) / "gridstatusio" / dataset / filter_value
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for lo, hi in utc_month_chunks(pd.Timestamp(start_utc), pd.Timestamp(end_utc)):
        path = cache / f"{lo:%Y%m%dT%H%M}_{hi:%Y%m%dT%H%M}.parquet"
        if path.exists() and not force:
            log(f"[cache] {path}")
            raw = pd.read_parquet(path)
        else:
            client = client or make_client()
            log(f"[fetch] {dataset} {filter_value} {lo} .. {hi}")
            raw = pd.DataFrame(client.get_dataset(
                dataset, start=lo.isoformat(), end=hi.isoformat(),
                filter_column=filter_column, filter_value=filter_value, verbose=False,
            ))
            raw.to_parquet(path, index=False)
            log(f"        {len(raw)} rows -> {path}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(
                f"{dataset} response is missing columns {missing}; got {list(raw.columns)}. "
                "Check the dataset schema before changing code."
            )
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def sced_series(raw: pd.DataFrame, value_col: str, ts_col: str = "sced_timestamp_utc") -> tuple[pd.Series, dict]:
    """One price per SCED run, sorted by SCED timestamp (UTC, ns).

    Exact duplicates are dropped and counted; the same run with two
    different prices raises. Rows without a price are dropped and counted.
    """
    df = pd.DataFrame({
        "ts": pd.to_datetime(raw[ts_col], utc=True).dt.as_unit("ns"),
        "v": pd.to_numeric(raw[value_col], errors="coerce").astype("float64"),
    })
    n_raw = len(df)
    n_nan = int(df["v"].isna().sum())
    df = df[df["v"].notna()]
    exact = df.duplicated()
    df = df[~exact]
    conflict = df["ts"].duplicated(keep=False)
    if conflict.any():
        raise ValueError(f"SCED runs with two different prices:\n{df[conflict].head(6).to_string(index=False)}")
    df = df.sort_values("ts")
    s = pd.Series(df["v"].to_numpy(), index=pd.DatetimeIndex(df["ts"]), name=value_col)
    return s, {"rows": n_raw, "rows_without_price": n_nan, "exact_duplicates": int(exact.sum()), "sced_runs": len(s)}


def five_minute_grid(
    s: pd.Series,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    clock: pd.DatetimeIndex | None = None,
    max_carry: pd.Timedelta = pd.Timedelta(minutes=60),
) -> pd.DataFrame:
    """Time-weighted price in effect for every 5-minute UTC interval in
    [start_utc, end_utc). ``clock`` is the SCED clock (default: the runs of
    ``s``). See the module docstring for the statuses."""
    start, end = pd.Timestamp(start_utc).as_unit("ns"), pd.Timestamp(end_utc).as_unit("ns")
    if start.floor(FIVE_MIN) != start or end.floor(FIVE_MIN) != end:
        raise ValueError("start and end must be on the 5-minute grid")
    clock = (s.index if clock is None else pd.DatetimeIndex(clock).union(s.index)).as_unit("ns")
    c = clock.asi8
    vc = s.reindex(clock).to_numpy(dtype=np.float64)  # NaN: SCED ran, price not in this series
    known = ~np.isnan(vc)
    # last known price at or before each run, and the run that published it
    idx = np.where(known, np.arange(len(c)), -1)
    idx = np.maximum.accumulate(idx) if len(idx) else idx
    kv = np.where(idx >= 0, vc[np.maximum(idx, 0)], np.nan)
    kv_ts = np.where(idx >= 0, c[np.maximum(idx, 0)], 0)

    edges = pd.date_range(start, end, freq=FIVE_MIN).as_unit("ns").asi8
    carry, width = max_carry.value, FIVE_MIN.value
    first = np.searchsorted(c, edges[:-1], side="left")
    after = np.searchsorted(c, edges[1:], side="left")
    n = len(edges) - 1
    value = np.full(n, np.nan)
    last_value = np.full(n, np.nan)
    runs = (after - first).astype(np.int64)
    status = np.empty(n, dtype=object)
    source = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)  # oldest run whose price filled the interval

    for k in range(n):
        b0, b1, i0, i1 = edges[k], edges[k + 1], first[k], after[k]
        seg_starts = [b0] + [int(x) for x in c[i0:i1]]
        seg_ends = seg_starts[1:] + [b1]
        seg_runs = [i0 - 1] + list(range(i0, i1))
        if seg_starts[0] == seg_ends[0]:  # a run exactly at the interval start
            seg_starts, seg_ends, seg_runs = seg_starts[1:], seg_ends[1:], seg_runs[1:]
        area = covered = 0.0
        imputed = False
        oldest = None
        for a, b, r in zip(seg_starts, seg_ends, seg_runs):
            if r < 0:
                continue  # before the first run in the data
            if known[r] and b - c[r] <= carry:
                price, published_at = vc[r], c[r]
            elif not known[r] and idx[r] >= 0 and b - kv_ts[r] <= carry:
                price, published_at, imputed = kv[r], kv_ts[r], True
            else:
                continue
            area += price * (b - a)
            covered += b - a
            if r < i0 or imputed:  # value came from before the interval or was imputed
                oldest = published_at if oldest is None else min(oldest, published_at)
        if i1 > i0:
            last_value[k] = vc[i1 - 1]
        if covered == 0:
            status[k] = "missing"
            continue
        value[k] = area / covered
        if covered < width:
            status[k] = "partial"
        elif imputed:
            status[k] = "imputed"
        elif runs[k] == 0:
            status[k] = "carried"
        else:
            status[k] = "rerun" if runs[k] > 1 else "ok"
        if status[k] in ("imputed", "carried") and oldest is not None:
            source[k] = oldest

    src = pd.DatetimeIndex(source.view("datetime64[ns]")).tz_localize("UTC")  # int64 min is NaT
    return pd.DataFrame({
        "interval_start_utc": pd.DatetimeIndex(edges[:-1], tz="UTC"),
        "interval_end_utc": pd.DatetimeIndex(edges[1:], tz="UTC"),
        "value": value,
        "last_sced_value": last_value,
        "sced_runs": runs,
        "status": status,
        "fill_source_sced_utc": src,
    })


def fifteen_minute_average(grid: pd.DataFrame) -> pd.Series:
    """Mean of the three 5-minute values of each 15-minute interval, indexed
    by interval start (UTC). NaN unless all three cover their full 5 minutes
    (status ok, rerun, carried or imputed)."""
    full = grid["status"].isin(["ok", "rerun", "carried", "imputed"])
    values = grid["value"].where(full)
    by = grid["interval_start_utc"].dt.floor("15min")
    counts = values.groupby(by).count()
    return values.groupby(by).mean().where(counts == 3)
