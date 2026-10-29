"""Five-minute real-time prices from SCED: energy LMPs and AS MCPCs.

ERCOT's Security-Constrained Economic Dispatch (SCED) runs about every five
minutes, but not on a fixed clock: a run can be late, skipped, or followed
by extra runs. Each run publishes an LMP per settlement point
(``ercot_lmp_by_settlement_point``) and, since RTC+B (2025-12-05), a Market
Clearing Price for Capacity per ancillary service (``ercot_mcpc_sced``).

Price in effect
    A run's price takes effect at its SCED timestamp and stays in effect
    until the next run under the reconstruction assumption. The 15-minute real-time
    price is a time-weighted aggregate of SCED prices during
    the interval. :func:`five_minute_grid` applies the same weighting to a
    regular 5-minute UTC grid, so the mean of the three 5-minute values of a
    15-minute interval is that interval's time-weighted input price. This
    reconstruction does not establish the settlement of variable-power
    strategies; adders, caps and settlement quantities require separate checks.

SCED clock
    The observed SCED clock: every SCED timestamp seen in any of the
    series exported together (energy LMPs have a row for every run; an AS
    series can lack some). A run missing from one series still happened,
    so its price is unknown, not "not set".

Statuses on the 5-minute grid (most severe first)
    ``missing``   no price known for any part of the interval: left empty
    ``partial``   part of the interval has no known price: averaged over
                  the known part only
    ``imputed``   SCED ran, but this series lacks that run's price; the
                  last known price of the series is used in its place
    ``carried``   no run appears on the supplied clock in the interval;
                  carry the previous price. This alone cannot prove that
                  ERCOT did not run SCED or that the provider missed no rows.
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
    offline: bool = True,
) -> pd.DataFrame:
    """Rows of a SCED dataset with SCED timestamps in [start_utc, end_utc).

    Cached as parquet, one file per UTC month, under
    ``<raw_dir>/gridstatusio/<dataset>/<filter_value>/``. Raises if the
    response lacks a required column. Cache-only by default: missing cache
    files raise before constructing an API client. Set offline=False to
    explicitly allow downloads.
    """
    if offline and force:
        raise ValueError("force cannot be used with offline=True")
    cache = Path(raw_dir) / "gridstatusio" / dataset / filter_value
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for lo, hi in utc_month_chunks(pd.Timestamp(start_utc), pd.Timestamp(end_utc)):
        path = cache / f"{lo:%Y%m%dT%H%M}_{hi:%Y%m%dT%H%M}.parquet"
        if path.exists() and not force:
            log(f"[cache] {path}")
            raw = pd.read_parquet(path)
        else:
            if offline:
                raise FileNotFoundError(
                    f"Offline mode: missing SCED cache {path}. Copy the existing "
                    "cloud cache here, or explicitly enable downloads. No API request was made."
                )
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
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("start/end must be timezone-aware and end must follow start")
    if pd.isna(max_carry) or max_carry <= pd.Timedelta(0):
        raise ValueError("max_carry must be positive")
    if start.floor(FIVE_MIN) != start or end.floor(FIVE_MIN) != end:
        raise ValueError("start and end must be on the 5-minute grid")
    clock = (s.index if clock is None else pd.DatetimeIndex(clock).union(s.index)).as_unit("ns")
    c = clock.asi8
    vc = s.reindex(clock).to_numpy(dtype=np.float64)  # NaN: SCED ran, price not in this series
    known = ~np.isnan(vc)
    # last known price at or before each run, and the run that published it
    idx = np.where(known, np.arange(len(c)), -1)
    idx = np.maximum.accumulate(idx) if len(idx) else idx
    kv = np.where(idx >= 0, vc[np.maximum(idx, 0)], np.nan) if len(c) else np.array([])
    kv_ts = np.where(idx >= 0, c[np.maximum(idx, 0)], 0) if len(c) else np.array([])

    edges = pd.date_range(start, end, freq=FIVE_MIN).as_unit("ns").asi8
    carry, width = max_carry.value, FIVE_MIN.value
    first = np.searchsorted(c, edges[:-1], side="left")
    after = np.searchsorted(c, edges[1:], side="left")
    n = len(edges) - 1
    value = np.full(n, np.nan)
    last_value = np.full(n, np.nan)
    coverage = np.zeros(n)
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
            segment_imputed = not known[r]
            if known[r]:
                price, published_at = vc[r], c[r]
            elif idx[r] >= 0:
                price, published_at = kv[r], kv_ts[r]
            else:
                continue
            # Retain the valid prefix when the age limit falls inside a segment.
            b = min(b, published_at + carry)
            if b <= a:
                continue
            imputed = imputed or segment_imputed
            area += price * (b - a)
            covered += b - a
            if r < i0 or segment_imputed:
                oldest = published_at if oldest is None else min(oldest, published_at)
        coverage[k] = covered / 1e9
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
        if status[k] in ("imputed", "carried", "partial") and oldest is not None:
            source[k] = oldest

    src = pd.DatetimeIndex(source.view("datetime64[ns]")).tz_localize("UTC")  # int64 min is NaT
    return pd.DataFrame({
        "interval_start_utc": pd.DatetimeIndex(edges[:-1], tz="UTC"),
        "interval_end_utc": pd.DatetimeIndex(edges[1:], tz="UTC"),
        "value": value,
        "last_sced_value": last_value,
        "sced_runs": runs,
        "status": status,
        "covered_seconds": coverage,
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
