"""Exercise the full export command on a synthetic, complete local cache."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from rtcb_gym.config import DEFAULT_CONFIG
from rtcb_gym.data import sced


def test_export_cli_uses_cache_without_api(tmp_path, monkeypatch):
    lo = pd.Timestamp("2026-03-02T06:00:00Z")
    hi = lo + pd.Timedelta(days=1)
    ts = pd.date_range(lo, hi, freq="5min", inclusive="left")
    raw = tmp_path / "raw"
    frames = [
        (sced.SCED_LMP_DATASET, "HB_WEST", pd.DataFrame({
            "sced_timestamp_utc": ts, "interval_start_utc": ts,
            "interval_end_utc": ts+sced.FIVE_MIN, "market": "REAL_TIME_SCED",
            "location": "HB_WEST", "lmp": 20.})),
        ("ercot_lmp_with_adders_by_settlement_point", "HB_WEST", pd.DataFrame({
            "sced_timestamp_utc": ts, "location": "HB_WEST", "lmp": 20.,
            "rtorpa": np.nan, "rtordpa": np.nan, "rtrdpa": 1., "lmp_with_adders": 21.})),
        (sced.SCED_MCPC_DATASET, "ECRS", pd.DataFrame({
            "sced_timestamp_utc": ts, "as_type": "ECRS", "mcpc": 3., "uncapped_mcpc": 3.})),
    ]
    for dataset, key, frame in frames:
        folder = raw / "gridstatusio" / dataset / key
        folder.mkdir(parents=True)
        frame.to_parquet(folder/f"{lo:%Y%m%dT%H%M}_{hi:%Y%m%dT%H%M}.parquet", index=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"extends": str(DEFAULT_CONFIG),
        "market": {"start_date": "2026-03-02", "end_date": "2026-03-02"},
        "data": {"raw_dir": str(raw), "processed_dir": str(tmp_path/"processed")}}))
    out = tmp_path / "exports"
    monkeypatch.setattr(sced, "make_client", lambda: pytest.fail("export constructed API client"))
    monkeypatch.delenv("GRIDSTATUS_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["export_5min.py", "--config", str(cfg),
                                     "--hubs", "HB_WEST", "--out", str(out)])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/export_5min.py"), run_name="__main__")
    energy = pd.read_csv(out/"energy_5min_hb_west.csv")
    assert len(energy) == 288 and (energy.energy_price_usd_per_mwh == 21.).all()
    assert (energy.lmp_status == "ok").all()
    report = json.loads((out/"report.json").read_text())
    assert report["period"]["five_minute_intervals"] == 288
    assert "cash-flow proxies" in " ".join(report["not_checked"])
    for line in (out/"SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == expected
