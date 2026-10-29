"""Validate the public offline entry point and its auditable artifacts."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rtcb_gym.demo import run_demo


def test_demo_reconciles_outputs_and_never_needs_api(tmp_path, monkeypatch):
    from rtcb_gym.data import fetch
    monkeypatch.delenv("GRIDSTATUS_API_KEY", raising=False)
    monkeypatch.setattr(fetch, "make_client", lambda: pytest.fail("demo constructed API client"))
    summary = run_demo(tmp_path)
    assert summary["data_kind"] == "synthetic" and summary["oracle_replay_verified"]
    trace = pd.read_csv(tmp_path/"trace.csv")
    assert len(trace) == 180
    for policy in summary["policies"]:
        rows = trace[trace.policy == policy["policy"]]
        assert rows.net_proxy_usd.sum() == pytest.approx(policy["net_proxy_usd"])
        assert np.isfinite(rows.soc_after).all()
        assert rows.soc_after.between(-1e-7, 1+1e-7).all()
        assert rows.soc_after.iloc[-1] >= .5-1e-7
    assert (trace.loc[trace.executed_energy_mw < -1e-8, "executed_ecrs_mw"] == 0).all()
    assert json.loads((tmp_path/"manifest.json").read_text())["source"].endswith("no download")
    for line in (tmp_path/"SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == expected
