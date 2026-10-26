"""Download, clean and store ERCOT prices for the configured hub.

    python scripts/download_data.py [--config configs/default.yaml] [--force]

Downloads real-time 15-minute prices and, if the config has a
``day_ahead`` section, hourly day-ahead prices. Needs the
GRIDSTATUS_API_KEY environment variable on the first run. Raw API
responses are cached under data/raw/, so later runs work offline.

Prints a SHA-256 of each cleaned data set. If ``expected_sha256`` is set
in the config, a different hash makes the script exit with an error: your
data is not the data behind the published results.
"""

from __future__ import annotations

import argparse
import json
import sys

from rtcb_gym.config import load_config, resolve_path
from rtcb_gym.data.clean import DA_INTERVAL, INTERVAL, clean_prices
from rtcb_gym.data.fetch import fetch_spp
from rtcb_gym.data.store import data_sha256, processed_path, write_prices


def download(cfg: dict, market: str, dataset: str, interval, expected: str | None, force: bool) -> bool:
    """Fetch, clean and store one market. Returns True if the hash is as expected."""
    m, d = cfg["market"], cfg["data"]
    print(f"\n== {market} ({dataset})")
    raw = fetch_spp(
        location=m["location"],
        start_date=m["start_date"],
        end_date=m["end_date"],
        raw_dir=resolve_path(d["raw_dir"]),
        dataset=dataset,
        market=market,
        tz=m["timezone"],
        sleep_sec=float(d.get("sleep_sec", 2)),
        force=force,
    )
    df, report = clean_prices(
        raw, m["location"], m["start_date"], m["end_date"], market=market, tz=m["timezone"], interval=interval
    )
    out = write_prices(df, processed_path(resolve_path(d["processed_dir"]), m["location"], market))
    digest = data_sha256(df)
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps({**report.to_dict(), "sha256": digest}, indent=2) + "\n")

    print(report.summary())
    print(f"\nwrote {out}\nwrote {report_path}")
    if report.n_missing_intervals:
        print("Gaps were NOT filled. Days with gaps are left out of episodes.")
    print(f"data sha256: {digest}")
    if not expected:
        return True
    if digest != expected:
        print(
            f"MISMATCH: expected {expected}.\n"
            "Your data differs from the data behind the published results "
            "(the source may have revised prices, or the config differs)."
        )
        return False
    print("matches expected_sha256")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--force", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, d = cfg["market"], cfg["data"]
    if d["provider"] != "gridstatusio":
        raise SystemExit(f"unsupported data provider {d['provider']!r} (gridstatusio only)")

    ok = download(cfg, m["market"], d["dataset"], INTERVAL, d.get("expected_sha256"), args.force)
    da = cfg.get("day_ahead")
    if da:
        ok &= download(cfg, da["market"], da["dataset"], DA_INTERVAL, da.get("expected_sha256"), args.force)
    if not ok:
        sys.exit("\nat least one data set does not match its expected_sha256")


if __name__ == "__main__":
    main()
