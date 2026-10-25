"""Download, clean and store ERCOT real-time prices for the configured hub.

    python scripts/download_data.py [--config configs/default.yaml] [--force]

Needs the GRIDSTATUS_API_KEY environment variable on the first run. Raw API
responses are cached under data/raw/, so later runs work offline.

Prints a SHA-256 of the cleaned data. If ``data.expected_sha256`` is set in
the config, a different hash makes the script exit with an error: your data
is not the data behind the published results.
"""

from __future__ import annotations

import argparse
import json
import sys

from rtcb_gym.config import load_config, resolve_path
from rtcb_gym.data.clean import clean_prices
from rtcb_gym.data.fetch import fetch_rt_spp
from rtcb_gym.data.store import data_sha256, processed_path, write_prices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--force", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, d = cfg["market"], cfg["data"]
    if d["provider"] != "gridstatusio":
        raise SystemExit(f"unsupported data provider {d['provider']!r} (v0.1: gridstatusio)")

    raw = fetch_rt_spp(
        location=m["location"],
        start_date=m["start_date"],
        end_date=m["end_date"],
        raw_dir=resolve_path(d["raw_dir"]),
        dataset=d["dataset"],
        market=m["market"],
        tz=m["timezone"],
        sleep_sec=float(d.get("sleep_sec", 2)),
        force=args.force,
    )
    df, report = clean_prices(
        raw, m["location"], m["start_date"], m["end_date"], market=m["market"], tz=m["timezone"]
    )
    out = write_prices(df, processed_path(resolve_path(d["processed_dir"]), m["location"], m["market"]))
    digest = data_sha256(df)
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps({**report.to_dict(), "sha256": digest}, indent=2) + "\n")

    print(report.summary())
    print(f"\nwrote {out}\nwrote {report_path}")
    if report.n_missing_intervals:
        print("\nGaps were NOT filled. Days with gaps are left out of episodes.")
    print(f"\ndata sha256: {digest}")
    expected = d.get("expected_sha256")
    if expected:
        if digest != expected:
            sys.exit(
                f"MISMATCH: expected {expected}.\n"
                "Your data differs from the data behind the published results "
                "(the source may have revised prices, or the config differs)."
            )
        print("matches data.expected_sha256")


if __name__ == "__main__":
    main()
