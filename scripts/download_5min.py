"""Download five-minute SCED prices (energy LMP, reliability adder, ECRS MCPC)
and the hourly day-ahead ancillary service prices of the same days.

    python scripts/download_5min.py --start 2026-07-01 --end 2026-10-24 [--hub HB_WEST]

Operating days are inclusive, in the configured timezone. Raw responses are
cached under data/raw/gridstatusio/ (git-ignored: Grid Status data is not
redistributed), one file per UTC month of the requested range; a chunk that
is already cached is not requested again. Needs GRIDSTATUS_API_KEY.

One hub and one month is about 9,000 rows per series, three series. The
free Grid Status plan allows 500,000 rows a month, so download the windows
you need instead of the whole year.
"""

from __future__ import annotations

import argparse

from rtcb_gym.config import load_config, resolve_path
from rtcb_gym.data import sced
from rtcb_gym.data.clean import check_post_rtcb, operating_day_bounds_utc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hub", default=None, help="default: the config's location")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m, d = cfg["market"], cfg["data"]
    hub = args.hub or m["location"]
    lo, _ = operating_day_bounds_utc(check_post_rtcb(args.start), m["timezone"])
    _, hi = operating_day_bounds_utc(args.end, m["timezone"])
    kw = dict(raw_dir=resolve_path(d["raw_dir"]), sleep_sec=float(d.get("sleep_sec", 2)), offline=False)
    sced.fetch_sced(sced.SCED_LMP_DATASET, lo, hi, filter_column="location", filter_value=hub,
                    required=sced.LMP_REQUIRED, **kw)
    sced.fetch_sced(sced.ADDERS_DATASET, lo, hi, filter_column="location", filter_value=hub,
                    required=("sced_timestamp_utc", "location", "lmp", "rtrdpa"), **kw)
    sced.fetch_sced(sced.SCED_MCPC_DATASET, lo, hi, filter_column="as_type", filter_value="ECRS",
                    required=sced.MCPC_REQUIRED, **kw)
    sced.fetch_sced(sced.DA_AS_DATASET, lo, hi, filter_column="market", filter_value="DAM",
                    required=sced.DA_AS_REQUIRED, **kw)


if __name__ == "__main__":
    main()
