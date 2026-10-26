"""Export five-minute real-time prices with provenance, gap/fill report and checksums.

    python scripts/export_5min.py [--config configs/default.yaml] [--hubs HB_WEST HB_HOUSTON] [--out DIR]

For the configured operating days it writes, under data/exports/ (git-ignored:
the data must not be redistributed):

    energy_5min_<hub>.csv   5-minute energy prices per hub: SCED LMP, RTRDPA and
                            their sum (the energy price that settles)
    ecrs_mcpc_5min.csv      5-minute real-time ECRS MCPC (system-wide)
    sced_runs.csv           every SCED run as published, no fills
    gaps_and_fills.csv      every 5-minute interval that is not a plain SCED value
    report.json, REPORT.md  sources, gap and fill report, validation
    SHA256SUMS              SHA-256 of every file above

All times are UTC. Each 5-minute value is the time-weighted average of the
price in effect (see rtcb_gym/data/sced.py). Raw API responses are cached
under data/raw/, so the script runs offline once the cache is complete.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import rtcb_gym
from rtcb_gym.config import REPO_ROOT, load_config, resolve_path
from rtcb_gym.data import sced
from rtcb_gym.data.clean import check_post_rtcb, operating_day_bounds_utc
from rtcb_gym.data.store import processed_path, read_prices

ADDERS_DATASET = "ercot_lmp_with_adders_by_settlement_point"
ADDERS_REQUIRED = ("sced_timestamp_utc", "location", "lmp", "rtorpa", "rtordpa", "rtrdpa", "lmp_with_adders")
SEVERITY = ["ok", "rerun", "carried", "imputed", "partial", "missing"]
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
DECIMALS = 6  # 5-minute averages are rounded to micro-dollars; published prices have 2 decimals

SOURCES = {
    sced.SCED_LMP_DATASET: {
        "what": "SCED LMP per settlement point, one row per SCED run",
        "grid_status_description": "The Locational Marginal Price (LMP) for each Settlement Point in ERCOT, "
                                   "normally produced by SCED every five minutes.",
        "unit": "USD/MWh",
    },
    ADDERS_DATASET: {
        "what": "SCED LMP with real-time price adders per settlement point, one row per SCED run "
                "(rtorpa and rtordpa are empty after RTC+B; rtrdpa is the Real-Time Reliability "
                "Deployment Price Adder)",
        "unit": "USD/MWh",
    },
    sced.SCED_MCPC_DATASET: {
        "what": "Real-time Market Clearing Price for Capacity per ancillary service, one row per SCED run",
        "grid_status_description": "Prices for Ancillary Services as reported by ERCOT for each SCED run in "
                                   "real-time. Implemented on December 5th, 2025, as part of ERCOT's RTC+B project.",
        "unit": "USD/MW per hour",
    },
}


def canonical_sha256(s: pd.Series, label: str) -> str:
    """SHA-256 of one SCED series in a fixed text form (independent of file formats):
    a header line, then ``<UTC timestamp>,<value>`` per run, values in Python's
    shortest round-trip form."""
    h = hashlib.sha256(f"rtcb-gym-sced-v1 {label}\n".encode())
    for ts, v in zip(s.index.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), s.tolist()):
        h.update(f"{ts},{v!r}\n".encode())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_frames(raw_dir: Path, dataset: str, key: str) -> tuple[pd.DataFrame | None, list[dict]]:
    """Whatever is in the cache for (dataset, key), without fetching."""
    files = sorted((raw_dir / "gridstatusio" / dataset / key).glob("*.parquet"))
    if not files:
        return None, []
    info = [{"file": str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f),
             "downloaded_utc": dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc).strftime(TS_FMT)}
            for f in files]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), info


def worst(a: pd.Series, b: pd.Series) -> pd.Series:
    rank = {s: i for i, s in enumerate(SEVERITY)}
    return pd.Series(np.where(a.map(rank) >= b.map(rank), a, b), index=a.index)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "rtcb_gym", "scripts"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("+uncommitted changes" if dirty else "")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--hubs", nargs="+", default=["HB_WEST", "HB_HOUSTON"])
    ap.add_argument("--out", default=None, help="default: data/exports/ercot_5min_<start>_<end>")
    ap.add_argument("--max-carry-minutes", type=float, default=60.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, d = cfg["market"], cfg["data"]
    start_day, end_day = check_post_rtcb(m["start_date"]), m["end_date"]
    lo, _ = operating_day_bounds_utc(start_day, m["timezone"])
    _, hi = operating_day_bounds_utc(end_day, m["timezone"])
    raw_dir = resolve_path(d["raw_dir"])
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "exports" / f"ercot_5min_{m['start_date']}_{end_day}"
    out.mkdir(parents=True, exist_ok=True)
    max_carry = pd.Timedelta(minutes=args.max_carry_minutes)
    hubs, ref = args.hubs, args.hubs[0]
    fetch_kw = dict(raw_dir=raw_dir, sleep_sec=float(d.get("sleep_sec", 2)))

    # ---------------------------------------------------------------- fetch (cache first)
    raw_lmp = {h: sced.fetch_sced(sced.SCED_LMP_DATASET, lo, hi, filter_column="location", filter_value=h,
                                  required=sced.LMP_REQUIRED, **fetch_kw) for h in hubs}
    raw_add = sced.fetch_sced(ADDERS_DATASET, lo, hi, filter_column="location", filter_value=ref,
                              required=ADDERS_REQUIRED, **fetch_kw)
    raw_ecrs = sced.fetch_sced(sced.SCED_MCPC_DATASET, lo, hi, filter_column="as_type", filter_value="ECRS",
                               required=sced.MCPC_REQUIRED, **fetch_kw)
    for h, r in raw_lmp.items():
        if set(r["market"]) != {"REAL_TIME_SCED"} or set(r["location"]) != {h}:
            raise ValueError(f"unexpected market/location in {h} LMP rows")
    if set(raw_ecrs["as_type"]) != {"ECRS"}:
        raise ValueError("unexpected as_type in ECRS rows")
    if raw_add["rtorpa"].notna().any() or raw_add["rtordpa"].notna().any():
        raise ValueError("rtorpa/rtordpa are not empty: the energy price would need them too")

    # ------------------------------------------------------------- one value per SCED run
    lmp, lmp_info = {}, {}
    for h in hubs:
        lmp[h], lmp_info[h] = sced.sced_series(raw_lmp[h], "lmp")
    rtrdpa, rtrdpa_info = sced.sced_series(raw_add, "rtrdpa")
    add_lmp, _ = sced.sced_series(raw_add, "lmp")
    common = add_lmp.index.intersection(lmp[ref].index)
    if not np.array_equal(add_lmp[common].to_numpy(), lmp[ref][common].to_numpy()):
        raise ValueError(f"{ref}: LMPs differ between {sced.SCED_LMP_DATASET} and {ADDERS_DATASET}")
    ecrs, ecrs_info = sced.sced_series(raw_ecrs, "mcpc")
    ecrs_uncapped = pd.to_numeric(raw_ecrs.set_index(pd.to_datetime(raw_ecrs["sced_timestamp_utc"], utc=True)
                                                     .dt.as_unit("ns"))["uncapped_mcpc"], errors="coerce")
    ecrs_uncapped = ecrs_uncapped[~ecrs_uncapped.index.duplicated()].sort_index()

    clock = ecrs.index.union(rtrdpa.index)
    for h in hubs:
        clock = clock.union(lmp[h].index)
    clock = clock[(clock >= lo) & (clock < hi)]

    # RTRDPA is system-wide: check it against any other hub's cached adders
    cross = {}
    for h in hubs[1:]:
        frame, _ = cached_frames(raw_dir, ADDERS_DATASET, h)
        if frame is None:
            continue
        other, _ = sced.sced_series(frame, "rtrdpa")
        c = other.index.intersection(rtrdpa.index)
        cross[h] = {"common_runs": len(c), "identical": bool(np.array_equal(other[c].to_numpy(), rtrdpa[c].to_numpy())),
                    "span_utc": [other.index.min().strftime(TS_FMT), other.index.max().strftime(TS_FMT)]}
        if not cross[h]["identical"]:
            raise ValueError(f"RTRDPA differs between {ref} and {h}: it is not system-wide")

    # --------------------------------------------------------------------- 5-minute grids
    grid = lambda s: sced.five_minute_grid(s, lo, hi, clock=clock, max_carry=max_carry)
    g_rtrdpa = grid(rtrdpa)
    g_ecrs = grid(ecrs)
    fills = []

    def fill_rows(series_name: str, g: pd.DataFrame) -> None:
        bad = g[~g["status"].isin(["ok", "rerun"])]
        for row in bad.itertuples():
            age = (row.interval_end_utc - row.fill_source_sced_utc).total_seconds() if pd.notna(row.fill_source_sced_utc) else None
            fills.append({"series": series_name, "interval_start_utc": row.interval_start_utc, "status": row.status,
                          "sced_runs": row.sced_runs, "value": row.value,
                          "fill_source_sced_utc": row.fill_source_sced_utc, "carry_age_s_at_interval_end": age})

    energy, status_counts, validation = {}, {}, {}
    for h in hubs:
        g = grid(lmp[h])
        e = pd.DataFrame({
            "interval_start_utc": g["interval_start_utc"],
            "interval_end_utc": g["interval_end_utc"],
            "sced_runs": g["sced_runs"],
            "lmp_usd_per_mwh": g["value"].round(DECIMALS),
            "rtrdpa_usd_per_mwh": g_rtrdpa["value"].round(DECIMALS),
            "energy_price_usd_per_mwh": (g["value"] + g_rtrdpa["value"]).round(DECIMALS),
            "lmp_status": g["status"],
            "rtrdpa_status": g_rtrdpa["status"],
        })
        energy[h] = e
        fill_rows(f"{h} lmp", g)
        status_counts[f"{h} lmp"] = g["status"].value_counts().reindex(SEVERITY, fill_value=0).to_dict()
        e.to_csv(out / f"energy_5min_{h.lower()}.csv", index=False, date_format=TS_FMT)

        # validate against the official 15-minute settlement point price, if we have it
        rt_path = processed_path(resolve_path(d["processed_dir"]), h, m["market"])
        if rt_path.exists():
            spp = read_prices(rt_path).set_index("interval_start_utc")["price_usd_per_mwh"]
            both_status = worst(g["status"], g_rtrdpa["status"])
            v = pd.DataFrame({"interval_start_utc": g["interval_start_utc"],
                              "value": g["value"] + g_rtrdpa["value"], "status": both_status})
            avg = sced.fifteen_minute_average(v)
            cmp = pd.concat([avg.rename("five_minute_average"), spp.rename("official_spp")], axis=1).dropna()
            diff = cmp["five_minute_average"] - cmp["official_spp"]
            off = cmp[diff.abs() > 0.01 + 1e-9].assign(diff=diff)
            st = v.set_index("interval_start_utc")["status"]
            validation[h] = {
                "intervals_compared": len(cmp),
                "median_abs_diff_usd": round(float(diff.abs().median()), 6),
                "share_within_1_cent": round(float((diff.abs() <= 0.01 + 1e-9).mean()), 6),
                "max_abs_diff_usd": round(float(diff.abs().max()), 6),
                "intervals_off_by_more_than_1_cent": [
                    {"interval_start_utc": t.strftime(TS_FMT), "diff_usd": round(float(r["diff"]), 4),
                     "five_minute_statuses": list(st.loc[t: t + pd.Timedelta(minutes=10)])}
                    for t, r in off.sort_values("diff", key=abs, ascending=False).iterrows()
                ],
                "official_spp_missing_but_reconstructable": [
                    t.strftime(TS_FMT) for t in avg.dropna().index.difference(spp.index)
                ],
            }

    fill_rows("rtrdpa (system-wide)", g_rtrdpa)
    fill_rows("ECRS mcpc", g_ecrs)
    status_counts["rtrdpa (system-wide)"] = g_rtrdpa["status"].value_counts().reindex(SEVERITY, fill_value=0).to_dict()
    status_counts["ECRS mcpc"] = g_ecrs["status"].value_counts().reindex(SEVERITY, fill_value=0).to_dict()
    pd.DataFrame({
        "interval_start_utc": g_ecrs["interval_start_utc"],
        "interval_end_utc": g_ecrs["interval_end_utc"],
        "sced_runs": g_ecrs["sced_runs"],
        "ecrs_mcpc_usd_per_mw_h": g_ecrs["value"].round(DECIMALS),
        "status": g_ecrs["status"],
    }).to_csv(out / "ecrs_mcpc_5min.csv", index=False, date_format=TS_FMT)

    runs = pd.DataFrame(index=clock)
    for h in hubs:
        runs[f"{h.lower()}_lmp_usd_per_mwh"] = lmp[h].reindex(clock)
    runs["rtrdpa_usd_per_mwh"] = rtrdpa.reindex(clock)
    runs["ecrs_mcpc_usd_per_mw_h"] = ecrs.reindex(clock)
    runs["ecrs_uncapped_mcpc_usd_per_mw_h"] = ecrs_uncapped.reindex(clock)
    runs.index.name = "sced_timestamp_utc"
    runs.reset_index().to_csv(out / "sced_runs.csv", index=False, date_format=TS_FMT)
    fills_df = pd.DataFrame(fills, columns=["series", "interval_start_utc", "status", "sced_runs", "value",
                                            "fill_source_sced_utc", "carry_age_s_at_interval_end"])
    fills_df = fills_df.sort_values(["interval_start_utc", "series"])
    fills_df.to_csv(out / "gaps_and_fills.csv", index=False, date_format=TS_FMT)

    # ------------------------------------------------------------------ runs per series
    missing_runs = {name: [t.strftime(TS_FMT) for t in clock.difference(s.index)]
                    for name, s in [*((f"{h} lmp", lmp[h]) for h in hubs), ("rtrdpa", rtrdpa), ("ECRS mcpc", ecrs)]}
    gaps_s = np.diff(clock.asi8) / 1e9
    long_gaps = [{"after_sced_utc": clock[i].strftime(TS_FMT), "minutes": round(gaps_s[i] / 60, 1)}
                 for i in np.argsort(gaps_s)[::-1] if gaps_s[i] > 360]

    caches = {}
    for name, key in [*((sced.SCED_LMP_DATASET, h) for h in hubs), (ADDERS_DATASET, ref), (sced.SCED_MCPC_DATASET, "ECRS")]:
        caches[f"{name}/{key}"] = cached_frames(raw_dir, name, key)[1]

    report = {
        "title": "ERCOT five-minute real-time prices",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(TS_FMT),
        "period": {"operating_days": [str(start_day), str(end_day)], "timezone_of_operating_days": m["timezone"],
                   "utc": [lo.strftime(TS_FMT), hi.strftime(TS_FMT)], "five_minute_intervals": int((hi - lo) / sced.FIVE_MIN)},
        "sources": {
            "provider": "Grid Status API (https://www.gridstatus.io), Python client gridstatusio",
            "credit": "Data provided by Grid Status.",
            "redistribution": "Not redistributable (Grid Status terms of use).",
            "datasets": {
                **{f"{sced.SCED_LMP_DATASET} (location={h})": {**SOURCES[sced.SCED_LMP_DATASET], "column": "lmp",
                                                              "cache_files": caches[f"{sced.SCED_LMP_DATASET}/{h}"]} for h in hubs},
                f"{ADDERS_DATASET} (location={ref})": {**SOURCES[ADDERS_DATASET], "column": "rtrdpa",
                                                      "cache_files": caches[f"{ADDERS_DATASET}/{ref}"]},
                f"{sced.SCED_MCPC_DATASET} (as_type=ECRS)": {**SOURCES[sced.SCED_MCPC_DATASET],
                                                            "column": "mcpc (uncapped_mcpc kept in sced_runs.csv only)",
                                                            "cache_files": caches[f"{sced.SCED_MCPC_DATASET}/ECRS"]},
            },
            "query": "get_dataset(dataset, start, end, filter_column, filter_value) per UTC month, "
                     "start/end on the SCED timestamp",
        },
        "method": {
            "price_in_effect": "A SCED run's price applies from its SCED timestamp until the next SCED run "
                               "(ERCOT's settlement rule). Each 5-minute value is the time-weighted average "
                               "of the price in effect over [interval_start, interval_end).",
            "sced_clock": "Union of the SCED timestamps of all exported series.",
            "energy_price": "energy_price = lmp + rtrdpa (5-minute averages). rtorpa and rtordpa are empty "
                            "after RTC+B. RTRDPA is system-wide and taken from the "
                            f"{ref} rows of {ADDERS_DATASET}.",
            "statuses": {
                "ok": "one SCED run started in the interval",
                "rerun": "more than one SCED run started in the interval",
                "carried": "no SCED run started in the interval; the previous run's price was still in effect "
                           "(ERCOT's rule, not an estimate)",
                "imputed": "SCED ran, but this series has no price for that run; the last known price of the "
                           "series is used instead (an estimate)",
                "partial": "part of the interval has no known price; averaged over the known part only",
                "missing": "no price known in the interval; left empty",
            },
            "max_carry_minutes": args.max_carry_minutes,
            "rounding": f"5-minute values rounded to {DECIMALS} decimals",
            "pre_rtcb": "No data before the RTC+B go-live (2025-12-05 00:00 CPT) is used, so the first "
                        "interval has no price.",
        },
        "sced_runs": {
            "runs_on_clock": len(clock),
            "per_series": {**{f"{h} lmp": lmp_info[h] for h in hubs}, "rtrdpa": rtrdpa_info, "ECRS mcpc": ecrs_info},
            "runs_missing_per_series": missing_runs,
            "gaps_longer_than_6_minutes": long_gaps,
            "seconds_between_runs": {"median": float(np.median(gaps_s)), "p1": float(np.percentile(gaps_s, 1)),
                                     "p99": float(np.percentile(gaps_s, 99)), "max": float(gaps_s.max())},
        },
        "gaps_and_fills": {"status_counts": status_counts, "detail_file": "gaps_and_fills.csv"},
        "rtrdpa_is_system_wide": cross or "no other hub's adders in the cache to compare",
        "validation_against_official_15_minute_spp": validation or "no processed real-time SPP found",
        "not_checked": [
            "ERCOT price corrections (ercot_lmp_by_settlement_point_price_corrections, "
            "ercot_mcpc_sced_price_corrections): whether the exported values include them was not checked.",
        ],
        "checksums": {
            "canonical_source_sha256": {
                **{f"{h} lmp": canonical_sha256(lmp[h], f"{sced.SCED_LMP_DATASET}:{h}:lmp") for h in hubs},
                "rtrdpa": canonical_sha256(rtrdpa, f"{ADDERS_DATASET}:{ref}:rtrdpa"),
                "ECRS mcpc": canonical_sha256(ecrs, f"{sced.SCED_MCPC_DATASET}:ECRS:mcpc"),
            },
            "canonical_form": "sha256 over 'rtcb-gym-sced-v1 <label>\\n' then '<UTC timestamp with microseconds>Z,"
                              "<value repr>\\n' per SCED run; independent of file formats and library versions",
            "files": "SHA256SUMS (sha256sum -c SHA256SUMS)",
        },
        "software": {"rtcb_gym": rtcb_gym.__version__, "git_commit": git_commit(), "python": platform.python_version(),
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (out / "REPORT.md").write_text(render_markdown(report) + "\n")

    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{file_sha256(out / n)}  {n}\n" for n in names))
    print(f"wrote {out}")
    for n in names + ["SHA256SUMS"]:
        print(f"  {n:28s} {(out / n).stat().st_size / 1e6:8.2f} MB")


def render_markdown(r: dict) -> str:
    p, s, gf = r["period"], r["sced_runs"], r["gaps_and_fills"]
    lines = [
        f"# {r['title']}",
        "",
        f"Operating days {p['operating_days'][0]} .. {p['operating_days'][1]} ({p['timezone_of_operating_days']}), "
        f"UTC {p['utc'][0]} .. {p['utc'][1]}: {p['five_minute_intervals']:,} five-minute intervals. "
        f"Created {r['created_utc']}. {r['sources']['credit']} {r['sources']['redistribution']}",
        "",
        "## Files",
        "",
        "| File | Content |",
        "|---|---|",
        "| `energy_5min_<hub>.csv` | per 5-minute interval: SCED runs, `lmp`, `rtrdpa`, `energy_price` (= lmp + rtrdpa, the energy price that settles), status of each |",
        "| `ecrs_mcpc_5min.csv` | per 5-minute interval: real-time ECRS MCPC (USD/MW per hour) and status |",
        "| `sced_runs.csv` | every SCED run as published, no fills (empty cell = not in that series) |",
        "| `gaps_and_fills.csv` | every interval whose status is not `ok` or `rerun`, with the SCED run its value came from |",
        "| `report.json` | everything in this report, machine-readable |",
        "| `SHA256SUMS` | SHA-256 of every file (`sha256sum -c SHA256SUMS`) |",
        "",
        "## Sources",
        "",
    ]
    for name, ds in r["sources"]["datasets"].items():
        files = ds["cache_files"]
        when = f"downloaded {files[0]['downloaded_utc']} .. {files[-1]['downloaded_utc']}" if files else ""
        lines.append(f"- `{name}`, column `{ds['column']}` ({ds['unit']}): {ds['what']}. {len(files)} monthly files, {when}.")
    lines += ["", f"Query: {r['sources']['query']}.", "", "## Method", ""]
    labels = {"price_in_effect": "Price in effect", "sced_clock": "SCED clock", "energy_price": "Energy price",
              "pre_rtcb": "Start", "rounding": "Rounding"}
    for k, label in labels.items():
        lines.append(f"- **{label}:** {r['method'][k]}")
    lines += [f"- A price is never used more than {r['method']['max_carry_minutes']:g} minutes after its SCED run.", "",
              "| Status | Meaning |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in r["method"]["statuses"].items()]
    lines += ["", "## Gaps and fills", "", "| Series | " + " | ".join(SEVERITY) + " |", "|---|" + "---:|" * len(SEVERITY)]
    for name, counts in gf["status_counts"].items():
        lines.append(f"| {name} | " + " | ".join(f"{counts[k]:,}" for k in SEVERITY) + " |")
    lines += ["", f"SCED runs on the clock: {s['runs_on_clock']:,}. Median spacing {s['seconds_between_runs']['median']:.0f} s, "
              f"longest {s['seconds_between_runs']['max'] / 60:.1f} min.", "",
              "Runs missing from a series (the run happened, the series has no price for it):", ""]
    for name, missing in s["runs_missing_per_series"].items():
        lines.append(f"- {name}: {len(missing)}" + (f" ({', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''})" if missing else ""))
    lines += ["", "SCED gaps longer than 6 minutes:", ""]
    lines += [f"- after {g['after_sced_utc']}: {g['minutes']} min" for g in s["gaps_longer_than_6_minutes"]]
    cross = r["rtrdpa_is_system_wide"]
    if isinstance(cross, dict):
        lines += ["", "## RTRDPA is system-wide", ""]
        lines += [f"- {h}: identical on all {c['common_runs']:,} common runs ({c['span_utc'][0]} .. {c['span_utc'][1]})" for h, c in cross.items()]
    val = r["validation_against_official_15_minute_spp"]
    if isinstance(val, dict):
        lines += ["", "## Validation against the official 15-minute settlement point price", "",
                  "The mean of the three 5-minute energy prices of each 15-minute interval, against the official "
                  "real-time SPP (`ercot_spp_real_time_15_min`, rounded to cents by ERCOT).", "",
                  "| Hub | Intervals | Median abs. diff | Within $0.01 | Max abs. diff |", "|---|---:|---:|---:|---:|"]
        for h, v in val.items():
            lines.append(f"| {h} | {v['intervals_compared']:,} | ${v['median_abs_diff_usd']:.4f} | "
                         f"{v['share_within_1_cent']:.3%} | ${v['max_abs_diff_usd']:.2f} |")
        for h, v in val.items():
            lines += ["", f"{h}: intervals off by more than $0.01 (5-minute statuses of lmp/rtrdpa combined):", ""]
            lines += [f"- {x['interval_start_utc']}: {x['diff_usd']:+.2f} ({'/'.join(x['five_minute_statuses'])})"
                      for x in v["intervals_off_by_more_than_1_cent"]]
            if v["official_spp_missing_but_reconstructable"]:
                lines += ["", f"{h}: official SPP missing, 5-minute average available: "
                          + ", ".join(v["official_spp_missing_but_reconstructable"])]
    lines += ["", "## Not checked", ""] + [f"- {x}" for x in r["not_checked"]]
    lines += ["", "## Checksums", "", f"Canonical SHA-256 of the source series ({r['checksums']['canonical_form']}):", ""]
    lines += [f"- {k}: `{v}`" for k, v in r["checksums"]["canonical_source_sha256"].items()]
    lines += ["", "File SHA-256: `SHA256SUMS`.", "", "## Software", ""]
    lines += [f"- {k}: {v}" for k, v in r["software"].items()]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
