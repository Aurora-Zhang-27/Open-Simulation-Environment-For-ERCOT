"""Score strategies in the five-minute energy + ECRS environment on real data.

    python scripts/run_market.py [--config configs/default.yaml] [--splits validation test]

RTCBEnv with five-minute dispatch, 15-minute settlement, ECRS, and the
day-ahead market for both products. Prices (all cache-only, see
scripts/download_5min.py and scripts/download_data.py):

    energy, real time   SCED LMP + reliability deployment adder at the hub,
                        time-weighted to five minutes; each step settles at
                        the mean of its 15-minute interval
    ECRS, real time     real-time MCPC, the same way
    energy, day-ahead   hourly day-ahead settlement point price
    ECRS, day-ahead     hourly day-ahead MCPC

Days with a five-minute price that is not a published price in effect
(imputed, partial, missing) are left out, and so is the day after.

Writes results/<hub>/energy_ecrs.md and .json. The ECRS cash flows follow
the reduced model in rtcb_gym/ancillary.py (docs/ecrs_gate.md): a
discharge-side reservation, full awards, no deployment. They are not
ERCOT's complete ancillary-service settlement.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from rtcb_gym.ancillary import ECRSRules
from rtcb_gym.baselines import DayAheadEnergyECRS, ECRSMPC, PerfectForesightMarket
from rtcb_gym.battery import BatteryParams
from rtcb_gym.config import REPO_ROOT, load_config, resolve_path
from rtcb_gym.data import sced
from rtcb_gym.data.clean import operating_day_bounds_utc
from rtcb_gym.data.store import data_sha256, processed_path, read_prices
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import check_replay, clean_nan, days_between, metrics_row, parse_splits, run_policy


def load_prices(cfg: dict, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    m, d = cfg["market"], cfg["data"]
    raw_dir, hub = resolve_path(d["raw_dir"]), m["location"]
    tape = sced.real_time_tape(
        sced.read_sced_cache(raw_dir, sced.SCED_LMP_DATASET, hub),
        sced.read_sced_cache(raw_dir, sced.ADDERS_DATASET, hub),
        sced.read_sced_cache(raw_dir, sced.SCED_MCPC_DATASET, "ECRS"), lo, hi,
    )
    counts = tape["status"].value_counts().to_dict()
    tape = tape[tape["status"].isin(sced.OBSERVED_STATUSES)]
    ts = pd.DatetimeIndex(tape["interval_start_utc"])
    rt5 = pd.DataFrame({"interval_start_utc": ts, "location": hub,
                        "price_usd_per_mwh": tape["energy_price_usd_per_mwh"].to_numpy()})
    ecrs = pd.Series(tape["ecrs_price_usd_per_mw_h"].to_numpy(), index=ts)
    da_ecrs = sced.day_ahead_ecrs_prices(sced.read_sced_cache(raw_dir, sced.DA_AS_DATASET, "DAM"))
    # the 15-minute mean of the five-minute energy price against the official settlement point price
    spp = read_prices(processed_path(resolve_path(d["processed_dir"]), hub, m["market"]))
    spp = spp.set_index("interval_start_utc")["price_usd_per_mwh"]
    grouped = rt5.groupby(rt5["interval_start_utc"].dt.floor("15min"))["price_usd_per_mwh"].agg(["mean", "count"])
    both = pd.concat([grouped.loc[grouped["count"] == 3, "mean"], spp], axis=1, join="inner")
    diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    quality = {
        "five_minute_rows": int(sum(counts.values())), "status_counts": counts,
        "settlement_price_vs_official_spp": {
            "intervals_compared": len(diff), "median_abs_diff_usd": round(float(diff.median()), 4),
            "share_within_1_cent": round(float((diff <= 0.01 + 1e-9).mean()), 5),
            "max_abs_diff_usd": round(float(diff.max()), 2),
        },
        "canonical_sha256": {
            "five-minute energy prices": data_sha256(pd.DataFrame({
                "interval_start_utc": ts, "interval_end_utc": ts + pd.Timedelta(minutes=5),
                "local_time": ts.tz_convert(m["timezone"]), "location": hub,
                "market": "REAL_TIME_5_MIN_LMP_PLUS_RTRDPA",
                "price_usd_per_mwh": rt5["price_usd_per_mwh"].round(6).to_numpy()})),
            "five-minute ECRS prices": data_sha256(pd.DataFrame({
                "interval_start_utc": ts, "interval_end_utc": ts + pd.Timedelta(minutes=5),
                "local_time": ts.tz_convert(m["timezone"]), "location": "ERCOT",
                "market": "REAL_TIME_5_MIN_ECRS_MCPC",
                "price_usd_per_mwh": ecrs.round(6).to_numpy()})),
        },
    }
    return rt5, ecrs, da_ecrs, quality


def score_split(job: tuple) -> tuple[str, list[dict], list[dict], dict]:
    """Run every strategy on one split."""
    split, cfg, rt5, ecrs, da_ecrs, da, params, rules, limit_days = job
    m, e, da_cfg = cfg["market"], cfg["energy_ecrs"], cfg["day_ahead"]
    splits = parse_splits(cfg["splits"])
    mpc = e["mpc"]
    env = RTCBEnv(
        rt5, params, days=days_between(*splits[split]), dt_hours=1 / 12, settlement_minutes=15,
        history_len=int(e["history_len"]), tz=m["timezone"], sample="sequential",
        da_prices=da, da_decision_time=str(da_cfg.get("decision_time", "10:00")),
        da_history_days=int(da_cfg.get("history_days", 7)),
        ecrs_prices=ecrs, ecrs_rules=rules, da_ecrs_prices=da_ecrs,
    )
    days = env.days[:limit_days] if limit_days else env.days
    print(f"{split}: {len(env.days)} episode days, {len(env.excluded_days)} excluded", flush=True)
    for day, reason in env.excluded_days.items():
        print(f"    excluded {day}: {reason}")

    def pf(name: str, **kw) -> PerfectForesightMarket:
        policy = PerfectForesightMarket(env, **kw)
        policy.name = name
        return policy

    bounds = [pf("Perfect foresight (DA + RT, energy + ECRS)"),
              pf("Perfect foresight (RT only, energy + ECRS)", day_ahead=False),
              pf("Perfect foresight (DA + RT, energy only)", use_ecrs=False),
              pf("Perfect foresight (RT only, energy only)", day_ahead=False, use_ecrs=False)]
    h = int(mpc["day_ahead_horizon_steps"])
    every = int(mpc["day_ahead_replan_every"])
    policies = bounds + [
        DayAheadEnergyECRS(env),
        ECRSMPC(env, horizon=h, forecast="day_ahead", replan_every=every),
        ECRSMPC(env, horizon=h, forecast="day_ahead", replan_every=every, energy_only=True),
        ECRSMPC(env, horizon=int(mpc["persistence_horizon_steps"])),
        ECRSMPC(env, horizon=int(mpc["persistence_horizon_steps"]), energy_only=True),
    ]
    results = []
    for policy in policies:
        r = run_policy(env, policy, days)
        results.append(r)
        print(f"  {split} {policy.name}: {r.revenue_usd:,.0f} (ECRS {r.ecrs_revenue_usd:,.0f})", flush=True)
    for policy, r in zip(bounds, results):
        check_replay(split, policy.name, r.revenue_usd, policy.planned_revenue_usd + policy.planned_da_usd)
    upper, rows = results[0], []
    for r in results:
        if r.revenue_usd > upper.revenue_usd + 1e-6:
            print(f"WARNING {split}: {r.policy} beats the bound")
        row = metrics_row(split, r, upper, params.energy_mwh)
        row.update(n_ecrs_clipped=r.n_ecrs_clipped, ecrs_mean_mw=r.ecrs_mw_hours / (r.n_steps * env.dt_hours),
                   ecrs_da_revenue_usd=r.ecrs_da_revenue_usd, ecrs_rt_revenue_usd=r.ecrs_rt_revenue_usd)
        rows.append(row)
    return split, rows, [r.to_dict() for r in results], env.excluded_days


def table(rows: list[dict]) -> str:
    lines = ["| Split | Strategy | Days | Revenue (USD) | of which ECRS | Capture rate (95% interval) | "
             "Cycles/day | ECRS award, mean MW | Clipped: energy / ECRS / DA h |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lo, hi = r["capture_ci95"]
        lines.append(
            f"| {r['split']} | {r['policy']} | {r['days']} | {round(r['revenue_usd']) + 0:,} | "
            f"{round(r['ecrs_revenue_usd']) + 0:,} | {r['capture_rate']:.1%} ({lo:.1%} to {hi:.1%}) | "
            f"{r['efc_per_day']:.2f} | {r['ecrs_mean_mw']:.1f} | "
            f"{r['n_clipped']} / {r['n_ecrs_clipped']} / {r['n_da_clipped_hours']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    ap.add_argument("--out-dir", default=None, help="default: results/<hub>")
    ap.add_argument("--limit-days", type=int, default=None, help="score only the first N days of each split (a quick check)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m, e = cfg["market"], cfg["energy_ecrs"]
    params = replace(BatteryParams.from_dict(cfg["battery"]), soc_init=float(e["soc_init"]),
                     soc_terminal_min=float(e["soc_terminal_min"]))
    rules = ECRSRules(float(e["duration_hours"]))
    splits = parse_splits(cfg["splits"])
    first = min(splits[s][0] for s in args.splits)
    last = max(splits[s][1] for s in args.splits)
    lo, _ = operating_day_bounds_utc(first, m["timezone"])
    _, hi = operating_day_bounds_utc(last, m["timezone"])
    rt5, ecrs, da_ecrs, quality = load_prices(cfg, lo, hi)
    da_cfg = cfg["day_ahead"]
    da = read_prices(processed_path(resolve_path(cfg["data"]["processed_dir"]), m["location"], da_cfg["market"]))
    hashes = {"day-ahead energy prices": data_sha256(da), **quality["canonical_sha256"]}

    rows, runs, excluded = [], {}, {}
    jobs = [(split, cfg, rt5, ecrs, da_ecrs, da, params, rules, args.limit_days) for split in args.splits]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:  # one process per split
        for split, split_rows, split_runs, split_excluded in pool.map(score_split, jobs):
            rows += split_rows
            runs[split] = split_runs
            excluded[split] = split_excluded

    md_table = table(rows)
    print("\n" + md_table + "\n")
    b, q = cfg["battery"], quality["settlement_price_vs_official_spp"]
    out_dir = resolve_path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / m["location"].lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    md = "\n".join([
        f"# RTCB-Gym baselines: {m['location']}, five-minute energy + ECRS",
        "",
        f"Battery: {b['power_mw']:g} MW / {b['energy_mwh']:g} MWh, {b['rte']:.0%} round-trip efficiency, "
        f"${b['degradation_cost_usd_per_mwh']:g}/MWh degradation cost; every day starts at SoC "
        f"{params.soc_init:g} and ends at or above {params.soc_terminal_min:g}. Five-minute dispatch, "
        f"15-minute settlement, ECRS duration requirement {rules.duration_hours:g} h, day-ahead bids at "
        f"{da_cfg.get('decision_time', '10:00')} the day before.",
        "",
        "ECRS follows the reduced model of `rtcb_gym/ancillary.py`: reserve from the discharge side only, "
        "full awards at the clearing price, no deployment. Not ERCOT's complete ancillary-service settlement.",
        "",
        f"Settlement price check: the 15-minute mean of the five-minute energy price is within $0.01 of the "
        f"official settlement point price in {q['share_within_1_cent']:.3%} of {q['intervals_compared']:,} "
        f"intervals (largest difference ${q['max_abs_diff_usd']:.2f}).",
        "",
        md_table,
        "",
        "Revenue includes ECRS. Capture rate = revenue / perfect foresight (DA + RT, energy + ECRS) on the "
        "same days; the interval is a block bootstrap over days. ECRS award = real-time award averaged over "
        "all intervals.",
        "",
        "Data provided by [Grid Status](https://www.gridstatus.io).",
    ])
    (out_dir / "energy_ecrs.md").write_text(md + "\n")
    payload = {"config": cfg, "data_sha256": hashes, "data_quality": quality, "excluded_days": excluded,
               "rule_id": rules.rule_id, "rows": rows, "runs": runs}
    (out_dir / "energy_ecrs.json").write_text(json.dumps(clean_nan(payload), indent=2, default=str) + "\n")
    print(f"wrote {out_dir / 'energy_ecrs.md'} and .json")


if __name__ == "__main__":
    main()
