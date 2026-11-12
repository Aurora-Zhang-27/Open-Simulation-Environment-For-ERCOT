"""Duration-sensitivity gate for the energy + ECRS reservation model.

    python scripts/run_ecrs_gate.py --source gridstatus --start 2026-08-25 --end 2026-10-23 \
        --panel-days 3 --out-dir results/ecrs_gate/real_test_holdout

Two price sources:

``gridstatus`` (recommended)
    SCED rows cached by scripts/download_5min.py: energy price = SCED LMP +
    reliability deployment adder at one hub, ECRS price = real-time MCPC, as
    time-weighted five-minute prices (rtcb_gym/data/sced.py). A panel is run
    only if every row of it and of its history is a published price (status
    ok, rerun or carried); other panels are listed as skipped, never filled.
``hybridbid``
    The third-party processed mirror used by the first pilots.

``--panel-days N`` cuts start..end into consecutive N-day panels, each
starting and ending at 50% SoC, and adds totals over the panels that ran.

Cash flows are LMP/MCPC proxies, not complete ERCOT settlement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from rtcb_gym.battery import BatteryParams, terminal_floor
from rtcb_gym.data import sced
from rtcb_gym.experimental.ecrs import (
    DT_HOURS, STEP, ECRSRules, ECRSReservationEnv, FiveMinuteTape, solve_reservation,
)

# MPC plans often tie (e.g. equal forecast prices); this small cost per MWh
# of activity picks the same plan on every solver version. It is a tenth of
# the 0.01 USD/MWh price tick. Oracles use none and stay exact.
MPC_TIE_BREAK_USD_PER_MWH = 1e-3


def profile_prices(root: Path) -> tuple[pd.DataFrame, dict]:
    """Audit full input files before any panel selection; no filling or dropping."""
    tables, records = {}, []
    for kind, col in [("energy_prices", "rt_lmp"), ("as_prices", "rt_mcpc_ecrs")]:
        frames = []
        for path in sorted((root/kind).glob("*.parquet")):
            frame = pd.read_parquet(path)
            if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
                raise ValueError(f"timezone-aware datetime index required: {path}")
            values = frame[col]
            records.append(dict(
                file=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=len(frame), column=col, nulls=int(values.isna().sum()),
                null_rate=float(values.isna().mean()), zeros=int((values == 0).sum()),
                duplicate_timestamps=int(frame.index.duplicated().sum()),
                start=str(frame.index.min()), end=str(frame.index.max()),
                quantiles={str(k):float(v) for k,v in values.quantile([0,.01,.5,.99,1]).items()},
            ))
            frames.append(frame[[col]])
        if not frames:
            raise FileNotFoundError(f"no {kind} Parquet files under {root}")
        combined = pd.concat(frames).sort_index()
        # pandas 3 keeps the file's time unit; compare indexes on one unit.
        combined.index = combined.index.as_unit("ns")
        if combined.index.has_duplicates:
            raise ValueError(f"duplicate keys across {kind} files")
        tables[kind] = combined
    if not tables["energy_prices"].index.equals(tables["as_prices"].index):
        raise ValueError("energy and ECRS timestamps do not match exactly")
    joined = tables["energy_prices"].join(tables["as_prices"], validate="one_to_one")
    quality = dict(grain="one five-minute UTC interval", file_profiles=records,
                   join_rows=len(joined), join_multiplier=1,
                   missing_policy="reject missing rows in evaluated panel and its history; no fill",
                   authority="third-party processed mirror, not independently reconciled to ERCOT",
                   cell_imputation_flags="not provided by source")
    return joined, quality


def mpc_action(obs, info, params, planning_rules, horizon, energy_only=False):
    """Past-day clock-aligned prices; no fitted parameters, no current prices.

    At a five-minute assumed publication lag, the most recent observed row
    is two steps old. The offset aligns forecasts with t-288 rather than
    treating the last 288 available values as a zero-lag day. If a long
    horizon exceeds yesterday's available suffix, wrap to older history.
    """
    if not np.all(obs["price_mask"] == 1):
        return np.zeros(2)
    n = min(horizon, int(info["steps_remaining"]))
    offset = int(round(obs["price_age_steps"][0])) - 1
    forecast = np.roll(obs["prices"], -offset, axis=0)[:n].copy()
    if energy_only:
        forecast[:, 1] = 0.
    floor = terminal_floor(params, int(info["steps_remaining"])-n, DT_HOURS)
    plan = solve_reservation(forecast, params, planning_rules, float(obs["soc"][0]),
                             soc_end_min=floor, tie_break_usd_per_mwh=MPC_TIE_BREAK_USD_PER_MWH)
    return plan.actions_mw[0]/params.power_mw


def run(env, name, policy):
    obs, info = env.reset()
    rows = []
    while True:
        action = policy(obs, info)
        obs, reward, done, _, info = env.step(action)
        request, actual = info["requested_mw"], info["actual_mw"]
        rows.append(dict(
            interval_start_utc=str(info["settled_interval_start_utc"]),
            requested_energy_mw=float(request[0]), actual_energy_mw=float(actual[0]),
            requested_ecrs_mw=float(request[1]), actual_ecrs_mw=float(actual[1]),
            ecrs_shortfall_mw=float(max(0.,request[1]-actual[1])),
            ecrs_clipped=int(info["ecrs_clipped"]), energy_clipped=int(info["energy_clipped"]),
            soc=info["soc"], **{k:info[k] for k in ["energy_proxy_usd","ecrs_proxy_usd","degradation_usd","net_proxy_usd"]},
        ))
        if done:
            break
    frame = pd.DataFrame(rows)
    summary = dict(policy=name, intervals=len(frame),
                   revenue_proxy_usd=float(frame.net_proxy_usd.sum()),
                   energy_proxy_usd=float(frame.energy_proxy_usd.sum()),
                   ecrs_proxy_usd=float(frame.ecrs_proxy_usd.sum()),
                   degradation_usd=float(frame.degradation_usd.sum()),
                   clipped_ecrs_intervals=int(frame.ecrs_clipped.sum()),
                   clipped_energy_intervals=int(frame.energy_clipped.sum()),
                   ecrs_shortfall_mw_intervals=float(frame.ecrs_shortfall_mw.sum()),
                   terminal_soc=float(frame.soc.iloc[-1]))
    return summary, frame



def load_gridstatus(raw_dir: Path, hub: str, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Five-minute tape on [lo, hi) from the SCED cache. No request is made."""
    raws, files = {}, []
    for name, dataset, key in [("lmp", sced.SCED_LMP_DATASET, hub), ("adders", sced.ADDERS_DATASET, hub),
                               ("mcpc", sced.SCED_MCPC_DATASET, "ECRS")]:
        raws[name] = sced.read_sced_cache(raw_dir, dataset, key)
        for path in sorted((raw_dir/"gridstatusio"/dataset/key).glob("*.parquet")):
            files.append(dict(file=f"gridstatusio/{dataset}/{key}/{path.name}",
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              rows=len(pd.read_parquet(path))))
    tape = sced.real_time_tape(raws["lmp"], raws["adders"], raws["mcpc"], lo, hi)
    joined = pd.DataFrame({"rt_lmp": tape["energy_price_usd_per_mwh"].to_numpy(),
                           "rt_mcpc_ecrs": tape["ecrs_price_usd_per_mw_h"].to_numpy(),
                           "status": tape["status"].to_numpy()},
                          index=pd.DatetimeIndex(tape["interval_start_utc"]).as_unit("ns"))
    observed = joined["status"].isin(sced.OBSERVED_STATUSES)
    joined.loc[~observed, ["rt_lmp", "rt_mcpc_ecrs"]] = np.nan  # estimates are not prices
    quality = dict(grain="one five-minute UTC interval", file_profiles=files,
                   rows=len(joined), status_counts=joined["status"].value_counts().to_dict(),
                   missing_policy="rows that are not a published price in effect (imputed, partial, missing) "
                                  "are treated as missing; a panel containing one is skipped, never filled",
                   authority="Grid Status API copy of ERCOT SCED reports; data provided by Grid Status",
                   energy_price="SCED LMP + Real-Time Reliability Deployment Price Adder, time-weighted",
                   ecrs_price="real-time ECRS MCPC (ercot_mcpc_sced), time-weighted")
    return joined[["rt_lmp", "rt_mcpc_ecrs"]], quality


def run_panel(joined, lo, hi, args, p, canonical, relaxed, price_basis, out=None):
    """Run every policy on one panel. Returns (results, None) or (None, reason)."""
    lag_steps = int(np.ceil(args.publication_lag_minutes/5))
    # Exactly 288 history values plus enough extra rows for the publication lag.
    history_lo = lo-pd.Timedelta(days=1)-lag_steps*STEP
    selected = joined.loc[(joined.index >= history_lo) & (joined.index < hi)].copy()
    expected = pd.date_range(history_lo, hi, freq=STEP, inclusive="left").as_unit("ns")
    if not selected.index.equals(expected) or selected.isna().any().any():
        n_bad = int(selected.isna().any(axis=1).sum()) + len(expected.difference(selected.index))
        return None, f"{n_bad} missing or estimated rows in the panel or its history"
    frame = selected.rename(columns={"rt_lmp": "energy_price_usd_per_mwh", "rt_mcpc_ecrs": "ecrs_price_usd_per_mw_h"})
    frame["interval_start_utc"] = frame.index
    frame["available_at_utc"] = frame.index+STEP+pd.Timedelta(minutes=args.publication_lag_minutes)
    tape = FiveMinuteTape(frame, dict(price_basis=price_basis,
                                      availability=f"ASSUMED interval end + {args.publication_lag_minutes} minutes"))
    start = int(tape.starts.searchsorted(lo))
    results = []
    policies = [("idle", lambda obs, info: np.zeros(2)),
                ("energy_mpc", lambda obs, info: mpc_action(obs, info, p, canonical, args.horizon, True)),
                ("ecrs_mpc_1h", lambda obs, info: mpc_action(obs, info, p, canonical, args.horizon)),
                ("ecrs_mpc_0.25h_executed_under_1h", lambda obs, info: mpc_action(obs, info, p, relaxed, args.horizon))]
    for name, policy in policies:
        summary, trajectory = run(ECRSReservationEnv(tape, p, canonical, start=start), name, policy)
        results.append(summary)
        if out is not None:
            trajectory.to_csv(out/f"{name}.csv", index=False)
    if not args.skip_oracle:
        for rules in [canonical, relaxed]:
            plan = solve_reservation(tape.prices[start:], p, rules, p.soc_init, time_limit=args.oracle_time_limit)
            counter = iter(plan.actions_mw/p.power_mw)
            name = f"oracle_{rules.duration_hours:g}h"
            summary, trajectory = run(ECRSReservationEnv(tape, p, rules, start=start), name, lambda obs, info: next(counter))
            if not np.isclose(summary["revenue_proxy_usd"], plan.net_proxy_usd, rtol=1e-7, atol=1e-5):
                raise RuntimeError("oracle replay did not match objective")
            if summary["clipped_ecrs_intervals"] or summary["clipped_energy_intervals"]:
                raise RuntimeError("oracle actions clipped during replay")
            summary["mip_gap"] = plan.mip_gap
            results.append(summary)
            if out is not None:
                trajectory.to_csv(out/f"{name}.csv", index=False)
    return results, None


ADDITIVE = ("intervals", "revenue_proxy_usd", "energy_proxy_usd", "ecrs_proxy_usd", "degradation_usd",
            "clipped_ecrs_intervals", "clipped_energy_intervals", "ecrs_shortfall_mw_intervals")


def totals(panels: list[dict]) -> list[dict]:
    """Sum each policy over the panels that ran, plus its per-panel revenues."""
    out: dict[str, dict] = {}
    for panel in panels:
        for r in panel["results"]:
            t = out.setdefault(r["policy"], dict(policy=r["policy"], panels=0, **{k: 0 for k in ADDITIVE},
                                                 panel_revenue_proxy_usd=[]))
            t["panels"] += 1
            for k in ADDITIVE:
                t[k] += r[k]
            t["panel_revenue_proxy_usd"].append(r["revenue_proxy_usd"])
    return list(out.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["hybridbid", "gridstatus"], default="hybridbid")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/third_party_hybridbid"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="SCED cache (gridstatus source)")
    parser.add_argument("--hub", default="HB_WEST", help="energy price hub (gridstatus source)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/ecrs_gate/february_pilot"))
    parser.add_argument("--start", default="2026-02-02")
    parser.add_argument("--end", default="2026-02-04", help="inclusive Central operating day")
    parser.add_argument("--panel-days", type=int, default=None,
                        help="cut start..end into consecutive panels of this many days (default: one panel)")
    parser.add_argument("--stage", default=None, help="analysis stage recorded in the protocol")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--power-mw", type=float, default=100.)
    parser.add_argument("--energy-mwh", type=float, default=200.)
    parser.add_argument("--publication-lag-minutes", type=int, default=5)
    parser.add_argument("--oracle-time-limit", type=float, default=600.)
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.horizon <= 288 or args.publication_lag_minutes < 0:
        parser.error("horizon must be 1..288 and publication lag nonnegative")
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    lo = pd.Timestamp(args.start, tz="US/Central").tz_convert("UTC")
    hi = (pd.Timestamp(args.end)+pd.Timedelta(days=1)).tz_localize("US/Central").tz_convert("UTC")
    if hi <= lo:
        parser.error("end before start")
    if args.source == "gridstatus":
        history_lo = lo-pd.Timedelta(days=1)-int(np.ceil(args.publication_lag_minutes/5))*STEP
        joined, quality = load_gridstatus(args.raw_dir, args.hub, history_lo, hi)
        price_basis = f"{args.hub}_5min_SCED_LMP_plus_RTRDPA_and_ECRS_MCPC_proxy"
        stage = args.stage or "exploratory_grid_status_sced"
    else:
        joined, quality = profile_prices(args.input_dir)
        price_basis = "HB_HUBAVG_5min_LMP_MCPC_proxy"
        stage = args.stage or "exploratory_third_party_mirror"
    days = pd.date_range(args.start, args.end, freq="D")
    step = args.panel_days or len(days)
    if step < 1 or len(days) % step:
        parser.error("start..end must be a whole number of panels")
    bounds = [(pd.Timestamp(d, tz="US/Central").tz_convert("UTC"),
               (d+pd.Timedelta(days=step)).tz_localize("US/Central").tz_convert("UTC")) for d in days[::step]]
    p = BatteryParams(power_mw=args.power_mw, energy_mwh=args.energy_mwh, soc_init=.5, soc_terminal_min=.5)
    canonical, relaxed = ECRSRules(), ECRSRules(.25)
    lock = dict(analysis_stage=stage, source=args.source, start=str(lo), end_exclusive=str(hi),
                intervals=int((hi-lo)/STEP), panels=len(bounds), panel_days=step,
                battery=asdict(p), canonical_rule=canonical.rule_id,
                comparison_rule=relaxed.rule_id, mpc_horizon_steps=args.horizon,
                mpc_tie_break_usd_per_mwh=MPC_TIE_BREAK_USD_PER_MWH,
                publication_lag_minutes=args.publication_lag_minutes, publication_lag_verified=False,
                price_basis=price_basis, as_deployment="none",
                software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                dependency_versions={name: importlib.metadata.version(name) for name in ["scipy", "gymnasium", "pyarrow"]},
                code_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [
                    "rtcb_gym/experimental/ecrs.py", "rtcb_gym/ancillary.py", "rtcb_gym/battery.py",
                    "rtcb_gym/data/sced.py", "scripts/run_ecrs_gate.py"]},
                input_files=quality["file_profiles"],
                interpretation="The 0.25h model is a sensitivity case reproducing a comparator parameter, not the historical pre-RTC+B ECRS rule.")
    # Freeze the chosen protocol before seeing outcomes.
    (out/"protocol_lock.json").write_text(json.dumps(lock, indent=2)+"\n")
    panels, skipped = [], []
    for a, b in bounds:
        results, reason = run_panel(joined, a, b, args, p, canonical, relaxed, price_basis,
                                    out if len(bounds) == 1 else None)
        if results is None:
            if len(bounds) == 1:
                raise ValueError(f"panel/history contains missing prices or timestamps: {reason}")
            skipped.append(dict(start=str(a), end_exclusive=str(b), reason=reason))
            print(f"skipped {a} .. {b}: {reason}", flush=True)
            continue
        panels.append(dict(start=str(a), end_exclusive=str(b), results=results))
        print(json.dumps(dict(start=str(a), revenue={r["policy"]: round(r["revenue_proxy_usd"], 2) for r in results})), flush=True)
    quality["selected_panels"] = dict(requested=len(bounds), run=len(panels), skipped=skipped,
                                      evaluated_intervals=sum(r["results"][0]["intervals"] for r in panels))
    (out/"data_quality.json").write_text(json.dumps(quality, indent=2)+"\n")
    scope = ("Reduced-model sensitivity on LMP/MCPC cash-flow proxies; not official settlement or "
             "deployment validation.")
    if len(bounds) == 1:
        payload = dict(protocol=lock, results=panels[0]["results"], conclusion_scope=scope)
    else:
        payload = dict(protocol=lock, totals=totals(panels), panels=panels, skipped_panels=skipped,
                       conclusion_scope=scope)
    (out/"summary.json").write_text(json.dumps(payload, indent=2)+"\n")


if __name__=="__main__":
    main()
