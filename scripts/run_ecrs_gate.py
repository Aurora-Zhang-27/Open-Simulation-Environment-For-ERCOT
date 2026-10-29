"""Exploratory duration-sensitivity gate on third-party five-minute prices.

This uses LMP/MCPC cash-flow proxies, not complete ERCOT settlement. It never
fills gaps and does not call the public mirror an independent held-out test.
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
from rtcb_gym.experimental.ecrs import (
    DT_HOURS, STEP, ECRSRules, ECRSReservationEnv, FiveMinuteTape, solve_reservation,
)


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
                             soc_end_min=floor)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("data/raw/third_party_hybridbid"))
    parser.add_argument("--out-dir",type=Path,default=Path("results/ecrs_gate/february_pilot"))
    parser.add_argument("--start",default="2026-02-02")
    parser.add_argument("--end",default="2026-02-04",help="inclusive Central operating day")
    parser.add_argument("--horizon",type=int,default=12)
    parser.add_argument("--power-mw",type=float,default=100.)
    parser.add_argument("--energy-mwh",type=float,default=200.)
    parser.add_argument("--publication-lag-minutes",type=int,default=5)
    parser.add_argument("--skip-oracle",action="store_true")
    args = parser.parse_args()
    if not 1 <= args.horizon <= 288 or args.publication_lag_minutes < 0:
        parser.error("horizon must be 1..288 and publication lag nonnegative")
    out=args.out_dir; out.mkdir(parents=True,exist_ok=True)
    joined, quality=profile_prices(args.input_dir)
    (out/"data_quality.json").write_text(json.dumps(quality,indent=2)+"\n")
    lo=pd.Timestamp(args.start,tz="US/Central").tz_convert("UTC")
    hi=(pd.Timestamp(args.end)+pd.Timedelta(days=1)).tz_localize("US/Central").tz_convert("UTC")
    if hi <= lo:
        parser.error("end before start")
    # Exactly 288 history values plus enough extra rows for the publication lag.
    history_lo=lo-pd.Timedelta(days=1)-int(np.ceil(args.publication_lag_minutes/5))*STEP
    selected=joined.loc[(joined.index>=history_lo)&(joined.index<hi)].copy()
    expected=pd.date_range(history_lo,hi,freq=STEP,inclusive="left").as_unit("ns")
    if not selected.index.equals(expected) or selected.isna().any().any():
        raise ValueError("panel/history contains missing prices or timestamps; see data_quality.json")
    quality["selected_panel"] = dict(start=str(lo), end_exclusive=str(hi),
                                     history_start=str(history_lo),
                                     evaluated_intervals=int(((selected.index>=lo)&(selected.index<hi)).sum()),
                                     rows_including_history=len(selected), nulls=0, timestamp_gaps=0)
    (out/"data_quality.json").write_text(json.dumps(quality,indent=2)+"\n")
    frame=selected.rename(columns={"rt_lmp":"energy_price_usd_per_mwh","rt_mcpc_ecrs":"ecrs_price_usd_per_mw_h"})
    frame["interval_start_utc"]=frame.index
    frame["available_at_utc"]=frame.index+STEP+pd.Timedelta(minutes=args.publication_lag_minutes)
    tape=FiveMinuteTape(frame,dict(price_basis="HB_HUBAVG_5min_LMP_MCPC_proxy",
                                  availability=f"ASSUMED interval end + {args.publication_lag_minutes} minutes"))
    start=int(tape.starts.searchsorted(lo))
    p=BatteryParams(power_mw=args.power_mw,energy_mwh=args.energy_mwh,soc_init=.5,soc_terminal_min=.5)
    canonical, relaxed=ECRSRules(),ECRSRules(.25)
    lock=dict(analysis_stage="exploratory_third_party_mirror",start=str(lo),end_exclusive=str(hi),
              intervals=len(tape.starts)-start,battery=asdict(p),canonical_rule=canonical.rule_id,
              comparison_rule=relaxed.rule_id,mpc_horizon_steps=args.horizon,
              publication_lag_minutes=args.publication_lag_minutes,publication_lag_verified=False,
              price_basis=tape.provenance["price_basis"],as_deployment="none",
              software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
              dependency_versions={name:importlib.metadata.version(name) for name in ["scipy","gymnasium","pyarrow"]},
              code_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [
                  "rtcb_gym/experimental/ecrs.py", "rtcb_gym/battery.py", "scripts/run_ecrs_gate.py"]},
              input_files=quality["file_profiles"],
              interpretation="The 0.25h model is a sensitivity case reproducing a comparator parameter, not the historical pre-RTC+B ECRS rule.")
    # Freeze the chosen protocol before seeing outcomes.
    (out/"protocol_lock.json").write_text(json.dumps(lock,indent=2)+"\n")
    results=[]
    policies=[("idle",lambda obs,info:np.zeros(2)),
              ("energy_mpc",lambda obs,info:mpc_action(obs,info,p,canonical,args.horizon,True)),
              ("ecrs_mpc_1h",lambda obs,info:mpc_action(obs,info,p,canonical,args.horizon)),
              ("ecrs_mpc_0.25h_executed_under_1h",lambda obs,info:mpc_action(obs,info,p,relaxed,args.horizon))]
    for name,policy in policies:
        summary,trajectory=run(ECRSReservationEnv(tape,p,canonical,start=start),name,policy)
        results.append(summary)
        trajectory.to_csv(out/f"{name}.csv",index=False)
        print(json.dumps(summary),flush=True)
    if not args.skip_oracle:
        for rules in [canonical,relaxed]:
            plan=solve_reservation(tape.prices[start:],p,rules,p.soc_init)
            counter=iter(plan.actions_mw/p.power_mw)
            name=f"oracle_{rules.duration_hours:g}h"
            summary,trajectory=run(ECRSReservationEnv(tape,p,rules,start=start),name,lambda obs,info:next(counter))
            if not np.isclose(summary["revenue_proxy_usd"],plan.net_proxy_usd,rtol=1e-7,atol=1e-5):
                raise RuntimeError("oracle replay did not match objective")
            if summary["clipped_ecrs_intervals"] or summary["clipped_energy_intervals"]:
                raise RuntimeError("oracle actions clipped during replay")
            summary["mip_gap"]=plan.mip_gap
            results.append(summary)
            trajectory.to_csv(out/f"{name}.csv",index=False)
            print(json.dumps(summary),flush=True)
    payload=dict(protocol=lock,results=results,
                 conclusion_scope="Exploratory reduced-model sensitivity; not official settlement, deployment validation, or independent held-out evidence.")
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":
    main()
