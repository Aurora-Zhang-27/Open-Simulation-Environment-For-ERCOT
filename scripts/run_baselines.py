"""Score the baselines on the validation and test splits.

    python scripts/run_baselines.py [--config configs/default.yaml] [--out-dir DIR]

Writes, under results/<hub>/ by default:

    real_time.md / .json        real-time energy only (v0.1 setting)
    two_settlement.md / .json   day-ahead + real time, if the config has a
                                day_ahead section

Each .md file is a self-contained results table (hub, dates, data hashes,
battery); the .json files hold the full numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rtcb_gym.baselines import (
    DayAheadForecastMPC,
    DayAheadPlusMPC,
    DayAheadSchedule,
    ForecastMPC,
    PerfectForesight,
    PerfectForesightDayAhead,
    PerfectForesightTwoSettlement,
    fit_rolling_threshold_rule,
    fit_threshold_rule,
)
from rtcb_gym.battery import BatteryParams
from rtcb_gym.config import REPO_ROOT, load_config, resolve_path
from rtcb_gym.data.store import data_sha256, processed_path, read_prices
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import (
    days_between,
    markdown_table,
    metrics_row,
    parse_splits,
    prices_in_days,
    run_policy,
)

SCORED = ("validation", "test")


def clean_nan(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_nan(v) for v in x]
    return x


def check_replay(split: str, label: str, replayed: float, planned: float) -> None:
    if abs(replayed - planned) > 1e-6 * max(1.0, abs(replayed)):
        print(f"WARNING {split} {label}: env replay {replayed:.2f} != optimizer objective {planned:.2f}")


def write_results(out: Path, title: str, cfg: dict, hashes: dict[str, str], table: str,
                  notes: list[str], payload: dict) -> None:
    m, b = cfg["market"], cfg["battery"]
    hash_lines = [f"- {name}: `{digest}`" for name, digest in hashes.items()]
    md = "\n".join([
        f"# RTCB-Gym baselines: {m['location']}, {title}",
        "",
        f"Operating days {m['start_date']} .. {m['end_date']}. "
        f"Battery: {b['power_mw']:g} MW / {b['energy_mwh']:g} MWh, {b['rte']:.0%} round-trip efficiency, "
        f"${b['degradation_cost_usd_per_mwh']:g}/MWh degradation cost, SoC {b['soc_init']:g} at the start "
        f"of each day.",
        "",
        "SHA-256 of the cleaned data:",
        "",
        *hash_lines,
        "",
        table,
        "",
        *notes,
        "Data provided by [Grid Status](https://www.gridstatus.io).",
    ])
    out.with_suffix(".md").write_text(md + "\n")
    out.with_suffix(".json").write_text(json.dumps(clean_nan(payload), indent=2, default=str) + "\n")
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")


def make_env(prices, params, cfg, start, end, da_prices=None) -> RTCBEnv:
    e, m, da = cfg["env"], cfg["market"], cfg.get("day_ahead") or {}
    return RTCBEnv(
        prices,
        params,
        days=days_between(start, end),
        episode_days=int(e["episode_days"]),
        history_len=int(e["history_len"]),
        dt_hours=float(e["dt_hours"]),
        tz=m["timezone"],
        sample="sequential",
        da_prices=da_prices,
        da_decision_time=str(da.get("decision_time", "10:00")),
        da_history_days=int(da.get("history_days", 7)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--out-dir", default=None, help="default: results/<hub>")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, d = cfg["market"], cfg["data"]
    params = BatteryParams.from_dict(cfg["battery"])
    processed = resolve_path(d["processed_dir"])
    prices = read_prices(processed_path(processed, m["location"], m["market"]))
    splits = parse_splits(cfg["splits"])
    hashes = {"real-time prices": data_sha256(prices)}
    for name, expected in (("real-time prices", d.get("expected_sha256")),):
        print(f"{name} sha256: {hashes[name]}")
        if expected and hashes[name] != expected:
            print(f"WARNING: expected {expected}; results will not match the published tables")
    out_dir = resolve_path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / m["location"].lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ real time
    envs = {name: make_env(prices, params, cfg, *span) for name, span in splits.items()}
    for name, env in envs.items():
        print(f"{name}: {len(env.days)} episode days, {len(env.excluded_days)} excluded")
        for day, reason in env.excluded_days.items():
            print(f"    excluded {day}: {reason}")

    b = cfg["baselines"]
    train_start, train_end = splits["train"]
    tr = b["threshold_rule"]
    threshold, search = fit_threshold_rule(
        envs["train"], prices_in_days(prices, train_start, train_end, m["timezone"]),
        tr["low_quantiles"], tr["high_quantiles"],
    )
    print(f"threshold rule fitted on train: charge <= {threshold.low_price:.2f}, "
          f"discharge >= {threshold.high_price:.2f} USD/MWh")
    rr = b["rolling_threshold"]
    rolling_fit, rolling_search = fit_rolling_threshold_rule(
        envs["train"], prices, rr["low_quantiles"], rr["high_quantiles"],
        int(rr["window_days"]), int(rr["min_window_days"]),
    )
    print(f"rolling threshold fitted on train: quantiles {rolling_fit.low_q} / {rolling_fit.high_q}, "
          f"window {rolling_fit.window_days} days")

    rows, runs, rolling_used = [], {}, {}
    for split in SCORED:
        env = envs[split]
        pf = PerfectForesight(prices, params, env.dt_hours)
        upper = run_policy(env, pf)
        check_replay(split, "perfect foresight", upper.revenue_usd, pf.planned_revenue_usd)
        rolling = fit_and_copy(rolling_fit, prices, params, env)
        results = [upper, run_policy(env, threshold), run_policy(env, rolling),
                   run_policy(env, ForecastMPC(params, env.obs_layout, env.dt_hours))]
        rows += [metrics_row(split, r, upper, params.energy_mwh) for r in results]
        runs[split] = [r.to_dict() for r in results]
        rolling_used[split] = rolling.history
    table = markdown_table(rows)
    print("\n" + table + "\n")
    write_results(
        out_dir / "real_time", "real-time energy", cfg, hashes, table,
        [f"Threshold rule: charge at or below ${threshold.low_price:.2f}/MWh, discharge at or above "
         f"${threshold.high_price:.2f}/MWh (training-split quantiles). Rolling threshold: quantiles "
         f"{rolling_fit.low_q} / {rolling_fit.high_q} of the previous {rolling_fit.window_days} days, "
         "refitted every day.", ""],
        {
            "config": cfg, "data_sha256": hashes,
            "threshold_rule": {"low_price": threshold.low_price, "high_price": threshold.high_price, "search": search},
            "rolling_threshold": {"low_q": rolling_fit.low_q, "high_q": rolling_fit.high_q,
                                  "window_days": rolling_fit.window_days, "search": rolling_search,
                                  "thresholds_by_day": rolling_used},
            "rows": rows, "runs": runs,
        },
    )

    # ------------------------------------------------------- two-settlement
    da_cfg = cfg.get("day_ahead")
    if not da_cfg:
        return
    da_prices = read_prices(processed_path(processed, m["location"], da_cfg["market"]))
    hashes["day-ahead prices"] = data_sha256(da_prices)
    print(f"day-ahead prices sha256: {hashes['day-ahead prices']}")
    if da_cfg.get("expected_sha256") and hashes["day-ahead prices"] != da_cfg["expected_sha256"]:
        print(f"WARNING: expected {da_cfg['expected_sha256']}; results will not match the published tables")

    rows, runs, no_forecast = [], {}, {}
    for split in SCORED:
        env = make_env(prices, params, cfg, *splits[split], da_prices=da_prices)
        print(f"{split} (two-settlement): {len(env.days)} episode days, {len(env.excluded_days)} excluded")
        pf2 = PerfectForesightTwoSettlement(prices, da_prices, params, env.dt_hours)
        upper = run_policy(env, pf2)
        check_replay(split, "perfect foresight (DA + RT)", upper.revenue_usd,
                     pf2.planned_da_usd + pf2.planned_revenue_usd)
        rt_only = PerfectForesight(prices, params, env.dt_hours)
        rt_only.name = "Perfect foresight (RT only)"
        da_only = PerfectForesightDayAhead(da_prices, params, env.obs_layout)
        delivered = DayAheadSchedule(params, env.obs_layout)
        combo = DayAheadPlusMPC(params, env.obs_layout, env.dt_hours)
        policies = [rt_only, da_only, delivered, combo, DayAheadForecastMPC(params, env.obs_layout, env.dt_hours),
                    ForecastMPC(params, env.obs_layout, env.dt_hours)]
        results = [upper] + [run_policy(env, p) for p in policies]
        check_replay(split, "perfect foresight (RT only)", results[1].revenue_usd, rt_only.planned_revenue_usd)
        check_replay(split, "perfect foresight (DA only)", results[2].revenue_usd, da_only.planned_revenue_usd)
        rows += [metrics_row(split, r, upper, params.energy_mwh) for r in results]
        runs[split] = [r.to_dict() for r in results]
        no_forecast[split] = {"DA schedule, delivered": delivered.n_days_without_forecast,
                              "DA schedule + MPC on DA prices": combo.schedule.n_days_without_forecast}
    table = markdown_table(rows, two_settlement=True)
    print("\n" + table + "\n")
    write_results(
        out_dir / "two_settlement", "day-ahead + real time (two-settlement)", cfg, hashes, table,
        ["Revenue = DA settlement + RT settlement - degradation. The day-ahead bid for each day is made "
         f"at {da_cfg.get('decision_time', '10:00')} local time the day before. \"Clipped actions\" counts "
         "real-time intervals, plus day-ahead hours clipped to what the battery could deliver.", ""],
        {"config": cfg, "data_sha256": hashes, "days_without_da_forecast": no_forecast, "rows": rows, "runs": runs},
    )


def fit_and_copy(fitted, prices, params, env):
    """A fresh rolling rule with the fitted quantiles, for a scored split."""
    from rtcb_gym.baselines import RollingThresholdRule

    return RollingThresholdRule(
        prices, fitted.low_q, fitted.high_q, params, env.obs_layout, env.dt_hours,
        window_days=fitted.window_days, min_window_days=fitted.min_window_days, tz=env.tz,
    )


if __name__ == "__main__":
    main()
