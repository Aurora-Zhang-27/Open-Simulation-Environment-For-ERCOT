"""Score the baselines on the validation and test splits.

    python scripts/run_baselines.py [--config configs/default.yaml] [--out-dir DIR]

Writes, under results/<hub>/ by default:

    real_time.md / .json        real-time energy only (v0.1 setting), with
                                power actions and with offer-curve actions
    two_settlement.md / .json   day-ahead + real time, if the config has a
                                day_ahead section
    cross_day.md / .json        two-settlement, half-full episode edges: one-day
                                episodes against episodes of several days
    sensitivity.md / .json      other battery durations

Each .md file is a self-contained results table (hub, dates, data hashes,
battery); the .json files hold the full numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtcb_gym.baselines import (
    DayAheadForecastMPC,
    DayAheadPlusMPC,
    DayAheadSchedule,
    ForecastMPC,
    LevelCorrectedMPC,
    OfferMPC,
    PerfectForesight,
    PerfectForesightDayAhead,
    PerfectForesightTwoSettlement,
    RollingThresholdOffer,
    fit_level_correction,
    fit_rolling_threshold_rule,
    fit_threshold_rule,
)
from rtcb_gym.battery import BatteryParams
from rtcb_gym.config import REPO_ROOT, load_config, resolve_path
from rtcb_gym.data.store import data_sha256, processed_path, read_prices
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import (
    check_replay,
    clean_nan,
    days_between,
    markdown_table,
    metrics_row,
    parse_splits,
    prices_in_days,
    run_policy,
)

SCORED = ("validation", "test")


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
        f"of each episode.",
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


def make_env(prices, params, cfg, start, end, da_prices=None, episode_days=None, action_mode="power") -> RTCBEnv:
    e, m, da = cfg["env"], cfg["market"], cfg.get("day_ahead") or {}
    return RTCBEnv(
        prices,
        params,
        days=days_between(start, end),
        episode_days=int(episode_days or e["episode_days"]),
        action_mode=action_mode,
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

    k, rho, level_search = fit_level_correction(envs["train"])
    print(f"level correction fitted on train: last {k} intervals, decay {rho}")
    # Offer curves: the same quantile search, with the thresholds as offer prices.
    offer_envs = {name: make_env(prices, params, cfg, *span, action_mode="offer") for name, span in splits.items()}
    offer_fit, offer_search = fit_rolling_threshold_rule(
        offer_envs["train"], prices, rr["low_quantiles"], rr["high_quantiles"],
        int(rr["window_days"]), int(rr["min_window_days"]), cls=RollingThresholdOffer,
    )
    print(f"rolling threshold offer fitted on train: quantiles {offer_fit.low_q} / {offer_fit.high_q}")

    rows, runs, rolling_used = [], {}, {}
    for split in SCORED:
        env, offers = envs[split], offer_envs[split]
        pf = PerfectForesight(prices, params, env.dt_hours)
        upper = run_policy(env, pf)
        check_replay(split, "perfect foresight", upper.revenue_usd, pf.planned_revenue_usd)
        rolling = fit_and_copy(rolling_fit, prices, params, env)
        results = [upper, run_policy(env, threshold), run_policy(env, rolling),
                   run_policy(env, ForecastMPC(params, env.obs_layout, env.dt_hours)),
                   run_policy(env, LevelCorrectedMPC(params, env.obs_layout, env.dt_hours, k=k, rho=rho)),
                   run_policy(offers, fit_and_copy(offer_fit, prices, params, offers, RollingThresholdOffer)),
                   run_policy(offers, OfferMPC(params, offers.obs_layout, offers.dt_hours))]
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
         f"refitted every day. Level-corrected forecast: last {k} intervals, decay {rho} (lowest "
         "training-split forecast error).", "",
         "The two offer strategies submit an offer curve instead of a power (`action_mode=\"offer\"`): the "
         "market dispatches them against the interval's own price. Rolling threshold offer: quantiles "
         f"{offer_fit.low_q} / {offer_fit.high_q}. The bound is the same: a curve cannot do more than the best power.", "",
         "Capture rate: revenue / perfect foresight, with a 95% interval from a block bootstrap over days, and "
         "again without the five days on which perfect foresight earns most.", ""],
        {
            "config": cfg, "data_sha256": hashes,
            "threshold_rule": {"low_price": threshold.low_price, "high_price": threshold.high_price, "search": search},
            "rolling_threshold": {"low_q": rolling_fit.low_q, "high_q": rolling_fit.high_q,
                                  "window_days": rolling_fit.window_days, "search": rolling_search,
                                  "thresholds_by_day": rolling_used},
            "level_correction": {"k": k, "rho": rho, "search": level_search},
            "rolling_threshold_offer": {"low_q": offer_fit.low_q, "high_q": offer_fit.high_q, "search": offer_search},
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
    if cfg.get("cross_day"):
        cross_day(cfg, prices, da_prices, splits, hashes, out_dir)
    if cfg.get("sensitivity"):
        sensitivity(cfg, prices, da_prices, splits, hashes, out_dir)


def two_settlement_policies(prices, da_prices, params, env):
    """The bound first, then the strategies of the two-settlement table."""
    pf2 = PerfectForesightTwoSettlement(prices, da_prices, params, env.dt_hours)
    rt_only = PerfectForesight(prices, params, env.dt_hours)
    rt_only.name = "Perfect foresight (RT only)"
    return [pf2, rt_only, DayAheadSchedule(params, env.obs_layout), DayAheadPlusMPC(params, env.obs_layout, env.dt_hours),
            DayAheadForecastMPC(params, env.obs_layout, env.dt_hours), ForecastMPC(params, env.obs_layout, env.dt_hours)]


def cross_day(cfg, prices, da_prices, splits, hashes, out_dir) -> None:
    """Half-full episode edges: one-day episodes against multi-day episodes
    on exactly the same operating days."""
    c = cfg["cross_day"]
    n = int(c["episode_days"])
    params = BatteryParams.from_dict({**cfg["battery"], "soc_init": c["soc_init"],
                                      "soc_terminal_min": c["soc_terminal_min"]})
    rows, runs = [], {}
    for split in SCORED:
        long_env = make_env(prices, params, cfg, *splits[split], da_prices=da_prices, episode_days=n)
        starts, last = [], None
        for day in long_env.days:  # consecutive, non-overlapping episodes
            if last is None or (day - last).days >= n:
                starts.append(day)
                last = day
        days = [d for start in starts for d in long_env.episode_operating_days(start)]
        day_env = make_env(prices, params, cfg, *splits[split], da_prices=da_prices, episode_days=1)
        print(f"{split} (cross-day): {len(starts)} episodes of {n} days")
        for label, env, run_days in ((f"{n}-day episodes", long_env, starts), ("1-day episodes", day_env, days)):
            policies = two_settlement_policies(prices, da_prices, params, env)
            results = [run_policy(env, p, run_days) for p in policies]
            check_replay(split, f"perfect foresight, {label}", results[0].revenue_usd,
                         policies[0].planned_da_usd + policies[0].planned_revenue_usd)
            for r in results:
                rows.append({**metrics_row(split, r, results[0], params.energy_mwh), "episodes": label})
            runs[f"{split}, {label}"] = [r.to_dict() for r in results]
    lines = ["| Split | Episodes | Strategy | Days | Revenue (USD) | Capture rate (95% interval) | Cycles/day | Clipped actions |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lo, hi = r["capture_ci95"]
        capture = f"{r['capture_rate']:.1%}" + ("" if r["policy"] == r["bound"] else f" ({lo * 100:.1f} to {hi * 100:.1f})")
        clipped = f"{r['n_clipped']}" + (f" + {r['n_da_clipped_hours']} DA h" if r["n_da_clipped_hours"] else "")
        lines.append(f"| {r['split']} | {r['episodes']} | {r['policy']} | {r['days']} | "
                     f"{round(r['revenue_usd']) + 0:,} | {capture} | {r['efc_per_day']:.2f} | {clipped} |")
    table = "\n".join(lines)
    print("\n" + table + "\n")
    write_results(
        out_dir / "cross_day", "two-settlement across days", {**cfg, "battery": params.to_dict()}, hashes, table,
        [f"Every episode starts at SoC {params.soc_init:g} and must end at or above {params.soc_terminal_min:g}. "
         f"In {n}-day episodes the SoC carries over midnight; one-day episodes cover the same operating days. "
         "Capture rate is against perfect foresight (DA + RT) of the same episode length. Each day's "
         "day-ahead bid is checked against what the battery could deliver from the starting SoC.", ""],
        {"config": cfg, "battery": params.to_dict(), "data_sha256": hashes, "rows": rows, "runs": runs},
    )


def sensitivity(cfg, prices, da_prices, splits, hashes, out_dir) -> None:
    """The main strategies at other battery durations."""
    rows = []
    power = float(cfg["battery"]["power_mw"])
    for energy in cfg["sensitivity"]["energy_mwh"]:
        params = BatteryParams.from_dict({**cfg["battery"], "energy_mwh": energy})
        for split in SCORED:
            env = make_env(prices, params, cfg, *splits[split], da_prices=da_prices)
            policies = two_settlement_policies(prices, da_prices, params, env)
            results = [run_policy(env, p) for p in policies]
            check_replay(split, f"perfect foresight, {energy:g} MWh", results[0].revenue_usd,
                         policies[0].planned_da_usd + policies[0].planned_revenue_usd)
            for r in results:
                rows.append({**metrics_row(split, r, results[0], params.energy_mwh), "energy_mwh": energy,
                             "duration_hours": energy / power,
                             "usd_per_mw_day": r.revenue_usd / power / r.n_days})
            print(f"{split} {energy:g} MWh: " + ", ".join(f"{r.policy} {r.revenue_usd:,.0f}" for r in results))
    lines = ["| Split | Duration | Strategy | Revenue (USD) | USD per MW-day | Capture rate | Cycles/day |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['split']} | {r['duration_hours']:g} h | {r['policy']} | {round(r['revenue_usd']) + 0:,} | "
                     f"{r['usd_per_mw_day']:.0f} | {r['capture_rate']:.1%} | {r['efc_per_day']:.2f} |")
    table = "\n".join(lines)
    print("\n" + table + "\n")
    write_results(
        out_dir / "sensitivity", "battery duration (two-settlement)", cfg, hashes, table,
        [f"The same {power:g} MW with other energy capacities; everything else as in the two-settlement table. "
         "Capture rate is against perfect foresight (DA + RT) of the same battery.", ""],
        {"config": cfg, "data_sha256": hashes, "rows": rows},
    )


def fit_and_copy(fitted, prices, params, env, cls=None):
    """A fresh rolling rule with the fitted quantiles, for a scored split."""
    from rtcb_gym.baselines import RollingThresholdRule

    return (cls or RollingThresholdRule)(
        prices, fitted.low_q, fitted.high_q, params, env.obs_layout, env.dt_hours,
        window_days=fitted.window_days, min_window_days=fitted.min_window_days, tz=env.tz,
    )


if __name__ == "__main__":
    main()
