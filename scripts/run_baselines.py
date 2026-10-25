"""Score the three baselines on the validation and test splits.

    python scripts/run_baselines.py [--config configs/default.yaml]

Prints one markdown table (for the README) and writes it to
results/baselines.md, with full numbers in results/baselines.json.
"""

from __future__ import annotations

import argparse
import json
import math

from rtcb_gym.baselines import ForecastMPC, PerfectForesight, fit_threshold_rule
from rtcb_gym.battery import BatteryParams
from rtcb_gym.config import REPO_ROOT, load_config, resolve_path
from rtcb_gym.data.store import data_sha256, processed_path, read_prices
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import days_between, markdown_table, metrics_row, parse_splits, prices_in_days, run_policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, e = cfg["market"], cfg["env"]
    params = BatteryParams.from_dict(cfg["battery"])
    prices = read_prices(processed_path(resolve_path(cfg["data"]["processed_dir"]), m["location"], m["market"]))
    splits = parse_splits(cfg["splits"])
    digest = data_sha256(prices)
    expected = cfg["data"].get("expected_sha256")
    print(f"data sha256: {digest}")
    if expected and digest != expected:
        print(f"WARNING: expected {expected}; results will not match the published table")

    envs = {}
    for name, (start, end) in splits.items():
        envs[name] = RTCBEnv(
            prices,
            params,
            days=days_between(start, end),
            episode_days=int(e["episode_days"]),
            history_len=int(e["history_len"]),
            dt_hours=float(e["dt_hours"]),
            tz=m["timezone"],
            sample="sequential",
        )
        excluded = envs[name].excluded_days
        print(f"{name}: {len(envs[name].days)} episode days, {len(excluded)} excluded")
        for day, reason in excluded.items():
            print(f"    excluded {day}: {reason}")

    tr = cfg["baselines"]["threshold_rule"]
    train_start, train_end = splits["train"]
    threshold, search = fit_threshold_rule(
        envs["train"],
        prices_in_days(prices, train_start, train_end, m["timezone"]),
        tr["low_quantiles"],
        tr["high_quantiles"],
    )
    print(f"threshold rule fitted on train: charge <= {threshold.low_price:.2f}, "
          f"discharge >= {threshold.high_price:.2f} USD/MWh")

    rows, runs = [], {}
    for split in ("validation", "test"):
        env = envs[split]
        pf = PerfectForesight(prices, params, env.dt_hours)
        upper = run_policy(env, pf)
        gap = abs(upper.revenue_usd - pf.planned_revenue_usd)
        if gap > 1e-6 * max(1.0, abs(upper.revenue_usd)):
            print(f"WARNING {split}: perfect-foresight replay {upper.revenue_usd:.2f} "
                  f"!= LP objective {pf.planned_revenue_usd:.2f}")
        mpc = ForecastMPC(params, env.obs_layout, env.dt_hours)
        results = [upper, run_policy(env, threshold), run_policy(env, mpc)]
        for r in results:
            rows.append(metrics_row(split, r, upper, params.energy_mwh))
        runs[split] = [r.to_dict() for r in results]

    table = markdown_table(rows)
    print()
    print(table)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines.md").write_text(table + "\n")
    payload = {
        "config": cfg,
        "data_sha256": digest,
        "threshold_rule": {"low_price": threshold.low_price, "high_price": threshold.high_price, "search": search},
        "rows": [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows],
        "runs": runs,
    }
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {out_dir / 'baselines.md'} and {out_dir / 'baselines.json'}")


if __name__ == "__main__":
    main()
