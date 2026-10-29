"""CPU-only, offline demonstration of the experimental energy + ECRS model.

Run ``python -m rtcb_gym.demo --out results/demo``. Synthetic prices are
generated locally. The outputs are cash-flow proxies, not ERCOT settlement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from rtcb_gym import __version__
from rtcb_gym.battery import BatteryParams
from rtcb_gym.experimental.ecrs import (
    STEP, ECRSRules, ECRSReservationEnv, FiveMinuteTape, solve_reservation,
)


def run_demo(out: str | Path) -> dict:
    """Replay two causal rules and a privileged oracle in the same model."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    k = np.arange(72)
    starts = pd.date_range("2026-02-02T06:00:00Z", periods=len(k), freq=STEP)
    frame = pd.DataFrame({
        "interval_start_utc": starts,
        "available_at_utc": starts + 2*STEP,
        "energy_price_usd_per_mwh": 40 + 70*np.sin(2*np.pi*k/24),
        "ecrs_price_usd_per_mw_h": 5 + 4*np.cos(2*np.pi*k/12),
    })
    tape = FiveMinuteTape(frame, {"price_basis": "synthetic LMP/MCPC cash-flow proxy"})
    battery = BatteryParams(power_mw=1., energy_mwh=2., soc_init=.5, soc_terminal_min=.5)
    rules = ECRSRules()
    start = 12
    plan = solve_reservation(tape.prices[start:], battery, rules, battery.soc_init)
    trace, summaries = [], []
    for name in ("idle", "lagged_price_rule", "perfect_foresight"):
        env = ECRSReservationEnv(tape, battery, rules, start=start, history_len=12)
        obs, _ = env.reset(seed=0)
        total, clipped = 0., 0
        for i in range(len(frame)-start):
            if name == "perfect_foresight":
                action = plan.actions_mw[i] / battery.power_mw
            elif name == "idle":
                action = np.zeros(2)
            else:
                visible = obs["prices"][obs["price_mask"] == 1]
                last = visible[-1, 0] if len(visible) else 40.
                energy = -1. if last < 20 else 1. if last > 60 else 0.
                action = np.array([energy, 0. if energy < 0 else 1.])
            obs, reward, done, _, info = env.step(action)
            total += reward
            clipped += int(info["energy_clipped"] or info["ecrs_clipped"])
            trace.append({
                "policy": name, "interval_start_utc": tape.starts[start+i],
                "requested_energy_mw": float(action[0]*battery.power_mw),
                "requested_ecrs_mw": float(action[1]*battery.power_mw),
                "executed_energy_mw": float(info["actual_mw"][0]),
                "executed_ecrs_mw": float(info["actual_mw"][1]),
                "soc_after": info["soc"],
                **{key: info[key] for key in ("energy_proxy_usd", "ecrs_proxy_usd",
                    "degradation_usd", "net_proxy_usd", "energy_clipped", "ecrs_clipped")},
            })
        if not done or info["soc"] < battery.soc_terminal_min-1e-7:
            raise RuntimeError("demo failed terminal-state verification")
        if total > plan.net_proxy_usd + 1e-6:
            raise RuntimeError("policy exceeded the model's oracle")
        if name == "perfect_foresight":
            if clipped or not np.isclose(total, plan.net_proxy_usd, rtol=1e-8, atol=1e-7):
                raise RuntimeError("oracle objective does not match executable replay")
        summaries.append({"policy": name, "steps": len(frame)-start,
                          "net_proxy_usd": total, "clipped_steps": clipped,
                          "terminal_soc": info["soc"],
                          "uses_future_prices": name == "perfect_foresight"})
    summary = {"schema_version": 1, "data_kind": "synthetic", "rule_id": rules.rule_id,
               "settlement_basis": "five-minute LMP/MCPC cash-flow proxy",
               "oracle_replay_verified": True, "policies": summaries}
    frame.to_csv(out / "input_prices.csv", index=False)
    pd.DataFrame(trace).to_csv(out / "trace.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    code_root = Path(__file__).resolve().parent
    code_hashes = {p: hashlib.sha256((code_root/p).read_bytes()).hexdigest()
                   for p in ("demo.py", "battery.py", "experimental/ecrs.py")}
    manifest = {"schema_version": 1, "rtcb_gym_version": __version__,
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
                "battery": asdict(battery), "rules": asdict(rules),
                "decision_start_utc": str(tape.starts[start]),
                "stop_exclusive_utc": str(tape.starts[-1]+STEP),
                "publication_delay_assumption": "interval end plus five minutes",
                "source": "deterministic synthetic formula in rtcb_gym.demo; no download",
                "code_sha256": code_hashes,
                "limitations": ["No market clearing or AS deployment",
                    "Full ECRS award assumed; no ECRS while charging",
                    "Cash-flow proxy, not a validated ERCOT settlement ledger",
                    "Perfect foresight sees future prices; other policies use observations only"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    names = ["input_prices.csv", "trace.csv", "summary.json", "manifest.json"]
    (out / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((out/n).read_bytes()).hexdigest()}  {n}\n" for n in names), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/demo", help="output directory (generated files are replaced)")
    args = parser.parse_args()
    summary = run_demo(args.out)
    print("Synthetic energy + ECRS demonstration; cash-flow proxies, no API or GPU.")
    for row in summary["policies"]:
        print(f"{row['policy']}: net_proxy_usd={row['net_proxy_usd']:.6f}, "
              f"clipped_steps={row['clipped_steps']}, terminal_soc={row['terminal_soc']:.6f}")
    print(f"Oracle replay verified. Outputs: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
