"""Score the reference and example agents on every benchmark task (real data)
and write the scorecards and the leaderboard.

    python scripts/run_benchmark.py [--config configs/default.yaml] [--splits validation test]

Writes results/benchmark/<hub>/<split>/<task>__<agent>.json and
results/benchmark/<hub>/LEADERBOARD.md. These agents are deliberately
simple: the leaderboard is a starting line, not a state of the art.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtcb_gym.benchmark import TASKS, evaluate, leaderboard
from rtcb_gym.config import REPO_ROOT, load_config
from rtcb_gym.evaluate import load_agent
from rtcb_gym.evaluation import clean_nan

EXAMPLES = REPO_ROOT / "examples" / "agents"
AGENTS = {
    "idle": "reference:idle",
    "persistence-mpc": "reference:persistence-mpc",
    "da-schedule": "reference:da-schedule",
    "example-threshold": str(EXAMPLES / "threshold.py"),
    "example-scripted-text": str(EXAMPLES / "text_scripted.py"),
}
RUNS = {
    "rt-energy": ["idle", "persistence-mpc", "example-threshold", "example-scripted-text"],
    "rt-energy-offer": ["persistence-mpc", "example-threshold"],
    "da-rt-energy": ["da-schedule", "persistence-mpc", "example-threshold", "example-scripted-text"],
    "da-rt-energy-7day": ["da-schedule", "persistence-mpc"],
    "da-rt-energy-ecrs-5min": ["da-schedule", "example-threshold"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    hub = cfg["market"]["location"].lower()
    out = REPO_ROOT / "results" / "benchmark" / hub
    cards = []
    assert set(RUNS) == set(TASKS)
    for split in args.splits:
        for task, agents in RUNS.items():
            for name in agents:
                card = evaluate(load_agent(AGENTS[name]), task, split=split, config=cfg)
                path = out / split / f"{task}__{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(clean_nan(card), indent=2, default=str) + "\n")
                cards.append(card)
                print(f"{split} {task} {card['agent']}: {card['revenue_usd']:,.0f} ({card['capture_rate']:.1%})", flush=True)
    table = leaderboard(cards)
    (out / "LEADERBOARD.md").write_text("\n".join([
        f"# RTCB-Gym benchmark: {cfg['market']['location']}",
        "",
        "Reference and example agents on every task. Capture rate = revenue / perfect foresight in the same "
        "task on the same days. These agents are simple on purpose; see docs/benchmark.md for how to add yours.",
        "",
        table,
        "",
        "Data provided by [Grid Status](https://www.gridstatus.io).",
    ]) + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
