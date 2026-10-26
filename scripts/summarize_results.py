"""Print the README summary tables from results/<hub>/*.json.

    python scripts/summarize_results.py [results/hb_west results/hb_houston ...]

One table per setting (real time, two-settlement), one column per hub and
split. Each cell is the revenue in thousands of USD and the capture rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtcb_gym.config import REPO_ROOT

SPLITS = ("validation", "test")
SETTINGS = {
    "real_time": "Real-time energy. Capture rate = revenue / perfect foresight on the same days.",
    "two_settlement": "Day-ahead + real time. Capture rate = revenue / perfect foresight (DA + RT).",
}


def cell(row: dict) -> str:
    return f"{row['revenue_usd'] / 1000:,.0f}k ({row['capture_rate']:.1%})"


def table(setting: str, hub_dirs: list[Path]) -> str:
    data = {d.name.upper(): json.loads((d / f"{setting}.json").read_text()) for d in hub_dirs
            if (d / f"{setting}.json").exists()}
    if not data:
        return ""
    columns = [(hub, split) for hub in data for split in SPLITS]
    policies = list(dict.fromkeys(r["policy"] for r in next(iter(data.values()))["rows"]))
    lines = [
        "| Strategy | " + " | ".join(f"{hub} {split}" for hub, split in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for policy in policies:
        cells = []
        for hub, split in columns:
            match = [r for r in data[hub]["rows"] if r["policy"] == policy and r["split"] == split]
            cells.append(cell(match[0]) if match else "")
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    days = {f"{hub} {split}": next(r["days"] for r in data[hub]["rows"] if r["split"] == split)
            for hub, split in columns}
    lines += ["", "Days scored: " + ", ".join(f"{k} {v}" for k, v in days.items()) + "."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="default: every folder under results/")
    args = ap.parse_args()
    dirs = [Path(d) for d in args.dirs] or sorted(p for p in (REPO_ROOT / "results").iterdir() if p.is_dir())
    dirs = sorted(dirs, key=lambda p: (p.name != "hb_west", p.name))  # default hub first
    for setting, caption in SETTINGS.items():
        t = table(setting, dirs)
        if t:
            print(f"<!-- {setting} -->\n{caption}\n\n{t}\n")


if __name__ == "__main__":
    main()
