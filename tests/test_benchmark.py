"""The benchmark layer: tasks, the agent contract, scorecards, the text
interface and the command line. Synthetic prices only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rtcb_gym.baselines import Policy
from rtcb_gym.benchmark import TASKS, evaluate, leaderboard, make_env, task_hash, task_spec
from rtcb_gym.data.synthetic import SYNTHETIC_SPLITS, synthetic_prices
from rtcb_gym.evaluate import REFERENCE, Idle, load_agent, main
from rtcb_gym.evaluation import parse_splits
from rtcb_gym.text import TextAgent, parse_action, render_observation

SYN = dict(source="synthetic")


def test_synthetic_prices_are_seeded_and_consistent():
    a, b, c = synthetic_prices(0), synthetic_prices(0), synthetic_prices(1)
    assert a["rt"].equals(b["rt"]) and not a["rt"].equals(c["rt"])
    means = a["rt5"]["price_usd_per_mwh"].to_numpy().reshape(-1, 3).mean(axis=1)
    np.testing.assert_allclose(means, a["rt"]["price_usd_per_mwh"], atol=1e-9)
    assert (a["ecrs"] >= 0).all() and (a["da_ecrs"] >= 0).all()
    parse_splits(SYNTHETIC_SPLITS)  # ordered, non-overlapping, post-RTC+B


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_scores_idle_at_zero_with_an_exact_bound(task):
    card = evaluate(Idle, task, limit_days=7, **SYN)
    assert card["task"] == task and card["partial"] and card["data"]["source"] == "synthetic"
    assert card["revenue_usd"] == 0 and card["capture_rate"] == 0
    assert card["bound_revenue_usd"] > 0 and card["bound_replay_error_usd"] < 1e-4
    assert card["invalid_actions"] == 0 and card["clipped_actions"] == 0
    assert len(card["daily_revenue_usd"]) == card["days"]
    json.dumps(card, default=str)


def test_spec_holds_training_prices_only_and_describes_the_rules():
    spec = task_spec("da-rt-energy-ecrs-5min", **SYN)
    train_end = parse_splits(SYNTHETIC_SPLITS)["train"][1]
    for key in ("rt5", "da", "ecrs", "da_ecrs"):
        x = spec.train[key]
        index = pd.DatetimeIndex(x["interval_start_utc"] if hasattr(x, "columns") else x.index)
        assert len(index) and max(index.tz_convert(spec.tz).date) <= train_end
    assert not hasattr(spec, "env")
    text = spec.describe()
    for phrase in ("100 MW", "ECRS", "Day-ahead", "never the price of the interval you act in"):
        assert phrase in text
    assert spec.action(0.5, 0.25).tolist() == [0.5, 0.25]
    offer = task_spec("rt-energy-offer", **SYN)
    assert "offer curve" in offer.describe() and offer.action(-1.0).shape == (4,)


def test_broken_agents_are_scored_not_crashed():
    class Broken(Policy):
        name = "broken"

        def __init__(self, spec):
            self.k = 0

        def day_ahead(self, context):
            return np.zeros(3)  # wrong length

        def act(self, obs, info):
            self.k += 1
            if self.k % 3 == 0:
                raise RuntimeError("model timed out")
            return np.array([np.nan]) if self.k % 3 == 1 else np.array([0.0, 0.0])

    card = evaluate(Broken, "da-rt-energy", limit_days=2, **SYN)
    assert card["invalid_actions"] == card["steps"] == 192
    assert card["invalid_day_ahead_bids"] == 2 and card["revenue_usd"] == 0
    assert any("timed out" in e for e in card["first_errors"]) and len(card["first_errors"]) <= 5


def test_agent_cannot_change_the_environment_through_what_it_is_given():
    class Vandal(Policy):
        name = "vandal"

        def __init__(self, spec):
            self.spec = spec

        def act(self, obs, info):
            obs[:] = 0.0
            info.clear()
            return self.spec.action(0.0)

    assert evaluate(Vandal, "rt-energy", limit_days=2, **SYN)["revenue_usd"] == 0
    honest = evaluate(REFERENCE["persistence-mpc"], "rt-energy", limit_days=2, **SYN)
    assert honest["revenue_usd"] > 0 and honest["capture_rate"] <= 1 + 1e-9


def test_reference_agents_and_task_hash():
    card = evaluate(REFERENCE["da-schedule"], "da-rt-energy-7day", **SYN)
    assert card["days"] == 7 and card["episodes"] == 1 and 0 < card["capture_rate"] <= 1
    with pytest.raises(ValueError, match="day-ahead"):
        evaluate(REFERENCE["da-schedule"], "rt-energy", **SYN)
    spec = task_spec("rt-energy", **SYN)
    h = task_hash(TASKS["rt-energy"], spec.battery)
    assert len(card["task_sha256"]) == 64 and card["task_sha256"] != h  # another task
    assert h == task_hash(TASKS["rt-energy"], spec.battery) != task_hash(TASKS["rt-energy-offer"], spec.battery)


def test_make_env_gives_the_training_split_for_learning_agents():
    env = make_env("da-rt-energy", split="train", **SYN)
    lo, hi = parse_splits(SYNTHETIC_SPLITS)["train"]
    assert all(lo <= d <= hi for d in env.days) and env.two_settlement
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_text_agent_round_trip_and_parse_errors():
    spec = task_spec("da-rt-energy-ecrs-5min", **SYN)
    assert parse_action(spec, 'Sure. {"energy": -0.5, "ecrs": 0}').tolist() == [-0.5, 0.0]
    assert parse_action(spec, "{}").tolist() == [0.0, 0.0]
    for bad in ("no json here", '{"power": 1}', '{"energy": "lots"}', "[1, 2]"):
        with pytest.raises(ValueError):
            parse_action(spec, bad)

    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        if "hours_to_bid" in prompt:
            n = json.loads(prompt.split("Current state:\n")[1].split("\n\n")[0])["hours_to_bid"]
            return json.dumps({"energy_mw": [0.0] * n, "ecrs_mw": [5.0] * n})
        return '{"energy": 0.0, "ecrs": 0.1}' if len(prompts) % 2 else "I am not sure."

    made = []
    card = evaluate(lambda s: made.append(TextAgent(s, complete, decide_every=12)) or made[-1],
                    "da-rt-energy-ecrs-5min", limit_days=1, **SYN)
    agent = made[0]
    assert agent.calls == len(prompts) and card["steps"] == 288
    assert 0 < card["invalid_actions"] < 288 and card["invalid_day_ahead_bids"] == 0
    assert card["ecrs_revenue_usd"] != 0
    state = json.loads(prompts[1].split("Current state:\n")[1].split("\n\n")[0])
    assert prompts[1].startswith("Task da-rt-energy-ecrs-5min") and state["interval_minutes"] == 5
    assert len(state["day_ahead_price_next_hours"]) == 24 and 0 <= state["soc"] <= 1


def test_rendered_observation_holds_nothing_from_the_current_interval():
    env = make_env("rt-energy", split="validation", sample="sequential", **SYN)
    spec = task_spec("rt-energy", **SYN)
    obs, info = env.reset()
    t = env._t
    state = render_observation(spec, obs, info)
    assert state["recent_prices_usd_per_mwh"][-1] == pytest.approx(env._price[t - 1], abs=0.005)
    assert state["price_history"]["max"] == pytest.approx(env._price[t - 96 : t].max(), abs=0.005)
    assert "price_usd_per_mwh" not in json.dumps(state)  # the env's key for the settled price of this step


def test_command_line_scores_a_file_agent_and_prints_a_leaderboard(tmp_path, capsys):
    agent = tmp_path / "mine.py"
    agent.write_text(
        "from rtcb_gym.baselines import Policy\n"
        "class A(Policy):\n"
        "    name = 'mine'\n"
        "    def __init__(self, spec): self.spec = spec\n"
        "    def act(self, obs, info): return self.spec.action(0.1)\n"
        "def build(spec): return A(spec)\n")
    out = tmp_path / "card.json"
    assert main(["--task", "rt-energy", "--agent", f"{agent}:build", "--source", "synthetic",
                 "--limit-days", "2", "--out", str(out)]) == 0
    card = json.loads(out.read_text())
    assert card["agent"] == "mine" and card["days"] == 2
    assert "not a result" in capsys.readouterr().out
    assert main(["--leaderboard", str(out)]) == 0
    table = capsys.readouterr().out
    assert "| rt-energy | validation (partial) | synthetic | mine | 2 |" in table
    assert main(["--list"]) == 0 and "da-rt-energy-ecrs-5min" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        load_agent("reference:nope")
    with pytest.raises(SystemExit):
        load_agent(f"{agent}:missing")
    assert leaderboard([card]).count("\n") == 2
