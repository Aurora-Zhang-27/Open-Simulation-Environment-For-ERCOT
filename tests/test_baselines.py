"""Baselines and the dispatch optimiser."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from rtcb_gym.baselines import (
    ForecastMPC,
    PerfectForesight,
    Policy,
    RollingThresholdRule,
    ThresholdRule,
    fit_rolling_threshold_rule,
    fit_threshold_rule,
    persistence_24h,
    solve_dispatch,
)
from rtcb_gym.battery import BatteryParams
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import markdown_table, metrics_row, parse_splits, prices_in_days, run_policy

HAND = BatteryParams(power_mw=100, energy_mwh=200, rte=0.81, soc_init=0.0,
                     degradation_cost_usd_per_mwh=0.0)
DEFAULT = BatteryParams()


def test_lp_hand_example():
    # Charge 100 MW at $10 (stores 22.5 MWh), then discharge all of it:
    # 22.5 * 0.9 = 20.25 MWh = 81 MW for 15 min at $100.
    # Revenue = -10 * 25 + 100 * 20.25 = 1775.
    plan = solve_dispatch(np.array([10.0, 100.0]), HAND, 0.0, 0.25)
    assert plan.power_mw == pytest.approx([-100.0, 81.0])
    assert plan.revenue_usd == pytest.approx(1775.0)
    assert plan.soc == pytest.approx([22.5 / 200, 0.0], abs=1e-9)
    assert not plan.used_milp


def test_lp_stays_idle_when_spread_does_not_cover_losses():
    # 10 -> 11 USD: buying 1 MWh costs 10, sells back 0.81 MWh for 8.91.
    plan = solve_dispatch(np.array([10.0, 11.0]), HAND, 0.0, 0.25)
    assert plan.power_mw == pytest.approx([0.0, 0.0], abs=1e-9)
    assert plan.revenue_usd == pytest.approx(0.0, abs=1e-9)


def test_lp_respects_soc_limits():
    rng = np.random.default_rng(1)
    prices = rng.normal(30, 40, 96)
    q = BatteryParams(soc_min=0.1, soc_max=0.9, soc_init=0.5)
    plan = solve_dispatch(prices, q, 0.5, 0.25)
    assert plan.soc.min() >= 0.1 - 1e-9 and plan.soc.max() <= 0.9 + 1e-9
    assert np.abs(plan.power_mw).max() <= q.power_mw + 1e-9


def test_milp_fallback_prevents_simultaneous_charge_and_discharge():
    # Full battery, very negative prices: the LP relaxation would charge and
    # discharge at once to burn energy. The MILP keeps it physical.
    prices = np.full(8, -1000.0)
    plan = solve_dispatch(prices, DEFAULT, 1.0, 0.25)
    assert plan.used_milp
    # Replaying the MILP plan in the env must give exactly its objective.
    p = make_prices(days=3)
    p.loc[:, "price_usd_per_mwh"] = -1000.0
    env = RTCBEnv(p, BatteryParams(soc_init=1.0))
    pf = PerfectForesight(p, env.battery_params)
    r = run_policy(env, pf, days=env.days[:1])
    assert r.revenue_usd == pytest.approx(pf.planned_revenue_usd, rel=1e-9)
    assert r.n_clipped == 0


def test_perfect_foresight_replay_matches_lp(prices):
    env = RTCBEnv(prices, DEFAULT)
    pf = PerfectForesight(prices, DEFAULT)
    r = run_policy(env, pf)
    assert r.n_days == len(env.days)
    assert r.revenue_usd == pytest.approx(pf.planned_revenue_usd, rel=1e-9)
    assert r.n_clipped == 0
    assert r.revenue_usd > 0


def test_perfect_foresight_is_an_upper_bound(prices):
    env = RTCBEnv(prices, DEFAULT)
    upper = run_policy(env, PerfectForesight(prices, DEFAULT))
    rule = ThresholdRule(20.0, 45.0, DEFAULT, env.obs_layout)
    mpc = ForecastMPC(DEFAULT, env.obs_layout)
    for res in (run_policy(env, rule), run_policy(env, mpc)):
        assert res.revenue_usd <= upper.revenue_usd + 1e-6
        row = metrics_row("test", res, upper, DEFAULT.energy_mwh)
        assert row["capture_rate"] <= 1.0
    # Every day, not only in total.
    rule_days = run_policy(env, rule).daily_revenue_usd
    assert all(a <= b + 1e-6 for a, b in zip(rule_days, upper.daily_revenue_usd))


def test_persistence_forecast_uses_past_only():
    past = np.arange(96.0)  # prices t-96 .. t-1
    assert persistence_24h(past, 96).tolist() == past.tolist()
    assert persistence_24h(past, 3).tolist() == [0.0, 1.0, 2.0]
    past2 = np.arange(120.0)
    assert persistence_24h(past2, 2).tolist() == [24.0, 25.0]
    with pytest.raises(ValueError):
        persistence_24h(past, 97)  # would need interval t itself
    with pytest.raises(ValueError):
        persistence_24h(np.arange(50.0), 10)


def test_mpc_matches_perfect_foresight_when_prices_repeat_daily():
    # If every day is identical, the 24h persistence forecast is exact, so
    # MPC must reach the perfect-foresight revenue.
    p = make_prices(days=4, periodic=True)
    env = RTCBEnv(p, DEFAULT, days=["2026-03-02", "2026-03-03"])
    upper = run_policy(env, PerfectForesight(p, DEFAULT))
    mpc = run_policy(env, ForecastMPC(DEFAULT, env.obs_layout))
    assert mpc.revenue_usd == pytest.approx(upper.revenue_usd, rel=1e-6)


def test_threshold_rule_never_requests_infeasible_power(prices):
    env = RTCBEnv(prices, DEFAULT)
    r = run_policy(env, ThresholdRule(25.0, 40.0, DEFAULT, env.obs_layout))
    assert r.n_clipped == 0
    assert r.storage_discharge_mwh > 0


def test_threshold_fit_uses_given_training_prices(prices):
    env = RTCBEnv(prices, DEFAULT, days=["2026-03-02", "2026-03-03", "2026-03-04"])
    train_prices = prices_in_days(prices, env.days[0], env.days[-1])
    assert len(train_prices) == 3 * 96
    rule, table = fit_threshold_rule(env, train_prices, [0.2, 0.3], [0.7, 0.8])
    assert len(table) == 4
    best = max(table, key=lambda r: r["train_revenue_usd"])
    assert rule.low_price == pytest.approx(np.quantile(train_prices, best["low_q"]))
    assert rule.high_price == pytest.approx(np.quantile(train_prices, best["high_q"]))


def test_runs_are_bit_for_bit_reproducible(prices):
    def run():
        env = RTCBEnv(prices, DEFAULT)
        return [run_policy(env, pol).to_dict() for pol in
                (PerfectForesight(prices, DEFAULT), ForecastMPC(DEFAULT, env.obs_layout))]

    assert run() == run()


def test_parse_splits_validation():
    good = {"train": ["2025-12-05", "2026-01-31"], "validation": ["2026-02-01", "2026-02-28"],
            "test": ["2026-03-01", "2026-03-31"]}
    assert parse_splits(good)["test"][0].isoformat() == "2026-03-01"
    overlap = dict(good, validation=["2026-01-15", "2026-02-28"])
    with pytest.raises(ValueError):
        parse_splits(overlap)
    early = dict(good, train=["2025-11-01", "2026-01-31"])
    with pytest.raises(ValueError, match="RTC\\+B"):
        parse_splits(early)


def test_markdown_table_format(prices):
    env = RTCBEnv(prices, DEFAULT, days=["2026-03-02"])
    upper = run_policy(env, PerfectForesight(prices, DEFAULT))
    table = markdown_table([metrics_row("test", upper, upper, DEFAULT.energy_mwh)])
    assert "| test | Perfect foresight | 1 |" in table and "100.0%" in table


CROSS_DAY = BatteryParams(soc_init=0.5, soc_terminal_min=0.5)


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        return self.rng.uniform(-1, 1, size=1)


def test_lp_end_target():
    prices = np.array([10.0] * 4 + [200.0] * 4)
    free = solve_dispatch(prices, CROSS_DAY, 0.5, 0.25, soc_end_min=0.0)
    held = solve_dispatch(prices, CROSS_DAY, 0.5, 0.25)  # default: params.soc_terminal_min
    assert free.soc[-1] == pytest.approx(0.0, abs=1e-9)
    assert held.soc[-1] >= 0.5 - 1e-9
    assert held.revenue_usd < free.revenue_usd


def test_upper_bound_holds_with_cross_day_end_target(prices):
    env = RTCBEnv(prices, CROSS_DAY)
    pf = PerfectForesight(prices, CROSS_DAY)
    upper = run_policy(env, pf)
    assert upper.revenue_usd == pytest.approx(pf.planned_revenue_usd, rel=1e-9)
    assert upper.n_clipped == 0
    others = [
        run_policy(env, ThresholdRule(20.0, 45.0, CROSS_DAY, env.obs_layout)),
        run_policy(env, ForecastMPC(CROSS_DAY, env.obs_layout)),
        run_policy(env, RandomPolicy(0)),
    ]
    for r in others:
        assert all(a <= b + 1e-6 for a, b in zip(r.daily_revenue_usd, upper.daily_revenue_usd))
    assert others[0].n_clipped == 0  # the rule respects the end target itself
    assert others[1].n_clipped == 0


def test_mpc_matches_perfect_foresight_with_end_target():
    p = make_prices(days=4, periodic=True)
    env = RTCBEnv(p, CROSS_DAY, days=["2026-03-02", "2026-03-03"])
    upper = run_policy(env, PerfectForesight(p, CROSS_DAY))
    mpc = run_policy(env, ForecastMPC(CROSS_DAY, env.obs_layout))
    assert mpc.revenue_usd == pytest.approx(upper.revenue_usd, rel=1e-6)


class ConstantAction(Policy):
    """Request the same normalised action at every interval."""

    def __init__(self, action):
        self.action = float(action)
        self.name = f"constant {self.action:+.1f}"

    def act(self, obs, info):
        return np.array([self.action])


# The results table reports zero clipped actions for all three baselines.
# These tests make sure that zero comes from the strategies, not from a
# counter that never counts.


@pytest.mark.parametrize(
    "soc_init, action, reason",
    [(1.0, -1.0, "soc_max"), (0.0, 1.0, "soc_min")],
    ids=["charge_full_battery", "discharge_empty_battery"],
)
def test_clip_counter_counts_every_infeasible_request(soc_init, action, reason):
    params = BatteryParams(soc_init=soc_init)
    env = RTCBEnv(make_prices(), params)
    env.reset(options={"day": env.days[0]})
    _, _, _, _, info = env.step(np.array([action]))
    assert info["clipped"] and info["clip_reason"] == reason

    r = run_policy(env, ConstantAction(action))
    assert r.n_steps > 0
    assert r.n_clipped == r.n_steps  # nothing is ever feasible
    assert r.revenue_usd == 0.0


def test_clip_counter_reaches_the_results_table():
    # From empty, charging at 100 MW stores 100 * 0.25 * sqrt(0.85) = 23.05 MWh
    # per interval, so 8 intervals fit into 200 MWh. The 9th is cut short and
    # every later one is refused: per day, all but 8 intervals are clipped.
    params = BatteryParams(soc_init=0.0)
    p = make_prices()
    env = RTCBEnv(p, params)
    r = run_policy(env, ConstantAction(-1.0))
    expected = r.n_steps - 8 * r.n_days
    assert 0 < expected < r.n_steps
    assert r.n_clipped == expected

    upper = run_policy(env, PerfectForesight(p, params))
    row = metrics_row("test", r, upper, params.energy_mwh)
    assert row["n_clipped"] == expected
    assert markdown_table([row]).rstrip("\n").endswith(f"| {expected} |")


# --------------------------------------------------------------------------
# Rolling threshold rule


def day_indexed_prices(days=40, start="2026-03-01"):
    """Synthetic prices where every interval of day k costs k USD/MWh, plus a
    small intraday shape so quantiles differ within a day."""
    p = make_prices(start=start, days=days, seed=None)
    day = p["local_time"].dt.date
    k = (pd.to_datetime(day) - pd.Timestamp(start)).dt.days.to_numpy()
    p["price_usd_per_mwh"] = k + 0.01 * (p["price_usd_per_mwh"] - 30.0)
    return p


def test_rolling_window_is_the_previous_n_days_only():
    p = day_indexed_prices()
    rule = RollingThresholdRule(p, 0.2, 0.8, DEFAULT, {}, window_days=30)
    day = pd.Timestamp("2026-04-05").date()  # day index 35
    window = rule.window_prices(day)
    day_index = np.round(window).astype(int)
    assert day_index.min() == 5 and day_index.max() == 34  # days 35-30 .. 35-1
    assert len(window) == 30 * 96 - 4  # one DST spring-forward day in the window


def test_rolling_rule_never_reads_the_episode_or_later_prices():
    p = make_prices(days=40)
    env = RTCBEnv(p, DEFAULT, days=["2026-04-02", "2026-04-03"], sample="sequential")
    changed = p.copy()
    late = changed["local_time"].dt.date >= pd.Timestamp("2026-04-02").date()
    changed.loc[late, "price_usd_per_mwh"] += 1000.0
    runs = []
    for table in (p, changed):
        rule = RollingThresholdRule(table, 0.3, 0.7, DEFAULT, env.obs_layout, window_days=30)
        runs.append((run_policy(env, rule, days=[env.days[0]]), rule.history))
    (a, ha), (b, hb) = runs
    assert ha == hb  # same thresholds for 2026-04-02
    assert a.daily_revenue_usd == b.daily_revenue_usd


def test_rolling_rule_idles_without_enough_history():
    p = make_prices(days=10)
    env = RTCBEnv(p, DEFAULT, days=["2026-03-04"])
    rule = RollingThresholdRule(p, 0.3, 0.7, DEFAULT, env.obs_layout, window_days=30, min_window_days=7)
    r = run_policy(env, rule)
    assert rule.history[0]["low_price"] == -np.inf and rule.history[0]["high_price"] == np.inf
    assert r.storage_discharge_mwh == 0.0 and r.revenue_usd == 0.0 and r.n_clipped == 0


def test_rolling_rule_follows_a_level_shift():
    p = make_prices(days=60, seed=1)
    shifted = p["local_time"].dt.date >= pd.Timestamp("2026-03-31").date()
    p.loc[shifted, "price_usd_per_mwh"] += 50.0
    env = RTCBEnv(p, DEFAULT, days=["2026-03-20", "2026-04-28"], sample="sequential")
    rule = RollingThresholdRule(p, 0.3, 0.7, DEFAULT, env.obs_layout, window_days=14)
    r = run_policy(env, rule)
    before, after = rule.history
    assert after["low_price"] - before["low_price"] == pytest.approx(50.0, abs=10.0)
    assert after["high_price"] - before["high_price"] == pytest.approx(50.0, abs=10.0)
    assert r.n_clipped == 0  # requests stay feasible, like the static rule


def test_fit_rolling_threshold_picks_the_best_training_pair():
    p = make_prices(days=30)
    days = [d for d in pd.date_range("2026-03-12", "2026-03-20").date]
    env = RTCBEnv(p, DEFAULT, days=days, sample="sequential")
    rule, table = fit_rolling_threshold_rule(env, p, [0.1, 0.3], [0.7, 0.9], window_days=7, min_window_days=3)
    best = max(table, key=lambda row: row["train_revenue_usd"])
    assert (rule.low_q, rule.high_q) == (best["low_q"], best["high_q"])
    assert len(table) == 4 and rule.name == "Rolling threshold (7 d)"
    assert rule.history == []  # a fresh rule, not one used during the search


def test_rolling_rule_rejects_bad_arguments():
    p = make_prices(days=3)
    with pytest.raises(ValueError):
        RollingThresholdRule(p, 0.7, 0.3, DEFAULT, {})
    with pytest.raises(ValueError):
        RollingThresholdRule(p, 0.3, 0.7, DEFAULT, {}, window_days=5, min_window_days=6)
