"""Baselines and the dispatch optimiser."""

import numpy as np
import pytest

from conftest import make_prices
from rtcb_gym.baselines import (
    ForecastMPC,
    PerfectForesight,
    ThresholdRule,
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
