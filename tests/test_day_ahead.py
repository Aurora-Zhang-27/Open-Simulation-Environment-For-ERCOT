"""Day-ahead market and two-settlement episodes (synthetic prices only)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from conftest import make_da_prices, make_prices
from rtcb_gym.baselines import (
    DayAheadForecastMPC,
    DayAheadPlusMPC,
    DayAheadSchedule,
    ForecastMPC,
    PerfectForesight,
    PerfectForesightTwoSettlement,
    Policy,
    solve_dispatch,
)
from rtcb_gym.battery import BatteryParams
from rtcb_gym.data.clean import INTERVAL
from rtcb_gym.day_ahead import (
    DA_AHEAD,
    build_context,
    decision_time_utc,
    feasible_day_ahead,
    persistence_da_forecast,
)
from rtcb_gym.env import RTCBEnv
from rtcb_gym.evaluation import markdown_table, metrics_row, run_policy

PARAMS = BatteryParams()
DAY = dt.date(2026, 3, 5)


@pytest.fixture
def rt():
    return make_prices(start="2026-03-01", days=10)


@pytest.fixture
def da():
    return make_da_prices(start="2026-03-01", days=10, offset=4.0)


def env2s(rt, da, **kw):
    return RTCBEnv(rt, PARAMS, da_prices=da, **kw)


class Random2S(Policy):
    name = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def day_ahead(self, context):
        return self.rng.uniform(-1, 1, context.n_hours) * PARAMS.power_mw

    def act(self, obs, info):
        return self.rng.uniform(-1, 1, size=1)


# ---------------------------------------------------------------- env basics


def test_passes_gymnasium_checker(rt, da):
    check_env(env2s(rt, da), skip_render_check=True)


def test_observation_layout(rt, da):
    env = env2s(rt, da)
    lay = env.obs_layout
    assert env.observation_space.shape == (10 + 96 + 3 * DA_AHEAD,)
    assert lay["da_price_ahead"] == slice(106, 106 + DA_AHEAD)
    assert lay["da_ahead_mask"].stop == env.observation_space.shape[0]
    assert env.two_settlement and not RTCBEnv(rt, PARAMS).two_settlement


def test_no_day_ahead_position_equals_real_time_only(rt, da):
    a, b = env2s(rt, da), RTCBEnv(rt, PARAMS)
    oa, ia = a.reset(options={"day": DAY})
    ob, _ = b.reset(options={"day": DAY})
    assert np.all(ia["da_award_mw"] == 0.0)
    np.testing.assert_array_equal(oa[: len(ob)], ob)  # same real-time part
    rng = np.random.default_rng(0)
    for _ in range(96):
        act = rng.uniform(-1, 1, size=1)
        oa, ra, *_ , info = a.step(act)
        ob, rb, *_ = b.step(act)
        assert ra == pytest.approx(rb, abs=1e-9)
        assert info["da_revenue_usd"] == 0.0
        np.testing.assert_array_equal(oa[: len(ob)], ob)


def test_two_settlement_needs_one_day_episodes_and_the_same_hub(rt, da):
    with pytest.raises(ValueError, match="one operating day"):
        env2s(rt, da, episode_days=2)
    with pytest.raises(ValueError, match="HB_WEST"):
        env2s(rt, make_da_prices(location="HB_HOUSTON"))


def test_days_without_complete_day_ahead_prices_are_excluded(rt, da):
    local_day = da["local_time"].dt.date
    holes = da.drop(index=da.index[local_day == DAY][:1])
    env = env2s(rt, holes)
    assert DAY not in env.days
    assert env.excluded_days[str(DAY)] == "incomplete day-ahead prices"
    with pytest.raises(ValueError, match="day-ahead"):
        env.reset(options={"day": DAY})


# ------------------------------------------------------ what is known when


def test_decision_time_is_10am_local_the_day_before():
    assert decision_time_utc("2026-03-05") == pd.Timestamp("2026-03-04 16:00", tz="UTC")  # CST
    assert decision_time_utc("2026-07-15") == pd.Timestamp("2026-07-14 15:00", tz="UTC")  # CDT
    assert decision_time_utc("2026-07-15", "13:30") == pd.Timestamp("2026-07-14 18:30", tz="UTC")
    with pytest.raises(ValueError):
        decision_time_utc("2026-07-15", "25:00")


def test_context_holds_only_what_is_known_at_the_decision(rt, da):
    env = env2s(rt, da)
    ctx = env.day_ahead_context(DAY)
    decision = ctx.decision_time_utc
    assert ctx.rt_history.index.max() + INTERVAL == decision  # last settled interval
    assert ctx.rt_history.index.min() >= decision - pd.Timedelta(days=7)
    da_days = set(ctx.da_history.index.tz_convert("US/Central").date)
    assert max(da_days) == DAY - dt.timedelta(days=1)
    assert ctx.n_hours == 24 and ctx.soc_init == PARAMS.soc_init


def test_context_ignores_later_prices(rt, da):
    decision = decision_time_utc(DAY)
    rt2, da2 = rt.copy(), da.copy()
    rt2.loc[rt2["interval_end_utc"] > decision, "price_usd_per_mwh"] += 500.0
    da2.loc[da2["local_time"].dt.date >= DAY, "price_usd_per_mwh"] -= 500.0
    a, b = env2s(rt, da).day_ahead_context(DAY), env2s(rt2, da2).day_ahead_context(DAY)
    pd.testing.assert_series_equal(a.rt_history, b.rt_history)
    pd.testing.assert_series_equal(a.da_history, b.da_history)

    rt3 = rt.copy()  # the interval that ended exactly at the decision is known
    rt3.loc[rt3["interval_end_utc"] == decision, "price_usd_per_mwh"] += 1.0
    c = env2s(rt3, da).day_ahead_context(DAY)
    assert c.rt_history.iloc[-1] == a.rt_history.iloc[-1] + 1.0


def test_day_ahead_blocks_in_the_observation(rt, da):
    env = env2s(rt, da)
    schedule = np.linspace(-50, 50, 24)
    obs, info = env.reset(options={"day": DAY, "da_schedule": schedule})
    lay = env.obs_layout
    award = info["da_award_mw"]
    da_day = da[da["local_time"].dt.date == DAY]["price_usd_per_mwh"].to_numpy()
    np.testing.assert_allclose(obs[lay["da_price_ahead"]], np.repeat(da_day, 4))
    np.testing.assert_allclose(obs[lay["da_award_ahead"]], np.repeat(award, 4) / PARAMS.power_mw)
    assert obs[lay["da_ahead_mask"]].sum() == 96
    for k in range(96):
        obs, _, done, _, info = env.step(np.array([0.0]))
        left = info["steps_remaining"]
        assert obs[lay["da_ahead_mask"]].sum() == left
        assert np.all(obs[lay["da_price_ahead"]][left:] == 0.0)
        if left:
            assert obs[lay["da_price_ahead"]][0] == da_day[(k + 1) // 4]
    assert done


def test_real_time_part_of_the_observation_is_unchanged(rt, da):
    a, b = env2s(rt, da), RTCBEnv(rt, PARAMS)
    oa, _ = a.reset(options={"day": DAY})
    ob, _ = b.reset(options={"day": DAY})
    assert np.all(oa[a.obs_layout["prices"]] == ob[b.obs_layout["prices"]])


# ------------------------------------------------------- the day-ahead bid


def test_infeasible_schedules_are_clipped_and_counted():
    sell, n, soc = feasible_day_ahead(np.full(24, 100.0), PARAMS, 0.0)
    assert np.all(sell == 0.0) and n == 24  # an empty battery cannot sell
    buy, n, soc = feasible_day_ahead(np.full(24, -100.0), PARAMS, 0.0)
    stored = 100.0 * np.sqrt(PARAMS.rte)  # MWh per full hour of charging
    assert buy[0] == buy[1] == -100.0
    assert buy[2] == pytest.approx(-(200.0 - 2 * stored) / np.sqrt(PARAMS.rte))
    assert np.all(np.abs(buy[3:]) < 1e-9) and n == 22
    assert soc[-1] == pytest.approx(1.0)


def test_reset_clips_the_bid_and_reports_it(rt, da):
    env = env2s(rt, da)
    _, info = env.reset(options={"day": DAY, "da_schedule": np.full(24, 100.0)})
    assert info["da_clipped_hours"] == 24 and np.all(info["da_award_mw"] == 0.0)
    assert np.all(info["da_requested_mw"] == 100.0)
    with pytest.raises(ValueError, match="24 hourly values"):
        env.reset(options={"day": DAY, "da_schedule": np.zeros(23)})


def test_day_ahead_policy_and_options(rt, da):
    # 5 MW for 24 h stores 5 * 24 * sqrt(0.85) = 111 MWh: feasible
    env = env2s(rt, da, day_ahead_policy=lambda ctx: np.full(ctx.n_hours, -5.0))
    _, info = env.reset(options={"day": DAY})
    assert np.all(info["da_award_mw"] == -5.0) and info["da_clipped_hours"] == 0
    _, info = env.reset(options={"day": DAY, "da_schedule": np.zeros(24)})
    assert np.all(info["da_award_mw"] == 0.0)  # options win over the policy


def test_dst_days_have_23_and_25_hours():
    rt = make_prices(start="2026-03-06", days=4)
    da = make_da_prices(start="2026-03-06", days=4)
    _, info = env2s(rt, da).reset(options={"day": "2026-03-08"})
    assert len(info["da_award_mw"]) == 23
    rt = make_prices(start="2026-10-30", days=4)
    da = make_da_prices(start="2026-10-30", days=4)
    env = env2s(rt, da)
    _, info = env.reset(options={"day": "2026-11-01", "da_schedule": np.full(25, -5.0)})
    assert len(info["da_award_mw"]) == 25
    total = 0.0
    done = False
    while not done:
        _, _, done, _, step = env.step(np.array([-0.05]))
        total += step["da_revenue_usd"]
    expected = (-5.0 * da[da["local_time"].dt.date == dt.date(2026, 11, 1)]["price_usd_per_mwh"]).sum()
    assert total == pytest.approx(expected)


# ------------------------------------------------------------- accounting


def test_delivering_a_feasible_award_needs_no_clipping(rt, da):
    env = env2s(rt, da, sample="sequential")
    r = run_policy(env, DayAheadSchedule(PARAMS, env.obs_layout))
    assert r.n_clipped == 0 and r.n_da_clipped_hours == 0
    assert abs(r.rt_revenue_usd) < 1e-6  # delivered exactly: no deviation
    assert r.da_revenue_usd != 0.0


def test_revenue_splits_into_spread_and_real_time_parts(rt, da):
    env = env2s(rt, da)
    rng = np.random.default_rng(3)
    obs, info = env.reset(options={"day": DAY, "da_schedule": rng.uniform(-100, 100, 24)})
    award = info["da_award_mw"]
    rewards, power, rt_price, da_price = [], [], [], []
    done = False
    while not done:
        obs, r, done, _, step = env.step(rng.uniform(-1, 1, size=1))
        rewards.append(r)
        power.append(step["actual_power_mw"])
        rt_price.append(step["price_usd_per_mwh"])
        da_price.append(step["da_price_usd_per_mwh"])
    power, rt_price, da_price = map(np.array, (power, rt_price, da_price))
    spread = da_price.reshape(24, 4)[:, 0] - rt_price.reshape(24, 4).mean(axis=1)
    day_ahead_part = float(award @ spread)
    real_time_part = float(power @ rt_price * 0.25 - np.abs(power).sum() * 0.25 * PARAMS.degradation_cost_usd_per_mwh)
    assert sum(rewards) == pytest.approx(day_ahead_part + real_time_part, rel=1e-9)


def test_run_policy_totals_add_up(rt, da):
    env = env2s(rt, da, sample="sequential")
    r = run_policy(env, Random2S(1))
    assert r.da_revenue_usd + r.rt_revenue_usd == pytest.approx(r.energy_revenue_usd)
    assert r.energy_revenue_usd - r.degradation_cost_usd == pytest.approx(r.revenue_usd)
    assert r.n_da_clipped_hours > 0  # random bids from an empty battery get clipped


# ---------------------------------------------------- perfect foresight


def test_perfect_foresight_replays_its_objective(rt, da):
    env = env2s(rt, da, sample="sequential")
    pf = PerfectForesightTwoSettlement(rt, da, PARAMS)
    r = run_policy(env, pf)
    assert r.revenue_usd == pytest.approx(pf.planned_da_usd + pf.planned_revenue_usd, rel=1e-6)
    assert r.n_clipped == 0 and r.n_da_clipped_hours == 0


def test_perfect_foresight_is_an_upper_bound(rt, da):
    env = env2s(rt, da, sample="sequential")
    upper = run_policy(env, PerfectForesightTwoSettlement(rt, da, PARAMS)).revenue_usd
    rt_only = PerfectForesight(rt, PARAMS)
    rt_only.name = "Perfect foresight (RT only)"
    others = [
        rt_only,
        DayAheadSchedule(PARAMS, env.obs_layout),
        DayAheadPlusMPC(PARAMS, env.obs_layout),
        DayAheadForecastMPC(PARAMS, env.obs_layout),
        ForecastMPC(PARAMS, env.obs_layout),
        *[Random2S(seed) for seed in range(5)],
    ]
    for policy in others:
        assert run_policy(env, policy).revenue_usd <= upper + 1e-6, policy.name


def test_upper_bound_holds_with_negative_spreads_and_an_end_target():
    rt = make_prices(start="2026-03-01", days=6, seed=7)
    da = make_da_prices(start="2026-03-01", days=6, offset=-80.0)  # deeply negative DA prices
    params = BatteryParams(soc_init=0.5, soc_terminal_min=0.5)
    env = RTCBEnv(rt, params, da_prices=da, sample="sequential")
    pf = PerfectForesightTwoSettlement(rt, da, params)
    upper = run_policy(env, pf)
    assert upper.revenue_usd == pytest.approx(pf.planned_da_usd + pf.planned_revenue_usd, rel=1e-6)
    for seed in range(5):
        assert run_policy(env, Random2S(seed)).revenue_usd <= upper.revenue_usd + 1e-6


# ------------------------------------------------------------ forecasts


def test_persistence_forecast_is_yesterdays_prices_by_clock_hour(rt, da):
    ctx = env2s(rt, da).day_ahead_context(DAY)
    yesterday = da[da["local_time"].dt.date == DAY - dt.timedelta(days=1)]["price_usd_per_mwh"]
    np.testing.assert_allclose(persistence_da_forecast(ctx), yesterday.to_numpy())


def test_persistence_forecast_on_dst_days():
    rt = make_prices(start="2026-03-06", days=5)
    da = make_da_prices(start="2026-03-06", days=5)
    env = env2s(rt, da)
    def by_hour(day):
        one = da[da["local_time"].dt.date == day]
        return pd.Series(one["price_usd_per_mwh"].to_numpy(), index=one["local_time"].dt.hour.to_numpy())

    spring = persistence_da_forecast(env.day_ahead_context("2026-03-08"))  # 23 hours
    prev = by_hour(dt.date(2026, 3, 7))
    np.testing.assert_allclose(spring, prev.drop(index=2).to_numpy())
    after = persistence_da_forecast(env.day_ahead_context("2026-03-09"))  # day before had no 2:00
    prev = by_hour(dt.date(2026, 3, 8))
    assert after[2] == prev.loc[1]  # filled from the hour before
    assert len(after) == 24

    rt = make_prices(start="2026-10-30", days=4)
    da = make_da_prices(start="2026-10-30", days=4)
    fall = persistence_da_forecast(env2s(rt, da).day_ahead_context("2026-11-01"))
    assert len(fall) == 25 and fall[1] == fall[2]  # the repeated 1:00 hour


def test_persistence_forecast_needs_the_previous_day(rt, da):
    env = env2s(rt, da)
    ctx = env.day_ahead_context(DAY)
    empty = build_context(DAY, ctx.rt_history, ctx.da_history.iloc[:0], PARAMS)
    with pytest.raises(ValueError, match="no complete day-ahead prices"):
        persistence_da_forecast(empty)
    policy = DayAheadSchedule(PARAMS, env.obs_layout)
    assert np.all(policy.day_ahead(empty) == 0.0) and policy.n_days_without_forecast == 1


def test_mpc_on_day_ahead_prices_is_optimal_when_they_equal_real_time():
    da = make_da_prices(start="2026-03-01", days=5)
    rt = make_prices(start="2026-03-01", days=5)
    rt["price_usd_per_mwh"] = np.repeat(da["price_usd_per_mwh"].to_numpy(), 4)  # RT == DA
    env = env2s(rt, da, sample="sequential")
    mpc = run_policy(env, DayAheadForecastMPC(PARAMS, env.obs_layout))
    upper = run_policy(env, PerfectForesight(rt, PARAMS))
    assert mpc.revenue_usd == pytest.approx(upper.revenue_usd, rel=1e-6)


def test_day_ahead_lp_on_hourly_prices_matches_the_hand_example():
    # Charge 1 h at $10 (stores 100 * 0.9 = 90 MWh with rte 0.81), sell it at $100.
    p = BatteryParams(rte=0.81, degradation_cost_usd_per_mwh=0.0)
    plan = solve_dispatch(np.array([10.0, 100.0]), p, 0.0, 1.0)
    assert plan.power_mw == pytest.approx([-100.0, 81.0])
    award, n, _ = feasible_day_ahead(plan.power_mw, p, 0.0)
    assert n == 0 and award == pytest.approx(plan.power_mw)


def test_two_settlement_table_format(rt, da):
    env = env2s(rt, da, days=[DAY])
    upper = run_policy(env, PerfectForesightTwoSettlement(rt, da, PARAMS))
    table = markdown_table([metrics_row("test", upper, upper, PARAMS.energy_mwh)], two_settlement=True)
    assert "DA settlement" in table and "| test | Perfect foresight (DA + RT) | 1 |" in table
