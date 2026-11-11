"""The options added to RTCBEnv in v0.4: five-minute dispatch with 15-minute
settlement, ECRS (real time, day-ahead, deployment), offer-curve actions,
bids with limit prices and two-settlement episodes of several days.

All prices are synthetic (see conftest)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rtcb_gym import BatteryParams, RTCBEnv
from rtcb_gym.ancillary import ECRSRules, solve_reservation
from rtcb_gym.baselines import (
    DayAheadEnergyECRS, ECRSMPC, ForecastMPC, LevelCorrectedMPC, OfferMPC, PerfectForesight,
    PerfectForesightMarket, PerfectForesightTwoSettlement, RollingThresholdOffer, env_action,
    fit_level_correction, solve_dispatch,
)
from rtcb_gym.day_ahead import DayAheadBid, clear_bid, feasible_day_ahead, feasible_day_ahead_with_ecrs
from rtcb_gym.env import OFFER_CAP, OFFER_FLOOR
from rtcb_gym.evaluation import bootstrap_ci, run_policy, top_days_share
from rtcb_gym.experimental.ecrs import ECRSReservationEnv, FiveMinuteTape
from tests.conftest import make_da_prices, make_prices

DAY = dt.date(2026, 3, 4)
PARAMS = BatteryParams()
HALF = BatteryParams(soc_init=0.5, soc_terminal_min=0.5)
FIVE = pd.Timedelta(minutes=5)


def five_minute(rt: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """Five-minute prices whose 15-minute means are exactly ``rt``'s prices."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 8, (len(rt), 3))
    noise -= noise.mean(axis=1, keepdims=True)
    start = pd.DatetimeIndex(rt["interval_start_utc"]).as_unit("ns").asi8
    ts = pd.DatetimeIndex((start[:, None] + np.arange(3) * FIVE.value).ravel(), tz="UTC")
    return pd.DataFrame({"interval_start_utc": ts, "location": "HB_WEST",
                         "price_usd_per_mwh": (rt["price_usd_per_mwh"].to_numpy()[:, None] + noise).ravel()})


def series(index: pd.DatetimeIndex, seed: int, lo: float = 0.0, hi: float = 12.0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).uniform(lo, hi, len(index)), index=index)


@pytest.fixture(scope="module")
def rt() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="module")
def da() -> pd.DataFrame:
    return make_da_prices()


@pytest.fixture(scope="module")
def rt5(rt) -> pd.DataFrame:
    return five_minute(rt)


def ecrs_env(rt5, da=None, params=HALF, deployment=False, da_ecrs=False, **kw) -> RTCBEnv:
    ts = pd.DatetimeIndex(rt5["interval_start_utc"])
    extra = {}
    if deployment:
        k = np.random.default_rng(5).uniform(0, 1, len(ts)) * (np.random.default_rng(6).uniform(size=len(ts)) < 0.3)
        extra["ecrs_deployment"] = pd.Series(k, index=ts)
    if da_ecrs:
        extra["da_ecrs_prices"] = series(pd.DatetimeIndex(da["interval_start_utc"]), 8)
    kw.setdefault("history_len", 288)
    return RTCBEnv(rt5, params, dt_hours=1 / 12, settlement_minutes=15, sample="sequential",
                   ecrs_prices=series(ts, 7), da_prices=da, **extra, **kw)


def random_run(env: RTCBEnv, day=DAY, seed=0, **options) -> tuple[float, dict]:
    obs, info = env.reset(options={"day": day, **options})
    rng = np.random.default_rng(seed)
    total, done = 0.0, False
    while not done:
        obs, reward, done, _, info = env.step(rng.uniform(env.action_space.low, env.action_space.high))
        total += reward
    return total, info


# ------------------------------------------------- five-minute dispatch
def test_default_env_is_unchanged_and_other_steps_are_refused(rt):
    env = RTCBEnv(rt, PARAMS)
    assert env.action_space.shape == (1,) and env.observation_space.shape == (106,)
    assert env.action_layout == {"energy": 0} and not env.ecrs
    with pytest.raises(ValueError, match="dt_hours"):
        RTCBEnv(rt, PARAMS, dt_hours=0.5)
    with pytest.raises(ValueError, match="settlement_minutes"):
        RTCBEnv(rt, PARAMS, settlement_minutes=5)
    with pytest.raises(ValueError, match="need ecrs_prices"):
        RTCBEnv(rt, PARAMS, da_ecrs_prices=pd.Series(dtype=float))


def test_five_minute_dispatch_with_15_minute_settlement_matches_the_15_minute_env(rt, rt5):
    slow = RTCBEnv(rt, PARAMS)
    fast = RTCBEnv(rt5, PARAMS, dt_hours=1 / 12, settlement_minutes=15, history_len=288)
    assert slow.days == fast.days
    slow.reset(options={"day": DAY})
    _, info = fast.reset(options={"day": DAY})
    assert info["steps_remaining"] == 288
    rng = np.random.default_rng(1)
    for _ in range(96):
        a = rng.uniform(-1, 1, 1)
        _, r, *_, i15 = slow.step(a)
        total = 0.0
        for _ in range(3):  # the same power held for the 15 minutes
            _, r5, *_, i5 = fast.step(a)
            total += r5
        assert total == pytest.approx(r, abs=1e-6)
        assert i5["soc"] == pytest.approx(i15["soc"], abs=1e-9)
        assert i5["price_usd_per_mwh"] == pytest.approx(i15["price_usd_per_mwh"])
        assert i5["dispatch_price_usd_per_mwh"] != i5["price_usd_per_mwh"]


def test_incomplete_settlement_interval_excludes_the_day(rt5):
    local = pd.DatetimeIndex(rt5["interval_start_utc"]).tz_convert("US/Central")
    hole = rt5.drop(index=np.flatnonzero(local.date == DAY)[100])
    env = RTCBEnv(hole, PARAMS, dt_hours=1 / 12, settlement_minutes=15, history_len=288)
    assert DAY not in env.days and "incomplete day" in env.excluded_days[str(DAY)]


def test_obs_lag_hides_the_newest_prices(rt):
    env = RTCBEnv(rt, PARAMS, obs_lag_steps=2)
    obs, _ = env.reset(options={"day": DAY})
    t = env._t
    np.testing.assert_array_equal(obs[env.obs_layout["prices"]], env._price[t - 98 : t - 2])


# ------------------------------------------------------------------ ECRS
def test_ecrs_env_passes_the_gymnasium_checker(rt5, da):
    check_env(ecrs_env(rt5, da, da_ecrs=True, deployment=True), skip_render_check=True)
    check_env(ecrs_env(rt5, action_mode="offer"), skip_render_check=True)


def test_ecrs_matches_the_gate_model_step_for_step(rt5):
    """Same actions, same rule: the unified env and the gate env agree."""
    ts = pd.DatetimeIndex(rt5["interval_start_utc"])
    ecrs = series(ts, 7)
    env = RTCBEnv(rt5, HALF, dt_hours=1 / 12, history_len=288, ecrs_prices=ecrs)  # settles each step
    _, info = env.reset(options={"day": DAY})
    i0 = env._t
    frame = pd.DataFrame({"interval_start_utc": ts, "available_at_utc": ts + FIVE,
                          "energy_price_usd_per_mwh": rt5["price_usd_per_mwh"].to_numpy(),
                          "ecrs_price_usd_per_mw_h": ecrs.to_numpy()})
    gate = ECRSReservationEnv(FiveMinuteTape(frame, {"price_basis": "synthetic"}), HALF,
                              start=i0, stop=i0 + 288, history_len=288)
    gobs, _ = gate.reset()
    obs = env._obs()
    np.testing.assert_array_equal(obs[env.obs_layout["prices"]], gobs["prices"][:, 0])
    np.testing.assert_array_equal(obs[env.obs_layout["ecrs_prices"]], gobs["prices"][:, 1])
    rng = np.random.default_rng(2)
    for _ in range(288):
        a = np.array([rng.uniform(-1, 1), rng.uniform(0, 1)])
        _, r, done, _, info = env.step(a)
        _, gr, gdone, _, ginfo = gate.step(a)
        assert r == pytest.approx(gr, abs=1e-9) and done == gdone
        assert info["ecrs_award_mw"] == pytest.approx(ginfo["actual_mw"][1])
        assert info["soc"] == pytest.approx(ginfo["soc"])


def test_ecrs_observation_and_action_do_not_see_the_current_interval(rt5):
    env = ecrs_env(rt5)
    obs, _ = env.reset(options={"day": DAY})
    t = env._t
    other = ecrs_env(rt5)
    for name, shift in (("_price", 1000.0), ("_settle", 1000.0), ("_ecrs_price", 50.0), ("_ecrs_settle", 50.0)):
        changed = getattr(other, name).copy()
        changed[t:] += shift
        setattr(other, name, changed)
    obs2, _ = other.reset(options={"day": DAY})
    np.testing.assert_array_equal(obs, obs2)


@pytest.mark.parametrize("opts", [
    dict(), dict(deployment=True), dict(two=True), dict(two=True, da_ecrs=True),
    dict(two=True, da_ecrs=True, deployment=True, action_mode="offer"),
    dict(two=True, da_ecrs=True, deployment=True, params=BatteryParams(soc_init=0.2, soc_terminal_min=0.6)),
])
def test_perfect_foresight_market_replays_its_objective_and_bounds_random_play(rt5, da, opts):
    opts = dict(opts)
    env = ecrs_env(rt5, da if opts.pop("two", False) else None, **opts)
    pf = PerfectForesightMarket(env)
    upper = run_policy(env, pf, [DAY])
    assert upper.n_clipped == 0 and upper.n_ecrs_clipped == 0 and upper.n_da_clipped_hours == 0
    assert upper.revenue_usd == pytest.approx(pf.planned_revenue_usd + pf.planned_da_usd, rel=1e-7, abs=1e-5)
    assert upper.revenue_usd == pytest.approx(
        upper.energy_revenue_usd + upper.ecrs_revenue_usd - upper.degradation_cost_usd)
    target = env.battery_params.soc_terminal_min
    for seed in range(3):
        options = {}
        if env.two_settlement:
            rng = np.random.default_rng(seed)
            options["da_schedule"] = DayAheadBid(
                energy_mw=rng.uniform(-100, 100, 24), ecrs_mw=rng.uniform(0, 100, 24) if env.da_ecrs else None)
        revenue, info = random_run(env, seed=seed, **options)
        assert revenue <= upper.revenue_usd + 1e-6
        assert info["soc"] >= target - 1e-9  # deployment never breaks the end target


def test_deployment_delivers_energy_and_drains_the_battery(rt5):
    ts = pd.DatetimeIndex(rt5["interval_start_utc"])
    env = RTCBEnv(rt5, HALF, dt_hours=1 / 12, history_len=288, ecrs_prices=series(ts, 7),
                  ecrs_deployment=pd.Series(0.5, index=ts))
    env.reset(options={"day": DAY})
    _, reward, *_, info = env.step(np.array([0.0, 1.0]))
    award = info["ecrs_award_mw"]
    assert award == pytest.approx(min(100.0, 0.5 * 200 * HALF.eta_discharge))  # one hour of the stored energy
    assert info["ecrs_deployed_mw"] == pytest.approx(0.5 * award)
    assert info["actual_power_mw"] == 0.0 and info["delivered_power_mw"] == pytest.approx(0.5 * award)
    assert info["soc"] == pytest.approx(0.5 - 0.5 * award / 12 / HALF.eta_discharge / 200)
    energy = 0.5 * award / 12
    assert info["energy_revenue_usd"] == pytest.approx(energy * info["price_usd_per_mwh"])
    assert info["degradation_cost_usd"] == pytest.approx(energy * 2.0)
    assert reward == pytest.approx(info["energy_revenue_usd"] - info["degradation_cost_usd"]
                                   + award / 12 * info["ecrs_price_usd_per_mw_h"])


def test_day_ahead_ecrs_award_settles_the_deviation_at_the_real_time_price(rt5, da):
    env = ecrs_env(rt5, da, da_ecrs=True)
    bid = DayAheadBid(energy_mw=np.zeros(24), ecrs_mw=np.full(24, 40.0))
    _, info = env.reset(options={"day": DAY, "da_schedule": bid})
    np.testing.assert_allclose(info["da_ecrs_award_mw"], 40.0)
    _, reward, *_, info = env.step(np.array([0.0, 0.1]))  # 10 MW in real time against a 40 MW award
    assert info["ecrs_da_revenue_usd"] == pytest.approx(40 / 12 * info["da_ecrs_price_usd_per_mw_h"])
    assert info["ecrs_rt_revenue_usd"] == pytest.approx((10 - 40) / 12 * info["ecrs_price_usd_per_mw_h"])
    assert reward == pytest.approx(info["ecrs_da_revenue_usd"] + info["ecrs_rt_revenue_usd"])


def test_day_ahead_ecrs_awards_are_clipped_to_the_reservation_rule():
    award, ecrs, n, soc = feasible_day_ahead_with_ecrs(np.zeros(4), np.full(4, 100.0), PARAMS, ECRSRules(), 0.0)
    assert np.all(ecrs == 0) and n == 4  # an empty battery holds no reserve
    award, ecrs, n, _ = feasible_day_ahead_with_ecrs(np.full(2, -100.0), np.full(2, 50.0), PARAMS, ECRSRules(), 0.0)
    assert np.all(ecrs == 0)  # none while charging
    sched = np.random.default_rng(0).uniform(-100, 100, 24)
    a, _, n, soc = feasible_day_ahead_with_ecrs(sched, np.zeros(24), HALF, ECRSRules(), 0.5)
    b, m, soc_b = feasible_day_ahead(sched, HALF, 0.5)
    np.testing.assert_array_equal(a, b)
    assert n == m


def test_ecrs_baselines_run_and_stay_below_the_bound(rt5, da):
    env = ecrs_env(rt5, da, da_ecrs=True)
    upper = run_policy(env, PerfectForesightMarket(env), [DAY])
    for policy in (ECRSMPC(env), ECRSMPC(env, energy_only=True), DayAheadEnergyECRS(env)):
        r = run_policy(env, policy, [DAY])
        assert r.revenue_usd <= upper.revenue_usd + 1e-6
        assert r.n_clipped == 0
    assert run_policy(env, ECRSMPC(env, energy_only=True), [DAY]).ecrs_mw_hours == 0


# ---------------------------------------------------------- offer curves
def test_offer_curve_is_applied_to_the_interval_price(rt):
    env = RTCBEnv(rt, HALF, action_mode="offer")
    assert env.action_space.shape == (4,)
    env.reset(options={"day": DAY})
    price = env._price[env._t]
    _, _, _, _, info = env.step(np.array([1.0, price + 1, 1.0, price - 1]))  # between the two prices
    assert info["actual_power_mw"] == 0.0 and not info["clipped"]
    price = env._price[env._t]
    _, _, _, _, info = env.step(np.array([0.5, price, 1.0, price - 1]))  # at the sell price
    assert info["actual_power_mw"] == pytest.approx(50.0)
    price = env._price[env._t]
    _, _, _, _, info = env.step(np.array([1.0, price + 1, 0.25, price]))  # at the buy price
    assert info["actual_power_mw"] == pytest.approx(-25.0)
    price = env._price[env._t]
    _, _, _, _, info = env.step(np.array([1.0, price - 5, 1.0, price + 5]))  # crossed: buy price lowered
    assert info["actual_power_mw"] == pytest.approx(100.0) and info["clip_reason"] == "action_bounds"


def test_offer_action_cannot_see_the_price_only_the_market_applies_it(rt):
    a, b = RTCBEnv(rt, HALF, action_mode="offer"), RTCBEnv(rt, HALF, action_mode="offer")
    oa, _ = a.reset(options={"day": DAY})
    oa, _ = a.reset(options={"day": DAY})
    t = a._t
    b._price = b._price.copy()
    b._price[t] += 500.0
    b._settle = b._price
    ob, _ = b.reset(options={"day": DAY})
    np.testing.assert_array_equal(oa, ob)
    curve = np.array([1.0, a._price[t] + 100, 1.0, OFFER_FLOOR])
    assert a.step(curve)[4]["actual_power_mw"] == 0.0
    assert b.step(curve)[4]["actual_power_mw"] == pytest.approx(100.0)


def test_price_taking_offers_reproduce_the_power_action(rt):
    power, offers = RTCBEnv(rt, HALF), RTCBEnv(rt, HALF, action_mode="offer")
    power.reset(options={"day": DAY})
    offers.reset(options={"day": DAY})
    rng = np.random.default_rng(0)
    for _ in range(96):
        x = rng.uniform(-1, 1)
        _, r, *_ = power.step(np.array([x]))
        _, ro, *_ = offers.step(env_action(offers, x))
        assert ro == pytest.approx(r, abs=1e-9)


def test_offer_baselines_stay_below_the_bound_and_inside_the_limits(rt):
    env = RTCBEnv(rt, HALF, action_mode="offer", sample="sequential")
    days = env.days[:3]
    upper = run_policy(env, PerfectForesightMarket(env), days)
    assert upper.revenue_usd == pytest.approx(run_policy(RTCBEnv(rt, HALF), PerfectForesight(rt, HALF), days).revenue_usd)
    rolling = RollingThresholdOffer(rt, 0.2, 0.8, HALF, env.obs_layout, window_days=3, min_window_days=1)
    for policy in (OfferMPC(HALF, env.obs_layout), rolling):
        r = run_policy(env, policy, days)
        assert r.n_clipped == 0
        assert 0 < r.revenue_usd <= upper.revenue_usd + 1e-6


# ------------------------------------------------- bids with limit prices
def test_limit_prices_clear_against_the_day_ahead_price():
    price = np.array([10.0, 20.0, 30.0, 40.0])
    bid = DayAheadBid(energy_mw=np.array([50.0, 50.0, -50.0, -50.0]),
                      energy_limit_price=np.array([15.0, 15.0, 35.0, 35.0]))
    offered, cleared, _, _ = clear_bid(bid, price)
    np.testing.assert_array_equal(offered, [50, 50, -50, -50])
    np.testing.assert_array_equal(cleared, [0, 50, -50, 0])
    # two segments an hour: 30 MW at any price above 5, 20 MW more above 25
    curve = DayAheadBid(energy_mw=np.tile([30.0, 20.0], (4, 1)), energy_limit_price=np.tile([5.0, 25.0], (4, 1)))
    np.testing.assert_array_equal(clear_bid(curve, price)[1], [30, 30, 50, 50])
    np.testing.assert_array_equal(clear_bid(np.array([1.0, 2, 3, 4]), price)[1], [1, 2, 3, 4])
    np.testing.assert_array_equal(clear_bid(None, price)[1], 0)
    ecrs = DayAheadBid(ecrs_mw=np.full(4, 10.0), ecrs_limit_price=np.full(4, 3.0))
    np.testing.assert_array_equal(clear_bid(ecrs, price, np.array([1.0, 3.0, 5.0, 2.0]))[3], [0, 10, 10, 0])
    with pytest.raises(ValueError, match="day-ahead ECRS prices"):
        clear_bid(ecrs, price)
    with pytest.raises(ValueError, match=">= 0"):
        clear_bid(DayAheadBid(ecrs_mw=np.full(4, -1.0)), price, price)


def test_env_awards_only_what_cleared(rt, da):
    env = RTCBEnv(rt, BatteryParams(soc_init=1.0), da_prices=da)
    ctx_price = env._da_price_rt[slice(*env._day_rows[DAY])][::4]
    limit = np.median(ctx_price)
    bid = DayAheadBid(energy_mw=np.full(24, 10.0), energy_limit_price=np.full(24, limit))
    _, info = env.reset(options={"day": DAY, "da_schedule": bid})
    np.testing.assert_array_equal(info["da_requested_mw"], 10.0)
    np.testing.assert_array_equal(info["da_cleared_mw"], np.where(ctx_price >= limit, 10.0, 0.0))
    assert 0 < (info["da_cleared_mw"] > 0).sum() < 24
    np.testing.assert_array_equal(info["da_award_mw"], info["da_cleared_mw"])  # all deliverable from full


# ------------------------------------------ two-settlement, several days
def test_multi_day_two_settlement_splits_into_spread_and_real_time(rt, da):
    env = RTCBEnv(rt, HALF, da_prices=da, episode_days=3)
    span = env.episode_operating_days(DAY)
    assert span == [DAY + dt.timedelta(days=k) for k in range(3)]
    rng = np.random.default_rng(4)
    bids = [rng.uniform(-100, 100, 24) for _ in span]
    with pytest.raises(ValueError, match="one bid per operating day"):
        env.reset(options={"day": DAY, "da_schedule": bids[:2]})
    _, info = env.reset(options={"day": DAY, "da_schedule": bids})
    assert len(info["da_award_mw"]) == 72 and info["steps_remaining"] == 288
    for k, bid in enumerate(bids):  # every day is checked from soc_init, as a one-day episode is
        np.testing.assert_array_equal(info["da_award_mw"][24 * k : 24 * k + 24], feasible_day_ahead(bid, HALF, 0.5)[0])
    award = np.repeat(info["da_award_mw"], 4)
    i0 = env._t
    rtp, dap = env._price[i0 : i0 + 288], env._da_price_rt[i0 : i0 + 288]
    total, power, done = 0.0, [], False
    while not done:
        _, r, done, _, step = env.step(rng.uniform(-1, 1, 1))
        total += r
        power.append(step["actual_power_mw"])
    power = np.array(power)
    expected = (award * (dap - rtp) * 0.25).sum() + (power * rtp * 0.25).sum() - np.abs(power).sum() * 0.25 * 2.0
    assert total == pytest.approx(expected, abs=1e-6)
    assert step["soc"] >= 0.5 - 1e-9


def test_next_days_day_ahead_results_appear_at_publication_time_only(rt, da):
    env = RTCBEnv(rt, HALF, da_prices=da, episode_days=2)
    obs, _ = env.reset(options={"day": DAY})
    lay, i0 = env.obs_layout, env._t
    np.testing.assert_array_equal(obs[lay["da_ahead_mask"]], 1.0)  # all of today, published yesterday
    for step in range(96):
        obs, *_ = env.step(np.zeros(1))
        local = env._ts[env._t].tz_convert("US/Central")
        left_today = 96 - (step + 1)
        visible = int(obs[lay["da_ahead_mask"]].sum())
        if local.date() == DAY and (local.hour, local.minute) < (13, 30):
            assert visible == left_today  # tomorrow is not published yet
        else:
            assert visible == 96
    # the day-ahead prices of tomorrow in the window are tomorrow's true ones
    np.testing.assert_array_equal(obs[lay["da_price_ahead"]], env._da_price_rt[i0 + 96 : i0 + 192])
    other = RTCBEnv(rt, HALF, da_prices=da, episode_days=2)
    other._da_price_rt = other._da_price_rt.copy()
    other._da_price_rt[i0 + 96 :] += 100.0
    oo, _ = other.reset(options={"day": DAY})
    o1, _ = env.reset(options={"day": DAY})
    np.testing.assert_array_equal(oo, o1)


def test_multi_day_perfect_foresight_is_exact_and_beats_daily_resets(rt, da):
    env = RTCBEnv(rt, HALF, da_prices=da, episode_days=3, sample="sequential")
    pf = PerfectForesightTwoSettlement(rt, da, HALF)
    upper = run_policy(env, pf, [DAY])
    assert upper.n_days == 3 and len(upper.daily_revenue_usd) == 3
    assert upper.n_clipped == 0 and upper.n_da_clipped_hours == 0
    assert upper.revenue_usd == pytest.approx(pf.planned_da_usd + pf.planned_revenue_usd, rel=1e-6)
    market = run_policy(env, PerfectForesightMarket(env), [DAY])
    assert market.revenue_usd == pytest.approx(upper.revenue_usd, rel=1e-6)
    daily = RTCBEnv(rt, HALF, da_prices=da, sample="sequential")
    by_day = run_policy(daily, PerfectForesightTwoSettlement(rt, da, HALF), env.episode_operating_days(DAY))
    assert upper.revenue_usd >= by_day.revenue_usd - 1e-6  # carrying energy overnight cannot hurt
    mpc = run_policy(env, ForecastMPC(HALF, env.obs_layout), [DAY])
    assert mpc.revenue_usd <= upper.revenue_usd


# --------------------------------------------------- solver and metrics
def test_long_horizon_solver_agrees_with_the_dense_one():
    rng = np.random.default_rng(0)
    prices = rng.normal(20, 40, 180)
    prices[::17] = -80.0  # negative enough to need the binaries
    for params in (PARAMS, HALF):
        dense = solve_dispatch(prices, params, params.soc_init, 0.25)
        long = solve_dispatch(np.tile(prices, 2), params, params.soc_init, 0.25)
        assert long.revenue_usd >= dense.revenue_usd - 1e-6
        from rtcb_gym.baselines import optimize
        sparse = optimize._solve_long(prices, params, params.soc_init * 200, 0.25, params.soc_terminal_min)
        assert sparse.revenue_usd == pytest.approx(dense.revenue_usd, rel=1e-9, abs=1e-6)
        assert sparse.used_milp == dense.used_milp


def test_reservation_oracle_with_deployment_matches_a_hand_case():
    # 10 MW / 20 MWh, full, no losses: 10 MW ECRS for one step, fully deployed, then sell the rest
    p = BatteryParams(power_mw=10, energy_mwh=20, rte=1.0, soc_init=1.0, degradation_cost_usd_per_mwh=0.0)
    prices = np.array([[50.0, 12.0], [100.0, 0.0]])
    plan = solve_reservation(prices, p, ECRSRules(), 1.0, dt_hours=1.0, deployment=np.array([1.0, 0.0]))
    np.testing.assert_allclose(plan.actions_mw, [[0, 10], [10, 0]], atol=1e-7)
    assert plan.net_proxy_usd == pytest.approx(10 * 12 + 10 * 50 + 10 * 100)
    np.testing.assert_allclose(plan.soc_after, [0.5, 0.0], atol=1e-9)


def test_level_corrected_forecast_uses_only_seen_prices_and_reduces_to_persistence(rt):
    env = RTCBEnv(rt, HALF, sample="sequential")
    plain = run_policy(env, ForecastMPC(HALF, env.obs_layout), [DAY])
    same = run_policy(env, LevelCorrectedMPC(HALF, env.obs_layout, rho=0.0), [DAY])
    assert same.revenue_usd == pytest.approx(plain.revenue_usd)
    k, rho, table = fit_level_correction(env, ks=(4,), rhos=(0.0, 0.9))
    assert k == 4 and rho in (0.0, 0.9) and len(table) == 2
    policy = LevelCorrectedMPC(HALF, env.obs_layout, k=4, rho=0.9)
    obs, info = env.reset(options={"day": DAY})
    policy.reset(obs, info)
    for _ in range(10):
        obs, _, _, _, info = env.step(policy.act(obs, info))
    t = env._t
    np.testing.assert_array_equal(policy._seen, env._price[t - 106 : t - 1])  # nothing from t-1 on
    assert len(policy._seen) == 96 + 9


def test_bootstrap_interval_and_top_days():
    daily = np.r_[np.full(20, 100.0), 2000.0]
    lo, hi = bootstrap_ci(daily)
    assert lo <= daily.sum() <= hi and lo < hi
    assert bootstrap_ci(daily) == bootstrap_ci(daily)  # seeded
    assert bootstrap_ci(daily, daily) == (1.0, 1.0)
    assert top_days_share(daily, 1) == pytest.approx(0.5)
    assert np.isnan(bootstrap_ci([])[0])


# ------------------------------------------------ packaging conveniences
def test_registered_with_gymnasium_and_version_matches_the_package_metadata(rt):
    import re
    from pathlib import Path

    import gymnasium as gym

    import rtcb_gym

    env = gym.make(rtcb_gym.ENV_ID, prices=rt, battery=HALF)
    obs, _ = env.reset(seed=0)
    assert isinstance(env.unwrapped, RTCBEnv) and obs.shape == (106,)
    pyproject = (Path(rtcb_gym.__file__).parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "(.+)"$', pyproject, re.M).group(1) == rtcb_gym.__version__


def test_scaled_observation_bounds_prices_and_keeps_everything_else(rt, da):
    from rtcb_gym import ScaledObservation

    raw = RTCBEnv(rt, HALF, da_prices=da)
    env = ScaledObservation(RTCBEnv(rt, HALF, da_prices=da), price_scale=25.0)
    check_env(env, skip_render_check=True)
    assert np.isfinite(env.observation_space.low).all() and np.isfinite(env.observation_space.high).all()
    o_raw, _ = raw.reset(options={"day": DAY})
    o, _ = env.reset(options={"day": DAY})
    lay = raw.obs_layout
    assert o.dtype == np.float32 and env.observation_space.contains(o)
    p = o_raw[lay["prices"]]
    np.testing.assert_allclose(o[lay["prices"]], np.sign(p) * np.log1p(np.abs(p) / 25.0), rtol=1e-6)
    for name in ("soc", "time_of_day", "day_of_week", "da_award_ahead", "da_ahead_mask"):
        np.testing.assert_allclose(o[lay[name]], o_raw[lay[name]], rtol=1e-6)
    _, r, *_ = env.step(np.array([0.3]))
    _, r_raw, *_ = raw.step(np.array([0.3]))
    assert r == r_raw  # rewards are not rescaled
