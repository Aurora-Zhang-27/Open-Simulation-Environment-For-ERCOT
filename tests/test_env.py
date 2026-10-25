"""Environment: API, look-ahead protection, DST, gaps, reproducibility."""

import datetime as dt

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from conftest import make_prices
from rtcb_gym.battery import BatteryParams
from rtcb_gym.env import RTCBEnv
from rtcb_gym.settlement import settle_real_time

PARAMS = BatteryParams(power_mw=100, energy_mwh=200, rte=0.85, soc_init=0.5,
                       degradation_cost_usd_per_mwh=2.0)


def test_passes_gymnasium_checker(prices):
    env = RTCBEnv(prices, PARAMS)
    check_env(env, skip_render_check=True)


def test_spaces_and_layout(prices):
    env = RTCBEnv(prices, PARAMS, history_len=96)
    assert env.observation_space.shape == (106,)
    assert env.action_space.shape == (1,)
    obs, info = env.reset(options={"day": "2026-03-02"})
    assert obs.shape == (106,) and obs.dtype == np.float64
    assert obs[env.obs_layout["soc"]][0] == 0.5
    assert info["steps_remaining"] == 96
    # 2026-03-02 is a Monday; first interval starts at local midnight.
    assert obs[env.obs_layout["day_of_week"]].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert obs[env.obs_layout["time_of_day"]] == pytest.approx([0.0, 1.0])


def test_first_day_excluded_for_missing_history(prices):
    env = RTCBEnv(prices, PARAMS, history_len=96)
    assert dt.date(2026, 3, 1) not in env.days
    assert "history" in env.excluded_days["2026-03-01"]
    assert env.days[0] == dt.date(2026, 3, 2)


def test_no_look_ahead_window_is_strictly_past(prices):
    # Price of row i is i itself, so the visible window tells us which
    # intervals leaked into the observation.
    p = prices.copy()
    p["price_usd_per_mwh"] = np.arange(len(p), dtype=np.float64)
    env = RTCBEnv(p, PARAMS, history_len=96)
    obs, _ = env.reset(options={"day": "2026-03-03"})
    terminated = False
    while not terminated:
        window = obs[env.obs_layout["prices"]]
        obs, _, terminated, _, info = env.step(np.array([0.3]))
        t = info["price_usd_per_mwh"]  # index of the interval just settled
        assert window.max() < t, "a price at or after the dispatch interval leaked"
        assert window.tolist() == list(np.arange(t - 96, t))


def test_no_look_ahead_future_changes_do_not_change_past_observations(prices):
    # Two price paths that agree up to interval k and differ afterwards must
    # give identical observations and rewards until interval k is settled.
    a = prices.copy()
    b = prices.copy()
    env_a = RTCBEnv(a, PARAMS)
    start, end = env_a._episodes[dt.date(2026, 3, 4)]
    k = start + 40
    b.loc[k:, "price_usd_per_mwh"] = b.loc[k:, "price_usd_per_mwh"] * -3.0 + 500.0
    env_b = RTCBEnv(b, PARAMS)
    oa, _ = env_a.reset(options={"day": "2026-03-04"})
    ob, _ = env_b.reset(options={"day": "2026-03-04"})
    rng = np.random.default_rng(0)
    for t in range(start, k + 1):
        np.testing.assert_array_equal(oa, ob)  # obs before dispatching t
        action = rng.uniform(-1, 1, size=1)
        oa, ra, *_ = env_a.step(action)
        ob, rb, *_ = env_b.step(action)
        if t < k:
            assert ra == rb
    assert ra != rb  # interval k itself differs once settled


def test_reward_matches_settlement(prices):
    env = RTCBEnv(prices, PARAMS)
    env.reset(options={"day": "2026-03-05"})
    for a in [-1.0, -0.5, 0.0, 0.7, 1.0]:
        _, reward, _, _, info = env.step(np.array([a]))
        s = settle_real_time(info["actual_power_mw"], 0.25, info["price_usd_per_mwh"], 2.0)
        assert reward == s.net_revenue_usd == info["reward_usd"]
        assert info["energy_revenue_usd"] == s.energy_revenue_usd


def test_action_scaling_and_sign(prices):
    env = RTCBEnv(prices, PARAMS)
    env.reset(options={"day": "2026-03-05"})
    _, _, _, _, info = env.step(np.array([0.5]))
    assert info["actual_power_mw"] == 50.0  # positive = discharge
    assert info["soc"] < 0.5
    _, _, _, _, info = env.step(np.array([-0.5]))
    assert info["actual_power_mw"] == -50.0  # negative = charge


def test_out_of_bounds_action_is_clipped_and_counted(prices):
    env = RTCBEnv(prices, PARAMS)
    env.reset(options={"day": "2026-03-05"})
    _, _, _, _, info = env.step(np.array([2.0]))
    assert info["actual_power_mw"] == 100.0
    assert info["clipped"] and info["clip_reason"] == "action_bounds"


def test_soc_clip_is_counted():
    p = make_prices()
    env = RTCBEnv(p, BatteryParams(soc_init=0.0))
    env.reset(options={"day": "2026-03-05"})
    _, _, _, _, info = env.step(np.array([1.0]))  # discharge an empty battery
    assert info["actual_power_mw"] == 0.0
    assert info["clipped"] and info["clip_reason"] == "soc_min"


def test_episode_terminates_and_step_after_end_raises(prices):
    env = RTCBEnv(prices, PARAMS)
    env.reset(options={"day": "2026-03-05"})
    n = 0
    terminated = False
    while not terminated:
        _, _, terminated, truncated, info = env.step(np.array([0.0]))
        n += 1
        assert not truncated
    assert n == 96 and info["steps_remaining"] == 0
    with pytest.raises(RuntimeError):
        env.step(np.array([0.0]))


def test_step_before_reset_raises(prices):
    with pytest.raises(RuntimeError):
        RTCBEnv(prices, PARAMS).step(np.array([0.0]))


def _episode_length(env, day):
    env.reset(options={"day": day})
    n, terminated = 0, False
    while not terminated:
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        n += 1
    return n


def test_dst_spring_forward_day_has_92_intervals(prices):
    env = RTCBEnv(prices, PARAMS)  # range includes 2026-03-08
    assert _episode_length(env, "2026-03-08") == 92
    assert _episode_length(env, "2026-03-09") == 96


def test_dst_fall_back_day_has_100_intervals():
    env = RTCBEnv(make_prices(start="2026-10-30", days=4), PARAMS)
    assert _episode_length(env, "2026-11-01") == 100


def test_days_with_gaps_are_excluded(prices):
    p = prices.copy()
    day = p["local_time"].dt.date == dt.date(2026, 3, 5)
    drop = p.index[day][10]
    p = p.drop(index=drop)
    env = RTCBEnv(p, PARAMS)
    assert dt.date(2026, 3, 5) not in env.days
    assert "incomplete" in env.excluded_days["2026-03-05"]
    # The next day's 96-interval history crosses the gap, so it is out too.
    assert dt.date(2026, 3, 6) not in env.days
    assert dt.date(2026, 3, 7) in env.days
    with pytest.raises(ValueError):
        env.reset(options={"day": "2026-03-05"})


def test_days_argument_restricts_episodes(prices):
    env = RTCBEnv(prices, PARAMS, days=["2026-03-04", "2026-03-05"])
    assert env.days == [dt.date(2026, 3, 4), dt.date(2026, 3, 5)]


def test_multi_day_episode(prices):
    env = RTCBEnv(prices, PARAMS, episode_days=2)
    assert _episode_length(env, "2026-03-02") == 192
    # SoC carries over midnight inside a multi-day episode.
    env.reset(options={"day": "2026-03-02"})
    for _ in range(96):
        _, _, _, _, info = env.step(np.array([-1.0]))
    assert info["soc"] == 1.0 and info["steps_remaining"] == 96


def test_reset_restores_soc_init(prices):
    env = RTCBEnv(prices, PARAMS)
    env.reset(options={"day": "2026-03-05"})
    env.step(np.array([1.0]))
    obs, info = env.reset(options={"day": "2026-03-05"})
    assert obs[0] == 0.5 and info["soc"] == 0.5


def test_seeded_reset_is_reproducible(prices):
    def days(seed):
        env = RTCBEnv(prices, PARAMS)
        out = [env.reset(seed=seed)[1]["operating_day"]]
        out += [env.reset()[1]["operating_day"] for _ in range(20)]
        return out

    assert days(7) == days(7)
    assert days(7) != days(8)


def test_sequential_sampling(prices):
    env = RTCBEnv(prices, PARAMS, sample="sequential")
    seen = [env.reset()[1]["operating_day"] for _ in range(len(env.days))]
    assert seen == [str(d) for d in env.days]


def test_rejects_bad_inputs(prices):
    with pytest.raises(ValueError):
        RTCBEnv(prices, PARAMS, dt_hours=1.0)
    two = prices.copy()
    two.loc[0, "location"] = "HB_HOUSTON"
    with pytest.raises(ValueError):
        RTCBEnv(two, PARAMS)
    env = RTCBEnv(prices, PARAMS)
    env.reset(options={"day": "2026-03-05"})
    with pytest.raises(ValueError):
        env.step(np.array([np.nan]))


CROSS_DAY = BatteryParams(soc_init=0.5, soc_terminal_min=0.5)


def test_end_target_always_met_by_random_actions(prices):
    env = RTCBEnv(prices, CROSS_DAY)
    rng = np.random.default_rng(0)
    for day in env.days:
        env.reset(options={"day": day})
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(rng.uniform(-1, 1, size=1))
        assert info["soc"] >= 0.5 - 1e-9


def test_end_target_forced_charging_is_counted(prices):
    env = RTCBEnv(prices, CROSS_DAY)
    env.reset(options={"day": "2026-03-05"})
    reasons, terminated = [], False
    while not terminated:  # always ask to discharge at full power
        _, _, terminated, _, info = env.step(np.array([1.0]))
        reasons.append(info["clip_reason"])
    assert "soc_terminal" in reasons
    assert info["soc"] == pytest.approx(0.5)
    assert info["actual_power_mw"] < 0  # the last steps were forced to charge


def test_end_target_applies_at_episode_end_only(prices):
    env = RTCBEnv(prices, CROSS_DAY, episode_days=2)
    env.reset(options={"day": "2026-03-02"})
    socs = []
    for _ in range(96):  # day 1: discharge all the way, allowed
        _, _, _, _, info = env.step(np.array([1.0]))
        socs.append(info["soc"])
    assert min(socs) == 0.0


def test_unreachable_end_target_rejected(prices):
    tiny = BatteryParams(power_mw=1.0, energy_mwh=200.0, soc_init=0.0, soc_terminal_min=1.0)
    with pytest.raises(ValueError, match="cannot be reached"):
        RTCBEnv(prices, tiny)
