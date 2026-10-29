"""Research-gate invariants: causal information, joint feasibility, oracle replay."""
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rtcb_gym.battery import BatteryParams, terminal_floor
from rtcb_gym.experimental.ecrs import (
    DT_HOURS, STEP, ECRSRules, ECRSReservationEnv, FiveMinuteTape,
    ecrs_capacity_mw, reservation_step, reservation_revenue, solve_reservation,
)


def make_tape(n=72, prices=None, lag=STEP):
    starts = pd.date_range("2026-03-05", periods=n, freq=STEP, tz="UTC")
    if prices is None:
        rng = np.random.default_rng(11)
        prices = np.column_stack([rng.normal(30, 50, n), rng.uniform(0, 8, n)])
    frame = pd.DataFrame({"interval_start_utc": starts,
                          "available_at_utc": starts + STEP + lag,
                          "energy_price_usd_per_mwh": prices[:, 0],
                          "ecrs_price_usd_per_mw_h": prices[:, 1]})
    return FiveMinuteTape(frame, {"price_basis": "synthetic"}), frame


def test_gym_contract_and_seeded_replay():
    tape, _ = make_tape()
    check_env(ECRSReservationEnv(tape, BatteryParams(), start=24, history_len=12), skip_render_check=True)


def test_publication_delay_and_future_perturbation():
    tape, frame = make_tape()
    env = ECRSReservationEnv(tape, BatteryParams(), start=24, history_len=12)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs["prices"], tape.prices[11:23])
    assert obs["price_age_steps"][0] == 2
    # Index 23 ended at the decision, but is not published for five more minutes.
    changed = frame.copy()
    changed.loc[23:, "energy_price_usd_per_mwh"] += 10000
    changed.loc[23:, "ecrs_price_usd_per_mw_h"] += 2000
    other = ECRSReservationEnv(FiveMinuteTape(changed, {"price_basis":"synthetic"}),
                               BatteryParams(), start=24, history_len=12)
    obs2, _ = other.reset()
    for key in obs:
        np.testing.assert_array_equal(obs[key], obs2[key])
    env.step(np.zeros(2))
    assert not env.tape.prices.flags.writeable


@pytest.mark.parametrize("bad", ["gap", "duplicate", "nan", "early", "naive"])
def test_bad_tapes_rejected(bad):
    _, f = make_tape()
    if bad == "gap": f = f.drop(5)
    if bad == "duplicate": f.loc[5, "interval_start_utc"] = f.loc[4,"interval_start_utc"]
    if bad == "nan": f.loc[5,"ecrs_price_usd_per_mw_h"] = np.nan
    if bad == "early": f["available_at_utc"] = f["interval_start_utc"]
    if bad == "naive": f["interval_start_utc"] = f["interval_start_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError): FiveMinuteTape(f, {"price_basis":"synthetic"})


def test_inventory_power_and_charging_boundaries():
    p = BatteryParams(power_mw=10,energy_mwh=20,soc_min=.1,soc_init=.2)
    assert ecrs_capacity_mw(p,ECRSRules(),0,.1) == 0
    assert ecrs_capacity_mw(p,ECRSRules(),-1,.9) == 0
    assert ecrs_capacity_mw(p,ECRSRules(),10,.9) == 0
    one = ecrs_capacity_mw(p,ECRSRules(),0,.2)
    quarter = ecrs_capacity_mw(p,ECRSRules(.25),0,.2)
    assert one == pytest.approx(2*p.eta_discharge)
    assert quarter == pytest.approx(4*one)
    assert ECRSRules(.25).rule_id.startswith("sensitivity")


def test_settlement_uses_executed_energy_and_award():
    p = BatteryParams(power_mw=10,energy_mwh=20,soc_init=0)
    executed, soc, info = reservation_step(p,ECRSRules(),0,np.array([10.,10.]),12)
    assert np.all(executed == 0) and info["ecrs_clipped"]
    assert reservation_revenue(executed,np.array([1000.,1000.]),p)["net_proxy_usd"] == 0
    ledger = reservation_revenue(np.array([-5.,0.]),np.array([-50.,1.]),p)
    assert ledger["net_proxy_usd"] == pytest.approx((250-10)*DT_HOURS)


@pytest.mark.parametrize("duration", [.25,1.,2.])
@pytest.mark.parametrize("end", [0.,.5])
def test_oracle_matches_execution_and_dominates_random(duration,end):
    tape, _ = make_tape(n=64)
    p=BatteryParams(power_mw=10,energy_mwh=20,soc_init=.5,soc_terminal_min=end)
    rules=ECRSRules(duration)
    prices=tape.prices[16:]
    plan=solve_reservation(prices,p,rules,p.soc_init)
    env=ECRSReservationEnv(tape,p,rules,start=16,history_len=12)
    env.reset(); total=0.
    for k,action in enumerate(plan.actions_mw):
        _,reward,done,_,info=env.step(action/p.power_mw)
        total+=reward
        assert not info["energy_clipped"] and not info["ecrs_clipped"]
        assert info["soc"] == pytest.approx(plan.soc_after[k],abs=1e-7)
    assert done and info["soc"] >= end-1e-7
    assert total == pytest.approx(plan.net_proxy_usd,rel=1e-8,abs=1e-7)
    rng=np.random.default_rng(4)
    for _ in range(3):
        env.reset(); random_total=0.
        for _ in prices:
            _,r,_,_,_=env.step(rng.uniform([-1,0],[1,1]))
            random_total+=r
        assert random_total <= total+1e-6


def test_full_battery_negative_price_cannot_burn_energy():
    p=BatteryParams(power_mw=10,energy_mwh=20,soc_init=1,degradation_cost_usd_per_mwh=0)
    plan=solve_reservation(np.array([[-100.,0.]]),p,ECRSRules(),1)
    assert plan.net_proxy_usd == pytest.approx(0)
    np.testing.assert_allclose(plan.actions_mw,0,atol=1e-8)


def test_tighter_duration_cannot_improve_same_model_oracle():
    tape,_=make_tape()
    p=BatteryParams(power_mw=10,energy_mwh=20,soc_init=.5,soc_terminal_min=.5)
    relaxed=solve_reservation(tape.prices,p,ECRSRules(.25),.5)
    canonical=solve_reservation(tape.prices,p,ECRSRules(),.5)
    assert canonical.net_proxy_usd <= relaxed.net_proxy_usd+1e-6


def test_terminal_charging_and_bad_actions():
    tape,_=make_tape()
    p=BatteryParams(soc_init=.5,soc_terminal_min=.8)
    env=ECRSReservationEnv(tape,p,start=16,history_len=8)
    env.reset()
    for _ in range(56): _,_,_,_,info=env.step(np.array([1.,1.]))
    assert info["soc"] >= .8-1e-9
    with pytest.raises(RuntimeError):env.step(np.zeros(2))
    env.reset()
    with pytest.raises(ValueError):env.step(np.array([np.nan,0.]))
    with pytest.raises(ValueError):env.step(np.ones(3))


def test_mpc_uses_clock_aligned_known_history(monkeypatch):
    from scripts import run_ecrs_gate as runner
    from types import SimpleNamespace
    captured=[]
    def record(prices,*args,**kwargs):
        captured.append(prices.copy())
        return SimpleNamespace(actions_mw=np.zeros((len(prices),2)))
    monkeypatch.setattr(runner,"solve_reservation",record)
    obs={"price_mask":np.ones(288), "price_age_steps":np.array([2.]),
         "prices":np.column_stack([np.arange(288),np.arange(288)]), "soc":np.array([.5])}
    runner.mpc_action(obs,{"steps_remaining":48},BatteryParams(),ECRSRules(),12)
    np.testing.assert_array_equal(captured[0][:,0],np.arange(1,13))
    runner.mpc_action(obs,{"steps_remaining":48},BatteryParams(),ECRSRules(),12,True)
    np.testing.assert_array_equal(captured[1][:,1],0)


def test_price_profile_preserves_missingness_and_rejects_join_mismatch(tmp_path):
    from scripts.run_ecrs_gate import profile_prices
    index=pd.date_range("2026-02-01",periods=12,freq="5min",tz="UTC")
    for kind,col in [("energy_prices","rt_lmp"),("as_prices","rt_mcpc_ecrs")]:
        (tmp_path/kind).mkdir()
        vals=np.arange(12,dtype=float);vals[3]=np.nan
        pd.DataFrame({col:vals},index=index).to_parquet(tmp_path/kind/"x.parquet")
    frame,quality=profile_prices(tmp_path)
    assert frame.isna().sum().to_dict()=={"rt_lmp":1,"rt_mcpc_ecrs":1}
    assert quality["join_rows"]==12 and quality["join_multiplier"]==1
    assert all(x["nulls"]==1 for x in quality["file_profiles"])
    pd.DataFrame({"rt_mcpc_ecrs":np.arange(11)},index=index[:-1]).to_parquet(tmp_path/"as_prices/x.parquet")
    with pytest.raises(ValueError,match="timestamps do not match"):
        profile_prices(tmp_path)


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_runner_accepts_complete_panel_in_any_time_unit(tmp_path, monkeypatch, unit):
    # pandas 3 keeps the file's time unit while date_range defaults to "us";
    # a complete panel must not be rejected as having missing timestamps.
    from pathlib import Path
    import json, sys
    from types import SimpleNamespace
    from scripts import run_ecrs_gate as runner
    index = pd.date_range("2026-02-01T05:50:00Z", "2026-02-03T06:00:00Z", freq="5min",
                          inclusive="left").as_unit(unit)
    for kind, col in [("energy_prices", "rt_lmp"), ("as_prices", "rt_mcpc_ecrs")]:
        (tmp_path/kind).mkdir()
        pd.DataFrame({col: np.linspace(1., 9., len(index))}, index=index).to_parquet(tmp_path/kind/"x.parquet")
    monkeypatch.setattr(runner, "solve_reservation",
                        lambda prices, *a, **k: SimpleNamespace(actions_mw=np.zeros((len(prices), 2))))
    monkeypatch.chdir(Path(runner.__file__).resolve().parents[1])
    out = tmp_path/"out"
    monkeypatch.setattr(sys, "argv", ["run_ecrs_gate.py", "--input-dir", str(tmp_path), "--out-dir", str(out),
                                      "--start", "2026-02-02", "--end", "2026-02-02", "--skip-oracle"])
    runner.main()
    quality = json.loads((out/"data_quality.json").read_text())
    assert quality["selected_panel"]["evaluated_intervals"] == 288
    assert quality["selected_panel"]["timestamp_gaps"] == 0


def test_tie_break_acts_least_and_latest_without_losing_value():
    # Four equal zero-price steps to charge in and two equal high-price steps
    # to sell in: many plans earn exactly the same.
    p = BatteryParams(power_mw=10, energy_mwh=1, soc_init=0)
    prices = np.array([[0., 0.]]*4 + [[100., 0.]]*2)
    exact = solve_reservation(prices, p, ECRSRules(), 0.)
    plan = solve_reservation(prices, p, ECRSRules(), 0., tie_break_usd_per_mwh=1e-3)
    energy = plan.actions_mw[:, 0]
    np.testing.assert_allclose(energy[:2], 0, atol=1e-7)      # no charging before it must
    assert energy[3] == pytest.approx(-10)                     # last zero-price step is full
    assert energy[5] == pytest.approx(10)                      # last high-price step is full
    np.testing.assert_allclose(plan.actions_mw[:, 1], 0, atol=1e-7)  # free reserve not requested
    assert plan.net_proxy_usd == pytest.approx(exact.net_proxy_usd, abs=1e-7)


def test_tie_break_must_be_nonnegative():
    with pytest.raises(ValueError, match="tie_break"):
        solve_reservation(np.ones((2, 2)), BatteryParams(), ECRSRules(), 0., tie_break_usd_per_mwh=-1.)
