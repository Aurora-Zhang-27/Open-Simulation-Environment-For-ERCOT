"""Battery physics. rte = 0.81 so that sqrt(rte) = 0.9 and hand math is easy."""

import math

import pytest

from rtcb_gym.battery import Battery, BatteryParams, battery_step, feasible_power_range

P = BatteryParams(power_mw=100, energy_mwh=200, rte=0.81, soc_min=0.0, soc_max=1.0,
                  soc_init=0.5, degradation_cost_usd_per_mwh=0.0)
DT = 0.25


def test_efficiency_split_is_sqrt_rte():
    assert P.eta_charge == pytest.approx(0.9)
    assert P.eta_discharge == pytest.approx(0.9)


def test_discharge_positive_power_lowers_soc():
    # 100 MW for 15 min = 25 MWh to the grid; storage loses 25 / 0.9 MWh.
    actual, soc, info = battery_step(P, 0.5, 100.0, DT)
    assert actual == 100.0
    assert soc == pytest.approx(0.5 - (25 / 0.9) / 200)
    assert not info["clipped"] and info["clip_reason"] is None
    assert info["grid_discharge_mwh"] == pytest.approx(25.0)
    assert info["storage_discharge_mwh"] == pytest.approx(25 / 0.9)


def test_charge_negative_power_raises_soc():
    # -100 MW for 15 min = 25 MWh from the grid; storage gains 25 * 0.9 MWh.
    actual, soc, info = battery_step(P, 0.5, -100.0, DT)
    assert actual == -100.0
    assert soc == pytest.approx(0.5 + 22.5 / 200)
    assert info["grid_charge_mwh"] == pytest.approx(25.0)
    assert info["storage_discharge_mwh"] == 0.0


def test_power_limit_clip():
    actual, _, info = battery_step(P, 0.5, 150.0, DT)
    assert actual == 100.0
    assert info["clipped"] and info["clip_reason"] == "power_limit"
    assert info["requested_power_mw"] == 150.0
    actual, _, info = battery_step(P, 0.5, -150.0, DT)
    assert actual == -100.0 and info["clip_reason"] == "power_limit"


def test_soc_min_clip():
    # 10 MWh stored; at most 10 * 0.9 = 9 MWh to the grid in 15 min = 36 MW.
    actual, soc, info = battery_step(P, 0.05, 100.0, DT)
    assert actual == pytest.approx(36.0)
    assert soc == pytest.approx(0.0)
    assert info["clipped"] and info["clip_reason"] == "soc_min"


def test_soc_max_clip():
    # 10 MWh headroom; at most 10 / 0.9 MWh from the grid in 15 min.
    actual, soc, info = battery_step(P, 0.95, -100.0, DT)
    assert actual == pytest.approx(-10 / 0.9 / DT)
    assert soc == pytest.approx(1.0)
    assert info["clipped"] and info["clip_reason"] == "soc_max"


def test_power_and_soc_clip_reports_binding_limit():
    actual, _, info = battery_step(P, 0.05, 500.0, DT)
    assert actual == pytest.approx(36.0)
    assert info["clip_reason"] == "soc_min"


def test_zero_action():
    actual, soc, info = battery_step(P, 0.5, 0.0, DT)
    assert actual == 0.0 and soc == 0.5 and not info["clipped"]


def test_full_cycle_returns_rte_of_energy():
    b = Battery(P, soc=0.0)
    bought = sold = 0.0
    while b.soc < 1.0 - 1e-12:
        actual, _, _ = b.step(-100.0, DT)
        bought += -actual * DT
    while b.soc > 1e-12:
        actual, _, _ = b.step(100.0, DT)
        sold += actual * DT
    assert bought == pytest.approx(200 / 0.9)
    assert sold == pytest.approx(200 * 0.9)
    assert sold / bought == pytest.approx(0.81)


def test_feasible_power_range():
    lo, hi = feasible_power_range(P, 0.5, DT)
    assert (lo, hi) == (-100.0, 100.0)
    lo, hi = feasible_power_range(P, 0.0, DT)
    assert hi == 0.0 and lo == -100.0
    lo, hi = feasible_power_range(P, 1.0, DT)
    assert lo == 0.0 and hi == 100.0


def test_soc_limits_respected_with_tight_band():
    q = BatteryParams(power_mw=100, energy_mwh=200, rte=0.85, soc_min=0.1, soc_max=0.9, soc_init=0.1)
    soc = q.soc_init
    for p in [100, -100] * 20 + [-100] * 20 + [100] * 20:
        _, soc, _ = battery_step(q, soc, p, DT)
        assert q.soc_min <= soc <= q.soc_max


def test_pure_and_deterministic():
    a = battery_step(P, 0.3, 42.0, DT)
    b = battery_step(P, 0.3, 42.0, DT)
    assert a == b
    with pytest.raises(Exception):
        P.power_mw = 1.0  # frozen


def test_battery_reset():
    b = Battery(P)
    b.step(100.0, DT)
    assert b.reset() == 0.5


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_power_rejected(bad):
    with pytest.raises(ValueError):
        battery_step(P, 0.5, bad, DT)


@pytest.mark.parametrize(
    "kwargs",
    [dict(power_mw=0), dict(energy_mwh=-1), dict(rte=1.2), dict(rte=0),
     dict(soc_min=0.6, soc_max=0.5), dict(soc_init=1.5), dict(degradation_cost_usd_per_mwh=-1)],
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        BatteryParams(**kwargs)
