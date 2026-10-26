"""Settlement, checked against hand-calculated numbers."""

import pytest

from rtcb_gym.battery import Battery, BatteryParams, battery_step
from rtcb_gym.settlement import settle_real_time, settle_two_settlement

DT = 0.25


def test_positive_price_discharge():
    # Sell 100 MW * 0.25 h = 25 MWh at $50: revenue $1250, degradation 25 * $2 = $50.
    s = settle_real_time(100.0, DT, 50.0, 2.0)
    assert s.energy_mwh == 25.0
    assert s.energy_revenue_usd == 1250.0
    assert s.degradation_cost_usd == 50.0
    assert s.net_revenue_usd == 1200.0


def test_positive_price_charge_costs_money():
    # Buy 25 MWh at $50: -$1250, degradation $50 -> -$1300.
    s = settle_real_time(-100.0, DT, 50.0, 2.0)
    assert s.net_revenue_usd == -1300.0


def test_negative_price_charging_earns_money():
    # Buy 25 MWh at -$20/MWh: we are paid $500; degradation $50 -> +$450.
    s = settle_real_time(-100.0, DT, -20.0, 2.0)
    assert s.energy_revenue_usd == 500.0
    assert s.net_revenue_usd == 450.0


def test_negative_price_discharging_costs_money():
    s = settle_real_time(100.0, DT, -20.0, 0.0)
    assert s.net_revenue_usd == -500.0


def test_zero_action():
    s = settle_real_time(0.0, DT, 1234.0, 2.0)
    assert s.energy_revenue_usd == 0.0
    assert s.degradation_cost_usd == 0.0
    assert s.net_revenue_usd == 0.0


def test_clipped_action_settles_on_actual_power():
    # rte 0.81 -> 0.9 each way. SoC 0.05 of 200 MWh = 10 MWh stored, so a
    # 100 MW request delivers only 36 MW (9 MWh). At $100: $900 - 9 * $2 = $882.
    p = BatteryParams(power_mw=100, energy_mwh=200, rte=0.81, soc_init=0.05)
    actual, _, info = battery_step(p, 0.05, 100.0, DT)
    assert info["clipped"]
    s = settle_real_time(actual, DT, 100.0, 2.0)
    assert s.energy_revenue_usd == pytest.approx(900.0)
    assert s.net_revenue_usd == pytest.approx(882.0)


def test_efficiency_loss_over_full_cycle():
    # Charge 0 -> full and discharge full -> 0 at a flat $10, no degradation.
    # Bought 200/0.9 = 222.22 MWh, sold 200*0.9 = 180 MWh: loss $422.22.
    p = BatteryParams(power_mw=100, energy_mwh=200, rte=0.81, soc_init=0.0,
                      degradation_cost_usd_per_mwh=0.0)
    b = Battery(p)
    total = 0.0
    while b.soc < 1.0 - 1e-12:
        actual, _, _ = b.step(-100.0, DT)
        total += settle_real_time(actual, DT, 10.0).net_revenue_usd
    while b.soc > 1e-12:
        actual, _, _ = b.step(100.0, DT)
        total += settle_real_time(actual, DT, 10.0).net_revenue_usd
    assert total == pytest.approx(-(200 / 0.9 - 200 * 0.9) * 10.0)
    assert total == pytest.approx(-422.2222222)


def test_degradation_charged_on_both_directions():
    a = settle_real_time(80.0, DT, 0.0, 3.0)
    b = settle_real_time(-80.0, DT, 0.0, 3.0)
    assert a.degradation_cost_usd == b.degradation_cost_usd == 60.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        settle_real_time(1.0, DT, float("nan"))
    with pytest.raises(ValueError):
        settle_real_time(1.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        settle_real_time(1.0, DT, 10.0, -1.0)


# --------------------------------------------------------------------------
# Two-settlement


def test_two_settlement_hand_example():
    # Sold 80 MW day-ahead at $50, delivered only 60 MW in real time at $90:
    # DA 80 * 0.25 * 50 = 1000; RT buys back 20 MW: -20 * 0.25 * 90 = -450.
    s = settle_two_settlement(60.0, 80.0, 0.25, 90.0, 50.0, degradation_cost_usd_per_mwh=2.0)
    assert s.da_revenue_usd == pytest.approx(1000.0)
    assert s.rt_revenue_usd == pytest.approx(-450.0)
    assert s.energy_revenue_usd == pytest.approx(550.0)
    assert s.degradation_cost_usd == pytest.approx(60 * 0.25 * 2.0)  # on delivered power only
    assert s.net_revenue_usd == pytest.approx(550.0 - 30.0)


def test_two_settlement_delivering_the_award_earns_the_day_ahead_price():
    s = settle_two_settlement(-40.0, -40.0, 0.25, 999.0, 20.0)
    assert s.rt_revenue_usd == 0.0
    assert s.energy_revenue_usd == pytest.approx(-40 * 0.25 * 20.0)


def test_two_settlement_without_award_equals_real_time_only():
    a = settle_two_settlement(75.0, 0.0, 0.25, 42.0, 10.0, degradation_cost_usd_per_mwh=2.0)
    b = settle_real_time(75.0, 0.25, 42.0, degradation_cost_usd_per_mwh=2.0)
    assert a.to_dict() == b.to_dict()


def test_two_settlement_position_without_delivery_is_a_spread_bet():
    # 50 MW sold day-ahead, battery idle: earns (DA - RT) on 50 MW, no degradation.
    s = settle_two_settlement(0.0, 50.0, 0.25, 30.0, 45.0, degradation_cost_usd_per_mwh=2.0)
    assert s.net_revenue_usd == pytest.approx(50 * 0.25 * (45.0 - 30.0))
    assert s.degradation_cost_usd == 0.0


def test_two_settlement_bad_inputs():
    with pytest.raises(ValueError):
        settle_two_settlement(0.0, 0.0, 0.25, 10.0, float("nan"))
    with pytest.raises(ValueError):
        settle_two_settlement(0.0, float("inf"), 0.25, 10.0, 10.0)
