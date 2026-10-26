"""Money.

Price-taker at one trading hub. Two ways to settle an interval:

Real-time only (v0.1, :func:`settle_real_time`)::

    energy_revenue   = actual_mw * dt * rt_price
    degradation_cost = |actual_mw| * dt * degradation_cost_usd_per_mwh
    net_revenue      = energy_revenue - degradation_cost

Two-settlement (v0.2, :func:`settle_two_settlement`), as in ERCOT: the
day-ahead award is paid at the day-ahead price, and the real-time
deviation from it at the real-time price::

    da_revenue       = da_award_mw * dt * da_price
    rt_revenue       = (actual_mw - da_award_mw) * dt * rt_price
    energy_revenue   = da_revenue + rt_revenue
    net_revenue      = energy_revenue - degradation_cost

Degradation is charged on the power actually delivered, never on the
day-ahead position (that is a financial position, not battery use). With
``da_award_mw = 0`` both functions give the same result.

Sign convention matches :mod:`rtcb_gym.battery`: positive power is discharge
(sell), negative is charge (buy). Charging at a negative price therefore
earns money: (-P) * dt * (negative price) > 0.

Not modelled: ancillary services (v0.5), resource-node vs hub basis, fees
and uplift charges.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Settlement:
    energy_mwh: float  # actual energy, + sold, - bought (grid side)
    energy_revenue_usd: float  # da_revenue_usd + rt_revenue_usd
    degradation_cost_usd: float
    net_revenue_usd: float
    da_revenue_usd: float = 0.0
    rt_revenue_usd: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _check(dt_hours: float, degradation_cost_usd_per_mwh: float, **prices: float) -> None:
    for name, value in prices.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if not dt_hours > 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
    if degradation_cost_usd_per_mwh < 0:
        raise ValueError("degradation cost must be >= 0")


def settle_real_time(
    actual_power_mw: float,
    dt_hours: float,
    rt_price_usd_per_mwh: float,
    degradation_cost_usd_per_mwh: float = 0.0,
) -> Settlement:
    """Settle one real-time interval on the power actually delivered."""
    _check(dt_hours, degradation_cost_usd_per_mwh, rt_price=rt_price_usd_per_mwh)
    energy = actual_power_mw * dt_hours
    revenue = energy * rt_price_usd_per_mwh
    degradation = abs(energy) * degradation_cost_usd_per_mwh
    return Settlement(
        energy_mwh=energy,
        energy_revenue_usd=revenue,
        degradation_cost_usd=degradation,
        net_revenue_usd=revenue - degradation,
        da_revenue_usd=0.0,
        rt_revenue_usd=revenue,
    )


def settle_two_settlement(
    actual_power_mw: float,
    da_award_mw: float,
    dt_hours: float,
    rt_price_usd_per_mwh: float,
    da_price_usd_per_mwh: float,
    degradation_cost_usd_per_mwh: float = 0.0,
) -> Settlement:
    """Settle one real-time interval against a day-ahead award.

    ``da_award_mw`` is the award of the hour this interval belongs to; the
    hour's day-ahead revenue is spread evenly over its intervals.
    """
    _check(
        dt_hours, degradation_cost_usd_per_mwh,
        rt_price=rt_price_usd_per_mwh, da_price=da_price_usd_per_mwh, da_award=da_award_mw,
    )
    energy = actual_power_mw * dt_hours
    da_revenue = da_award_mw * dt_hours * da_price_usd_per_mwh
    rt_revenue = (actual_power_mw - da_award_mw) * dt_hours * rt_price_usd_per_mwh
    degradation = abs(energy) * degradation_cost_usd_per_mwh
    revenue = da_revenue + rt_revenue
    return Settlement(
        energy_mwh=energy,
        energy_revenue_usd=revenue,
        degradation_cost_usd=degradation,
        net_revenue_usd=revenue - degradation,
        da_revenue_usd=da_revenue,
        rt_revenue_usd=rt_revenue,
    )
