"""Money.

v0 settles real-time energy only, as a price-taker at one trading hub:

    energy_revenue   = actual_power_mw * dt_hours * rt_price
    degradation_cost = |actual_power_mw| * dt_hours * degradation_cost_usd_per_mwh
    net_revenue      = energy_revenue - degradation_cost

Sign convention matches :mod:`rtcb_gym.battery`: positive power is discharge
(sell), negative is charge (buy). Charging at a negative price therefore
earns money: (-P) * dt * (negative price) > 0.

Not modelled in v0: day-ahead two-settlement (v0.2), ancillary services
(v0.5), resource-node vs hub basis, fees and uplift charges.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Settlement:
    energy_mwh: float  # + sold, - bought (grid side)
    energy_revenue_usd: float
    degradation_cost_usd: float
    net_revenue_usd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def settle_real_time(
    actual_power_mw: float,
    dt_hours: float,
    rt_price_usd_per_mwh: float,
    degradation_cost_usd_per_mwh: float = 0.0,
) -> Settlement:
    """Settle one real-time interval on the power actually delivered."""
    if not math.isfinite(rt_price_usd_per_mwh):
        raise ValueError(f"rt price must be finite, got {rt_price_usd_per_mwh}")
    if not dt_hours > 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
    if degradation_cost_usd_per_mwh < 0:
        raise ValueError("degradation cost must be >= 0")
    energy = actual_power_mw * dt_hours
    revenue = energy * rt_price_usd_per_mwh
    degradation = abs(energy) * degradation_cost_usd_per_mwh
    return Settlement(
        energy_mwh=energy,
        energy_revenue_usd=revenue,
        degradation_cost_usd=degradation,
        net_revenue_usd=revenue - degradation,
    )
