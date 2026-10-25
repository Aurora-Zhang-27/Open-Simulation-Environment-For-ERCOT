"""Physical battery model.

Sign convention (used everywhere in RTCB-Gym):
    power > 0  -> discharge (sell energy to the grid)
    power < 0  -> charge    (buy energy from the grid)

Power is measured at the grid connection. Round-trip efficiency is split
evenly: sqrt(rte) on charge and sqrt(rte) on discharge, so
    charging   |P| MW for dt hours stores    |P| * dt * sqrt(rte) MWh
    discharging  P MW for dt hours removes     P * dt / sqrt(rte) MWh

State of charge (SoC) is a fraction of ``energy_mwh``.

Episode SoC rule: ``soc_init`` is the SoC at the start of an episode and
``soc_terminal_min`` the lowest SoC allowed at its end. The end target is
enforced step by step: when ``steps_left`` is given, the battery may not go
below the SoC from which the target can still be reached by charging at full
power for the remaining intervals. Near the end this can limit discharge or
force charging. Every trajectory the environment allows therefore ends at or
above the target, which keeps perfect foresight an exact upper bound.

The model is pure and deterministic: no prices, no randomness, no hidden state
beyond the SoC held by :class:`Battery`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

# Requested and delivered power that differ by less than this are not
# counted as clipped. It absorbs floating-point and LP-solver round-off
# (0.1 kW on a 100 MW battery).
CLIP_TOL_MW = 1e-4


@dataclass(frozen=True)
class BatteryParams:
    power_mw: float = 100.0
    energy_mwh: float = 200.0
    rte: float = 0.85
    soc_min: float = 0.0
    soc_max: float = 1.0
    soc_init: float = 0.0
    soc_terminal_min: float = 0.0
    degradation_cost_usd_per_mwh: float = 2.0

    def __post_init__(self) -> None:
        if not self.power_mw > 0:
            raise ValueError(f"power_mw must be > 0, got {self.power_mw}")
        if not self.energy_mwh > 0:
            raise ValueError(f"energy_mwh must be > 0, got {self.energy_mwh}")
        if not 0 < self.rte <= 1:
            raise ValueError(f"rte must be in (0, 1], got {self.rte}")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError(
                f"need 0 <= soc_min < soc_max <= 1, got {self.soc_min}, {self.soc_max}"
            )
        if not self.soc_min <= self.soc_init <= self.soc_max:
            raise ValueError(
                f"soc_init {self.soc_init} outside [{self.soc_min}, {self.soc_max}]"
            )
        # A target at or below soc_min means "no end target".
        if not 0 <= self.soc_terminal_min <= self.soc_max:
            raise ValueError(
                f"soc_terminal_min {self.soc_terminal_min} outside [0, {self.soc_max}]"
            )
        if self.degradation_cost_usd_per_mwh < 0:
            raise ValueError("degradation_cost_usd_per_mwh must be >= 0")

    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.rte)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.rte)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatteryParams":
        return cls(**{k: float(v) for k, v in d.items()})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def terminal_floor(params: BatteryParams, steps_after: int, dt_hours: float) -> float:
    """Lowest SoC from which ``soc_terminal_min`` is still reachable by charging
    at full power for ``steps_after`` more intervals (never below soc_min)."""
    gain_per_step = params.power_mw * dt_hours * params.eta_charge / params.energy_mwh
    return max(params.soc_min, params.soc_terminal_min - steps_after * gain_per_step)


def _power_range(
    params: BatteryParams, soc: float, dt_hours: float, steps_left: int | None
) -> tuple[float, float, str]:
    if not dt_hours > 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
    e = params.energy_mwh
    floor, floor_name = params.soc_min, "soc_min"
    if steps_left is not None:
        f = terminal_floor(params, steps_left - 1, dt_hours)
        if f > floor:
            floor, floor_name = f, "soc_terminal"
    max_charge = (params.soc_max - soc) * e / (params.eta_charge * dt_hours)
    lo = -min(params.power_mw, max(max_charge, 0.0))
    if soc >= floor:
        hi = min(params.power_mw, (soc - floor) * e * params.eta_discharge / dt_hours)
    else:  # below the floor: must charge at least this much
        hi = -(floor - soc) * e / (params.eta_charge * dt_hours)
    return lo, max(hi, lo), floor_name


def feasible_power_range(
    params: BatteryParams, soc: float, dt_hours: float, steps_left: int | None = None
) -> tuple[float, float]:
    """Return (min_power_mw, max_power_mw) the battery may deliver this interval
    without breaking power or SoC limits. min is <= 0 (charging).

    With ``steps_left`` (intervals left in the episode, this one included) the
    end-of-episode target ``soc_terminal_min`` is enforced too; max can then
    be negative, meaning the battery must charge.
    """
    lo, hi, _ = _power_range(params, soc, dt_hours, steps_left)
    return lo, hi


def battery_step(
    params: BatteryParams,
    soc: float,
    power_mw: float,
    dt_hours: float,
    steps_left: int | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Apply one interval of requested power.

    Returns ``(actual_power_mw, new_soc, info)``. The request is clipped first
    to the power limit, then to what the SoC limits allow (and, if
    ``steps_left`` is given, the end-of-episode target); ``info`` records
    whether clipping happened and which limit was binding.
    """
    requested = float(power_mw)
    if not math.isfinite(requested):
        raise ValueError(f"power_mw must be finite, got {power_mw}")
    if not dt_hours > 0:
        raise ValueError(f"dt_hours must be > 0, got {dt_hours}")

    p = requested
    reason = None
    if abs(p) > params.power_mw:
        p = math.copysign(params.power_mw, p)
        reason = "power_limit"

    lo, hi, floor_name = _power_range(params, soc, dt_hours, steps_left)
    if p > hi:
        p = hi
        reason = floor_name  # soc_min, or soc_terminal: keep the end target reachable
    elif p < lo:
        p = lo
        reason = "soc_max"  # not enough headroom to charge more

    if p >= 0:
        storage_change_mwh = -p * dt_hours / params.eta_discharge
    else:
        storage_change_mwh = -p * dt_hours * params.eta_charge
    new_soc = soc + storage_change_mwh / params.energy_mwh
    # Guard against round-off pushing SoC a hair outside its limits.
    new_soc = min(max(new_soc, params.soc_min), params.soc_max)

    clipped = abs(requested - p) > CLIP_TOL_MW
    info = {
        "requested_power_mw": requested,
        "actual_power_mw": p,
        "clipped": clipped,
        "clip_reason": reason if clipped else None,
        "soc_before": soc,
        "soc_after": new_soc,
        # Grid-side energy this interval (always >= 0).
        "grid_discharge_mwh": max(p, 0.0) * dt_hours,
        "grid_charge_mwh": max(-p, 0.0) * dt_hours,
        # Storage-side energy removed this interval (used for cycle counting).
        "storage_discharge_mwh": max(-storage_change_mwh, 0.0),
    }
    return p, new_soc, info


class Battery:
    """Holds the SoC and steps the pure model in :func:`battery_step`."""

    def __init__(self, params: BatteryParams, soc: float | None = None) -> None:
        self.params = params
        self.soc = params.soc_init if soc is None else float(soc)

    def reset(self, soc: float | None = None) -> float:
        self.soc = self.params.soc_init if soc is None else float(soc)
        return self.soc

    def step(
        self, power_mw: float, dt_hours: float, steps_left: int | None = None
    ) -> tuple[float, float, dict[str, Any]]:
        actual, self.soc, info = battery_step(self.params, self.soc, power_mw, dt_hours, steps_left)
        return actual, self.soc, info
