"""Mission analysis module.

Public API
----------
performance(aircraft, weight_result, altitude=0.0) -> DragPolar
    Fit a drag polar at the given altitude.

envelope(aircraft, weight_result, CL_max=1.4) -> EnvelopeResult
    Compute full flight performance envelope.

analyze(aircraft, weight_result, mission) -> MissionResult
    Run mission energy budget analysis.
"""
from __future__ import annotations

from aerisplane.core.aircraft import Aircraft
from aerisplane.mission.envelope import EnvelopeResult, compute_envelope
from aerisplane.mission.performance import (
    DragPolar,
    fit_drag_polar,
    power_required,
    _estimate_efficiency,
)
from aerisplane.mission.result import MissionResult, SegmentResult
from aerisplane.mission.segments import (
    Climb, Cruise, Descent, Loiter, Mission, Return,
)
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult

G = 9.81


def performance(
    aircraft: Aircraft,
    weight_result: WeightResult,
    altitude: float = 0.0,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> DragPolar:
    """Fit a drag polar at the given altitude."""
    return fit_drag_polar(aircraft, weight_result, altitude, aero_method, **aero_kwargs)


def envelope(
    aircraft: Aircraft,
    weight_result: WeightResult,
    CL_max: float = 1.4,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> EnvelopeResult:
    """Compute full flight performance envelope."""
    return compute_envelope(
        aircraft, weight_result, CL_max=CL_max,
        aero_method=aero_method, **aero_kwargs,
    )


def analyze(
    aircraft: Aircraft,
    weight_result: WeightResult,
    mission: Mission,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> MissionResult:
    """Run mission energy budget analysis.

    For each segment, computes duration, distance, and energy required.
    Compares total energy to battery capacity.
    """
    mass = weight_result.total_mass
    W = mass * G
    prop = aircraft.propulsion

    # Fit drag polar once at representative altitude
    polar = fit_drag_polar(aircraft, weight_result, altitude=0.0,
                           aero_method=aero_method, **aero_kwargs)

    E_battery = prop.battery.energy() if prop else 0.0

    seg_results = []
    alt_current = mission.start_altitude
    total_energy = 0.0
    total_time = 0.0
    total_distance = 0.0

    for seg in mission.segments:
        sr = _analyze_segment(seg, alt_current, polar, mass, prop)
        seg_results.append(sr)
        total_energy += sr.energy
        total_time += sr.duration
        total_distance += sr.distance
        alt_current = sr.altitude_end

    energy_margin = max(0.0, 1.0 - total_energy / E_battery) if E_battery > 0 else 0.0
    feasible = total_energy <= E_battery

    return MissionResult(
        total_energy=total_energy,
        total_time=total_time,
        total_distance=total_distance,
        battery_energy_available=E_battery,
        energy_margin=energy_margin,
        feasible=feasible,
        segments=seg_results,
    )


def _analyze_segment(seg, alt_start, polar, mass, prop):
    """Compute energy for a single mission segment."""
    W = mass * G

    if isinstance(seg, Climb):
        alt_end = seg.to_altitude
        dh = alt_end - alt_start
        duration = abs(dh) / seg.climb_rate if seg.climb_rate > 0 else 0.0
        distance = seg.velocity * duration

        # Power = D*V + W*climb_rate
        pr_level = power_required(seg.velocity, polar, mass, alt_start)
        pr_climb = pr_level + W * seg.climb_rate

        eta = _estimate_efficiency(prop, seg.velocity, alt_start) if prop else 0.5
        p_elec = pr_climb / eta if eta > 0 else pr_climb
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    elif isinstance(seg, (Cruise, Return)):
        alt_end = seg.altitude
        duration = seg.distance / seg.velocity if seg.velocity > 0 else 0.0
        distance = seg.distance

        pr = power_required(seg.velocity, polar, mass, seg.altitude)
        eta = _estimate_efficiency(prop, seg.velocity, seg.altitude) if prop else 0.5
        p_elec = pr / eta if eta > 0 else pr
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    elif isinstance(seg, Loiter):
        alt_end = seg.altitude
        duration = seg.duration
        distance = seg.velocity * duration

        pr = power_required(seg.velocity, polar, mass, seg.altitude)
        eta = _estimate_efficiency(prop, seg.velocity, seg.altitude) if prop else 0.5
        p_elec = pr / eta if eta > 0 else pr
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    elif isinstance(seg, Descent):
        alt_end = seg.to_altitude
        dh = alt_start - alt_end
        duration = abs(dh) / seg.descent_rate if seg.descent_rate > 0 else 0.0
        distance = seg.velocity * duration

        # Partial power descent: assume 50% of level power
        pr = power_required(seg.velocity, polar, mass, alt_start) * 0.5
        eta = _estimate_efficiency(prop, seg.velocity, alt_start) if prop else 0.5
        p_elec = pr / eta if eta > 0 else pr
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    else:
        raise ValueError(f"Unknown segment type: {type(seg)}")


__all__ = [
    "analyze", "performance", "envelope",
    "MissionResult", "EnvelopeResult", "DragPolar",
]
