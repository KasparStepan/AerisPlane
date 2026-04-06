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
    """Fit a drag polar at the given altitude.

    Parameters
    ----------
    aircraft : Aircraft
    weight_result : WeightResult
    altitude : float, optional
        Altitude for drag polar fitting [m]. Default 0.0.
    aero_method : str, optional
        Aero solver. Default ``"vlm"``.

    Returns
    -------
    DragPolar
        Fitted CD0, k (induced drag factor), and reference conditions.
    """
    return fit_drag_polar(aircraft, weight_result, altitude, aero_method, **aero_kwargs)


def envelope(
    aircraft: Aircraft,
    weight_result: WeightResult,
    CL_max: float = 1.4,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> EnvelopeResult:
    """Compute the full flight performance envelope.

    Parameters
    ----------
    aircraft : Aircraft
    weight_result : WeightResult
    CL_max : float, optional
        Maximum lift coefficient for stall speed calculation. Default 1.4.
    aero_method : str, optional
        Aero solver. Default ``"vlm"``.

    Returns
    -------
    EnvelopeResult
        Stall speed, minimum power speed, best range speed, best climb speed,
        and service ceiling.
    """
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
    """Run segment-by-segment mission energy budget analysis.

    Fits a drag polar once at sea level, then evaluates each segment
    in order for duration, distance, and electrical energy required.
    Compares total energy to the battery capacity.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft with a ``propulsion`` attribute (used for battery energy and
        efficiency estimation).
    weight_result : WeightResult
        Weight analysis result (provides total mass for power-required calculation).
    mission : Mission
        Mission profile with ordered segments and start altitude.
    aero_method : str, optional
        Aero solver used for drag polar fitting. Default ``"vlm"``.
    **aero_kwargs
        Additional keyword arguments passed to ``aero.analyze()``.

    Returns
    -------
    MissionResult
        Total energy [J], time [s], distance [m], battery margin, feasibility flag,
        and per-segment breakdown.

    Examples
    --------
    >>> from aerisplane.mission import analyze
    >>> from aerisplane.mission.segments import Mission, Climb, Cruise, Loiter
    >>> mission = Mission(
    ...     start_altitude=0.0,
    ...     segments=[
    ...         Climb(name="climb", velocity=14.0, climb_rate=2.0, to_altitude=100.0),
    ...         Cruise(name="cruise", velocity=18.0, altitude=100.0, distance=5000.0),
    ...         Loiter(name="loiter", velocity=16.0, altitude=100.0, duration=600.0),
    ...     ]
    ... )
    >>> result = analyze(aircraft, weight_result, mission)
    >>> print(f"Feasible: {result.feasible}  margin: {result.energy_margin:.1%}")
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
