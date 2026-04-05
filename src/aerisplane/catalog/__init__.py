"""AerisPlane hardware and airfoil catalog."""
from __future__ import annotations


def get_airfoil(name: str):
    from aerisplane.core.airfoil import Airfoil
    af = Airfoil(name)
    if af.coordinates is None:
        raise ValueError(
            f"Airfoil '{name}' not found in catalog. "
            "Check catalog/airfoils/ for available .dat files."
        )
    return af


def list_motors():
    import aerisplane.catalog.motors as _m
    from aerisplane.core.propulsion import Motor
    return [v for v in vars(_m).values() if isinstance(v, Motor)]


def list_batteries():
    import aerisplane.catalog.batteries as _b
    from aerisplane.core.propulsion import Battery
    return [v for v in vars(_b).values() if isinstance(v, Battery)]


def list_propellers():
    import aerisplane.catalog.propellers as _p
    from aerisplane.core.propulsion import Propeller
    return [v for v in vars(_p).values() if isinstance(v, Propeller)]


def list_servos():
    import aerisplane.catalog.servos as _s
    from aerisplane.core.control_surface import Servo
    return [v for v in vars(_s).values() if isinstance(v, Servo)]


__all__ = [
    "get_airfoil",
    "list_motors",
    "list_batteries",
    "list_propellers",
    "list_servos",
]
