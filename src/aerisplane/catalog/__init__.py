"""AerisPlane hardware and airfoil catalog."""
from __future__ import annotations


def get_airfoil(name: str):
    """Load an airfoil from the catalog by name.

    Parameters
    ----------
    name : str
        Airfoil name (case-insensitive), e.g. ``"ag35"``, ``"naca2412"``.

    Returns
    -------
    Airfoil

    Raises
    ------
    ValueError
        If no matching ``.dat`` file is found in the catalog.
    """
    from aerisplane.core.airfoil import Airfoil
    af = Airfoil(name)
    if af.coordinates is None:
        raise ValueError(
            f"Airfoil '{name}' not found in catalog. "
            "Check catalog/airfoils/ for available .dat files."
        )
    return af


def list_motors() -> list:
    """Return all motors in the catalog.

    Returns
    -------
    list of Motor
        All :class:`~aerisplane.core.propulsion.Motor` instances in
        ``aerisplane.catalog.motors``.

    Examples
    --------
    >>> from aerisplane.catalog import list_motors
    >>> motors = list_motors()
    >>> for m in motors:
    ...     print(f"{m.name}  KV={m.kv}")
    """
    import aerisplane.catalog.motors as _m
    from aerisplane.core.propulsion import Motor
    return [v for v in vars(_m).values() if isinstance(v, Motor)]


def list_batteries() -> list:
    """Return all batteries in the catalog.

    Returns
    -------
    list of Battery
        All :class:`~aerisplane.core.propulsion.Battery` instances in
        ``aerisplane.catalog.batteries``.
    """
    import aerisplane.catalog.batteries as _b
    from aerisplane.core.propulsion import Battery
    return [v for v in vars(_b).values() if isinstance(v, Battery)]


def list_propellers() -> list:
    """Return all propellers in the catalog.

    Returns
    -------
    list of Propeller
        All :class:`~aerisplane.core.propulsion.Propeller` instances in
        ``aerisplane.catalog.propellers``.
    """
    import aerisplane.catalog.propellers as _p
    from aerisplane.core.propulsion import Propeller
    return [v for v in vars(_p).values() if isinstance(v, Propeller)]


def list_servos() -> list:
    """Return all servos in the catalog.

    Returns
    -------
    list of Servo
        All :class:`~aerisplane.core.control_surface.Servo` instances in
        ``aerisplane.catalog.servos``.
    """
    import aerisplane.catalog.servos as _s
    from aerisplane.core.control_surface import Servo
    return [v for v in vars(_s).values() if isinstance(v, Servo)]


def list_aircraft() -> dict:
    """Return all reference aircraft factory functions keyed by name.

    Returns
    -------
    dict
        Mapping of aircraft name → factory callable.  Call the factory with no
        arguments to obtain a fresh :class:`~aerisplane.core.aircraft.Aircraft`
        instance, e.g.::

            from aerisplane.catalog import list_aircraft
            factories = list_aircraft()
            ac = factories["trainer"]()

    Examples
    --------
    >>> from aerisplane.catalog import list_aircraft
    >>> for name, factory in list_aircraft().items():
    ...     ac = factory()
    ...     print(f"{name}: {ac.name}")
    """
    from aerisplane.catalog.aircraft import AIRCRAFT_CATALOG
    return AIRCRAFT_CATALOG


def get_aircraft(name: str):
    """Return a reference aircraft by name.

    Parameters
    ----------
    name : str
        One of ``"small_uav"``, ``"trainer"``, ``"ultralight"``,
        ``"glider"``, ``"business_jet"``, ``"transport"``.

    Returns
    -------
    Aircraft

    Raises
    ------
    KeyError
        If ``name`` is not in the catalog.

    Examples
    --------
    >>> from aerisplane.catalog import get_aircraft
    >>> ac = get_aircraft("trainer")
    """
    from aerisplane.catalog.aircraft import AIRCRAFT_CATALOG
    if name not in AIRCRAFT_CATALOG:
        available = ", ".join(AIRCRAFT_CATALOG)
        raise KeyError(f"Aircraft '{name}' not in catalog. Available: {available}")
    return AIRCRAFT_CATALOG[name]()


__all__ = [
    "get_airfoil",
    "list_motors",
    "list_batteries",
    "list_propellers",
    "list_servos",
    "list_aircraft",
    "get_aircraft",
]
