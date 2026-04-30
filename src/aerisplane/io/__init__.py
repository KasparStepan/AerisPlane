"""File I/O for aircraft geometry.

Native AerisPlane format
------------------------
Use :func:`save_aircraft` / :func:`load_aircraft` to round-trip a full
:class:`~aerisplane.core.aircraft.Aircraft` (wings, control surfaces,
fuselages, propulsion, payload, structures) to a versioned JSON file.

Exchange formats
----------------
:func:`from_avl` / :func:`to_avl` interoperate with AVL.
:func:`to_openvsp` writes a ``.vspscript`` for OpenVSP.

These exchange formats are geometry-only — propulsion, payload, control
surfaces (where unsupported), and structural fields are dropped on export
and remain unset on import.

Future targets (XFLR5 ``.xml``, Flow5) will use the same ``from_<fmt>`` /
``to_<fmt>`` naming.
"""

from aerisplane.io.aerisplane import load_aircraft, save_aircraft
from aerisplane.io.avl import from_avl, to_avl
from aerisplane.io.openvsp import to_openvsp

__all__ = [
    "load_aircraft",
    "save_aircraft",
    "from_avl",
    "to_avl",
    "to_openvsp",
]
