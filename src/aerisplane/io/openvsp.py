"""OpenVSP export via the ``.vspscript`` file format.

The script is loaded inside OpenVSP via *File → Run Script…*. It uses the
public OpenVSP API (``AddGeom``, ``SetParmVal``, ``SetAirfoilPnts``, …) to
rebuild the geometry parametrically — wings as ``WING`` Geoms with
``XS_FILE_AIRFOIL`` cross-sections, fuselages as ``FUSELAGE`` Geoms with
``XS_SUPER_ELLIPSE`` cross-sections.

API call patterns are based on the AeroSandbox OpenVSP exporter
(MIT-licensed, by Peter Sharpe), tested against OpenVSP 3.36.

Lossy on export: control surfaces, spars/skins, materials, propulsion, and
payload have no representation in VSP's parametric Geom model. They are
dropped with a warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import indent

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.fuselage import Fuselage
from aerisplane.core.wing import Wing

OPENVSP_VERSION = "3.36"
"""OpenVSP version the API call patterns were last verified against."""


# --------------------------------------------------------------------------- #
# Wing
# --------------------------------------------------------------------------- #


def _airfoil_block(xsec_index: int, airfoil: Airfoil | None) -> str:
    """Emit the ``SetAirfoilPnts`` call that loads explicit coordinates into XSec_i."""
    if airfoil is None or airfoil.coordinates is None:
        # Flat-plate fallback: a degenerate airfoil VSP can still ingest.
        upper = np.array([[0.0, 0.0], [1.0, 0.0]])
        lower = np.array([[0.0, 0.0], [1.0, 0.0]])
    else:
        upper = airfoil.upper_coordinates()                                 # LE→TE
        lower = airfoil.lower_coordinates()                                 # LE→TE

    up_pts = ", ".join(f"vec3d({p[0]:.8g}, {p[1]:.8g}, 0.0)" for p in upper)
    lo_pts = ", ".join(f"vec3d({p[0]:.8g}, {p[1]:.8g}, 0.0)" for p in lower)

    return f"""\
{{
    array<vec3d> up_pnt_vec = {{ {up_pts} }};
    array<vec3d> lo_pnt_vec = {{ {lo_pts} }};
    string xsec_surf = GetXSecSurf( wid, {xsec_index} );
    string xsec = GetXSec( xsec_surf, {xsec_index} );
    SetParmVal( wid, "ThickChord", "XSecCurve_{xsec_index}", 0.5 );
    SetParmVal( wid, "ThickChord", "XSecCurve_{xsec_index}", 1.0 );
    SetAirfoilPnts( xsec, up_pnt_vec, lo_pnt_vec );
    Update();
}}
"""


def _wing_block(wing: Wing) -> str:
    if len(wing.xsecs) < 2:
        raise ValueError(f"Wing '{wing.name}' must have at least 2 cross-sections.")

    sym_flag = "2.0" if wing.symmetric else "0.0"                           # 2 = XZ symmetry
    x0, y0, z0 = wing.xsecs[0].xyz_le

    parts = [
        f'//==== Add Wing "{wing.name}" ====//',
        'string wid = AddGeom("WING");',
        'ChangeXSecShape( GetXSecSurf( wid, 0 ), 0, XS_FILE_AIRFOIL );',
        'ChangeXSecShape( GetXSecSurf( wid, 1 ), 1, XS_FILE_AIRFOIL );',
        '',
        '//==== Generate Blank Wing Sections ====//',
    ]
    for i in range(len(wing.xsecs) - 1):
        parts.append(
            f'InsertXSec( wid, 1, XS_FILE_AIRFOIL ); // WingXSecs {i} to {i + 1}'
        )
    parts += [
        '',
        '//==== Cut The Original Section ====//',
        'CutXSec( wid, 1 );',
        '',
        f'SetParmVal( wid, "GeomName", "Design", "{wing.name}" );',
        f'SetParmVal( wid, "X_Rel_Location", "XForm", {x0} );',
        f'SetParmVal( wid, "Y_Rel_Location", "XForm", {y0} );',
        f'SetParmVal( wid, "Z_Rel_Location", "XForm", {z0} );',
        f'SetParmVal( wid, "Twist", "XSec_0", {wing.xsecs[0].twist} );',
        'SetParmVal( wid, "Twist_Location", "XSec_0", 0.0 );',
        'SetParmVal( wid, "Sym_Ancestor", "Sym", 0.0 );',
        f'SetParmVal( wid, "Sym_Planar_Flag", "Sym", {sym_flag} );',
        'SetParmVal( wid, "RotateAirfoilMatchDideralFlag", "WingGeom", 1.0 );',
        'Update();',
        '',
        '//==== Set Wing Section Options ====//',
    ]

    xyz_le = np.stack([xs.xyz_le for xs in wing.xsecs], axis=0)
    dxyz = np.diff(xyz_le, axis=0)
    dx, dy, dz = dxyz[:, 0], dxyz[:, 1], dxyz[:, 2]
    dyz = np.sqrt(dy**2 + dz**2)
    dihedrals = np.degrees(np.arctan2(dz, dy))
    sweeps_le = np.degrees(np.arctan2(dx, dyz))

    for i, (a, b) in enumerate(zip(wing.xsecs[:-1], wing.xsecs[1:])):
        parts.append(f"// Section {i} (XSec_{i + 1}: from xsec {i} to {i + 1})")
        parts.append(f'SetParmVal( wid, "Root_Chord", "XSec_{i + 1}", {a.chord} );')
        parts.append(f'SetParmVal( wid, "Tip_Chord", "XSec_{i + 1}", {b.chord} );')
        parts.append(f'SetParmVal( wid, "Span", "XSec_{i + 1}", {dyz[i]} );')
        parts.append(f'SetParmVal( wid, "Sweep", "XSec_{i + 1}", {sweeps_le[i]} );')
        parts.append(f'SetParmVal( wid, "Sweep_Location", "XSec_{i + 1}", 0.0 );')
        parts.append(f'SetParmVal( wid, "Twist", "XSec_{i + 1}", {b.twist} );')
        parts.append(f'SetParmVal( wid, "Twist_Location", "XSec_{i + 1}", 0.0 );')
        parts.append(f'SetParmVal( wid, "Dihedral", "XSec_{i + 1}", {dihedrals[i]} );')
        parts.append(f'SetParmVal( wid, "SectTess_U", "XSec_{i + 1}", 10 );')
        parts.append('Update();')
        parts.append('')

    parts.append('//==== Set Airfoils ====//')
    for i, xsec in enumerate(wing.xsecs):
        parts.append(_airfoil_block(i, xsec.airfoil))

    parts += [
        'SetParmVal( wid, "Tess_W", "Shape", 100 );',
        'Update();',
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Fuselage
# --------------------------------------------------------------------------- #


def _fuselage_block(fuselage: Fuselage) -> str:
    if len(fuselage.xsecs) < 2:
        raise ValueError(
            f"Fuselage '{fuselage.name}' must have at least 2 cross-sections."
        )

    x_local = np.array([xs.x for xs in fuselage.xsecs])
    length = float(x_local[-1] - x_local[0])
    if length <= 0.0:
        raise ValueError(
            f"Fuselage '{fuselage.name}' has non-increasing x-stations; "
            "VSP requires monotonic XLocPercent."
        )

    parts = [
        f'//==== Add Fuselage "{fuselage.name}" ====//',
        'string fid = AddGeom("FUSELAGE");',
        'ChangeXSecShape( GetXSecSurf( fid, 0 ), 0, XS_SUPER_ELLIPSE );',
        'ChangeXSecShape( GetXSecSurf( fid, 4 ), 4, XS_SUPER_ELLIPSE );',
        '',
        '//==== Generate Blank Fuselage Sections ====//',
        # Move the trailing default xsec to x/L=1 to avoid collisions while inserting.
        'SetParmVal( fid, "XLocPercent", "XSec_3", 1.0 );',
    ]
    for i in range(len(fuselage.xsecs) - 1):
        parts.append(
            f'InsertXSec( fid, 3, XS_SUPER_ELLIPSE ); // FuselageXSecs {i} to {i + 1}'
        )
    parts += [
        '',
        '//==== Cut The Original Sections ====//',
        'CutXSec( fid, 0 );',
        'CutXSec( fid, 0 );',
        'CutXSec( fid, 0 );',
        'CutXSec( fid, 0 );',
        '',
        f'SetParmVal( fid, "GeomName", "Design", "{fuselage.name}" );',
        f'SetParmVal( fid, "X_Rel_Location", "XForm", {fuselage.x_le + x_local[0]} );',
        f'SetParmVal( fid, "Y_Rel_Location", "XForm", {fuselage.y_le} );',
        f'SetParmVal( fid, "Z_Rel_Location", "XForm", {fuselage.z_le} );',
        f'SetParmVal( fid, "Length", "Design", {length} );',
        'Update();',
        '',
        '//==== Set Fuselage Section Locations ====//',
    ]

    for i, xs in enumerate(fuselage.xsecs):
        x_pct = float((xs.x - x_local[0]) / length)
        parts += [
            f"// FuselageXSec {i}",
            f'SetParmVal( fid, "XLocPercent", "XSec_{i}", {x_pct} );',
            f'SetParmVal( fid, "YLocPercent", "XSec_{i}", 0.0 );',
            f'SetParmVal( fid, "ZLocPercent", "XSec_{i}", 0.0 );',
            f'SetParmVal( fid, "SectTess_U", "XSec_{i}", 10 );',
            f'SetParmVal( fid, "AllSym", "XSec_{i}", 1.0 );',
            'Update();',
            '',
        ]

    parts.append('//==== Set Cross-Section Shapes ====//')
    for i, xs in enumerate(fuselage.xsecs):
        # VSP super-ellipse: |x/W|^M + |y/H|^N = 1. Our shape exponent applies
        # equally to both axes (Super_M = Super_N = shape).
        parts.append(f"""\
{{
    string xsec_surf = GetXSecSurf( fid, {i} );
    string xsec = GetXSec( xsec_surf, {i} );
    SetParmVal( fid, "Super_Width",  "XSecCurve_{i}", {xs.width} );
    SetParmVal( fid, "Super_Height", "XSecCurve_{i}", {xs.height} );
    SetParmVal( fid, "Super_M",      "XSecCurve_{i}", {xs.shape} );
    SetParmVal( fid, "Super_N",      "XSecCurve_{i}", {xs.shape} );
    Update();
}}""")

    parts += [
        'SetParmVal( fid, "Tess_W", "Shape", 100 );',
        'Update();',
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #


def _wrap_main(body: str, aircraft_name: str) -> str:
    """Wrap geometry blocks in a vspscript ``main()`` with API error reporting."""
    epilogue = """\

//==== Check For API Errors ====//
while ( GetNumTotalErrors() > 0 )
{
    ErrorObj err = PopLastError();
    Print( err.GetErrorString() );
}

{
    array<string> @geomids = FindGeoms();
    for (uint i = 0; i < geomids.length(); i++)
    {
        SetGeomDrawType( geomids[i], GEOM_DRAW_SHADE );
    }
}
"""
    inner = body + epilogue
    return f"""\
// AerisPlane → OpenVSP export — aircraft "{aircraft_name}"
// API calls verified against OpenVSP {OPENVSP_VERSION}.
// Open OpenVSP, then File → Run Script… and select this file.

void main()
{{
{indent(inner, "    ")}}}
"""


def _warn_dropped(aircraft: Aircraft) -> None:
    dropped: list[str] = []
    if aircraft.propulsion is not None:
        dropped.append("propulsion")
    if aircraft.payload is not None:
        dropped.append("payload")
    if any(any(xs.spar is not None or xs.skin is not None for xs in w.xsecs)
           for w in aircraft.wings):
        dropped.append("wing structures (spars/skins)")
    if any(w.control_surfaces for w in aircraft.wings):
        dropped.append("control surfaces")
    if dropped:
        warnings.warn(
            "OpenVSP export drops fields the parametric Geom model does not "
            "represent: " + ", ".join(dropped)
        )


def to_openvsp(aircraft: Aircraft, path: str | Path) -> Path:
    """Write *aircraft* as an OpenVSP ``.vspscript`` file.

    Open the file in OpenVSP via *File → Run Script…* — VSP will rebuild the
    wings and fuselages as parametric Geoms.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _warn_dropped(aircraft)

    blocks: list[str] = []
    for wing in aircraft.wings:
        blocks.append("{\n" + indent(_wing_block(wing), "    ") + "\n}")
    for fuselage in aircraft.fuselages:
        blocks.append("{\n" + indent(_fuselage_block(fuselage), "    ") + "\n}")

    out.write_text(_wrap_main("\n\n".join(blocks) + "\n", aircraft.name))
    return out
