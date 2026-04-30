"""AVL (Athena Vortex Lattice) file format — geometry-only round-trip.

AVL covers lifting surfaces (SURFACE/SECTION/CONTROL) and simple axisymmetric
bodies (BODY/BFILE). Anything outside that — propulsion, structures, payload,
servos, non-circular fuselage cross-sections — is dropped on export and stays
unset on import. A warning is emitted for each category that gets lost.

Reference: Mark Drela & Harold Youngren, *AVL User Primer* (avl_doc.txt).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface
from aerisplane.core.fuselage import Fuselage, FuselageXSec
from aerisplane.core.wing import Wing, WingXSec
from aerisplane.io._common import resolve_airfoil

# AVL keywords that introduce a new block or modifier. Used by the parser to
# detect when a multi-line data block (e.g. AIRFOIL coordinates) ends.
_KEYWORDS = {
    "SURFACE", "BODY", "SECTION", "CONTROL", "DESIGN",
    "NACA", "AFILE", "AIRFOIL", "BFILE",
    "YDUPLICATE", "YDUP", "SCALE", "TRANSLATE", "ANGLE",
    "INDEX", "COMPONENT", "CLAF", "CDCL",
    "NOWAKE", "NOALBE", "NOLOAD",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strip_comments(raw: str) -> str:
    """Remove AVL comments (after `#` or `!`) and surrounding whitespace."""
    for ch in "#!":
        idx = raw.find(ch)
        if idx >= 0:
            raw = raw[:idx]
    return raw.strip()


def _clean_lines(text: str) -> list[str]:
    """Return non-blank, comment-free lines from an AVL file body."""
    return [s for s in (_strip_comments(line) for line in text.splitlines()) if s]


def _is_keyword(line: str) -> bool:
    return bool(line) and line.split()[0].upper() in _KEYWORDS


def _floats(line: str, n: int | None = None) -> list[float]:
    parts = line.split()
    if n is not None and len(parts) < n:
        raise ValueError(f"AVL: expected {n} numbers, got {len(parts)}: {line!r}")
    return [float(p) for p in (parts[:n] if n else parts)]


@dataclass
class _Cursor:
    """Line-cursor over the cleaned AVL token stream."""

    lines: list[str]
    i: int = 0

    def at_end(self) -> bool:
        return self.i >= len(self.lines)

    def peek(self) -> str | None:
        return None if self.at_end() else self.lines[self.i]

    def take(self) -> str:
        line = self.lines[self.i]
        self.i += 1
        return line


# --------------------------------------------------------------------------- #
# Parser internal state
# --------------------------------------------------------------------------- #


@dataclass
class _AvlSection:
    """Raw SECTION data; resolved into a WingXSec after the surface is known."""

    xle: float
    yle: float
    zle: float
    chord: float
    ainc: float = 0.0
    airfoil_name: str | None = None
    airfoil_coords: np.ndarray | None = None
    controls: list[tuple[str, float, float, float]] = field(default_factory=list)
    """Each entry: (name, gain, xhinge, sgn_dup)."""


@dataclass
class _AvlSurface:
    name: str
    sections: list[_AvlSection] = field(default_factory=list)
    yduplicate: float | None = None
    translate: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    angle: float = 0.0


@dataclass
class _AvlBody:
    name: str
    translate: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    bfile: Path | None = None


@dataclass
class _AvlDocument:
    title: str = ""
    xref: float = 0.0
    yref: float = 0.0
    zref: float = 0.0
    surfaces: list[_AvlSurface] = field(default_factory=list)
    bodies: list[_AvlBody] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _parse_header(cur: _Cursor) -> _AvlDocument:
    doc = _AvlDocument()
    doc.title = cur.take()
    _floats(cur.take(), 1)                                                  # Mach
    _floats(cur.take(), 3)                                                  # iYsym iZsym Zsym
    _floats(cur.take(), 3)                                                  # Sref Cref Bref
    xref, yref, zref = _floats(cur.take(), 3)
    doc.xref, doc.yref, doc.zref = xref, yref, zref
    # Optional CDp line: present if the next token is a number, not a keyword.
    nxt = cur.peek()
    if nxt is not None and not _is_keyword(nxt):
        try:
            _floats(nxt, 1)
            cur.take()
        except ValueError:
            pass
    return doc


def _parse_section(cur: _Cursor) -> _AvlSection:
    nums = _floats(cur.take())
    if len(nums) < 5:
        raise ValueError(f"AVL: SECTION needs 5+ numbers, got {nums}")
    sec = _AvlSection(xle=nums[0], yle=nums[1], zle=nums[2], chord=nums[3], ainc=nums[4])

    while not cur.at_end():
        nxt = cur.peek()
        if not _is_keyword(nxt):
            break
        kw = nxt.split()[0].upper()
        if kw in {"SECTION", "SURFACE", "BODY"}:
            break

        cur.take()
        if kw == "NACA":
            digits = cur.take().split()[0]
            sec.airfoil_name = f"naca{digits}"
        elif kw == "AFILE":
            sec.airfoil_name = cur.take().split()[0]
        elif kw == "AIRFOIL":
            coords = []
            while not cur.at_end() and not _is_keyword(cur.peek()):
                coords.append(_floats(cur.take(), 2))
            sec.airfoil_coords = np.array(coords, dtype=float) if coords else None
        elif kw == "CONTROL":
            parts = cur.take().split()
            if len(parts) < 6:
                raise ValueError(f"AVL: CONTROL needs 6 fields, got {parts}")
            name, gain, xhinge = parts[0], float(parts[1]), float(parts[2])
            sgn_dup = float(parts[6]) if len(parts) >= 7 else 1.0
            sec.controls.append((name, gain, xhinge, sgn_dup))
        elif kw in {"CLAF", "CDCL", "DESIGN"}:
            cur.take()                                                       # discard data
        elif kw in {"NOWAKE", "NOALBE", "NOLOAD"}:
            pass                                                             # flag-only
        else:
            warnings.warn(f"AVL: skipping unknown keyword in SECTION: {kw}")
    return sec


def _parse_surface(cur: _Cursor) -> _AvlSurface:
    surf = _AvlSurface(name=cur.take())
    cur.take()                                                              # discretization line

    while not cur.at_end():
        nxt = cur.peek()
        if not _is_keyword(nxt):
            break
        kw = nxt.split()[0].upper()
        if kw in {"SURFACE", "BODY"}:
            break

        cur.take()
        if kw == "SECTION":
            surf.sections.append(_parse_section(cur))
        elif kw in {"YDUPLICATE", "YDUP"}:
            surf.yduplicate = _floats(cur.take(), 1)[0]
        elif kw == "SCALE":
            surf.scale = np.array(_floats(cur.take(), 3))
        elif kw == "TRANSLATE":
            surf.translate = np.array(_floats(cur.take(), 3))
        elif kw == "ANGLE":
            surf.angle = _floats(cur.take(), 1)[0]
        elif kw in {"INDEX", "COMPONENT"}:
            cur.take()                                                       # one int
        elif kw in {"CLAF", "CDCL", "DESIGN"}:
            cur.take()
        elif kw in {"NOWAKE", "NOALBE", "NOLOAD"}:
            pass
        else:
            warnings.warn(f"AVL: skipping unknown keyword in SURFACE: {kw}")
    return surf


def _parse_body(cur: _Cursor, source_dir: Path) -> _AvlBody:
    body = _AvlBody(name=cur.take())
    cur.take()                                                              # Nbody Bspace

    while not cur.at_end():
        nxt = cur.peek()
        if not _is_keyword(nxt):
            break
        kw = nxt.split()[0].upper()
        if kw in {"SURFACE", "BODY"}:
            break

        cur.take()
        if kw == "BFILE":
            body.bfile = source_dir / cur.take().split()[0]
        elif kw in {"YDUPLICATE", "YDUP"}:
            cur.take()
        elif kw == "SCALE":
            body.scale = np.array(_floats(cur.take(), 3))
        elif kw == "TRANSLATE":
            body.translate = np.array(_floats(cur.take(), 3))
        elif kw in {"NOWAKE", "NOALBE", "NOLOAD"}:
            pass
        else:
            warnings.warn(f"AVL: skipping unknown keyword in BODY: {kw}")
    return body


def _read_bfile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an AVL .bfi body profile into (x, radius) arrays."""
    lines = _clean_lines(path.read_text())
    pairs = []
    for ln in lines:
        try:
            xy = _floats(ln, 2)
        except ValueError:
            continue                                                        # title line
        pairs.append(xy)
    arr = np.array(pairs, dtype=float)
    return arr[:, 0], np.abs(arr[:, 1])


# --------------------------------------------------------------------------- #
# AVL → Aircraft assembly
# --------------------------------------------------------------------------- #


def _build_xsec(sec: _AvlSection, surf: _AvlSurface, source_dir: Path) -> WingXSec:
    le = surf.translate + np.array([sec.xle, sec.yle, sec.zle]) * surf.scale
    chord = sec.chord * surf.scale[0]
    twist = sec.ainc + surf.angle

    if sec.airfoil_coords is not None:
        airfoil = Airfoil(name=f"{surf.name}_section", coordinates=sec.airfoil_coords)
    elif sec.airfoil_name is not None:
        airfoil = resolve_airfoil(sec.airfoil_name, search_dir=source_dir)
    else:
        airfoil = None

    return WingXSec(xyz_le=le, chord=chord, twist=twist, airfoil=airfoil)


def _build_control_surfaces(surf: _AvlSurface, wing: Wing) -> list[ControlSurface]:
    """Aggregate per-section CONTROL declarations into one entry per surface name."""
    if not wing.xsecs or not any(sec.controls for sec in surf.sections):
        return []

    semispan = wing.semispan()
    if semispan == 0.0:
        return []

    y_stations = np.array([xsec.xyz_le[1] for xsec in wing.xsecs])
    z_stations = np.array([xsec.xyz_le[2] for xsec in wing.xsecs])
    use_z = (np.max(z_stations) - np.min(z_stations)) > (np.max(y_stations) - np.min(y_stations))
    span_coords = z_stations - z_stations[0] if use_z else y_stations - y_stations.min()

    grouped: dict[str, list[tuple[int, float, float]]] = {}
    for i, sec in enumerate(surf.sections):
        for name, _gain, xhinge, sgn_dup in sec.controls:
            grouped.setdefault(name, []).append((i, xhinge, sgn_dup))

    out: list[ControlSurface] = []
    for name, hits in grouped.items():
        idxs = [h[0] for h in hits]
        span_start = float(span_coords[min(idxs)] / semispan)
        span_end = float(span_coords[max(idxs)] / semispan)
        if span_end <= span_start:
            warnings.warn(
                f"AVL: control '{name}' on '{surf.name}' has zero span; skipping."
            )
            continue
        chord_fraction = 1.0 - float(np.mean([h[1] for h in hits]))
        symmetric = float(np.mean([h[2] for h in hits])) >= 0.0
        out.append(ControlSurface(
            name=name,
            span_start=span_start,
            span_end=span_end,
            chord_fraction=chord_fraction,
            symmetric=symmetric,
        ))
    return out


def _build_wing(surf: _AvlSurface, source_dir: Path) -> Wing:
    xsecs = [_build_xsec(sec, surf, source_dir) for sec in surf.sections]
    wing = Wing(
        name=surf.name,
        xsecs=xsecs,
        symmetric=surf.yduplicate is not None,
    )
    wing.control_surfaces = _build_control_surfaces(surf, wing)
    return wing


def _build_fuselage(body: _AvlBody) -> Fuselage | None:
    if body.bfile is None or not body.bfile.exists():
        warnings.warn(f"AVL: BODY '{body.name}' has no readable BFILE; skipping.")
        return None
    x, r = _read_bfile(body.bfile)
    xsecs = [
        FuselageXSec(x=float(xi), width=2.0 * float(ri), height=2.0 * float(ri), shape=2.0)
        for xi, ri in zip(x, r)
    ]
    return Fuselage(
        name=body.name,
        xsecs=xsecs,
        x_le=float(body.translate[0]),
        y_le=float(body.translate[1]),
        z_le=float(body.translate[2]),
    )


def from_avl(path: str | Path) -> Aircraft:
    """Load an aircraft from an AVL ``.avl`` file.

    Sidecar airfoil ``.dat`` files (referenced by ``AFILE``) are searched in
    the same directory as the AVL file. NACA 4-digit profiles are generated
    analytically.

    Lossy on import: AVL has no propulsion, structures, payload, servos, or
    non-circular fuselage cross-sections. Those fields stay at their defaults.
    """
    p = Path(path)
    cur = _Cursor(_clean_lines(p.read_text()))
    doc = _parse_header(cur)

    while not cur.at_end():
        kw = cur.peek().split()[0].upper()
        cur.take()
        if kw == "SURFACE":
            doc.surfaces.append(_parse_surface(cur))
        elif kw == "BODY":
            doc.bodies.append(_parse_body(cur, source_dir=p.parent))
        else:
            raise ValueError(f"AVL: unexpected top-level keyword {kw!r}")

    wings = [_build_wing(s, source_dir=p.parent) for s in doc.surfaces]
    fuselages = [f for f in (_build_fuselage(b) for b in doc.bodies) if f is not None]

    return Aircraft(
        name=doc.title or p.stem,
        wings=wings,
        fuselages=fuselages,
        xyz_ref=[doc.xref, doc.yref, doc.zref],
    )


# --------------------------------------------------------------------------- #
# Aircraft → AVL writer
# --------------------------------------------------------------------------- #


def _airfoil_avl_block(airfoil: Airfoil | None, sidecar_dir: Path) -> list[str]:
    """Emit the per-section airfoil lines (NACA or AFILE)."""
    if airfoil is None:
        return []

    name = airfoil.name.lower()
    if name.startswith("naca"):
        digits = name[4:].strip()
        if len(digits) == 4 and digits.isdigit():
            return ["NACA", digits]

    if airfoil.coordinates is None:
        warnings.warn(
            f"AVL: airfoil '{airfoil.name}' has no coordinates and no NACA name; "
            "leaving section without an airfoil reference."
        )
        return []

    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in airfoil.name)
    dat_path = sidecar_dir / f"{safe}.dat"
    if not dat_path.exists():
        lines = [airfoil.name]
        for x, y in airfoil.coordinates:
            lines.append(f"{x:>10.6f} {y:>10.6f}")
        dat_path.write_text("\n".join(lines) + "\n")
    return ["AFILE", dat_path.name]


def _control_lines_for_section(
    wing: Wing,
    section_idx: int,
    span_coords: np.ndarray,
    semispan: float,
) -> list[str]:
    """Emit CONTROL lines for any control surface that covers this section."""
    out: list[str] = []
    if semispan == 0.0:
        return out
    y_here = float(span_coords[section_idx])
    for cs in wing.control_surfaces:
        y0 = cs.span_start * semispan
        y1 = cs.span_end * semispan
        if y0 - 1e-9 <= y_here <= y1 + 1e-9:
            xhinge = 1.0 - cs.chord_fraction
            sgn_dup = 1.0 if cs.symmetric else -1.0
            out.append("CONTROL")
            out.append(f"{cs.name}  1.0  {xhinge:.4f}  0.0 0.0 0.0  {sgn_dup:+.1f}")
    return out


def _surface_block(wing: Wing, sidecar_dir: Path) -> list[str]:
    lines = ["", "#" + "-" * 70, "SURFACE", wing.name, "12  1.0  20  -1.5"]
    if wing.symmetric:
        lines += ["YDUPLICATE", "0.0"]

    y_stations = np.array([xsec.xyz_le[1] for xsec in wing.xsecs])
    z_stations = np.array([xsec.xyz_le[2] for xsec in wing.xsecs])
    use_z = (np.max(z_stations) - np.min(z_stations)) > (np.max(y_stations) - np.min(y_stations))
    span_coords = z_stations - z_stations[0] if use_z else y_stations - y_stations.min()
    semispan = float(np.max(span_coords) - np.min(span_coords))

    for i, xsec in enumerate(wing.xsecs):
        x, y, z = xsec.xyz_le
        lines += [
            "SECTION",
            f"{x:.6f}  {y:.6f}  {z:.6f}  {xsec.chord:.6f}  {xsec.twist:.4f}",
        ]
        lines += _airfoil_avl_block(xsec.airfoil, sidecar_dir)
        lines += _control_lines_for_section(wing, i, span_coords, semispan)
    return lines


def _body_block(fuselage: Fuselage, sidecar_dir: Path) -> list[str]:
    """Emit a BODY referencing an axisymmetric BFILE generated from the fuselage."""
    if not fuselage.xsecs:
        return []

    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in fuselage.name)
    bfile = sidecar_dir / f"{safe}.bfi"
    body_lines = [fuselage.name]
    for xs in fuselage.xsecs:
        body_lines.append(f"{xs.x:>10.6f} {xs.equivalent_radius():>10.6f}")
    bfile.write_text("\n".join(body_lines) + "\n")

    return [
        "",
        "#" + "-" * 70,
        "BODY",
        fuselage.name,
        "20  1.0",
        "TRANSLATE",
        f"{fuselage.x_le:.6f}  {fuselage.y_le:.6f}  {fuselage.z_le:.6f}",
        "BFILE",
        bfile.name,
    ]


def _warn_dropped(aircraft: Aircraft) -> None:
    dropped: list[str] = []
    if aircraft.propulsion is not None:
        dropped.append("propulsion")
    if aircraft.payload is not None:
        dropped.append("payload")
    if any(any(xs.spar is not None or xs.skin is not None for xs in w.xsecs)
           for w in aircraft.wings):
        dropped.append("wing structures (spars/skins)")
    if any(w.control_surfaces and any(cs.servo is not None for cs in w.control_surfaces)
           for w in aircraft.wings):
        dropped.append("servo specs")
    if dropped:
        warnings.warn(
            "AVL export drops fields the format does not represent: "
            + ", ".join(dropped)
        )


def to_avl(aircraft: Aircraft, path: str | Path) -> Path:
    """Write *aircraft* to an AVL ``.avl`` file plus any required sidecars.

    Sidecar files written next to the AVL file:

    - ``<airfoil>.dat`` — Selig-format airfoil for any non-NACA section.
    - ``<fuselage>.bfi`` — axisymmetric body profile per fuselage.
    """
    out = Path(path)
    sidecar_dir = out.parent
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    _warn_dropped(aircraft)

    main = aircraft.main_wing()
    sref = main.area() if main is not None else 1.0
    cref = main.mean_aerodynamic_chord() if main is not None else 1.0
    bref = main.span() if main is not None else 1.0
    xref, yref, zref = aircraft.xyz_ref

    lines = [
        aircraft.name,
        "#Mach",
        "0.0",
        "#IYsym  IZsym  Zsym",
        "0  0  0.0",
        "#Sref  Cref  Bref",
        f"{sref:.6f}  {cref:.6f}  {bref:.6f}",
        "#Xref  Yref  Zref",
        f"{xref:.6f}  {yref:.6f}  {zref:.6f}",
    ]
    for wing in aircraft.wings:
        lines += _surface_block(wing, sidecar_dir)
    for fuse in aircraft.fuselages:
        lines += _body_block(fuse, sidecar_dir)

    out.write_text("\n".join(lines) + "\n")
    return out
