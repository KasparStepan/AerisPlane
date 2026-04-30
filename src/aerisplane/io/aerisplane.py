"""Native AerisPlane JSON format — full :class:`Aircraft` round-trip.

File extension is ``.apl.json``. The schema is versioned (``"version": 1``);
older files keep loading as the schema grows.

Conventions
-----------
- Lengths in metres, angles in degrees, mass in kg.
- Aircraft frame: x aft, y starboard, z up (same as ``core/``).
- Airfoils, materials, and servos are deduplicated by name into a top-level
  ``embedded`` block; the aircraft tree references them as ``{"ref": name}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface, Servo
from aerisplane.core.fuselage import Fuselage, FuselageXSec
from aerisplane.core.payload import Payload
from aerisplane.core.propulsion import (
    ESC,
    Battery,
    Motor,
    Propeller,
    PropellerPerfData,
    PropulsionSystem,
)
from aerisplane.core.structures import Material, Skin, Spar, TubeSection
from aerisplane.core.wing import Wing, WingXSec

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #


class _Encoder:
    """Walk an Aircraft tree, collecting referenced objects into registries."""

    def __init__(self) -> None:
        self.airfoils: dict[str, dict] = {}
        self.materials: dict[str, dict] = {}
        self.servos: dict[str, dict] = {}

    # ----- ref-collected types ----- #

    def airfoil_ref(self, af: Airfoil) -> dict:
        if af.name not in self.airfoils:
            entry: dict = {}
            if af.coordinates is not None:
                entry["coordinates"] = af.coordinates.tolist()
            self.airfoils[af.name] = entry
        return {"ref": af.name}

    def material_ref(self, mat: Material) -> dict:
        if mat.name not in self.materials:
            self.materials[mat.name] = {
                "density": mat.density,
                "E": mat.E,
                "yield_strength": mat.yield_strength,
                "poisson_ratio": mat.poisson_ratio,
                "shear_modulus": mat.shear_modulus,
            }
        return {"ref": mat.name}

    def servo_ref(self, servo: Servo) -> dict:
        if servo.name not in self.servos:
            self.servos[servo.name] = {
                "torque": servo.torque,
                "speed": servo.speed,
                "voltage": servo.voltage,
                "mass": servo.mass,
            }
        return {"ref": servo.name}

    # ----- inline types ----- #

    def tube_section(self, ts: TubeSection) -> dict:
        return {
            "outer_diameter": ts.outer_diameter,
            "wall_thickness": ts.wall_thickness,
        }

    def spar(self, sp: Spar) -> dict:
        return {
            "position": sp.position,
            "material": self.material_ref(sp.material),
            "section": self.tube_section(sp.section),
        }

    def skin(self, sk: Skin) -> dict:
        return {
            "material": self.material_ref(sk.material),
            "thickness": sk.thickness,
        }

    def wing_xsec(self, x: WingXSec) -> dict:
        return {
            "xyz_le": np.asarray(x.xyz_le, dtype=float).tolist(),
            "chord": x.chord,
            "twist": x.twist,
            "airfoil": self.airfoil_ref(x.airfoil) if x.airfoil is not None else None,
            "spar": self.spar(x.spar) if x.spar is not None else None,
            "skin": self.skin(x.skin) if x.skin is not None else None,
        }

    def control_surface(self, cs: ControlSurface) -> dict:
        return {
            "name": cs.name,
            "span_start": cs.span_start,
            "span_end": cs.span_end,
            "chord_fraction": cs.chord_fraction,
            "max_deflection": cs.max_deflection,
            "min_deflection": cs.min_deflection,
            "symmetric": cs.symmetric,
            "servo": self.servo_ref(cs.servo) if cs.servo is not None else None,
        }

    def wing(self, w: Wing) -> dict:
        return {
            "name": w.name,
            "symmetric": w.symmetric,
            "color": w.color,
            "xsecs": [self.wing_xsec(x) for x in w.xsecs],
            "control_surfaces": [self.control_surface(cs) for cs in w.control_surfaces],
        }

    def fuselage_xsec(self, x: FuselageXSec) -> dict:
        return {
            "x": x.x,
            "width": x.width,
            "height": x.height,
            "shape": x.shape,
        }

    def fuselage(self, f: Fuselage) -> dict:
        return {
            "name": f.name,
            "x_le": f.x_le,
            "y_le": f.y_le,
            "z_le": f.z_le,
            "wall_thickness": f.wall_thickness,
            "color": f.color,
            "material": self.material_ref(f.material) if f.material is not None else None,
            "xsecs": [self.fuselage_xsec(x) for x in f.xsecs],
        }

    def motor(self, m: Motor) -> dict:
        return {
            "name": m.name,
            "kv": m.kv,
            "resistance": m.resistance,
            "no_load_current": m.no_load_current,
            "max_current": m.max_current,
            "mass": m.mass,
        }

    def propeller_perf(self, pd: PropellerPerfData) -> dict:
        return {
            "J": np.asarray(pd.J, dtype=float).tolist(),
            "CT": np.asarray(pd.CT, dtype=float).tolist(),
            "CP": np.asarray(pd.CP, dtype=float).tolist(),
            "source": pd.source,
        }

    def propeller(self, p: Propeller) -> dict:
        return {
            "diameter": p.diameter,
            "pitch": p.pitch,
            "mass": p.mass,
            "num_blades": p.num_blades,
            "performance_data": (
                self.propeller_perf(p.performance_data)
                if p.performance_data is not None
                else None
            ),
        }

    def battery(self, b: Battery) -> dict:
        return {
            "name": b.name,
            "capacity_ah": b.capacity_ah,
            "nominal_voltage": b.nominal_voltage,
            "cell_count": b.cell_count,
            "c_rating": b.c_rating,
            "mass": b.mass,
            "internal_resistance": b.internal_resistance,
        }

    def esc(self, e: ESC) -> dict:
        return {
            "name": e.name,
            "max_current": e.max_current,
            "mass": e.mass,
            "has_telemetry": e.has_telemetry,
        }

    def propulsion(self, ps: PropulsionSystem) -> dict:
        return {
            "motor": self.motor(ps.motor),
            "propeller": self.propeller(ps.propeller),
            "battery": self.battery(ps.battery),
            "esc": self.esc(ps.esc),
            "position": np.asarray(ps.position, dtype=float).tolist(),
            "direction": np.asarray(ps.direction, dtype=float).tolist(),
        }

    def payload(self, p: Payload) -> dict:
        return {
            "name": p.name,
            "mass": p.mass,
            "cg": np.asarray(p.cg, dtype=float).tolist(),
        }

    def aircraft(self, ac: Aircraft) -> dict:
        return {
            "name": ac.name,
            "xyz_ref": list(map(float, ac.xyz_ref)),
            "wings": [self.wing(w) for w in ac.wings],
            "fuselages": [self.fuselage(f) for f in ac.fuselages],
            "propulsion": self.propulsion(ac.propulsion) if ac.propulsion is not None else None,
            "payload": self.payload(ac.payload) if ac.payload is not None else None,
        }


def save_aircraft(aircraft: Aircraft, path: str | Path, *, indent: int = 2) -> Path:
    """Write *aircraft* to a versioned ``.apl.json`` file.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft to serialize.
    path : str or Path
        Destination file. Parent directory must exist.
    indent : int
        JSON indent. Default 2 (human-readable). Pass ``None`` for compact.
    """
    enc = _Encoder()
    document = {
        "format": "aerisplane",
        "version": SCHEMA_VERSION,
        "aircraft": enc.aircraft(aircraft),
        "embedded": {
            "airfoils": enc.airfoils,
            "materials": enc.materials,
            "servos": enc.servos,
        },
    }
    if not path.endswith(".apl.json"):
        path += ".apl.json"
    out = Path(path)
    out.write_text(json.dumps(document, indent=indent))
    return out


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #


class _Decoder:
    """Resolve refs in a parsed document and rebuild an Aircraft tree."""

    def __init__(self, embedded: dict) -> None:
        self._airfoil_data = embedded.get("airfoils", {})
        self._material_data = embedded.get("materials", {})
        self._servo_data = embedded.get("servos", {})
        self._airfoil_cache: dict[str, Airfoil] = {}
        self._material_cache: dict[str, Material] = {}
        self._servo_cache: dict[str, Servo] = {}

    def airfoil(self, ref: dict) -> Airfoil:
        name = ref["ref"]
        if name in self._airfoil_cache:
            return self._airfoil_cache[name]

        entry = self._airfoil_data.get(name, {})
        coords = entry.get("coordinates")
        if coords is not None:
            af = Airfoil(name=name, coordinates=np.asarray(coords, dtype=float))
        else:
            af = Airfoil(name=name)
        self._airfoil_cache[name] = af
        return af

    def material(self, ref: dict) -> Material:
        name = ref["ref"]
        if name in self._material_cache:
            return self._material_cache[name]
        d = self._material_data[name]
        mat = Material(
            name=name,
            density=d["density"],
            E=d["E"],
            yield_strength=d["yield_strength"],
            poisson_ratio=d.get("poisson_ratio", 0.3),
            shear_modulus=d.get("shear_modulus"),
        )
        self._material_cache[name] = mat
        return mat

    def servo(self, ref: dict) -> Servo:
        name = ref["ref"]
        if name in self._servo_cache:
            return self._servo_cache[name]
        d = self._servo_data[name]
        s = Servo(
            name=name,
            torque=d["torque"],
            speed=d["speed"],
            voltage=d["voltage"],
            mass=d["mass"],
        )
        self._servo_cache[name] = s
        return s

    def spar(self, d: dict) -> Spar:
        return Spar(
            position=d["position"],
            material=self.material(d["material"]),
            section=TubeSection(
                outer_diameter=d["section"]["outer_diameter"],
                wall_thickness=d["section"]["wall_thickness"],
            ),
        )

    def skin(self, d: dict) -> Skin:
        return Skin(material=self.material(d["material"]), thickness=d["thickness"])

    def wing_xsec(self, d: dict) -> WingXSec:
        return WingXSec(
            xyz_le=np.asarray(d["xyz_le"], dtype=float),
            chord=d["chord"],
            twist=d.get("twist", 0.0),
            airfoil=self.airfoil(d["airfoil"]) if d.get("airfoil") else None,
            spar=self.spar(d["spar"]) if d.get("spar") else None,
            skin=self.skin(d["skin"]) if d.get("skin") else None,
        )

    def control_surface(self, d: dict) -> ControlSurface:
        return ControlSurface(
            name=d["name"],
            span_start=d["span_start"],
            span_end=d["span_end"],
            chord_fraction=d["chord_fraction"],
            max_deflection=d.get("max_deflection", 25.0),
            min_deflection=d.get("min_deflection", -25.0),
            symmetric=d.get("symmetric", True),
            servo=self.servo(d["servo"]) if d.get("servo") else None,
        )

    def wing(self, d: dict) -> Wing:
        return Wing(
            name=d["name"],
            symmetric=d.get("symmetric", True),
            color=d.get("color"),
            xsecs=[self.wing_xsec(x) for x in d["xsecs"]],
            control_surfaces=[self.control_surface(cs) for cs in d.get("control_surfaces", [])],
        )

    def fuselage_xsec(self, d: dict) -> FuselageXSec:
        return FuselageXSec(
            x=d["x"],
            width=d.get("width", 0.0),
            height=d.get("height", 0.0),
            shape=d.get("shape", 2.0),
        )

    def fuselage(self, d: dict) -> Fuselage:
        return Fuselage(
            name=d["name"],
            xsecs=[self.fuselage_xsec(x) for x in d["xsecs"]],
            x_le=d.get("x_le", 0.0),
            y_le=d.get("y_le", 0.0),
            z_le=d.get("z_le", 0.0),
            wall_thickness=d.get("wall_thickness", 0.001),
            color=d.get("color"),
            material=self.material(d["material"]) if d.get("material") else None,
        )

    def motor(self, d: dict) -> Motor:
        return Motor(
            name=d["name"], kv=d["kv"], resistance=d["resistance"],
            no_load_current=d["no_load_current"], max_current=d["max_current"],
            mass=d["mass"],
        )

    def propeller(self, d: dict) -> Propeller:
        perf = d.get("performance_data")
        return Propeller(
            diameter=d["diameter"], pitch=d["pitch"], mass=d.get("mass", 0.03),
            num_blades=d.get("num_blades", 2),
            performance_data=(
                PropellerPerfData(
                    J=np.asarray(perf["J"], dtype=float),
                    CT=np.asarray(perf["CT"], dtype=float),
                    CP=np.asarray(perf["CP"], dtype=float),
                    source=perf.get("source", ""),
                )
                if perf is not None
                else None
            ),
        )

    def battery(self, d: dict) -> Battery:
        return Battery(
            name=d["name"], capacity_ah=d["capacity_ah"],
            nominal_voltage=d["nominal_voltage"], cell_count=d["cell_count"],
            c_rating=d["c_rating"], mass=d["mass"],
            internal_resistance=d.get("internal_resistance", 0.0),
        )

    def esc(self, d: dict) -> ESC:
        return ESC(
            name=d["name"], max_current=d["max_current"], mass=d["mass"],
            has_telemetry=d.get("has_telemetry", False),
        )

    def propulsion(self, d: dict) -> PropulsionSystem:
        return PropulsionSystem(
            motor=self.motor(d["motor"]),
            propeller=self.propeller(d["propeller"]),
            battery=self.battery(d["battery"]),
            esc=self.esc(d["esc"]),
            position=np.asarray(d.get("position", [0.0, 0.0, 0.0]), dtype=float),
            direction=np.asarray(d.get("direction", [-1.0, 0.0, 0.0]), dtype=float),
        )

    def payload(self, d: dict) -> Payload:
        return Payload(
            name=d.get("name", "payload"),
            mass=d["mass"],
            cg=np.asarray(d.get("cg", [0.0, 0.0, 0.0]), dtype=float),
        )

    def aircraft(self, d: dict) -> Aircraft:
        return Aircraft(
            name=d["name"],
            wings=[self.wing(w) for w in d.get("wings", [])],
            fuselages=[self.fuselage(f) for f in d.get("fuselages", [])],
            propulsion=self.propulsion(d["propulsion"]) if d.get("propulsion") else None,
            payload=self.payload(d["payload"]) if d.get("payload") else None,
            xyz_ref=[float(v) for v in d.get("xyz_ref", [0.0, 0.0, 0.0])],
        )


def load_aircraft(path: str | Path) -> Aircraft:
    """Load an aircraft from a ``.apl.json`` file written by :func:`save_aircraft`."""
    document: dict[str, Any] = json.loads(Path(path).read_text())

    fmt = document.get("format")
    if fmt != "aerisplane":
        raise ValueError(
            f"{path}: not an AerisPlane file (format={fmt!r}); expected 'aerisplane'."
        )

    version = document.get("version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported AerisPlane schema version {version}; "
            f"this build understands version {SCHEMA_VERSION}."
        )

    decoder = _Decoder(document.get("embedded", {}))
    return decoder.aircraft(document["aircraft"])
