"""Propeller catalog entries. All diameters and pitches converted from inches to meters."""
from aerisplane.core.propulsion import Propeller

_IN = 0.0254  # meters per inch

apc_10x4_7sf = Propeller(diameter=10 * _IN, pitch=4.7 * _IN, mass=0.018, num_blades=2)
apc_11x4_7sf = Propeller(diameter=11 * _IN, pitch=4.7 * _IN, mass=0.021, num_blades=2)
apc_13x4_7sf = Propeller(diameter=13 * _IN, pitch=4.7 * _IN, mass=0.030, num_blades=2)
apc_10x7e = Propeller(diameter=10 * _IN, pitch=7.0 * _IN, mass=0.019, num_blades=2)
apc_12x6e = Propeller(diameter=12 * _IN, pitch=6.0 * _IN, mass=0.027, num_blades=2)
apc_14x8_3mf = Propeller(diameter=14 * _IN, pitch=8.3 * _IN, mass=0.040, num_blades=2)
master_airscrew_10x5 = Propeller(diameter=10 * _IN, pitch=5.0 * _IN, mass=0.020, num_blades=2)
master_airscrew_11x7 = Propeller(diameter=11 * _IN, pitch=7.0 * _IN, mass=0.026, num_blades=2)
master_airscrew_14x7 = Propeller(diameter=14 * _IN, pitch=7.0 * _IN, mass=0.042, num_blades=2)
tjd_14x8_5 = Propeller(diameter=14 * _IN, pitch=8.5 * _IN, mass=0.022, num_blades=2)
