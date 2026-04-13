"""VLM draw() example — simple tractor aircraft with wing and horizontal tail.

Run:
    python vlm_draw_example.py               # pyvista interactive window
    python vlm_draw_example.py --plotly      # plotly (opens in browser / Jupyter)
    python vlm_draw_example.py --no-streamlines
"""

import argparse
import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.wing import Wing, WingXSec
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.aero.solvers.vlm import VortexLatticeMethod


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Airfoils
naca2412 = Airfoil("naca2412")   # cambered — main wing
naca0009 = Airfoil("naca0009")   # symmetric — tail

# Main wing  (2.4 m span, 0.25 m root chord, mild taper and sweep)
#
#   LE sweep ≈ 10°  |  AR ≈ 8  |  taper ratio ≈ 0.6
#
main_wing = Wing(
    name="main_wing",
    symmetric=True,
    xsecs=[
        WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=naca2412),
        WingXSec(xyz_le=[0.04, 0.60, 0.00], chord=0.24, airfoil=naca2412),
        WingXSec(xyz_le=[0.13, 1.20, 0.00], chord=0.17, airfoil=naca2412),
    ],
)

# Horizontal tail  (0.8 m span, set 1.4 m aft, 0.15 m above wing plane)
#
#   Symmetric NACA 0009  |  small incidence (−2°) for trim
#
htail = Wing(
    name="htail",
    symmetric=True,
    xsecs=[
        WingXSec(xyz_le=[1.40, 0.00, 0.15], chord=0.14, twist=-2.0, airfoil=naca0009),
        WingXSec(xyz_le=[1.44, 0.40, 0.15], chord=0.11, twist=-2.0, airfoil=naca0009),
    ],
)

aircraft = Aircraft(
    name="SimpleTrainer",
    wings=[main_wing, htail],
    xyz_ref=[0.07, 0.0, 0.0],   # approx CG at 25 % MAC
)

# ---------------------------------------------------------------------------
# Flight condition
# ---------------------------------------------------------------------------

condition = FlightCondition(
    velocity=28.0,    # m/s  ≈ 100 km/h
    altitude=500.0,   # m
    alpha=4.0,        # deg AoA
)

# ---------------------------------------------------------------------------
# VLM solve
# ---------------------------------------------------------------------------

vlm = VortexLatticeMethod(
    aircraft=aircraft,
    condition=condition,
    spanwise_resolution=8,
    chordwise_resolution=4,
    verbose=True,
)

result = vlm.run()

print(f"\n{'─'*40}")
print(f"  CL  = {result['CL']:.4f}")
print(f"  CD  = {result['CD']:.5f}")
print(f"  Cm  = {result['Cm']:.4f}")
print(f"{'─'*40}\n")

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--plotly", action="store_true")
parser.add_argument("--no-streamlines", action="store_true")
args = parser.parse_args()

vlm.draw(
    backend="plotly" if args.plotly else "pyvista",
    draw_streamlines=not args.no_streamlines,
)
