"""Lifting-line draw() example — trainer aircraft from the catalog.

Run:
    python ll_draw_example.py               # pyvista interactive window
    python ll_draw_example.py --plotly      # plotly (opens in browser / Jupyter)
    python ll_draw_example.py --no-streamlines
"""

import argparse

from aerisplane.catalog import get_aircraft
from aerisplane.catalog.aircraft import trainer_condition
from aerisplane.aero.solvers.lifting_line import LiftingLine


# ---------------------------------------------------------------------------
# Aircraft + flight condition from catalog
# ---------------------------------------------------------------------------

aircraft = get_aircraft("trainer")
condition = trainer_condition()

# ---------------------------------------------------------------------------
# Lifting-line solve
# ---------------------------------------------------------------------------

ll = LiftingLine(
    aircraft=aircraft,
    condition=condition,
    spanwise_resolution=12,
    verbose=True,
)

result = ll.run()

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

ll.draw(
    backend="plotly" if args.plotly else "pyvista",
    draw_streamlines=not args.no_streamlines,
)
