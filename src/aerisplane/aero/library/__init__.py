# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
"""Aerodynamic engineering library functions.

Pure-math correlations — no geometry types, no solvers.
"""
from aerisplane.aero.library.viscous import Cf_flat_plate, Cd_cylinder
from aerisplane.aero.library.inviscid import oswalds_efficiency, CL_over_Cl, induced_drag
from aerisplane.aero.library.transonic import (
    sears_haack_drag_from_volume,
    approximate_CD_wave,
    Cd_wave_Korn,
    mach_crit_Korn,
)
