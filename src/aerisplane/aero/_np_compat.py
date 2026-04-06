# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
"""numpy compatibility helpers for vendored AeroSandbox solvers.

Provides the non-standard functions that aerosandbox.numpy adds on top of
plain numpy.  In vendored files, replace:

    import aerosandbox.numpy as np
    np.blend(...)  →  blend(...)

with:

    import numpy as np
    from aerisplane.aero._np_compat import blend, softmax, cosd, sind, tand, arccosd
"""
import numpy as _np


def cosd(x):
    """Cosine of angle given in degrees."""
    return _np.cos(_np.radians(x))


def sind(x):
    """Sine of angle given in degrees."""
    return _np.sin(_np.radians(x))


def tand(x):
    """Tangent of angle given in degrees."""
    return _np.tan(_np.radians(x))


def arccosd(x):
    """Inverse cosine; result in degrees."""
    return _np.degrees(_np.arccos(x))


def blend(switch, value_switch_high, value_switch_low):
    """Smooth sigmoid blend between two values.

    Uses tanh so the transition is differentiable everywhere.

    When switch → +∞  →  value_switch_high
    When switch → −∞  →  value_switch_low
    When switch = 0   →  mean of the two values
    """
    weight = _np.tanh(switch) / 2.0 + 0.5
    return value_switch_high * weight + value_switch_low * (1.0 - weight)


def softmax(*args, hardness=None, softness=None):
    """Element-wise smooth maximum (log-sum-exp) of two or more arrays.

    Based on the log-sum-exp identity: softmax(a, b) = log(exp(a) + exp(b)).
    Numerically stable via the max-shift trick.

    Parameters
    ----------
    *args : scalar or array
        Values to take the smooth maximum of (two or more).
    hardness : float, optional
        Higher value → closer to true maximum. Inverse of softness.
    softness : float, optional
        Lower value → closer to true maximum. Default 1.0 when neither is given.
    """
    if hardness is not None and softness is not None:
        raise ValueError("Provide at most one of `hardness` or `softness`.")
    if hardness is None and softness is None:
        softness = 1.0
    if hardness is not None:
        softness = 1.0 / hardness

    scaled = [arg / softness for arg in args]

    # Numerically stable: shift by current max before exponentiating
    m = scaled[0]
    for s in scaled[1:]:
        m = _np.fmax(m, s)

    out = m + _np.log(sum(_np.exp(_np.maximum(s - m, -500)) for s in scaled))
    return out * softness
