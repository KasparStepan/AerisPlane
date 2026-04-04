from aerisplane.catalog import get_airfoil
from aerisplane.core.airfoil import Airfoil
import pytest

def test_get_airfoil_naca():
    af = get_airfoil("naca2412")
    assert isinstance(af, Airfoil)
    assert af.coordinates is not None
    assert af.name == "naca2412"

def test_get_airfoil_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        get_airfoil("totally_fake_airfoil_xyz")
