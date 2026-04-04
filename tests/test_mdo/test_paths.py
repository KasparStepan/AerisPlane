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


import numpy as np
import aerisplane as ap
from aerisplane.mdo._paths import (
    _tokenize, _get_dv_value, _set_dv_value,
    _resolve_to_obj, _get_result_value,
    _pack, _unpack, _build_pool_entries, _integrality_array,
)
from aerisplane.mdo.problem import DesignVar, AirfoilPool


@pytest.fixture
def simple_aircraft():
    wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25,
                        airfoil=ap.Airfoil("naca2412")),
            ap.WingXSec(xyz_le=[0.03, 0.6, 0.04], chord=0.14,
                        airfoil=ap.Airfoil("naca2412")),
        ],
        symmetric=True,
    )
    return ap.Aircraft(name="test", wings=[wing])


def test_tokenize_simple():
    assert _tokenize("chord") == ["chord"]

def test_tokenize_nested():
    assert _tokenize("wings[0].xsecs[1].chord") == ["wings", 0, "xsecs", 1, "chord"]

def test_get_dv_value(simple_aircraft):
    assert abs(_get_dv_value(simple_aircraft, "wings[0].xsecs[0].chord") - 0.25) < 1e-9

def test_set_dv_value(simple_aircraft):
    _set_dv_value(simple_aircraft, "wings[0].xsecs[0].chord", 0.30)
    assert abs(simple_aircraft.wings[0].xsecs[0].chord - 0.30) < 1e-9

def test_pack_unpack_roundtrip(simple_aircraft):
    dvars = [
        DesignVar("wings[0].xsecs[0].chord", lower=0.1, upper=0.5),
        DesignVar("wings[0].xsecs[1].chord", lower=0.1, upper=0.3),
    ]
    x = _pack(simple_aircraft, dvars, pool_entries=[])
    assert x.shape == (2,)
    assert abs(x[0] - 0.25) < 1e-9

    ac2 = _unpack(simple_aircraft, dvars, pool_entries=[], x=np.array([0.30, 0.16]))
    assert abs(ac2.wings[0].xsecs[0].chord - 0.30) < 1e-9
    # original unchanged
    assert abs(simple_aircraft.wings[0].xsecs[0].chord - 0.25) < 1e-9

def test_get_result_value():
    from unittest.mock import MagicMock
    mock_stab = MagicMock()
    mock_stab.static_margin = 0.08
    results = {"stability": mock_stab}
    assert _get_result_value(results, "stability.static_margin") == 0.08

def test_get_result_value_nested():
    from unittest.mock import MagicMock
    mock_wing = MagicMock()
    mock_wing.bending_margin = 2.1
    mock_struct = MagicMock()
    mock_struct.wings = [mock_wing]
    results = {"structures": mock_struct}
    assert _get_result_value(results, "structures.wings[0].bending_margin") == 2.1

def test_pool_entries_all_xsecs(simple_aircraft):
    pools = {"wings[0]": AirfoilPool(options=["naca2412", "naca4412"])}
    entries = _build_pool_entries(simple_aircraft, pools)
    assert len(entries) == 2   # 2 xsecs on wings[0]
    assert entries[0] == ("wings[0]", 0, pools["wings[0]"])
    assert entries[1] == ("wings[0]", 1, pools["wings[0]"])

def test_pool_entries_specific_xsecs(simple_aircraft):
    pools = {"wings[0]": AirfoilPool(options=["naca2412", "naca4412"], xsecs=[0])}
    entries = _build_pool_entries(simple_aircraft, pools)
    assert len(entries) == 1
    assert entries[0][1] == 0   # only xsec 0

def test_integrality_array():
    dvars = [DesignVar("a", 0, 1), DesignVar("b", 0, 1)]
    pool_entries = [("wings[0]", 0, None), ("wings[0]", 1, None)]
    arr = _integrality_array(n_dvars=2, pool_entries=pool_entries)
    assert list(arr) == [False, False, True, True]
