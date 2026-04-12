"""Tests for Aircraft geometry methods."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.wing import Wing, WingXSec


@pytest.fixture
def simple_aircraft():
    af = ap.Airfoil.from_naca("0012")
    main_wing = Wing(
        name="main_wing",
        xsecs=[
            WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.3, airfoil=af),
            WingXSec(xyz_le=[0.02, 0.75, 0.0], chord=0.15, airfoil=af),
        ],
        symmetric=True,
    )
    htail = Wing(
        name="htail",
        xsecs=[
            WingXSec(xyz_le=[0.9, 0.0, 0.0], chord=0.12, airfoil=af),
            WingXSec(xyz_le=[0.92, 0.25, 0.0], chord=0.08, airfoil=af),
        ],
        symmetric=True,
    )
    return Aircraft(name="test_ac", wings=[main_wing, htail])


class TestAircraftAerodynamicCenter:
    def test_returns_array(self, simple_aircraft):
        ac = simple_aircraft.aerodynamic_center()
        assert isinstance(ac, np.ndarray)
        assert ac.shape == (3,)

    def test_ac_dominated_by_main_wing(self, simple_aircraft):
        # Main wing has much more area, so AC should be near main wing AC
        ac = simple_aircraft.aerodynamic_center()
        main_ac = simple_aircraft.wings[0].aerodynamic_center()
        htail_ac = simple_aircraft.wings[1].aerodynamic_center()
        # AC x should be between main wing AC and htail AC
        assert min(main_ac[0], htail_ac[0]) <= ac[0] <= max(main_ac[0], htail_ac[0])

    def test_empty_aircraft_returns_origin(self):
        ac = Aircraft(name="empty").aerodynamic_center()
        assert np.allclose(ac, [0.0, 0.0, 0.0])


class TestAircraftIsEntirelySymmetric:
    def test_symmetric_aircraft(self, simple_aircraft):
        assert simple_aircraft.is_entirely_symmetric() is True

    def test_asymmetric_if_any_wing_asymmetric(self, simple_aircraft):
        af = ap.Airfoil.from_naca("0012")
        vtail = Wing(
            name="vtail",
            xsecs=[
                WingXSec(xyz_le=[0.9, 0.0, 0.0], chord=0.12, airfoil=af),
                WingXSec(xyz_le=[0.92, 0.0, 0.25], chord=0.08, airfoil=af),
            ],
            symmetric=False,
        )
        ac = Aircraft(name="test", wings=[simple_aircraft.wings[0], vtail])
        assert ac.is_entirely_symmetric() is False
