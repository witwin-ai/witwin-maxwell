"""The surface-integral solver is exposed only under its MoM names."""
import importlib

import pytest

import witwin.maxwell as mw


@pytest.mark.parametrize("name", ["SIEConfig", "SIEPeriodicConfig", "SIEDyadicPeriodicConfig", "SIERuntime"])
def test_old_public_names_are_absent(name):
    assert not hasattr(mw, name)
    assert name not in mw.__all__


def test_old_simulation_entry_and_method_are_absent():
    assert not hasattr(mw.Simulation, "sie")
    assert not hasattr(mw.SimulationMethod, "SIE")
    with pytest.raises(ValueError):
        mw.SimulationMethod("sie")


def test_old_package_is_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("witwin.maxwell.sie")


def test_mom_names_are_public():
    assert mw.SimulationMethod("mom") is mw.SimulationMethod.MOM
    assert callable(mw.Simulation.mom)
    for name in ("MoMConfig", "MoMPeriodicConfig", "MoMDyadicPeriodicConfig"):
        assert hasattr(mw, name) and name in mw.__all__
    from witwin.maxwell.mom.runtime import MoMRuntime
    assert MoMRuntime.__module__ == "witwin.maxwell.mom.runtime"
