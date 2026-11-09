"""CPU coverage for dispatching ModalPortSweep through Simulation(excitations=...)."""

from __future__ import annotations

import pytest

import witwin.maxwell as mw
import witwin.maxwell.modal_sweep as modal_sweep


_FREQUENCIES = (2.9e8, 3.0e8, 3.1e8)


def _scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    )
    scene.add_port(
        mw.ModePort("left", position=(-0.5, 0.0, 0.0), size=(0.0, 0.6, 0.6),
                    direction="+", polarization="Ez", monitor_offset=0.1)
    )
    scene.add_port(
        mw.ModePort("right", position=(0.5, 0.0, 0.0), size=(0.0, 0.6, 0.6),
                    direction="-", polarization="Ez", monitor_offset=-0.1)
    )
    return scene


def test_fdtd_simulation_stores_modal_port_sweep_without_rf_excitations():
    sweep = mw.ModalPortSweep(modes={"left": (0, 1)})
    simulation = mw.Simulation.fdtd(_scene(), frequencies=_FREQUENCIES, excitations=sweep)

    assert simulation.modal_port_sweep is sweep
    assert simulation.excitations == ()
    assert simulation.port_sweep is None


def test_simulation_without_modal_sweep_has_no_modal_port_sweep():
    simulation = mw.Simulation.fdtd(_scene(), frequencies=_FREQUENCIES)

    assert simulation.modal_port_sweep is None


def test_non_fdtd_backends_reject_modal_port_sweep():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=2),
        device="cpu",
    )
    with pytest.raises(ValueError, match=r"Simulation\.fdtd\(\.\.\.\) only"):
        mw.Simulation(
            scene=scene,
            method=mw.SimulationMethod.FDFD,
            frequencies=(3.0e8,),
            excitations=mw.ModalPortSweep(),
        )


def test_run_dispatches_to_run_modal_port_sweep(monkeypatch):
    sweep = mw.ModalPortSweep(run_only=("left::Ez0",))
    simulation = mw.Simulation.fdtd(_scene(), frequencies=_FREQUENCIES, excitations=sweep)
    calls = []
    sentinel = object()

    def fake_run(sim, requested_sweep, **kwargs):
        calls.append((sim, requested_sweep, kwargs))
        return sentinel

    monkeypatch.setattr(modal_sweep, "run_modal_port_sweep", fake_run)

    assert simulation.run() is sentinel
    assert calls == [(simulation, sweep, {})]
