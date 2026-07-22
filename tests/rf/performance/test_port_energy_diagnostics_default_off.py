"""Guard that port lumped runtimes keep energy diagnostics default-off (S2.2).

The per-step energy/branch bookkeeping (dissipated/stored energy, per-branch
currents, source work) is consumed only by validation. On the port hot path it
must stay disabled by default so ``apply_lumped_runtime`` issues the minimal
in-place kernel schedule. This test pins that default across every port
construction path (matched excitation, passive SeriesRLC / ParallelRLC
termination) so a future change that flips diagnostics on -- silently
reintroducing the audit's per-step overhead -- turns red.

Falsification (recorded 2026-07-18): forwarding ``diagnostics=True`` in
``prepare_port_runtimes``'s ``prepare_lumped_runtime`` calls flips
``diagnostics_enabled`` to True and fails every assertion here.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Port diagnostics-default-off guard requires CUDA.",
)


def _port(name, *, termination=None, offset=0.0):
    return mw.LumpedPort(
        name=name,
        positive=(offset, 0.0, 0.005),
        negative=(offset, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(offset, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
        reference_impedance=50.0,
        termination=termination,
    )


def _solver(ports, *, excitations=()):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1), (-0.02, 0.02), (-0.02, 0.02))),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=ports,
        device="cuda",
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(3.0e9,),
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
        excitations=excitations,
    )
    return simulation.prepare().solver


def test_series_rlc_termination_runtime_defaults_to_diagnostics_off():
    solver = _solver((_port("p", termination=mw.SeriesRLC(r=60.0, l=1.2e-9, c=0.8e-12)),))
    lumped = solver._port_runtimes[0].lumped
    assert lumped is not None
    assert lumped.diagnostics_enabled is False


def test_parallel_rlc_termination_runtime_defaults_to_diagnostics_off():
    solver = _solver((_port("p", termination=mw.ParallelRLC(r=75.0, l=2.0e-9, c=0.5e-12)),))
    lumped = solver._port_runtimes[0].lumped
    assert lumped is not None
    assert lumped.diagnostics_enabled is False


def test_matched_excitation_runtime_defaults_to_diagnostics_off():
    excitation = mw.PortExcitation(
        port_name="p",
        source_time=mw.GaussianPulse(frequency=3.0e9, fwidth=1.0e9, amplitude=1.0),
    )
    solver = _solver((_port("p"),), excitations=(excitation,))
    lumped = solver._port_runtimes[0].lumped
    assert lumped is not None
    assert lumped.diagnostics_enabled is False


def test_all_port_lumped_runtimes_default_to_diagnostics_off():
    solver = _solver(
        (
            _port("a", termination=mw.SeriesRLC(r=50.0), offset=0.0),
            _port("b", termination=mw.ParallelRLC(r=50.0), offset=0.02),
        )
    )
    lumped_runtimes = [
        runtime.lumped
        for runtime in solver._port_runtimes
        if runtime.lumped is not None
    ]
    assert lumped_runtimes
    assert all(lumped.diagnostics_enabled is False for lumped in lumped_runtimes)
