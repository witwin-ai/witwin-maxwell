"""Checkpoint/resume of a generic rational surface-impedance run (S1.2 fail-open fix).

A generic ``SurfaceImpedanceMedium`` steps a per-edge Z-form ADE whose state is dynamic
solver memory. Before this slice the resume path was fail-open: the ADE state was neither
captured nor part of the resume fingerprint, so ``run_until(k) + run(resume_from=...)``
silently restarted the surface memory from zero mid-run and diverged from the
uninterrupted result. The scene here is the checkpoint/resume-supported public workflow
(a ``LumpedPort`` + ``Circuit`` drive, no full-field DFT or observers) with a generic
good-conductor surface-impedance block added, i.e. the exact fail-open path. These gates
hold three contracts:

* round-trip: resume reproduces the uninterrupted final fields and port data, and the
  per-edge ADE state is actually captured into the checkpoint;
* drop-one falsification: zeroing the captured ADE state before resume genuinely breaks
  the result (proving the restored state is load-bearing, not incidental);
* fingerprint: a checkpoint captured against a *different* rational surface model with the
  same geometry is rejected before any field is mutated (the discrete A/B/C/D coefficients
  are part of the resume identity).
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.media import RationalSurfaceImpedance, SurfaceImpedanceMedium
from witwin.maxwell.fdtd.surface_impedance_reference import good_conductor_surface_impedance

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="generic surface-impedance resume requires CUDA.",
)

_BAND = (0.5e9, 5.0e9)


def _good_conductor_medium(sigma=50.0, order=6):
    freqs = torch.logspace(math.log10(_BAND[0]), math.log10(_BAND[1]), 120, dtype=torch.float64)
    admittance = (1.0 / good_conductor_surface_impedance(sigma, freqs)).to(torch.complex128)
    model = RationalSurfaceImpedance.fit(freqs, admittance, order=order, band=_BAND)
    return SurfaceImpedanceMedium(impedance=model, name="coating")


def _simulation(*, steps=64, frequency=3.0e9, sigma=50.0):
    port = mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
        reference_impedance=50.0,
    )
    circuit = mw.Circuit("pulse_interconnect")
    input_node = circuit.node("input")
    output_node = circuit.node("output")
    circuit.add(mw.Resistor("R1", input_node, output_node, 35.0))
    circuit.add(mw.Capacitor("C1", output_node, circuit.ground, 1.2e-12))
    circuit.add(
        mw.CurrentSource(
            "I1",
            circuit.ground,
            input_node,
            waveform=mw.PiecewiseLinearWaveform(
                (0.0, 4.0e-11, 8.0e-11, 1.4e-10, 2.0e-10),
                (0.0, 0.0, 0.02, -0.01, 0.0),
            ),
        )
    )
    circuit.bind_port("feed", positive=output_node, negative=circuit.ground)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(port,),
        circuits=(circuit,),
        # A boundary-flush good-conductor surface-impedance half-space on +x: the
        # transverse extent overflows the domain (masked full), leaving the -x face
        # exposed with a generic per-edge ADE surface write.
        structures=(
            mw.Structure(
                geometry=Box(position=(0.0125, 0.0, 0.0), size=(0.015, 0.08, 0.08)),
                material=_good_conductor_medium(sigma=sigma),
            ),
        ),
        device="cuda",
    )
    return mw.Simulation.fdtd(
        scene,
        frequency=frequency,
        run_time=mw.TimeConfig(time_steps=steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )


def _surface_state_keys(checkpoint):
    return [name for name in checkpoint.physics.tensors if name.startswith("surface_ade_")]


def _assert_fields_close(actual, expected, *, rtol, atol):
    for name in ("Ex", "Ey", "Ez"):
        a = actual.field(name)
        b = expected.field(name)
        a = a["data"] if isinstance(a, dict) else a
        b = b["data"] if isinstance(b, dict) else b
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def test_generic_surface_resume_matches_uninterrupted_run():
    expected = _simulation().run()
    checkpoint = _simulation().prepare().run_until(32)

    keys = _surface_state_keys(checkpoint)
    assert keys, "generic surface ADE state was not captured into the checkpoint"
    assert any(
        float(checkpoint.physics.tensors[name].abs().max()) > 0.0 for name in keys
    ), "captured ADE state is all-zero at the checkpoint step"

    actual = _simulation().prepare().run(resume_from=checkpoint)

    assert checkpoint.step == 32
    _assert_fields_close(actual, expected, rtol=1e-6, atol=2e-8)
    torch.testing.assert_close(
        actual.port("feed").voltage, expected.port("feed").voltage, rtol=1e-6, atol=1e-9
    )


def test_zeroing_captured_ade_state_breaks_resume():
    """Falsification: the restored ADE state is load-bearing, not incidental.

    Dropping (zeroing) the captured per-edge surface memory before resume must make the
    resumed field diverge from the uninterrupted run -- the exact silent-reset failure the
    fix closes. If resume ignored the ADE state this would spuriously still match.
    """
    expected = _simulation().run()
    checkpoint = _simulation().prepare().run_until(32)
    keys = _surface_state_keys(checkpoint)
    assert keys
    for name in keys:
        checkpoint.physics.tensors[name].zero_()

    corrupted = _simulation().prepare().run(resume_from=checkpoint)

    ez_expected = expected.field("Ez")
    ez_expected = ez_expected["data"] if isinstance(ez_expected, dict) else ez_expected
    ez_corrupt = corrupted.field("Ez")
    ez_corrupt = ez_corrupt["data"] if isinstance(ez_corrupt, dict) else ez_corrupt
    max_abs = float(ez_expected.abs().max())
    rel = float((ez_corrupt - ez_expected).abs().max()) / max_abs
    assert rel > 1e-3, (
        "zeroing the captured surface ADE state left the resumed field unchanged, so the "
        f"restored state is not actually driving the surface (rel diff {rel:.2e})"
    )


def test_resume_rejects_a_different_surface_model_with_identical_geometry():
    """The discrete A/B/C/D coefficients are part of the resume identity.

    Two good-conductor surfaces with the same box geometry but different conductivity fit
    to the same static field/material tensors and the same state layout, so only the ADE
    coefficient fingerprint distinguishes them. A checkpoint from one must not resume into
    the other.
    """
    checkpoint = _simulation(sigma=50.0).prepare().run_until(32)
    prepared = _simulation(sigma=120.0).prepare()
    before = prepared.solver.Ez.clone()

    with pytest.raises(ValueError, match="does not match the prepared solver layout"):
        prepared.run(resume_from=checkpoint)

    torch.testing.assert_close(prepared.solver.Ez, before, rtol=0.0, atol=0.0)
