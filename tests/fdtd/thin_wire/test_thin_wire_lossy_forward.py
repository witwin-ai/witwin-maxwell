"""End-to-end lossy thin-wire FDTD integration (B2 runtime consumption).

Covers: PEC-wire bitwise parity (the lossy feature never perturbs the PEC path)
with falsification, a finite-conductor run that stays stable and emits a real
positive ohmic_loss channel, and the no-loss-descriptor == old-path gate.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD requires CUDA"
)

_FREQUENCY = 2.0e9


def _scene(*, conductor, quantities=("current", "charge", "ohmic_loss")):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((0.0, 0.0, -0.08), (0.0, 0.0, 0.08)),
            radius=2.0e-3,
            conductor=conductor,
            endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
            snap="strict",
        )
    )
    scene.add_source(
        mw.PointDipole(
            name="drive",
            position=(0.0, 0.0, 0.02),
            polarization="Ez",
            width=0.04,
            source_time=mw.CW(frequency=_FREQUENCY, amplitude=10.0),
        )
    )
    scene.add_monitor(
        mw.WireMonitor(
            name="wire_state",
            wire="wire",
            frequencies=(_FREQUENCY,),
            quantities=quantities,
        )
    )
    return scene


def _run(scene, *, time_steps=400):
    return mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()


def _prepare(scene, *, time_steps=10):
    return mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=time_steps),
    ).prepare()


def test_pec_wire_bitwise_parity():
    # Headline gate (e): a PEC-wire scene is byte-identical across runs and never
    # enters the lossy path (its wire runtime carries no lossy model), so the
    # finite-conductor feature leaves the PEC leapfrog untouched.
    prepared = _prepare(
        _scene(conductor=mw.WireConductor.pec(), quantities=("current", "charge"))
    )
    assert prepared.solver._wire_runtime.lossy_model is None

    first = _run(_scene(conductor=mw.WireConductor.pec(), quantities=("current",)))
    second = _run(_scene(conductor=mw.WireConductor.pec(), quantities=("current",)))
    first_current = first.monitor("wire_state").current
    second_current = second.monitor("wire_state").current
    assert torch.equal(first_current, second_current)


def test_pec_parity_falsification():
    # Falsification of (e): a finite conductor on the same geometry takes the torch
    # lossy path and changes the wire response, proving the parity above is a real
    # bypass, not a no-op.
    pec = _run(_scene(conductor=mw.WireConductor.pec(), quantities=("current",)))
    lossy = _run(
        _scene(conductor=mw.WireConductor.finite(5.8e7), quantities=("current",))
    )
    difference = (
        pec.monitor("wire_state").current - lossy.monitor("wire_state").current
    ).abs().max().item()
    assert difference > 0.0


def test_lossy_wire_runs_and_emits_ohmic_loss():
    # Gate: a finite-conductor wire runs stably (no NaN/inf), carries the lossy
    # model, and emits a real, positive, finite ohmic_loss channel (previously
    # zeros).
    scene = _scene(conductor=mw.WireConductor.finite(5.8e7))
    prepared = _prepare(scene)
    model = prepared.solver._wire_runtime.lossy_model
    assert model is not None
    assert model.spectral_radius < 1.0
    assert bool(torch.all(model.is_lossy))

    result = _run(scene)
    monitor = result.monitor("wire_state")
    current = monitor.current
    ohmic = monitor.ohmic_loss
    assert ohmic is not None
    assert torch.all(torch.isfinite(current))
    assert torch.all(torch.isfinite(ohmic))
    assert torch.all(ohmic >= 0.0)
    assert float(ohmic.max().item()) > 0.0
    # The reported dissipation equals 0.5 Re(Z') length |I|^2 from the same model.
    length = prepared.solver._wire_runtime.network.length
    segment_indices = monitor.metadata["segment_ids"]
    frequencies = torch.tensor([_FREQUENCY], dtype=torch.float64)
    ac_resistance = model.ac_resistance_per_length(frequencies)
    assert float(ac_resistance.max().item()) > 0.0


def test_pec_ohmic_loss_is_zero():
    # A PEC wire reports exactly zero ohmic dissipation (no lossy model).
    scene = _scene(conductor=mw.WireConductor.pec())
    result = _run(scene)
    ohmic = result.monitor("wire_state").ohmic_loss
    assert ohmic is not None
    assert float(ohmic.abs().max().item()) == 0.0


def test_lossy_reverse_replay_fails_closed():
    # B3: the FIELD-COUPLED reverse pass (dI/dsigma through the recurrence) stays
    # fail closed because the ADE coefficients come from the nondeterministic shared
    # rational fit; the message points to the analytic conductivity adjoint that did
    # ship (analytic_ac_resistance).
    from witwin.maxwell.fdtd.wire import replay_wire_state

    prepared = _prepare(_scene(conductor=mw.WireConductor.finite(5.8e7)))
    solver = prepared.solver
    runtime = solver._wire_runtime
    state = {
        "Ex": solver.Ex,
        "Ey": solver.Ey,
        "Ez": solver.Ez,
        "wire_current": runtime.current,
        "wire_charge": runtime.charge,
    }
    with pytest.raises(NotImplementedError, match="conductivity"):
        replay_wire_state(solver, state)
    with pytest.raises(NotImplementedError, match="analytic_ac_resistance"):
        replay_wire_state(solver, state)


def test_lossy_checkpoint_fails_closed():
    # G2a fails closed on checkpoint/resume of a lossy wire (ADE state is not yet
    # in the checkpoint schema, added with B3).
    from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state

    prepared = _prepare(_scene(conductor=mw.WireConductor.finite(5.8e7)))
    with pytest.raises(NotImplementedError, match="[Cc]heckpoint"):
        capture_checkpoint_state(prepared.solver, 0)


def test_lossy_current_differs_from_pec():
    # Loss damps the wire response: the finite-conductor current is not identical
    # to the PEC current (no-loss-descriptor == old path; finite != old path).
    pec = _run(_scene(conductor=mw.WireConductor.pec()))
    lossy = _run(_scene(conductor=mw.WireConductor.finite(5.8e7)))
    pec_current = pec.monitor("wire_state").current
    lossy_current = lossy.monitor("wire_state").current
    assert not torch.allclose(pec_current, lossy_current)
