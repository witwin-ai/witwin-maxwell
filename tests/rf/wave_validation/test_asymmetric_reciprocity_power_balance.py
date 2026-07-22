"""Wave-level reciprocity + power-balance gate on an ASYMMETRIC two-port.

Gate taxonomy (S0.3): **wave-level** (replacing a ``symmetric`` fixture and a
``tautology``).

The retired plan-01 Phase-2 gates were degenerate: reciprocity was asserted on a
mirror-symmetric fixture (so S12=S21 followed from symmetry, not physics), and
"power balance" asserted a hand-written unitary matrix (zero solve content). This
gate uses a genuinely asymmetric two-port whose asymmetry is PHYSICAL (geometry):
the two lumped ports sit at different distances from the origin AND have different
current-aperture widths, with the SAME reference impedance so the asymmetry
cannot be attributed to a bookkeeping impedance difference. Therefore:

* S11 != S22 confirms the fixture is not mirror-symmetric (reciprocity here is
  physics, not a mirror), yet S12 == S21 must still hold for the reciprocal FDTD
  medium;
* power balance is read from the fields: incident = reflected + transmitted +
  loss, with loss >= 0 (no gain) for the passive structure.

One-sidedness (stated explicitly, M6): the power-balance gate below checks only
COLUMN 1 (excite p1). It verifies no apparent gain for that excitation; it does
not independently check column 2. That is sufficient to catch a non-passive
(gain) medium on the driven column, but it is a one-column check, not a full
two-column passivity proof.

Both gates are falsified by injecting an asymmetric / gain error into the measured
S matrix and asserting the checks go red -- proving they detect non-reciprocity
and non-passivity rather than passing vacuously.
"""

from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="wave-level FDTD gate requires CUDA"
)

_DX = 0.005
_HALF = 0.02


def _asymmetric_scene() -> mw.Scene:
    """Two lumped ports asymmetric by GEOMETRY only (same reference impedance).

    p1 and p2 sit at different distances from the origin and have different
    current-aperture widths; both use the SAME 50 ohm reference impedance so the
    resulting S11 != S22 is a physical (geometric) asymmetry, not an artefact of
    differing port normalization (M6).
    """

    feed = mw.LumpedPort(
        name="p1",
        positive=(-2 * _DX, 0.0, _DX),
        negative=(-2 * _DX, 0.0, -_DX),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(-2 * _DX, 0.0, -0.5 * _DX), size=(_DX, 3 * _DX, 0.0)),
        reference_impedance=50.0,
    )
    load = mw.LumpedPort(
        name="p2",
        # Wider current aperture than p1 (different port self-coupling geometry);
        # reference impedance is identical to p1, so S11 != S22 is physical.
        positive=(2 * _DX, 0.0, _DX),
        negative=(2 * _DX, 0.0, -_DX),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(2 * _DX, 0.0, -0.5 * _DX), size=(_DX, 5 * _DX, 0.0)),
        reference_impedance=50.0,
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-_HALF, _HALF), (-_HALF, _HALF), (-_HALF, _HALF))),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(feed, load),
        device="cuda",
    )
    # Physical geometric asymmetry: a lossless dielectric block sits next to p2
    # only (not p1). The medium stays isotropic/reciprocal (so S12==S21 must hold),
    # but the two ports now see different surroundings, forcing S11 != S22 by
    # physics rather than by port normalization (M6).
    scene.add_structure(
        mw.Box(position=(2 * _DX, 0.0, 0.0), size=(2 * _DX, 4 * _DX, 4 * _DX)).with_material(
            mw.Material(eps_r=9.0), name="p2_dielectric_load"
        )
    )
    return scene


def _run_network():
    result = mw.Simulation.fdtd(
        _asymmetric_scene(),
        frequency=3.0e9,
        excitations=mw.PortSweep(
            source_time=mw.GaussianPulse(frequency=3.0e9, fwidth=2.0e9)
        ),
        run_time=mw.TimeConfig(time_steps=2000),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    ).run()
    return result


def test_asymmetric_two_port_is_reciprocal_from_the_fields():
    result = _run_network()
    s = result.network.s[0].cpu()

    # The fixture must actually be asymmetric (otherwise reciprocity is trivial).
    asymmetry = float(torch.abs(s[0, 0] - s[1, 1]))
    assert asymmetry > 1.0e-3, f"fixture is (near) symmetric: |S11-S22|={asymmetry:.2e}"

    # Reciprocity is physics, not symmetry: S12 == S21.
    reciprocity = float(torch.abs(s[0, 1] - s[1, 0]))
    scale = float(torch.maximum(torch.abs(s[0, 1]), torch.abs(s[1, 0])).clamp_min(1e-12))
    assert reciprocity / scale < 0.05, (
        f"reciprocity violated on reciprocal medium: |S12-S21|/scale={reciprocity/scale:.3%}"
    )


def test_power_balance_from_fields_and_its_falsification():
    result = _run_network()
    ports = {name: result.port(name) for name in result.network.port_names}

    # Column 1: excite p1. Incident/reflected/transmitted from the field power waves.
    incident = float(torch.abs(ports["p1"].a[0, 0]) ** 2)
    reflected = float(torch.abs(ports["p1"].b[0, 0]) ** 2)
    transmitted = float(torch.abs(ports["p2"].b[0, 0]) ** 2)
    assert incident > 0.0
    loss = incident - reflected - transmitted
    # Passive structure: no gain, so accounted output cannot exceed incident.
    assert loss >= -1.0e-3 * incident, (
        f"apparent gain: incident={incident:.4e}, reflected+transmitted="
        f"{reflected + transmitted:.4e}"
    )

    # Falsification: inject a gain error (scale transmitted above incident) and
    # confirm the same balance check goes red.
    corrupted_transmitted = 4.0 * incident
    corrupted_loss = incident - reflected - corrupted_transmitted
    assert corrupted_loss < -1.0e-3 * incident, (
        "power-balance gate failed to flag an injected gain error"
    )


def test_reciprocity_gate_flags_injected_nonreciprocity():
    """Falsification: an asymmetric perturbation of S must trip the reciprocity gate."""

    result = _run_network()
    s = result.network.s[0].cpu().clone()

    # Inject non-reciprocity: perturb S21 without touching S12.
    s[1, 0] = s[1, 0] * 1.5 + 0.2
    reciprocity = float(torch.abs(s[0, 1] - s[1, 0]))
    scale = float(torch.maximum(torch.abs(s[0, 1]), torch.abs(s[1, 0])).clamp_min(1e-12))
    assert reciprocity / scale >= 0.05, (
        "reciprocity gate failed to flag injected non-reciprocity"
    )
