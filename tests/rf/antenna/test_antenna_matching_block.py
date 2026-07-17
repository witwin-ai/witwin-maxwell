"""Antenna matching-block realized-gain bookkeeping on a real FDTD solve.

An antenna's realized gain is its intrinsic gain scaled by the port mismatch
efficiency ``1 - |Gamma|^2``. Inserting a lossless external matching network in
front of the feed transforms an input reflection ``Gamma_L`` to a new value
``Gamma_in`` and, at constant accepted (radiated) power, leaves the intrinsic
gain untouched while lifting the realized gain by
``(1 - |Gamma_in|^2)/(1 - |Gamma_L|^2)``.

This test exercises that relationship through ``Result.antenna`` on a **real**
FDTD solve: a driven z-directed lumped port radiates inside a small PML box, a
first-class ``ClosedSurfaceMonitor`` captures the near field, and
``Result.antenna`` transforms the genuine monitor payload to a far field via
Stratton-Chu. The lossless matching two-port is a real ``mw.NetworkData`` built
through the public ``NetworkData.from_y`` constructor; its predicted matched
reflection is obtained from the network's public S-parameters (``NetworkData.s``)
cascaded with the load.

Scope / plan gap (plan 03 corpus honesty): this validates the *bookkeeping* that
``Result.antenna`` performs on a real radiated far field -- intrinsic gain
referenced to accepted power (invariant under an external match) and realized
gain referenced to incident power (scaling by the network mismatch ratio). It
does **not** validate an input-impedance-driven match against the radiator's own
port impedance: extracting a reliable input impedance from an electrically small
lumped-port radiator is a separate validation left out of scope (the unified
trapezoidal coupling makes the raw port reflection passive, |Gamma| < 1, but it
sits at ~0.996 -- nearly total reflection -- so it is no useful match target, see
the assertion below). The antenna load ``Gamma_L`` presented to the matching
network is therefore modelled as a representative reactive load rather than read
back from the port.

The public network surface exposes ``NetworkData.from_y`` (construction) and
``NetworkData.s`` (S-parameters) but no two-port/one-port cascade or termination
helper, so the single load-cascade expression below is the smallest building
block layered on top of the public S-matrix.
"""

from __future__ import annotations

import math

import torch

import witwin.maxwell as mw


_FREQUENCY = 3.0e9
_Z0 = 50.0
_DEVICE = "cuda"


def _radiator_scene() -> mw.Scene:
    """A driven z-directed lumped port radiating inside a small PML box."""

    port = mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
        reference_impedance=_Z0,
    )
    surface = mw.ClosedSurfaceMonitor.box(
        "nf2ff",
        position=(0.0, 0.0, 0.0),
        size=(0.04, 0.04, 0.04),
        frequencies=(_FREQUENCY,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.08, 0.08),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        ports=(port,),
        device=_DEVICE,
    )
    scene.add_monitor(surface)
    return scene


def _real_far_field_result():
    simulation = mw.Simulation.fdtd(
        _radiator_scene(),
        frequency=_FREQUENCY,
        excitations=mw.PortExcitation(
            "feed",
            amplitude=1.0,
            source_impedance="matched",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=1.0e9),
        ),
        run_time=mw.TimeConfig.auto(steady_cycles=16, transient_cycles=32),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    )
    return simulation.run()


def _port_from_reflection(gamma: complex, *, accepted_power: float) -> mw.PortData:
    """Model a driven port with the given input reflection and accepted power."""

    magnitude = abs(gamma)
    incident = accepted_power / (1.0 - magnitude * magnitude)
    a = torch.tensor(
        [complex(math.sqrt(incident), 0.0)], dtype=torch.complex128, device=_DEVICE
    )
    b = a * torch.tensor([gamma], dtype=torch.complex128, device=_DEVICE)
    return mw.PortData.from_power_waves(
        port_name="feed",
        frequencies=torch.tensor([_FREQUENCY], dtype=torch.float64, device=_DEVICE),
        a=a,
        b=b,
        z0=_Z0,
    )


def _series_reactance_block(reactance_ohms: float) -> mw.NetworkData:
    """A lossless series-reactance two-port built from its public admittance matrix.

    A series impedance ``jX`` between the two ports has the indefinite admittance
    matrix ``Y = (1/jX) * [[1, -1], [-1, 1]]``. Constructing the network through
    the public ``NetworkData.from_y`` avoids hand-writing the scattering matrix;
    the cascade prediction below then reads the network's public S-parameters.
    """

    y = (1.0 / (1j * reactance_ohms)) * torch.tensor(
        [[1.0, -1.0], [-1.0, 1.0]], dtype=torch.complex128
    )
    return mw.NetworkData.from_y(
        frequencies=torch.tensor([_FREQUENCY], dtype=torch.float64),
        y=y.unsqueeze(0),
        z0=_Z0,
        port_names=("in", "out"),
    )


def _input_reflection_through_block(block: mw.NetworkData, load_gamma: complex) -> complex:
    """Cascade a two-port matching block (public S) with a load reflection Gamma_L.

    ``NetworkData`` exposes no public cascade/termination helper, so this is the
    smallest load-cascade expression on top of the public ``block.s`` accessor.
    """

    s = block.s[0]
    s11 = complex(s[0, 0])
    s12 = complex(s[0, 1])
    s21 = complex(s[1, 0])
    s22 = complex(s[1, 1])
    return s11 + s12 * s21 * load_gamma / (1.0 - s22 * load_gamma)


def _antenna(result, port):
    return result.antenna(
        surface="nf2ff",
        driven_port=port,
        theta_points=15,
        phi_points=21,
        radius=4.0,
    )


def test_realized_gain_lift_matches_network_mismatch_ratio_on_a_real_solve():
    result = _real_far_field_result()

    # The unified trapezoidal port coupling samples the port voltage and current
    # at the same magnetic half-step, so the crude radiator's raw port reflection
    # is now passive (|Gamma| < 1) instead of the non-passive |Gamma| > 1 the old
    # half-step-staggered sampling produced. It is passive but still not a useful
    # match target: an electrically small lumped-port radiator reflects almost all
    # of the incident power (|Gamma| ~ 0.996, mismatch efficiency < 1%), so the
    # matched load below is still modelled externally rather than read back from
    # the port. This assertion pins the restored passivity and documents that
    # scope boundary instead of hiding it.
    raw_reflection = float(torch.abs(result.port("feed").reflection_coefficient)[0])
    assert raw_reflection < 1.0
    assert raw_reflection > 0.99

    # Modelled representative antenna load: 50 + j60 Ohm, a reactive mismatch a
    # lossless series reactance can conjugate-match. The far field the matching
    # bookkeeping operates on is the genuine FDTD-radiated pattern.
    z_load = complex(50.0, 60.0)
    load_gamma = (z_load - _Z0) / (z_load + _Z0)
    accepted_power = 1.0 - abs(load_gamma) ** 2

    block = _series_reactance_block(-60.0)
    matched_gamma = _input_reflection_through_block(block, load_gamma)
    # The block genuinely improves the match toward a conjugate match.
    assert abs(matched_gamma) < abs(load_gamma)
    assert abs(matched_gamma) < 1.0e-9

    bare_port = _port_from_reflection(load_gamma, accepted_power=accepted_power)
    matched_port = _port_from_reflection(matched_gamma, accepted_power=accepted_power)

    bare = _antenna(result, bare_port)
    matched = _antenna(result, matched_port)

    # The far field came from the real solve: six real closed-surface faces feed a
    # strictly positive, finite radiated power.
    assert bare.surface_currents is not None
    assert len(bare.surface_currents[0].surfaces) == 6
    assert torch.all(torch.isfinite(bare.p_rad))
    assert float(bare.p_rad[0]) > 0.0

    # Intrinsic gain (referenced to accepted/radiated power) is invariant under an
    # external lossless match; only the realized gain moves.
    torch.testing.assert_close(matched.gain, bare.gain, rtol=1.0e-10, atol=0.0)

    predicted_ratio = (1.0 - abs(matched_gamma) ** 2) / (1.0 - abs(load_gamma) ** 2)
    measured_ratio = matched.realized_gain / bare.realized_gain
    torch.testing.assert_close(
        measured_ratio,
        torch.full_like(measured_ratio, predicted_ratio),
        rtol=1.0e-9,
        atol=0.0,
    )

    # The realized-gain lift equals the mismatch-efficiency improvement that the
    # network cascade predicts, reported through Result.antenna.
    torch.testing.assert_close(
        matched.mismatch_efficiency / bare.mismatch_efficiency,
        torch.full_like(bare.mismatch_efficiency, predicted_ratio),
        rtol=1.0e-9,
        atol=0.0,
    )
    assert float(bare.mismatch_efficiency[0]) < 0.8
    assert float(matched.mismatch_efficiency[0]) > 0.999
