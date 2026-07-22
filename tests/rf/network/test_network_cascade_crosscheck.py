"""Independent raw-sample S-cascade cross-check vs time-domain embedding.

This breaks the fit-model-class circularity of the embedded ``< 0.02`` gate.
Two fully independent computations of the same input reflection are compared:

* Reference path: measure the bare three-port device S with a port sweep, then
  connect a network's RAW sampled S (read directly from a Touchstone file, NOT
  the rational fit) across two device ports using the first-principles
  ``NetworkData.cascade`` algebra. This path never touches the field-embedding
  code.
* Embedded path: rationally fit the same Touchstone samples, realize a
  state-space model, embed it in the time-domain FDTD run, and read the input
  reflection at the free port. This path never touches the cascade algebra or
  the raw samples.

The two share no code, so agreement to the pre-registered tolerance is genuine
cross-validation of both the connection algebra and the time-domain embedding.

Pre-registered tolerance: ``|dS| < 1e-5`` across the band on two networks (one
lossy attenuator-like, one reactive). Observed residuals are ~1e-8; the network
changes the input reflection by ~2e-4 (>> the tolerance), so the gate has teeth
(a corrupted cascade or embedding is caught).
"""

import os
import tempfile

import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"
# 121-point 0..6 GHz sample grid (0.05 GHz step) covers the source effective
# band; 2.0/2.5/3.0 GHz land exactly on grid indices 40/50/60.
FULL_GRID = torch.linspace(0.0, 6.0e9, 121, dtype=torch.float64, device=DEVICE)
MEASURE_INDICES = (40, 50, 60)
MEASURE_FREQUENCIES = tuple(float(FULL_GRID[i]) for i in MEASURE_INDICES)
SOURCE_TIME = mw.GaussianPulse(frequency=2.5e9, fwidth=1.0e9)
TIME_STEPS = 2000
# Pre-registered cross-check tolerance (see module docstring).
CROSS_CHECK_TOLERANCE = 1.0e-5

PORT_XS = (-0.005, 0.0, 0.005)
PORT_NAMES = ("d0", "d1", "d2")


def _lumped_port(name: str, x: float) -> mw.LumpedPort:
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(x, 0.0, -0.0025), size=(0.005, 0.005, 0.0)),
        reference_impedance=50.0,
    )


def _scene(networks=()):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.03, 0.03), (-0.02, 0.02), (-0.02, 0.02))),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        sources=(),
        ports=tuple(_lumped_port(n, x) for n, x in zip(PORT_NAMES, PORT_XS)),
        networks=networks,
        device=DEVICE,
    )


def _measure_bare_three_port() -> mw.NetworkData:
    result = mw.Simulation.fdtd(
        _scene(),
        frequencies=MEASURE_FREQUENCIES,
        excitations=mw.PortSweep(source_time=SOURCE_TIME),
        run_time=mw.TimeConfig(time_steps=TIME_STEPS),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    bare = result.network
    assert bare is not None
    # A well-behaved passive linear device (no stray sources) has bounded S.
    assert torch.max(torch.linalg.svdvals(bare.s)).item() <= 1.01
    return bare


def _lossy_model() -> mw.StateSpaceNetwork:
    conductance = torch.tensor([[0.03, -0.02], [-0.02, 0.03]], dtype=torch.float64, device=DEVICE)
    return mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64, device=DEVICE),
        B=torch.zeros((0, 2), dtype=torch.float64, device=DEVICE),
        C=torch.zeros((2, 0), dtype=torch.float64, device=DEVICE),
        D=conductance,
        representation="Y",
        port_order=("n0", "n1"),
    )


def _reactive_model() -> mw.StateSpaceNetwork:
    resistance = 50.0
    capacitance = 5.0e-12
    shunt = 2.0e-3
    incidence = torch.tensor((1.0, -1.0), dtype=torch.float64, device=DEVICE)
    return mw.StateSpaceNetwork(
        A=torch.tensor(((-1.0 / (resistance * capacitance),),), dtype=torch.float64, device=DEVICE),
        B=(incidence / (resistance * capacitance)).reshape(1, 2),
        C=(-incidence / resistance).reshape(2, 1),
        D=torch.outer(incidence, incidence) / resistance
        + shunt * torch.eye(2, dtype=torch.float64, device=DEVICE),
        representation="Y",
        port_order=("n0", "n1"),
    )


def _write_touchstone(model: mw.StateSpaceNetwork, path: str) -> mw.NetworkData:
    full = mw.NetworkData.from_y(
        frequencies=FULL_GRID,
        y=model.evaluate(FULL_GRID),
        z0=50.0,
        port_names=("n0", "n1"),
    )
    full.to_touchstone(path, format="ri")
    return full


def _raw_samples_at_measure(path: str) -> mw.NetworkData:
    """Read the raw Touchstone samples and select the measurement frequencies."""

    raw = mw.NetworkData.from_touchstone(path, device=DEVICE)
    idx = torch.tensor(MEASURE_INDICES, device=DEVICE)
    return mw.NetworkData(
        frequencies=raw.frequencies[idx],
        s=raw.s[idx],
        z0=raw.z0[idx],
        port_names=raw.port_names,
    )


def _embedded_input_reflection(path: str, order: int) -> torch.Tensor:
    block = mw.TouchstoneNetwork(
        name="dut",
        path=path,
        connections={"n0": "d1", "n1": "d2"},
        fit=mw.RationalFitConfig(order=order),
        device=DEVICE,
    )
    result = mw.Simulation.fdtd(
        _scene(networks=(block,)),
        frequencies=MEASURE_FREQUENCIES,
        excitations=mw.PortExcitation("d0", source_time=SOURCE_TIME),
        run_time=mw.TimeConfig(time_steps=TIME_STEPS),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    port = result.port("d0")
    return (port.b / port.a).reshape(-1)


@pytest.mark.parametrize(
    "tag, model_factory, order",
    (
        ("lossy_attenuator", _lossy_model, 1),
        ("reactive", _reactive_model, 2),
    ),
)
def test_raw_sample_cascade_matches_embedded_run(tag, model_factory, order, tmp_path):
    # Bare three-port device S measured once per network (the sweep is cheap;
    # keeping it inside the test keeps every network self-contained).
    bare = _measure_bare_three_port()

    path = os.path.join(str(tmp_path), f"{tag}.s2p")
    _write_touchstone(model_factory(), path)
    raw = _raw_samples_at_measure(path)

    # Reference path: raw samples + first-principles connection algebra.
    reference = bare.cascade(raw, port_map={"d1": "n0", "d2": "n1"})
    assert reference.port_names == ("d0",)
    reference_s11 = reference.s[:, 0, 0]

    # Independence-of-effect check: the connected network must actually change
    # the input reflection well above the tolerance, otherwise the cross-check
    # would pass trivially even if the connection algebra were wrong.
    network_effect = torch.max(torch.abs(bare.s[:, 0, 0] - reference_s11)).item()
    assert network_effect > 10.0 * CROSS_CHECK_TOLERANCE

    # Embedded path: rational fit + state-space time-domain embedding.
    embedded_s11 = _embedded_input_reflection(path, order)

    residual = torch.max(torch.abs(embedded_s11 - reference_s11)).item()
    assert residual < CROSS_CHECK_TOLERANCE, (
        f"{tag}: |dS|={residual:.3e} exceeds {CROSS_CHECK_TOLERANCE:.1e} "
        f"(network effect {network_effect:.3e})"
    )
