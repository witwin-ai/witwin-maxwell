from pathlib import Path

import pytest
import torch

import witwin.maxwell as mw


def _port(index: int, count: int) -> mw.LumpedPort:
    x = -0.03 + index * (0.06 / max(count - 1, 1))
    return mw.LumpedPort(
        name=f"field_{index}",
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, -0.0025),
            size=(0.005, 0.005, 0.0),
        ),
        reference_impedance=50.0,
    )


def _scene(port_count: int, *, network=None, device: str = "cpu") -> mw.Scene:
    return mw.Scene(
        domain=mw.Domain(
            bounds=((-0.04, 0.04), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.5e9),
            ),
        ),
        ports=tuple(_port(index, port_count) for index in range(port_count)),
        networks=() if network is None else (network,),
        device=device,
    )


def _two_port_filter(frequencies: torch.Tensor) -> mw.NetworkData:
    resistance = 50.0
    capacitance = 5.0e-12
    shunt = 2.0e-3
    incidence = torch.tensor((1.0, -1.0), dtype=torch.float64)
    model = mw.StateSpaceNetwork(
        A=torch.tensor(((-1.0 / (resistance * capacitance),),), dtype=torch.float64),
        B=(incidence / (resistance * capacitance)).reshape(1, 2),
        C=(-incidence / resistance).reshape(2, 1),
        D=(
            torch.outer(incidence, incidence) / resistance
            + shunt * torch.eye(2, dtype=torch.float64)
        ),
        representation="Y",
        port_order=("network_0", "network_1"),
    )
    return mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies),
        z0=50.0,
        port_names=model.port_order,
    )


def _four_port_direct(frequencies: torch.Tensor) -> mw.NetworkData:
    conductance = (
        0.015 * torch.eye(4, dtype=torch.float64)
        + 0.001 * torch.ones((4, 4), dtype=torch.float64)
    )
    values = conductance.to(torch.complex128).expand(frequencies.numel(), -1, -1)
    return mw.NetworkData.from_y(
        frequencies=frequencies,
        y=values,
        z0=50.0,
        port_names=tuple(f"network_{index}" for index in range(4)),
    )


@pytest.mark.parametrize("port_count", (2, 4))
def test_touchstone_multiport_fit_matches_independent_network_reference(
    tmp_path: Path,
    port_count: int,
) -> None:
    frequencies = torch.linspace(0.1e9, 5.0e9, 49, dtype=torch.float64)
    reference = (
        _two_port_filter(frequencies)
        if port_count == 2
        else _four_port_direct(frequencies)
    )
    path = tmp_path / f"fixture.s{port_count}p"
    reference.to_touchstone(path, format="ri")
    block = mw.TouchstoneNetwork(
        name="fixture",
        path=path,
        connections={
            f"network_{index}": f"field_{index}" for index in range(port_count)
        },
        fit=mw.RationalFitConfig(
            order=1,
            iterations=2,
            relative_tolerance=2.0e-3,
        ),
        device="cpu",
    )
    scene = _scene(port_count, network=block)

    (compiled,) = scene.compile_networks(dt=1.0e-12, device="cpu")
    predicted = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=compiled.continuous.evaluate(frequencies),
        z0=reference.z0,
        port_names=reference.port_names,
    )

    assert torch.max(torch.abs(predicted.s - reference.s)) < 0.02
    assert compiled.port_order == reference.port_names
    assert compiled.connection_names == tuple(
        f"field_{index}" for index in range(port_count)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_four_port_fdtd_currents_match_embedded_admittance() -> None:
    frequencies = torch.linspace(1.0e9, 5.0e9, 17, dtype=torch.float64)
    data = _four_port_direct(frequencies)
    conductance = data.to_y()[0].real
    model = mw.StateSpaceNetwork(
        A=torch.zeros((0, 0), dtype=torch.float64),
        B=torch.zeros((0, 4), dtype=torch.float64),
        C=torch.zeros((4, 0), dtype=torch.float64),
        D=conductance,
        representation="Y",
        port_order=data.port_names,
    )
    block = mw.NetworkBlock(
        name="four_port_load",
        network=data,
        connections={
            f"network_{index}": f"field_{index}" for index in range(4)
        },
        fit=False,
        model=model,
    )
    requested = (2.5e9, 3.0e9)

    result = mw.Simulation.fdtd(
        _scene(4, network=block, device="cuda"),
        frequencies=requested,
        run_time=mw.TimeConfig(time_steps=96),
        spectral_sampler=mw.SpectralSampler(window="none"),
        cuda_graph=True,
    ).run()

    diagnostics = result.embedded_network("four_port_load")
    expected_current = conductance.to(
        device=diagnostics.voltage.device,
        dtype=diagnostics.voltage.dtype,
    ) @ diagnostics.voltage
    scale = torch.max(torch.abs(expected_current))
    relative_error = torch.max(torch.abs(diagnostics.current - expected_current)) / scale
    assert relative_error < 0.02
    signed_total = torch.sum(
        0.5 * torch.real(
            diagnostics.voltage * torch.conj(diagnostics.current)
        ),
        dim=0,
    )
    incident_total = torch.stack(
        tuple(result.port(f"field_{index}").incident_power for index in range(4))
    ).sum(dim=0)
    assert torch.all(signed_total >= -1.0e-5 * incident_total)
    assert result.stats()["network_cuda_graph_active"] is True
