from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.ports import (
    accumulate_port_observers,
    apply_port_runtimes,
    prepare_port_spectral_accumulators,
)
from witwin.maxwell.fdtd.observers import _compute_plane_flux


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD requires CUDA"
)

_FREQUENCY = 1.0e9


def _wire_port_scene(*, termination=None) -> mw.Scene:
    wire = mw.ThinWire(
        name="dipole",
        points=(
            (-0.08, -0.08, 0.0),
            (-0.04, -0.04, 0.0),
            (0.04, 0.04, 0.0),
            (0.08, 0.08, 0.0),
        ),
        radius=2.0e-3,
        conductor=mw.WireConductor.pec(),
        endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
        snap="strict",
    )
    binding = mw.WirePortBinding.gap(
        negative=mw.WireNodeRef("dipole", 1),
        positive=mw.WireNodeRef("dipole", 2),
    )
    port = mw.LumpedPort(
        "feed",
        wire_binding=binding,
        reference_impedance=50.0,
        termination=termination,
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        thin_wires=(wire,),
        ports=(port,),
        device="cuda",
    )


def _wire_node_port_scene(*, termination=None) -> mw.Scene:
    negative = mw.ThinWire(
        "negative_arm",
        ((-0.08, -0.08, 0.0), (-0.04, -0.04, 0.0)),
        2.0e-3,
        mw.WireConductor.pec(),
        endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
    )
    positive = mw.ThinWire(
        "positive_arm",
        ((0.04, 0.04, 0.0), (0.08, 0.08, 0.0)),
        2.0e-3,
        mw.WireConductor.pec(),
        endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
    )
    port = mw.TerminalPort(
        "feed",
        wire_binding=mw.WirePortBinding.nodes(
            negative=mw.WireNodeRef("negative_arm", 1),
            positive=mw.WireNodeRef("positive_arm", 0),
        ),
        termination=termination,
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        thin_wires=(negative, positive),
        ports=(port,),
        device="cuda",
    )


def _generalized_coordinate(solver, port_runtime) -> torch.Tensor:
    provider = port_runtime.wire_provider
    assert provider is not None
    geometry = provider.geometry
    values = solver.Ex.new_empty(geometry.edge_offsets.numel())
    for component, name in enumerate(("Ex", "Ey", "Ez")):
        selected = torch.nonzero(
            geometry.edge_components == component,
            as_tuple=False,
        ).reshape(-1)
        if selected.numel() == 0:
            continue
        values.index_copy_(
            0,
            selected,
            getattr(solver, name)
            .reshape(-1)
            .index_select(
                0,
                geometry.edge_offsets.index_select(0, selected),
            ),
        )
    node_voltage = solver._wire_runtime.charge.index_select(
        0, provider.node_ids
    ) / provider.node_capacitance
    return torch.cat((values, node_voltage))


def test_gap_compiler_builds_cuda_csr_and_removes_source_segment_from_wire_state():
    scene = _wire_port_scene()
    network = scene.compile_thin_wires()
    geometry = scene.compile_ports()[0]

    assert network.port_binding_names == ("feed",)
    assert network.port_binding_kinds == ("gap",)
    torch.testing.assert_close(
        network.port_gap_offsets,
        torch.tensor((0, geometry.edge_weights.numel()), device="cuda"),
    )
    torch.testing.assert_close(
        geometry.edge_components,
        network.port_gap_edge_components,
    )
    torch.testing.assert_close(geometry.edge_offsets, network.port_gap_edge_offsets)
    torch.testing.assert_close(geometry.edge_weights, network.port_gap_weights)
    assert geometry.edge_weights.device.type == "cuda"
    assert geometry.edge_weights.numel() > 2

    # Source segment 1 is the declared feed gap. Its grid fragments belong only
    # to the port row; the ordinary I/q recurrence retains the two conductor arms.
    torch.testing.assert_close(
        network.segment_source_ids,
        torch.tensor((0, 2), device="cuda"),
    )
    assert network.segment_count == 2

    component_sums = torch.zeros(3, device="cuda", dtype=torch.float64)
    component_sums.index_add_(
        0,
        geometry.edge_components.to(dtype=torch.int64),
        geometry.edge_weights.to(dtype=torch.float64),
    )
    torch.testing.assert_close(
        component_sums,
        torch.tensor((0.08, 0.08, 0.0), device="cuda", dtype=torch.float64),
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def test_passive_gap_runtime_conserves_charge_and_closes_one_step_energy_on_cuda():
    solver = mw.Simulation.fdtd(
        _wire_port_scene(termination=mw.SeriesRLC(r=50.0)),
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=4),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver
    port_runtime = solver._port_runtimes[0]
    provider = port_runtime.wire_provider
    lumped = port_runtime.lumped

    assert provider is not None
    assert lumped is not None
    assert provider.coordinate.device.type == "cuda"
    assert provider.energy_masses.device.type == "cuda"
    assert provider.node_ids.device.type == "cuda"
    assert lumped.linear_indices.device.type == "cuda"

    geometry = provider.geometry
    prescribed = torch.linspace(
        40.0,
        120.0,
        geometry.edge_offsets.numel(),
        device="cuda",
        dtype=solver.Ex.dtype,
    )
    for component, name in enumerate(("Ex", "Ey", "Ez")):
        selected = torch.nonzero(
            geometry.edge_components == component,
            as_tuple=False,
        ).reshape(-1)
        if selected.numel() == 0:
            continue
        getattr(solver, name).reshape(-1).index_copy_(
            0,
            geometry.edge_offsets.index_select(0, selected),
            prescribed.index_select(0, selected),
        )

    charge_before = solver._wire_runtime.charge.clone()
    coordinate_before = _generalized_coordinate(solver, port_runtime)
    masses = provider.energy_masses.reshape(-1)
    energy_before = 0.5 * torch.sum(masses * coordinate_before.square())

    apply_port_runtimes(solver)

    coordinate_after = _generalized_coordinate(solver, port_runtime)
    energy_after = 0.5 * torch.sum(masses * coordinate_after.square())
    branch_current = lumped.last_branch_current.clone()
    charge_after = solver._wire_runtime.charge
    negative_node, positive_node = provider.node_ids

    torch.testing.assert_close(
        charge_after[negative_node] - charge_before[negative_node],
        -solver.dt * branch_current,
        rtol=2.0e-5,
        atol=1.0e-18,
    )
    torch.testing.assert_close(
        charge_after[positive_node] - charge_before[positive_node],
        solver.dt * branch_current,
        rtol=2.0e-5,
        atol=1.0e-18,
    )
    torch.testing.assert_close(
        torch.sum(charge_after - charge_before),
        torch.zeros((), device="cuda", dtype=charge_after.dtype),
        rtol=0.0,
        atol=1.0e-18,
    )
    torch.testing.assert_close(
        energy_after - energy_before,
        lumped.last_field_energy_change,
        rtol=3.0e-5,
        atol=1.0e-16,
    )
    assert float(lumped.last_dissipated_energy) > 0.0


def test_active_gap_port_runs_finite_fdtd_and_returns_wire_provider_port_data():
    simulation = mw.Simulation.fdtd(
        _wire_port_scene(),
        frequencies=(_FREQUENCY, 1.25e9),
        excitations=mw.PortExcitation(
            "feed",
            source_impedance="matched",
            source_time=mw.GaussianPulse(frequency=_FREQUENCY, fwidth=0.5e9),
        ),
        run_time=mw.TimeConfig(time_steps=64),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )

    result = simulation.run()
    data = result.port("feed")

    assert data.frequencies.device.type == "cuda"
    assert data.voltage.device.type == "cuda"
    assert data.current.device.type == "cuda"
    assert data.voltage.shape == data.current.shape == (2,)
    assert torch.all(torch.isfinite(data.voltage))
    assert torch.all(torch.isfinite(data.current))
    assert torch.any(torch.abs(data.voltage) > 0.0)
    assert torch.any(torch.abs(data.current) > 0.0)
    assert data.available_power is not None
    assert torch.all(data.available_power > 0.0)
    assert torch.all(data.accepted_power >= 0.0)
    assert data.metadata["provider"] == "wire_gap"
    assert data.metadata["current_convention"] == "entering_wire_network"
    assert data.metadata["wire_binding"]["source_segment"] == 1
    assert result.stats()["thin_wire_enabled"] is True
    assert result.stats()["thin_wire_segments"] == 2
    assert result.stats()["num_ports"] == 1


def test_node_bound_port_is_charge_conservative_and_has_no_gap_field_coordinate():
    solver = mw.Simulation.fdtd(
        _wire_node_port_scene(termination=mw.SeriesRLC(r=50.0)),
        frequency=_FREQUENCY,
        run_time=mw.TimeConfig(time_steps=2),
    ).prepare().solver
    runtime = solver._port_runtimes[0]
    provider = runtime.wire_provider
    lumped = runtime.lumped

    assert provider is not None
    assert lumped is not None
    assert provider.field_count == 0
    assert provider.geometry.edge_weights.numel() == 0
    assert provider.voltage_weights.tolist() == [1.0, -1.0]

    charge = solver._wire_runtime.charge
    negative_node, positive_node = provider.node_ids
    charge[negative_node] = provider.node_capacitance[0]
    charge[positive_node] = -provider.node_capacitance[1]
    charge_before = charge.clone()
    apply_port_runtimes(solver)

    delta = charge - charge_before
    torch.testing.assert_close(
        delta[negative_node],
        -solver.dt * lumped.last_branch_current,
        rtol=2.0e-5,
        atol=1.0e-18,
    )
    torch.testing.assert_close(
        delta[positive_node],
        solver.dt * lumped.last_branch_current,
        rtol=2.0e-5,
        atol=1.0e-18,
    )
    torch.testing.assert_close(delta.sum(), torch.zeros_like(delta.sum()), atol=1.0e-18, rtol=0.0)


def test_wire_gap_port_step_has_no_scalar_sync_or_host_device_copy():
    solver = mw.Simulation.fdtd(
        _wire_port_scene(),
        frequency=_FREQUENCY,
        excitations=mw.PortExcitation(
            "feed",
            source_time=mw.CW(_FREQUENCY),
        ),
        run_time=mw.TimeConfig(time_steps=32),
    ).prepare().solver
    prepare_port_spectral_accumulators(solver, 32, "none")

    for _ in range(4):
        apply_port_runtimes(solver)
        accumulate_port_observers(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        acc_events=True,
    ) as profile:
        for _ in range(16):
            apply_port_runtimes(solver)
            accumulate_port_observers(solver)
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    assert "aten::item" not in event_names
    assert "aten::_local_scalar_dense" not in event_names
    assert not any(
        "Memcpy HtoD" in name or "Memcpy DtoH" in name
        for name in event_names
    )


def test_pec_wire_port_accepted_power_closes_against_radiated_surface_flux():
    frequency = 3.0e9
    arm_segments = 32
    negative = torch.linspace(-0.025, -0.00025, arm_segments + 1)
    positive = torch.linspace(0.00025, 0.025, arm_segments + 1)
    negative_wire = mw.ThinWire(
        "negative_arm",
        tuple((0.0, 0.0, float(value)) for value in negative),
        1.0e-4,
        mw.WireConductor.pec(),
        snap="continuous",
    )
    positive_wire = mw.ThinWire(
        "positive_arm",
        tuple((0.0, 0.0, float(value)) for value in positive),
        1.0e-4,
        mw.WireConductor.pec(),
        snap="continuous",
    )
    port = mw.LumpedPort(
        "feed",
        wire_binding=mw.WirePortBinding.nodes(
            negative=mw.WireNodeRef("negative_arm", arm_segments),
            positive=mw.WireNodeRef("positive_arm", 0),
        ),
    )
    surface = mw.ClosedSurfaceMonitor.box(
        "radiation",
        position=(0.0, 0.0, 0.0),
        size=(0.09, 0.09, 0.09),
        frequencies=(frequency,),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.05, 0.05),) * 3),
        grid=mw.GridSpec.uniform(1.25e-3),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        thin_wires=(negative_wire, positive_wire),
        ports=(port,),
        monitors=(surface,),
        device="cuda",
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequency=frequency,
        excitations=mw.PortExcitation(
            "feed",
            source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.5e9),
        ),
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    prepared = simulation.prepare()
    simulation.config.run_time = mw.TimeConfig(
        time_steps=math.ceil(8.0e-9 / float(prepared.solver.dt))
    )
    result = prepared.run()
    port_data = result.port("feed")
    accepted = port_data.accepted_power[0]
    payload = result.monitor("radiation")
    radiated = sum(
        _compute_plane_flux(face)
        for face in payload["faces"].values()
    )
    closure = torch.abs(accepted - radiated) / torch.abs(accepted)
    assert float(accepted) > 0.0
    assert float(radiated) > 0.0
    assert float(closure) < 0.01
