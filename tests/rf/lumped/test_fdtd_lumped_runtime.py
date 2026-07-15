from __future__ import annotations

import pytest
import torch

from witwin.maxwell.compiler.ports import CompiledPortGeometry
from witwin.maxwell.fdtd.lumped import apply_lumped_runtime, prepare_lumped_runtime
from witwin.maxwell.network import voltage_current_to_power_waves
from witwin.maxwell.lumped import ParallelRLC, SeriesRLC


def _geometry(device: torch.device, *, orientation: int = 1) -> CompiledPortGeometry:
    indices = torch.tensor(
        ((0, 0, 0), (1, 0, 0)),
        device=device,
        dtype=torch.int64,
    )
    weights = orientation * torch.tensor((0.4, 0.6), device=device, dtype=torch.float64)
    return CompiledPortGeometry(
        port_name="p1",
        axis="x",
        direction=orientation,
        voltage_component="Ex",
        voltage_indices=indices,
        voltage_weights=weights,
        current_components=(),
        current_indices=(),
        current_weights=(),
        reference_impedance=50.0,
    )


def _materials(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    eps_edge = torch.tensor(
        (
            ((2.0, 1.2), (1.5, 1.1)),
            ((3.0, 1.4), (1.6, 1.3)),
            ((2.5, 1.8), (1.7, 1.9)),
        ),
        device=device,
        dtype=torch.float64,
    )
    volume = torch.tensor(
        (
            ((0.5, 0.7), (0.6, 0.8)),
            ((0.25, 0.9), (0.55, 0.65)),
            ((0.4, 0.75), (0.45, 0.85)),
        ),
        device=device,
        dtype=torch.float64,
    )
    return eps_edge, volume


def _field_energy(
    field: torch.Tensor,
    eps_edge: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * torch.sum(eps_edge * volume * field.square())


def _prepare(
    device: torch.device,
    *,
    orientation: int = 1,
    resistance: float | torch.Tensor | None = None,
    termination=None,
    thevenin_voltage: float = 0.0,
):
    eps_edge, volume = _materials(device)
    resolved_resistance = 0.8 if resistance is None and termination is None else resistance
    runtime = prepare_lumped_runtime(
        _geometry(device, orientation=orientation),
        dt=torch.tensor(0.2, device=device, dtype=torch.float64),
        eps_edge=eps_edge,
        yee_control_volume=volume,
        resistance=(
            None
            if resolved_resistance is None
            else torch.as_tensor(resolved_resistance, device=device, dtype=torch.float64)
        ),
        termination=termination,
        thevenin_voltage=torch.tensor(thevenin_voltage, device=device, dtype=torch.float64),
    )
    return runtime, eps_edge, volume


def test_prepare_builds_energy_consistent_injection_vector():
    device = torch.device("cpu")
    geometry = _geometry(device)
    runtime, eps_edge, volume = _prepare(device)
    local_capacitance = torch.tensor(
        (eps_edge[0, 0, 0] * volume[0, 0, 0], eps_edge[1, 0, 0] * volume[1, 0, 0])
    )
    expected_q = runtime.dt * geometry.voltage_weights / local_capacitance

    torch.testing.assert_close(runtime.injection, expected_q)
    torch.testing.assert_close(
        runtime.coupling_impedance,
        torch.dot(geometry.voltage_weights, expected_q),
    )
    assert runtime.injection.device == eps_edge.device
    assert runtime.edge_buffer.device == eps_edge.device
    assert runtime.correction_buffer.device == eps_edge.device


def test_resistive_termination_removes_exactly_its_dissipated_energy():
    device = torch.device("cpu")
    runtime, eps_edge, volume = _prepare(device, resistance=0.7)
    field = torch.tensor(
        (
            ((0.8, 0.1), (-0.2, 0.05)),
            ((-0.35, 0.02), (0.15, -0.1)),
            ((0.4, -0.25), (0.3, 0.12)),
        ),
        dtype=torch.float64,
    )
    energy_before = _field_energy(field, eps_edge, volume)

    apply_lumped_runtime(runtime, field)
    energy_after = _field_energy(field, eps_edge, volume)

    torch.testing.assert_close(
        energy_after - energy_before,
        -runtime.last_dissipated_energy,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert runtime.last_dissipated_energy > 0.0
    torch.testing.assert_close(runtime.last_source_work, torch.zeros_like(runtime.last_source_work))


def test_thevenin_excitation_supplies_field_and_resistor_energy():
    device = torch.device("cpu")
    runtime, eps_edge, volume = _prepare(
        device,
        resistance=0.4,
        thevenin_voltage=1.25,
    )
    field = torch.zeros_like(eps_edge)
    energy_before = _field_energy(field, eps_edge, volume)

    branch_current = apply_lumped_runtime(runtime, field)
    energy_after = _field_energy(field, eps_edge, volume)

    assert branch_current < 0.0
    assert runtime.last_source_work > 0.0
    assert runtime.last_dissipated_energy > 0.0
    torch.testing.assert_close(
        energy_after - energy_before,
        runtime.last_source_work - runtime.last_dissipated_energy,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def _port_subspace_field(runtime, *, sign: float = 1.0) -> torch.Tensor:
    field = torch.zeros(runtime.field_shape, device=runtime.injection.device, dtype=runtime.field_dtype)
    field.view(-1).index_copy_(
        0,
        runtime.linear_indices,
        sign * runtime.injection,
    )
    return field


def test_open_short_and_discrete_matched_limits():
    device = torch.device("cpu")
    probe, _, _ = _prepare(device)
    matched_resistance = probe.discrete_port_impedance

    open_runtime, _, _ = _prepare(device, resistance=float("inf"))
    open_field = _port_subspace_field(open_runtime)
    open_before = open_field.clone()
    apply_lumped_runtime(open_runtime, open_field)
    torch.testing.assert_close(open_field, open_before)
    torch.testing.assert_close(
        open_runtime.last_branch_current,
        torch.zeros_like(open_runtime.last_branch_current),
    )

    short_runtime, _, _ = _prepare(device, resistance=0.0)
    short_field = _port_subspace_field(short_runtime)
    short_before_voltage = torch.dot(short_runtime.voltage_weights, short_runtime.injection)
    apply_lumped_runtime(short_runtime, short_field)
    torch.testing.assert_close(short_field, -_port_subspace_field(short_runtime))
    torch.testing.assert_close(short_runtime.last_voltage_after, -short_before_voltage)
    torch.testing.assert_close(
        short_runtime.last_dissipated_energy,
        torch.zeros_like(short_runtime.last_dissipated_energy),
        atol=1.0e-15,
        rtol=0.0,
    )

    matched_runtime, _, _ = _prepare(device, resistance=matched_resistance)
    matched_field = _port_subspace_field(matched_runtime)
    apply_lumped_runtime(matched_runtime, matched_field)
    torch.testing.assert_close(matched_field, torch.zeros_like(matched_field), atol=1.0e-15, rtol=0.0)
    torch.testing.assert_close(
        matched_runtime.last_voltage_after,
        torch.zeros_like(matched_runtime.last_voltage_after),
        atol=1.0e-15,
        rtol=0.0,
    )


def test_fifty_ohm_match_and_open_short_power_wave_exit_gate():
    device = torch.device("cpu")
    cases = ((50.0, 0.0), (float("inf"), 1.0), (0.0, -1.0))

    for resistance, expected_gamma in cases:
        runtime, _, _ = _prepare(device, resistance=resistance)
        field = _port_subspace_field(runtime)
        apply_lumped_runtime(runtime, field)
        voltage = torch.complex(
            runtime.last_voltage_midpoint,
            torch.zeros_like(runtime.last_voltage_midpoint),
        ).unsqueeze(0)
        current = torch.complex(
            runtime.last_branch_current,
            torch.zeros_like(runtime.last_branch_current),
        ).unsqueeze(0)
        incident, reflected = voltage_current_to_power_waves(voltage, current, 50.0)
        gamma = reflected / incident

        torch.testing.assert_close(
            gamma,
            torch.full_like(gamma, expected_gamma),
            rtol=0.0,
            atol=1.0e-12,
        )
        if resistance == 50.0:
            return_loss_db = -20.0 * torch.log10(torch.clamp(torch.abs(gamma), min=1.0e-30))
            assert return_loss_db > 30.0


def test_orientation_reversal_preserves_field_update_and_energy():
    device = torch.device("cpu")
    forward, _, _ = _prepare(device, orientation=1, resistance=0.9)
    reverse, _, _ = _prepare(device, orientation=-1, resistance=0.9)
    initial = torch.tensor(
        (
            ((0.7, -0.1), (0.2, 0.3)),
            ((-0.25, 0.15), (-0.05, 0.4)),
            ((0.1, -0.2), (0.05, 0.12)),
        ),
        dtype=torch.float64,
    )
    forward_field = initial.clone()
    reverse_field = initial.clone()

    apply_lumped_runtime(forward, forward_field)
    apply_lumped_runtime(reverse, reverse_field)

    torch.testing.assert_close(reverse_field, forward_field)
    torch.testing.assert_close(reverse.last_branch_current, -forward.last_branch_current)
    torch.testing.assert_close(reverse.last_voltage_before, -forward.last_voltage_before)
    torch.testing.assert_close(reverse.last_dissipated_energy, forward.last_dissipated_energy)


def _stored_branch_energy(runtime) -> torch.Tensor:
    return (
        0.5 * runtime.inductance * runtime.inductor_current.square()
        + 0.5 * runtime.capacitance * runtime.capacitor_voltage.square()
    )


@pytest.mark.parametrize(
    "termination",
    (
        SeriesRLC(l=0.7),
        SeriesRLC(c=0.9),
        SeriesRLC(r=0.25, l=0.7, c=0.9),
    ),
)
def test_series_rlc_midpoint_coupling_balances_field_storage_and_loss(termination):
    device = torch.device("cpu")
    runtime, eps_edge, volume = _prepare(device, termination=termination)
    field = _port_subspace_field(runtime)

    for _ in range(8):
        field_energy_before = _field_energy(field, eps_edge, volume)
        branch_energy_before = _stored_branch_energy(runtime)
        apply_lumped_runtime(runtime, field)
        field_energy_after = _field_energy(field, eps_edge, volume)
        branch_energy_after = _stored_branch_energy(runtime)

        torch.testing.assert_close(
            (field_energy_after - field_energy_before)
            + (branch_energy_after - branch_energy_before),
            -runtime.last_dissipated_energy,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            runtime.last_stored_energy_change,
            branch_energy_after - branch_energy_before,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


@pytest.mark.parametrize(
    "termination",
    (
        ParallelRLC(l=0.7),
        ParallelRLC(c=0.9),
        ParallelRLC(r=0.25, l=0.7, c=0.9),
    ),
)
def test_parallel_rlc_midpoint_coupling_balances_energy_and_branch_currents(termination):
    device = torch.device("cpu")
    runtime, eps_edge, volume = _prepare(device, termination=termination)
    field = _port_subspace_field(runtime)

    for _ in range(8):
        field_energy_before = _field_energy(field, eps_edge, volume)
        branch_energy_before = _stored_branch_energy(runtime)
        apply_lumped_runtime(runtime, field)
        field_energy_after = _field_energy(field, eps_edge, volume)
        branch_energy_after = _stored_branch_energy(runtime)

        torch.testing.assert_close(
            runtime.last_branch_current,
            runtime.last_resistor_current
            + runtime.last_capacitor_current
            + runtime.last_inductor_current_midpoint,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            (field_energy_after - field_energy_before)
            + (branch_energy_after - branch_energy_before),
            -runtime.last_dissipated_energy,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def _run_driven_steps(device: torch.device, topology: str):
    if topology == "resistor":
        runtime, _, _ = _prepare(device, resistance=0.65, thevenin_voltage=0.0)
    else:
        model_type = SeriesRLC if topology == "series" else ParallelRLC
        termination = model_type(
            r=torch.tensor(0.65, device=device, dtype=torch.float64),
            l=torch.tensor(0.35, device=device, dtype=torch.float64),
            c=torch.tensor(0.8, device=device, dtype=torch.float64),
        )
        runtime, _, _ = _prepare(
            device,
            termination=termination,
            thevenin_voltage=0.0,
        )
    field = torch.linspace(-0.4, 0.7, 12, device=device, dtype=torch.float64).reshape(3, 2, 2)
    drives = torch.tensor((0.1, -0.25, 0.4, 0.0, 0.2), device=device, dtype=torch.float64)
    edge_buffer_address = runtime.edge_buffer.data_ptr()
    correction_buffer_address = runtime.correction_buffer.data_ptr()
    inductor_state_address = runtime.inductor_current.data_ptr()
    capacitor_state_address = runtime.capacitor_voltage.data_ptr()
    for drive in drives.unbind():
        apply_lumped_runtime(runtime, field, thevenin_voltage=drive)
    assert runtime.edge_buffer.data_ptr() == edge_buffer_address
    assert runtime.correction_buffer.data_ptr() == correction_buffer_address
    assert runtime.inductor_current.data_ptr() == inductor_state_address
    assert runtime.capacitor_voltage.data_ptr() == capacitor_state_address
    return runtime, field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("topology", ("resistor", "series", "parallel"))
def test_lumped_runtime_cpu_cuda_parity_and_buffer_reuse(topology):
    cpu_runtime, cpu_field = _run_driven_steps(torch.device("cpu"), topology)
    cuda_runtime, cuda_field = _run_driven_steps(torch.device("cuda"), topology)

    torch.testing.assert_close(cuda_field, cpu_field.to("cuda"), rtol=1.0e-13, atol=1.0e-13)
    torch.testing.assert_close(
        cuda_runtime.last_branch_current,
        cpu_runtime.last_branch_current.to("cuda"),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        cuda_runtime.last_field_energy_change,
        cpu_runtime.last_field_energy_change.to("cuda"),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
