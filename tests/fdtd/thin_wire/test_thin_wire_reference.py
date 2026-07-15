import math

import pytest
import torch

from witwin.maxwell.fdtd.thin_wire_reference import (
    ACCEPTANCE_BUDGET,
    EPSILON_0,
    MU_0,
    AxisAlignedWireReference,
    WireReferenceState,
    assemble_axis_aligned_coefficients,
    coupling_distance,
    wire_per_unit_parameters,
)


def _ring_reference(*, dt_fraction=0.4):
    dtype = torch.float64
    incidence = torch.tensor(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
        dtype=dtype,
    )
    provisional = AxisAlignedWireReference(
        incidence=incidence,
        sampling=torch.empty((4, 0), dtype=dtype),
        segment_inductance=torch.full((4,), 2.0, dtype=dtype),
        node_capacitance=torch.full((4,), 0.5, dtype=dtype),
        field_mass=torch.empty((0,), dtype=dtype),
        dt=1.0,
    )
    return AxisAlignedWireReference(
        incidence=incidence,
        sampling=provisional.sampling,
        segment_inductance=provisional.segment_inductance,
        node_capacitance=provisional.node_capacitance,
        field_mass=provisional.field_mass,
        dt=dt_fraction * provisional.maximum_stable_dt(),
    )


def test_acceptance_budget_is_frozen_to_the_plan_limits():
    assert ACCEPTANCE_BUDGET.reference_rtol == 1.0e-5
    assert ACCEPTANCE_BUDGET.analytic_relative_error == 2.0e-2
    assert ACCEPTANCE_BUDGET.energy_charge_relative_error == 1.0e-2
    assert ACCEPTANCE_BUDGET.gradient_relative_error == 2.0e-2
    assert ACCEPTANCE_BUDGET.convergence_levels == 3
    assert ACCEPTANCE_BUDGET.no_wire_runtime_regression == 1.0e-2


def test_kernel_distance_is_tied_to_coupling_support_not_voxel_radius():
    spacing = torch.tensor([1.0e-3, 1.0e-3], dtype=torch.float64)
    selected = coupling_distance(spacing, method="bspline")
    historical = coupling_distance(spacing, method="legacy_edge")

    assert 0.2 * spacing[0] < selected < spacing[0]
    assert historical == pytest.approx(0.230e-3)
    assert not torch.isclose(selected, historical)


def test_square_bspline_kernel_distance_matches_analytic_oracle():
    spacing = torch.tensor([1.0e-3, 1.0e-3], dtype=torch.float64)
    measured = coupling_distance(spacing, method="bspline") / spacing[0]
    expected = torch.tensor(
        math.exp(math.log(2.0) / 3.0 + math.pi / 3.0 - 25.0 / 12.0),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        measured,
        expected,
        rtol=ACCEPTANCE_BUDGET.reference_rtol,
        atol=0.0,
    )


def test_fixed_physical_radius_remains_distinct_across_tenfold_grid_sweep():
    radius = torch.tensor(1.0e-5, dtype=torch.float64)
    spacings = (5.0e-4, 2.0e-3, 5.0e-3)
    values = [wire_per_unit_parameters(radius, (spacing, spacing)) for spacing in spacings]

    assert spacings[-1] / spacings[0] == 10.0
    expected_speed = torch.tensor(1.0 / math.sqrt(MU_0 * EPSILON_0), dtype=torch.float64)
    for value in values:
        expected_inductance = MU_0 * torch.log(value.coupling_distance / radius) / (2.0 * math.pi)
        expected_capacitance = MU_0 * EPSILON_0 / expected_inductance
        torch.testing.assert_close(value.inductance, expected_inductance)
        torch.testing.assert_close(value.capacitance, expected_capacitance)
        torch.testing.assert_close(value.wave_speed, expected_speed)
    assert all(left.inductance < right.inductance for left, right in zip(values, values[1:]))
    assert all(left.capacitance > right.capacitance for left, right in zip(values, values[1:]))
    assert all(
        not torch.isclose(left.impedance, right.impedance)
        for left, right in zip(values, values[1:])
    )


def test_radius_sweep_preserves_wave_speed_and_uses_physical_radius():
    radii = torch.logspace(-5, -4, 9, dtype=torch.float64)
    values = wire_per_unit_parameters(
        radii,
        torch.tensor([1.0e-3, 1.5e-3], dtype=torch.float64),
    )
    expected_speed = torch.tensor(1.0 / math.sqrt(MU_0 * EPSILON_0), dtype=torch.float64)
    torch.testing.assert_close(values.wave_speed, expected_speed.expand_as(values.wave_speed))
    assert torch.all(values.impedance[:-1] > values.impedance[1:])
    assert torch.unique(values.impedance).numel() == radii.numel()


def test_radius_gradient_matches_the_analytic_impedance_derivative():
    radius = torch.tensor(4.0e-5, dtype=torch.float64, requires_grad=True)
    parameters = wire_per_unit_parameters(radius, (1.0e-3, 1.5e-3))
    (gradient,) = torch.autograd.grad(parameters.impedance, radius)
    eta = math.sqrt(MU_0 / EPSILON_0)
    expected = torch.tensor(-eta / (2.0 * math.pi * float(radius.detach())), dtype=torch.float64)
    torch.testing.assert_close(gradient, expected)


def test_per_length_coefficients_are_integrated_on_segments_and_dual_nodes():
    parameters = wire_per_unit_parameters(2.0e-5, (1.0e-3, 1.5e-3))
    segment_lengths = torch.tensor([0.1, 0.25], dtype=torch.float64)
    node_dual_lengths = torch.tensor([0.05, 0.175, 0.125], dtype=torch.float64)
    coefficients = assemble_axis_aligned_coefficients(
        parameters,
        segment_lengths,
        node_dual_lengths,
    )

    torch.testing.assert_close(
        coefficients.segment_inductance,
        parameters.inductance * segment_lengths,
    )
    torch.testing.assert_close(
        coefficients.node_capacitance,
        parameters.capacitance * node_dual_lengths,
    )

    vector_parameters = wire_per_unit_parameters(
        torch.tensor([2.0e-5, 3.0e-5], dtype=torch.float64),
        (1.0e-3, 1.5e-3),
    )
    with pytest.raises(ValueError, match="requires scalar"):
        assemble_axis_aligned_coefficients(
            vector_parameters,
            segment_lengths,
            node_dual_lengths,
        )


def test_reference_rejects_nonfloating_nonfinite_and_mixed_coefficients():
    incidence = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    common = {
        "incidence": incidence,
        "sampling": torch.empty((1, 0), dtype=torch.float64),
        "segment_inductance": torch.ones((1,), dtype=torch.float64),
        "node_capacitance": torch.ones((2,), dtype=torch.float64),
        "field_mass": torch.empty((0,), dtype=torch.float64),
        "dt": 0.1,
    }
    with pytest.raises(TypeError, match="real floating-point"):
        AxisAlignedWireReference(**(common | {"incidence": incidence.to(torch.int64)}))
    with pytest.raises(ValueError, match="finite"):
        AxisAlignedWireReference(
            **(common | {"segment_inductance": torch.tensor([math.nan], dtype=torch.float64)})
        )
    with pytest.raises(ValueError, match="share one dtype and device"):
        AxisAlignedWireReference(
            **(common | {"node_capacitance": torch.ones((2,), dtype=torch.float32)})
        )
    with pytest.raises(ValueError, match="dt must be finite"):
        AxisAlignedWireReference(**(common | {"dt": math.nan}))
    with pytest.raises(ValueError, match="dt must be a scalar"):
        AxisAlignedWireReference(**(common | {"dt": torch.tensor([0.1], dtype=torch.float64)}))


@pytest.mark.parametrize("value", [True, 1.0 + 2.0j, torch.tensor(1.0 + 2.0j)])
def test_physical_parameters_reject_boolean_and_complex_values(value):
    with pytest.raises(TypeError, match="must (?:be real|not be boolean)"):
        wire_per_unit_parameters(value, (1.0e-3, 1.0e-3))


def test_reference_rejects_state_shape_dtype_and_nonfinite_values():
    reference = _ring_reference()
    valid = WireReferenceState(
        torch.empty((0,), dtype=torch.float64),
        torch.zeros((4,), dtype=torch.float64),
        torch.zeros((4,), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="electric must have shape"):
        reference.step(
            WireReferenceState(torch.zeros((1,), dtype=torch.float64), valid.charge, valid.current_half)
        )
    with pytest.raises(ValueError, match="share the coefficient dtype"):
        reference.step(
            WireReferenceState(valid.electric, valid.charge.float(), valid.current_half)
        )
    with pytest.raises(ValueError, match="current_half must be finite"):
        reference.step(
            WireReferenceState(
                valid.electric,
                valid.charge,
                torch.full((4,), math.nan, dtype=torch.float64),
            )
        )


def _fundamental_phase_velocity(segment_count: int) -> tuple[float, float]:
    dtype = torch.float64
    physical_length = 1.0
    step = physical_length / segment_count
    radius = 1.0e-5
    parameters = wire_per_unit_parameters(radius, (step, step))
    coefficients = assemble_axis_aligned_coefficients(
        parameters,
        torch.full((segment_count,), step, dtype=dtype),
        torch.full((segment_count,), step, dtype=dtype),
    )
    incidence = torch.zeros((segment_count, segment_count), dtype=dtype)
    indices = torch.arange(segment_count)
    incidence[indices, indices] = 1.0
    incidence[(indices + 1) % segment_count, indices] = -1.0
    provisional = AxisAlignedWireReference(
        incidence=incidence,
        sampling=torch.empty((segment_count, 0), dtype=dtype),
        segment_inductance=coefficients.segment_inductance,
        node_capacitance=coefficients.node_capacitance,
        field_mass=torch.empty((0,), dtype=dtype),
        dt=1.0,
    )
    dt = 0.5 * provisional.maximum_stable_dt()
    reference = AxisAlignedWireReference(
        incidence=incidence,
        sampling=provisional.sampling,
        segment_inductance=provisional.segment_inductance,
        node_capacitance=provisional.node_capacitance,
        field_mass=provisional.field_mass,
        dt=dt,
    )
    coordinate = torch.arange(segment_count, dtype=dtype) * step
    mode = torch.cos(2.0 * math.pi * coordinate / physical_length)
    charge = coefficients.node_capacitance * mode
    electric = torch.empty((0,), dtype=dtype)
    current = torch.zeros((segment_count,), dtype=dtype)
    state = WireReferenceState(
        electric,
        charge,
        reference.initialize_current_half(electric, charge, current),
    )
    after = reference.step(state)
    cosine = torch.dot(after.charge, mode) / torch.dot(charge, mode)
    phase = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    wavenumber = 2.0 * math.pi / physical_length
    return float(phase / (dt * wavenumber)), float(dt)


def test_reference_wave_speed_converges_over_three_grid_and_time_levels():
    levels = (16, 64, 256)
    measured = [_fundamental_phase_velocity(level) for level in levels]
    expected = 1.0 / math.sqrt(MU_0 * EPSILON_0)
    errors = [abs(speed / expected - 1.0) for speed, _ in measured]
    time_steps = [dt for _, dt in measured]

    assert levels[-1] / levels[0] == 16
    assert time_steps[0] > time_steps[1] > time_steps[2]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < ACCEPTANCE_BUDGET.analytic_relative_error


def test_sampling_and_deposition_are_exact_transposes():
    generator = torch.Generator().manual_seed(91)
    sampling = torch.randn((7, 11), dtype=torch.float64, generator=generator)
    electric = torch.randn((11,), dtype=torch.float64, generator=generator)
    current = torch.randn((7,), dtype=torch.float64, generator=generator)
    torch.testing.assert_close(
        torch.dot(sampling @ electric, current),
        torch.dot(electric, sampling.mT @ current),
    )


def test_axis_aligned_single_wire_exchanges_field_energy_without_growth():
    dtype = torch.float64
    provisional = AxisAlignedWireReference(
        incidence=torch.tensor([[1.0], [-1.0]], dtype=dtype),
        sampling=torch.tensor([[0.25]], dtype=dtype),
        segment_inductance=torch.tensor([0.8], dtype=dtype),
        node_capacitance=torch.tensor([0.6, 0.6], dtype=dtype),
        field_mass=torch.tensor([1.2], dtype=dtype),
        dt=1.0,
    )
    reference = AxisAlignedWireReference(
        incidence=provisional.incidence,
        sampling=provisional.sampling,
        segment_inductance=provisional.segment_inductance,
        node_capacitance=provisional.node_capacitance,
        field_mass=provisional.field_mass,
        dt=0.5 * provisional.maximum_stable_dt(),
    )
    electric = torch.tensor([1.0], dtype=dtype)
    charge = torch.zeros((2,), dtype=dtype)
    current = torch.zeros((1,), dtype=dtype)
    state = WireReferenceState(
        electric,
        charge,
        reference.initialize_current_half(electric, charge, current),
    )
    energies = []
    for _ in range(500):
        after = reference.step(state)
        energies.append(reference.staggered_energy(state, after.current_half))
        torch.testing.assert_close(
            reference.continuity_residual(state, after),
            torch.zeros_like(charge),
        )
        state = after

    energy = torch.stack(energies)
    assert (energy.amax() - energy.amin()) / energy.abs().mean() < 1.0e-12
    assert state.charge.abs().amax() > 0


def test_closed_lossless_network_conserves_staggered_energy_and_charge():
    reference = _ring_reference()
    electric = torch.empty((0,), dtype=torch.float64)
    charge = torch.tensor([0.3, -0.2, 0.1, -0.2], dtype=torch.float64)
    current_integer = torch.tensor([0.1, -0.05, 0.02, -0.03], dtype=torch.float64)
    state = WireReferenceState(
        electric,
        charge,
        reference.initialize_current_half(electric, charge, current_integer),
    )
    energies = []
    total_charge = state.charge.sum()
    for _ in range(2000):
        after = reference.step(state)
        energies.append(reference.staggered_energy(state, after.current_half))
        torch.testing.assert_close(reference.continuity_residual(state, after), torch.zeros_like(charge))
        torch.testing.assert_close(after.charge.sum(), total_charge)
        state = after

    energy = torch.stack(energies)
    relative_span = (energy.amax() - energy.amin()) / energy.abs().mean()
    assert relative_span < 1.0e-11


def test_stability_limit_separates_bounded_and_unstable_steps():
    stable = _ring_reference(dt_fraction=0.99)
    unstable = _ring_reference(dt_fraction=1.01)
    electric = torch.empty((0,), dtype=torch.float64)
    charge = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    current = torch.zeros((4,), dtype=torch.float64)

    def peak(reference):
        state = WireReferenceState(
            electric,
            charge,
            reference.initialize_current_half(electric, charge, current),
        )
        values = []
        for _ in range(80):
            state = reference.step(state)
            values.append(state.charge.abs().amax())
        return torch.stack(values).amax()

    assert peak(stable) < 2.0
    assert peak(unstable) > 1.0e3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reference_recurrence_remains_device_resident_on_cuda():
    cpu_reference = _ring_reference()
    reference = AxisAlignedWireReference(
        incidence=cpu_reference.incidence.cuda(),
        sampling=cpu_reference.sampling.cuda(),
        segment_inductance=cpu_reference.segment_inductance.cuda(),
        node_capacitance=cpu_reference.node_capacitance.cuda(),
        field_mass=cpu_reference.field_mass.cuda(),
        dt=cpu_reference.dt.cuda(),
    )
    electric = torch.empty((0,), dtype=torch.float64, device="cuda")
    charge = torch.tensor([0.3, -0.2, 0.1, -0.2], dtype=torch.float64, device="cuda")
    current = torch.zeros((4,), dtype=torch.float64, device="cuda")
    state = WireReferenceState(
        electric,
        charge,
        reference.initialize_current_half(electric, charge, current),
    )
    for _ in range(8):
        state = reference.step(state)

    assert state.electric.is_cuda
    assert state.charge.is_cuda
    assert state.current_half.is_cuda
