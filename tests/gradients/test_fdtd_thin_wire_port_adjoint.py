from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import lumped_state_name
from witwin.maxwell.fdtd.ports import (
    prepare_port_runtimes,
    pullback_port_runtimes,
    replay_port_runtimes,
)
from tests.gradients.finite_difference_gate import assert_finite_difference_agrees


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD requires CUDA"
)

_FREQUENCY = 1.0e9

_assert_finite_difference_agrees = assert_finite_difference_agrees

_STRICT_POINTS = (
    (-0.08, -0.08, 0.0),
    (-0.04, -0.04, 0.0),
    (0.04, 0.04, 0.0),
    (0.08, 0.08, 0.0),
)

_CONTINUOUS_POINTS = (
    (-0.11, -0.07, 0.005),
    (-0.06, -0.079, 0.01),
    (0.039, 0.021, 0.025),
    (0.038, 0.075, 0.035),
)


def _scene(
    *,
    radius=2.0e-3,
    termination=None,
    points=_STRICT_POINTS,
    snap="strict",
) -> mw.Scene:
    wire = mw.ThinWire(
        name="dipole",
        points=points,
        radius=radius,
        conductor=mw.WireConductor.pec(),
        endpoints=(mw.WireEnd.open(), mw.WireEnd.open()),
        snap=snap,
    )
    port = mw.LumpedPort(
        "feed",
        wire_binding=mw.WirePortBinding.gap(
            negative=mw.WireNodeRef("dipole", 1),
            positive=mw.WireNodeRef("dipole", 2),
        ),
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


def _float64_local_solver(*, mode: str):
    if mode == "termination":
        parameter = torch.tensor(
            0.8, device="cuda", dtype=torch.float64, requires_grad=True
        )
        termination = mw.SeriesRLC(r=parameter)
        excitations = ()
        semantic_key = ("port", "feed", "r")
    elif mode == "excitation":
        parameter = torch.tensor(
            0.45, device="cuda", dtype=torch.float64, requires_grad=True
        )
        termination = None
        excitations = (
            mw.PortExcitation(
                "feed",
                amplitude=parameter,
                source_time=mw.CW(_FREQUENCY),
            ),
        )
        semantic_key = ("excitation", "feed", "amplitude")
    else:
        raise AssertionError(f"unknown mode {mode!r}")

    continuous_points = torch.tensor(
        _CONTINUOUS_POINTS,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    solver = mw.Simulation.fdtd(
        _scene(
            termination=termination,
            points=continuous_points,
            snap="continuous",
        ),
        frequency=_FREQUENCY,
        excitations=excitations,
        run_time=mw.TimeConfig(time_steps=2),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver
    solver.dt = 0.2

    eps_by_field = {}
    for index, name in enumerate(("Ex", "Ey", "Ez")):
        field = getattr(solver, name).to(dtype=torch.float64)
        setattr(solver, name, field)
        eps = torch.linspace(
            8.0e3 + 1.0e3 * index,
            1.2e4 + 1.0e3 * index,
            field.numel(),
            device="cuda",
            dtype=torch.float64,
        ).reshape_as(field)
        eps_by_field[name] = eps.requires_grad_()
        setattr(solver, f"eps_{name}", eps_by_field[name])

    wire_runtime = solver._wire_runtime
    wire_runtime.charge = wire_runtime.charge.to(dtype=torch.float64)
    node_capacitance = torch.linspace(
        0.7,
        1.1,
        wire_runtime.network.node_count,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    wire_runtime.coefficients["node_capacitance"] = node_capacitance
    prepare_port_runtimes(solver, (_FREQUENCY,), excitations)
    gap_weights = solver._port_runtimes[0].wire_provider.geometry.edge_weights
    assert gap_weights.requires_grad
    return (
        solver,
        eps_by_field,
        node_capacitance,
        gap_weights,
        parameter,
        semantic_key,
    )


@pytest.mark.parametrize("mode", ("termination", "excitation"))
def test_wire_gap_port_replay_pullback_matches_dense_float64_cuda_autograd(mode):
    torch.manual_seed(731)
    (
        solver,
        eps_by_field,
        node_capacitance,
        gap_weights,
        parameter,
        semantic_key,
    ) = _float64_local_solver(mode=mode)
    runtime = solver._port_runtimes[0]
    provider = runtime.wire_provider
    assert provider is not None
    assert runtime.lumped is not None
    assert runtime.lumped.field_dtype == torch.float64

    fields = {
        name: torch.linspace(
            -0.35 + 0.1 * index,
            0.45 + 0.1 * index,
            getattr(solver, name).numel(),
            device="cuda",
            dtype=torch.float64,
        )
        .reshape_as(getattr(solver, name))
        .requires_grad_()
        for index, name in enumerate(("Ex", "Ey", "Ez"))
    }
    charge = (
        node_capacitance.detach()
        * torch.linspace(
            -0.25,
            0.3,
            node_capacitance.numel(),
            device="cuda",
            dtype=torch.float64,
        )
    ).requires_grad_()
    old_inductor = torch.tensor(
        0.07, device="cuda", dtype=torch.float64, requires_grad=True
    )
    old_capacitor = torch.tensor(
        -0.09, device="cuda", dtype=torch.float64, requires_grad=True
    )
    old_last_voltage = torch.tensor(
        0.11, device="cuda", dtype=torch.float64, requires_grad=True
    )
    inductor_name = lumped_state_name("port", 0, "inductor_current")
    capacitor_name = lumped_state_name("port", 0, "capacitor_voltage")
    last_voltage_name = lumped_state_name("port", 0, "last_voltage_after")
    state = {
        "wire_charge": charge,
        inductor_name: old_inductor,
        capacitor_name: old_capacitor,
        last_voltage_name: old_last_voltage,
    }
    captured = []
    time_value = -0.5 * float(solver.dt)
    output_fields, output_auxiliary = replay_port_runtimes(
        solver,
        fields,
        state,
        time_value=time_value,
        capture=captured,
    )
    traces = captured[0]
    trace = traces[0]

    field_seeds = {
        name: torch.linspace(
            -0.4 + 0.05 * index,
            0.5 + 0.05 * index,
            value.numel(),
            device="cuda",
            dtype=torch.float64,
        ).reshape_as(value)
        for index, (name, value) in enumerate(output_fields.items())
    }
    charge_seed = torch.linspace(
        0.3,
        -0.2,
        charge.numel(),
        device="cuda",
        dtype=torch.float64,
    )
    inductor_seed = torch.tensor(0.23, device="cuda", dtype=torch.float64)
    capacitor_seed = torch.tensor(-0.19, device="cuda", dtype=torch.float64)
    last_voltage_seed = torch.tensor(0.17, device="cuda", dtype=torch.float64)
    voltage_seed = torch.tensor(0.31, device="cuda", dtype=torch.float64)
    current_seed = torch.tensor(-0.27, device="cuda", dtype=torch.float64)
    objective = sum(
        torch.sum(output_fields[name] * field_seeds[name])
        for name in ("Ex", "Ey", "Ez")
    )
    objective = objective + torch.sum(
        output_auxiliary["wire_charge"] * charge_seed
    )
    objective = objective + output_auxiliary[inductor_name] * inductor_seed
    objective = objective + output_auxiliary[capacitor_name] * capacitor_seed
    objective = objective + output_auxiliary[last_voltage_name] * last_voltage_seed
    objective = objective + trace.lumped.voltage_midpoint * voltage_seed
    objective = objective - trace.lumped.branch_current * current_seed

    inputs = (
        fields["Ex"],
        fields["Ey"],
        fields["Ez"],
        charge,
        eps_by_field["Ex"],
        eps_by_field["Ey"],
        eps_by_field["Ez"],
        node_capacitance,
        gap_weights,
        parameter,
        old_inductor,
        old_capacitor,
        old_last_voltage,
    )
    expected_raw = torch.autograd.grad(objective, inputs, allow_unused=True)
    expected = tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(inputs, expected_raw)
    )

    (
        actual_state,
        actual_eps,
        semantic_grads,
        actual_capacitance,
        actual_gap_weights,
    ) = (
        pullback_port_runtimes(
            solver,
            traces,
            {
                **field_seeds,
                "wire_charge": charge_seed,
                inductor_name: inductor_seed,
                capacitor_name: capacitor_seed,
                last_voltage_name: last_voltage_seed,
            },
            port_sample_adjoints={
                0: (
                    voltage_seed,
                    current_seed,
                    torch.zeros_like(voltage_seed),
                )
            },
            eps_by_field=eps_by_field,
            time_value=time_value,
        )
    )
    actual = (
        actual_state["Ex"],
        actual_state["Ey"],
        actual_state["Ez"],
        actual_state["wire_charge"],
        actual_eps["Ex"],
        actual_eps["Ey"],
        actual_eps["Ez"],
        actual_capacitance,
        actual_gap_weights,
        semantic_grads[semantic_key],
        actual_state[inductor_name],
        actual_state[capacitor_name],
        actual_state[last_voltage_name],
    )
    for found, reference in zip(actual, expected):
        torch.testing.assert_close(
            found,
            reference,
            rtol=3.0e-12,
            atol=3.0e-12,
        )


def _end_to_end_objective(scene, excitation):
    result = mw.Simulation.fdtd(
        scene,
        frequency=_FREQUENCY,
        excitations=excitation,
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).run()
    data = result.port("feed")
    objective = data.voltage.abs().square().sum()
    objective = objective + (50.0 * data.current).abs().square().sum()
    return objective, result


def _central_differences(parameter, center, steps, objective):
    values = []
    for step in steps:
        with torch.no_grad():
            parameter.fill_(center + step)
        plus = float(objective()[0].detach())
        with torch.no_grad():
            parameter.fill_(center - step)
        minus = float(objective()[0].detach())
        values.append((plus - minus) / (2.0 * step))
    with torch.no_grad():
        parameter.fill_(center)
    return values


def _assert_same_wire_port_stencil(reference, candidate):
    for name in (
        "tail",
        "head",
        "segment_source_ids",
        "edge_components",
        "edge_offsets",
        "segment_offsets",
        "fragment_offsets",
        "fragment_segment_ids",
        "fragment_cell_indices",
        "target_components",
        "target_offsets",
        "edge_group_offsets",
        "contribution_segments",
        "source_point_node_ids",
        "port_negative_node_ids",
        "port_positive_node_ids",
        "port_gap_offsets",
        "port_gap_edge_components",
        "port_gap_edge_offsets",
    ):
        torch.testing.assert_close(
            getattr(candidate, name),
            getattr(reference, name),
            rtol=0.0,
            atol=0.0,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )


def test_wire_gap_port_end_to_end_radius_and_amplitude_adjoint_match_finite_difference():
    radius_center = 2.0e-3
    amplitude_center = 0.8
    radius = torch.tensor(
        radius_center,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    amplitude = torch.tensor(
        amplitude_center,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    scene = _scene(radius=radius)
    excitation = mw.PortExcitation(
        "feed",
        amplitude=amplitude,
        source_time=mw.CW(_FREQUENCY),
    )

    objective, result = _end_to_end_objective(scene, excitation)
    objective.backward()
    adjoints = {
        "radius": float(radius.grad),
        "amplitude": float(amplitude.grad),
    }
    assert all(torch.isfinite(value) for value in (radius.grad, amplitude.grad))
    assert all(abs(value) > 1.0e-12 for value in adjoints.values())

    finite_differences = {
        "radius": _central_differences(
            radius,
            radius_center,
            (2.0e-4, 1.0e-4, 5.0e-5),
            lambda: _end_to_end_objective(scene, excitation),
        ),
        "amplitude": _central_differences(
            amplitude,
            amplitude_center,
            (8.0e-2, 4.0e-2, 2.0e-2),
            lambda: _end_to_end_objective(scene, excitation),
        ),
    }
    for name, adjoint in adjoints.items():
        errors = tuple(
            abs(value - adjoint) / max(abs(value), abs(adjoint), 1.0e-30)
            for value in finite_differences[name]
        )
        _assert_finite_difference_agrees(
            errors,
            context=(
                f"{name}: adjoint={adjoint}, "
                f"finite_differences={finite_differences[name]}, "
            ),
        )
    assert result.port("feed").voltage.grad_fn is not None
    assert result.port("feed").current.grad_fn is not None


def test_continuous_oblique_gap_coordinate_adjoint_matches_fixed_stencil_finite_difference():
    base_points = torch.tensor(
        _CONTINUOUS_POINTS,
        device="cuda",
        dtype=torch.float64,
    )
    points = base_points.clone().requires_grad_()
    scene = _scene(points=points, snap="continuous")
    excitation = mw.PortExcitation(
        "feed",
        amplitude=0.8,
        source_time=mw.CW(_FREQUENCY),
    )
    reference_network = scene.compile_thin_wires()
    assert reference_network.metadata["validity"]["coordinate_gradient"] == (
        "fixed_stencil_only"
    )
    assert reference_network.port_gap_weights.requires_grad

    objective, result = _end_to_end_objective(scene, excitation)
    objective.backward()
    point_index, axis = 1, 1
    adjoint = float(points.grad[point_index, axis])
    assert torch.isfinite(points.grad).all()
    assert abs(adjoint) > 1.0e-12

    center = float(base_points[point_index, axis])
    finite_differences = []
    # The objective comes from a float32 forward run, so its ~1e-7 relative noise
    # sets a central-difference roundoff floor of roughly |objective| * 1e-7 /
    # step that grows as the step shrinks. Across this whole sweep that floor
    # already dominates truncation error, so all three steps sit at the roundoff
    # floor: each lands inside the 2% budget but their ordering carries no signal
    # (the shared gate makes its monotonicity clause conditional for exactly this
    # reason). The steps are kept large enough that the floor stays within budget
    # and small enough to stay below the 1.6e-3 displacement at which the port
    # stencil changes cell.
    for step in (8.0e-4, 4.0e-4, 2.0e-4):
        with torch.no_grad():
            points[point_index, axis] = center + step
        plus_network = scene.compile_thin_wires()
        _assert_same_wire_port_stencil(reference_network, plus_network)
        plus = float(_end_to_end_objective(scene, excitation)[0].detach())

        with torch.no_grad():
            points[point_index, axis] = center - step
        minus_network = scene.compile_thin_wires()
        _assert_same_wire_port_stencil(reference_network, minus_network)
        minus = float(_end_to_end_objective(scene, excitation)[0].detach())
        finite_differences.append((plus - minus) / (2.0 * step))
    with torch.no_grad():
        points[point_index, axis] = center

    errors = tuple(
        abs(value - adjoint) / max(abs(value), abs(adjoint), 1.0e-30)
        for value in finite_differences
    )
    _assert_finite_difference_agrees(
        errors,
        context=f"adjoint={adjoint}, finite_differences={finite_differences}, ",
    )
    assert result.port("feed").voltage.grad_fn is not None
