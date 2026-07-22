from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from tests.rf.network.test_network_multiport_runtime import (
    _solver as _multiport_solver,
)
from tests.rf.network.test_network_runtime import _fdtd_scene, _solver
from witwin.maxwell.fdtd.adjoint import _FDTDGradientBridge, _replay_segment_states
from witwin.maxwell.fdtd.checkpoint import (
    network_carried_voltage_name,
    network_state_name,
)
from witwin.maxwell.fdtd.networks import (
    pullback_network_runtimes,
    replay_network_runtimes,
)


def test_network_step_pullback_matches_autograd_oracle() -> None:
    solver = _solver(torch.device("cpu"))
    runtime = solver._network_runtimes[0]
    state_name = network_state_name(0)
    carried_name = network_carried_voltage_name(0)
    old_state = torch.tensor((0.5,), dtype=torch.float64)
    # Slice U2: the trapezoidal network interface carries the previous step's
    # post-step port voltage; exercise a nonzero carry so the reverse threads it.
    old_carried = torch.tensor((0.3,), dtype=torch.float64)
    input_field = solver.Ex.clone()
    traces = []
    output_fields, next_state = replay_network_runtimes(
        solver,
        {"Ex": input_field},
        {state_name: old_state, carried_name: old_carried},
        capture=traces,
    )

    field_seed = torch.tensor([0.7, -0.2], dtype=torch.float64).reshape(2, 1, 1)
    state_seed = torch.tensor((0.9,), dtype=torch.float64)
    carried_seed = torch.tensor((0.15,), dtype=torch.float64)
    voltage_seed = torch.tensor(0.35, dtype=torch.float64)
    current_seed = torch.tensor(-0.4, dtype=torch.float64)
    pulled, _grad_eps, matrix_grads = pullback_network_runtimes(
        solver,
        traces[0],
        {"Ex": field_seed, state_name: state_seed, carried_name: carried_seed},
        port_sample_adjoints={
            0: (voltage_seed, current_seed, torch.zeros_like(voltage_seed))
        },
        eps_by_field={"Ex": torch.ones_like(input_field)},
    )

    field = input_field.detach().clone().requires_grad_(True)
    state = old_state.detach().clone().requires_grad_(True)
    carried = old_carried.detach().clone().requires_grad_(True)
    matrices = tuple(
        getattr(runtime, name).detach().clone().requires_grad_(True)
        for name in ("A", "B", "C", "D")
    )
    matrix_a, matrix_b, matrix_c, matrix_d = matrices
    lumped = runtime.port_runtime.lumped
    free_voltage = torch.dot(
        field.reshape(-1).index_select(0, lumped.linear_indices),
        lumped.voltage_weights,
    ).reshape(1)
    coupling_voltage = 0.5 * (carried + free_voltage)
    loop = torch.eye(1, dtype=field.dtype) + matrix_d * runtime.feedback_impedance
    branch_current = torch.linalg.solve(
        loop,
        matrix_c @ state + matrix_d @ coupling_voltage,
    )
    network_voltage = coupling_voltage - runtime.feedback_impedance * branch_current
    advanced_state = matrix_a @ state + matrix_b @ network_voltage
    next_carried = free_voltage - runtime.coupling_impedance * branch_current
    corrected_field = field.clone()
    corrected_field.reshape(-1).index_add_(
        0,
        lumped.linear_indices,
        -lumped.injection * branch_current[0],
    )
    objective = (
        torch.sum(corrected_field * field_seed)
        + torch.sum(advanced_state * state_seed)
        + next_carried[0] * carried_seed[0]
        + network_voltage[0] * voltage_seed
        - branch_current[0] * current_seed
    )
    expected = torch.autograd.grad(objective, (field, state, carried, *matrices))

    torch.testing.assert_close(output_fields["Ex"], corrected_field.detach())
    torch.testing.assert_close(next_state[state_name], advanced_state.detach())
    torch.testing.assert_close(next_state[carried_name], next_carried.detach())
    torch.testing.assert_close(pulled["Ex"], expected[0])
    torch.testing.assert_close(pulled[state_name], expected[1])
    torch.testing.assert_close(pulled[carried_name], expected[2])
    for name, gradient in zip(("A", "B", "C", "D"), expected[3:]):
        torch.testing.assert_close(
            matrix_grads[("network", runtime.name, name)],
            gradient,
        )


def test_multiport_network_step_pullback_matches_autograd_oracle() -> None:
    solver = _multiport_solver(4, torch.device("cpu"))
    runtime = solver._network_runtimes[0]
    state_name = network_state_name(0)
    carried_name = network_carried_voltage_name(0)
    old_state = runtime.state.clone()
    # Slice U2: nonzero per-port carried post-step voltage exercises the reverse.
    old_carried = torch.tensor((0.12, -0.05, 0.2, -0.15), dtype=torch.float64)
    input_field = solver.Ex.clone()
    traces = []
    replay_network_runtimes(
        solver,
        {"Ex": input_field},
        {state_name: old_state, carried_name: old_carried},
        capture=traces,
    )

    field_seed = torch.linspace(
        -0.4,
        0.7,
        input_field.numel(),
        dtype=torch.float64,
    ).reshape_as(input_field)
    state_seed = torch.tensor((0.2, -0.5, 0.8), dtype=torch.float64)
    carried_seed = torch.tensor((0.11, -0.22, 0.33, -0.14), dtype=torch.float64)
    voltage_seed = torch.tensor((0.1, 0.3, -0.2, 0.4), dtype=torch.float64)
    current_seed = torch.tensor((-0.3, 0.2, 0.5, -0.1), dtype=torch.float64)
    pulled, _grad_eps, matrix_grads = pullback_network_runtimes(
        solver,
        traces[0],
        {"Ex": field_seed, state_name: state_seed, carried_name: carried_seed},
        port_sample_adjoints={
            index: (
                voltage_seed[index],
                current_seed[index],
                torch.zeros((), dtype=torch.float64),
            )
            for index in range(4)
        },
        eps_by_field={"Ex": torch.ones_like(input_field)},
    )

    field = input_field.detach().clone().requires_grad_(True)
    state = old_state.detach().clone().requires_grad_(True)
    carried = old_carried.detach().clone().requires_grad_(True)
    matrices = tuple(
        getattr(runtime, name).detach().clone().requires_grad_(True)
        for name in ("A", "B", "C", "D")
    )
    matrix_a, matrix_b, matrix_c, matrix_d = matrices
    free_voltage = torch.stack(
        tuple(
            torch.dot(
                field.reshape(-1).index_select(0, port_runtime.lumped.linear_indices),
                port_runtime.lumped.voltage_weights,
            )
            for port_runtime in runtime.port_runtimes
        )
    )
    coupling_voltage = 0.5 * (carried + free_voltage)
    loop = torch.eye(4, dtype=field.dtype) + matrix_d * runtime.feedback_impedance.unsqueeze(0)
    branch_current = torch.linalg.solve(
        loop,
        matrix_c @ state + matrix_d @ coupling_voltage,
    )
    network_voltage = coupling_voltage - runtime.feedback_impedance * branch_current
    advanced_state = matrix_a @ state + matrix_b @ network_voltage
    next_carried = free_voltage - runtime.coupling_impedance * branch_current
    corrected_field = field.clone()
    for index, port_runtime in enumerate(runtime.port_runtimes):
        corrected_field.reshape(-1).index_add_(
            0,
            port_runtime.lumped.linear_indices,
            -port_runtime.lumped.injection * branch_current[index],
        )
    objective = (
        torch.sum(corrected_field * field_seed)
        + torch.sum(advanced_state * state_seed)
        + torch.sum(next_carried * carried_seed)
        + torch.sum(network_voltage * voltage_seed)
        - torch.sum(branch_current * current_seed)
    )
    expected = torch.autograd.grad(objective, (field, state, carried, *matrices))

    torch.testing.assert_close(pulled["Ex"], expected[0])
    torch.testing.assert_close(pulled[state_name], expected[1])
    torch.testing.assert_close(pulled[carried_name], expected[2])
    for name, gradient in zip(("A", "B", "C", "D"), expected[3:]):
        torch.testing.assert_close(
            matrix_grads[("network", runtime.name, name)],
            gradient,
        )


def test_conjugate_residue_gradient_preserves_real_model_constraint() -> None:
    residues = torch.tensor(
        (((2.0 + 3.0j, 2.0 - 3.0j),),),
        dtype=torch.complex128,
        requires_grad=True,
    )
    model = mw.RationalModel(
        poles=torch.tensor((-4.0 + 5.0j, -4.0 - 5.0j), dtype=torch.complex128),
        residues=residues,
        direct=0.1,
        representation="Y",
    )
    state_space = model.to_state_space(port_order=("load",))
    weights = torch.tensor(((1.5, -0.75),), dtype=torch.float64)
    objective = torch.sum(state_space.C * weights)
    objective.backward()

    first, second = residues.grad.reshape(-1)
    assert abs(first) > 0.0
    torch.testing.assert_close(second, first.conj())
    updated = residues.detach() - 0.05 * residues.grad
    torch.testing.assert_close(updated[..., 1], updated[..., 0].conj())


def _rational_network(
    *,
    residue_value: float,
    direct_value: float,
    trainable: bool,
):
    device = torch.device("cuda")
    poles = torch.tensor((-8.0e9 + 0.0j,), device=device, dtype=torch.complex128)
    residues = torch.tensor(
        (((residue_value + 0.0j,),),),
        device=device,
        dtype=torch.complex128,
        requires_grad=trainable,
    )
    direct = torch.tensor(
        ((direct_value + 0.0j,),),
        device=device,
        dtype=torch.complex128,
        requires_grad=trainable,
    )
    model = mw.RationalModel(
        poles=poles,
        residues=residues,
        direct=direct,
        representation="Y",
    )
    frequencies = torch.linspace(
        0.25e9,
        6.0e9,
        33,
        device=device,
        dtype=torch.float64,
    )
    network = mw.NetworkData.from_y(
        frequencies=frequencies,
        y=model.evaluate(frequencies).detach(),
        z0=50.0,
        port_names=("load",),
    )
    block = mw.NetworkBlock(
        name="differentiable_load",
        network=network,
        connections={"load": "feed"},
        fit=False,
        model=model,
    )
    return block, residues, direct


def _network_objective(*, residue_value: float, direct_value: float, trainable: bool):
    block, residues, direct = _rational_network(
        residue_value=residue_value,
        direct_value=direct_value,
        trainable=trainable,
    )
    result = mw.Simulation.fdtd(
        _fdtd_scene(network=block),
        frequency=2.75e9,
        run_time=mw.TimeConfig(time_steps=48),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    voltage = result.embedded_network("differentiable_load").voltage
    torch.testing.assert_close(voltage[0], result.port("feed").voltage)
    objective = voltage.abs().square().sum()
    return objective, residues, direct


def _network_material_objective(*, logit_value: float, trainable: bool):
    block, _residues, _direct = _rational_network(
        residue_value=2.0e8,
        direct_value=1.0e-2,
        trainable=False,
    )
    logit = torch.tensor(
        logit_value,
        device="cuda",
        dtype=torch.float64,
        requires_grad=trainable,
    )
    scene = _fdtd_scene(network=block)
    scene.add_material_region(
        mw.MaterialRegion(
            name="near_terminal",
            geometry=mw.Box(position=(0.005, 0.0, 0.005), size=(0.005, 0.005, 0.005)),
            density=torch.sigmoid(logit).reshape(1, 1, 1),
            eps_bounds=(1.0, 4.0),
            mu_bounds=(1.0, 1.0),
        )
    )
    result = mw.Simulation.fdtd(
        scene,
        frequency=2.75e9,
        run_time=mw.TimeConfig(time_steps=48),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    objective = result.port("feed").voltage.abs().square().sum()
    return objective, logit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("parameter_name", "step_sizes"),
    (
        ("residues", (2.0e5, 5.0e5, 1.0e6)),
        ("direct", (1.0e-5, 2.0e-5, 5.0e-5)),
    ),
)
def test_rational_network_gradient_matches_three_step_central_difference(
    parameter_name: str,
    step_sizes: tuple[float, ...],
) -> None:
    residue_value = 2.0e8
    direct_value = 1.0e-2
    objective, residues, direct = _network_objective(
        residue_value=residue_value,
        direct_value=direct_value,
        trainable=True,
    )
    objective.backward()
    tensor = residues if parameter_name == "residues" else direct
    adjoint = float(tensor.grad.real.reshape(-1)[0])
    assert abs(adjoint) > 1.0e-20

    differences = []
    value_key = "residue_value" if parameter_name == "residues" else "direct_value"
    for step in step_sizes:
        plus = dict(residue_value=residue_value, direct_value=direct_value)
        minus = dict(plus)
        plus[value_key] += step
        minus[value_key] -= step
        plus_value = _network_objective(trainable=False, **plus)[0]
        minus_value = _network_objective(trainable=False, **minus)[0]
        differences.append(float((plus_value - minus_value) / (2.0 * step)))

    relative_errors = tuple(
        abs(value - adjoint) / max(abs(value), abs(adjoint), 1.0e-30)
        for value in differences
    )
    # Require every step size to agree, not just the best one: max() over the
    # three central differences means a regression that corrupts any one step
    # is caught. The three FD sizes bracket the truncation/round-off optimum so
    # all three legitimately land far under the gate today.
    assert max(relative_errors) < 0.02, (
        f"adjoint={adjoint}, central_differences={differences}, "
        f"relative_errors={relative_errors}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_network_connected_material_gradient_matches_three_step_central_difference() -> None:
    objective, logit = _network_material_objective(logit_value=0.1, trainable=True)
    objective.backward()
    adjoint = float(logit.grad)
    assert abs(adjoint) > 1.0e-20

    differences = []
    for step in (2.0e-3, 5.0e-3, 1.0e-2):
        plus = _network_material_objective(logit_value=0.1 + step, trainable=False)[0]
        minus = _network_material_objective(logit_value=0.1 - step, trainable=False)[0]
        differences.append(float((plus - minus) / (2.0 * step)))
    relative_errors = tuple(
        abs(value - adjoint) / max(abs(value), abs(adjoint), 1.0e-30)
        for value in differences
    )
    # Require every step size to agree (see the note on the residue/direct case).
    assert max(relative_errors) < 0.02, (
        f"adjoint={adjoint}, central_differences={differences}, "
        f"relative_errors={relative_errors}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_network_checkpoint_replay_restores_terminal_state_without_per_step_storage() -> None:
    block, _residues, _direct = _rational_network(
        residue_value=2.0e8,
        direct_value=1.0e-2,
        trainable=True,
    )
    simulation = mw.Simulation.fdtd(
        _fdtd_scene(network=block),
        frequency=2.75e9,
        run_time=mw.TimeConfig(time_steps=36),
        spectral_sampler=mw.SpectralSampler(window="none"),
    )
    bridge = _FDTDGradientBridge(simulation)
    bridge.forward(bridge.trainable_inputs)

    # Slice U2: the trapezoidal network interface carries the post-step port
    # voltage as dynamic state alongside the state-space vector.
    assert bridge._checkpoint_schema.network_state_names == (
        network_state_name(0),
        network_carried_voltage_name(0),
    )
    assert len(bridge._last_checkpoints) < bridge._time_steps
    replayed = _replay_segment_states(
        bridge._last_solver,
        bridge._last_checkpoints[0],
        0,
        bridge._time_steps,
    )
    torch.testing.assert_close(
        replayed[-1][network_state_name(0)],
        bridge._last_solver._network_runtimes[0].state,
    )


def test_network_adjoint_rejects_explicit_delay_state() -> None:
    bridge = object.__new__(_FDTDGradientBridge)
    solver = SimpleNamespace(
        scene=SimpleNamespace(
            networks=(SimpleNamespace(delay_seconds=(1.0e-9,), model=None),)
        )
    )
    with pytest.raises(NotImplementedError, match="explicit delay state"):
        bridge._validate_supported_configuration(solver)


@pytest.mark.parametrize("model_kind", ("proportional", "state_space"))
def test_network_adjoint_rejects_unsupported_trainable_model_parameters(
    model_kind: str,
) -> None:
    if model_kind == "proportional":
        model = mw.RationalModel(
            poles=torch.tensor((-2.0 + 0.0j,), dtype=torch.complex128),
            residues=torch.tensor((((1.0 + 0.0j,),),), dtype=torch.complex128),
            direct=0.1,
            proportional=torch.zeros(
                (1, 1),
                dtype=torch.complex128,
                requires_grad=True,
            ),
            representation="Y",
        )
        message = "residues and direct terms only"
    else:
        model = mw.StateSpaceNetwork(
            A=torch.tensor(((-2.0,),), dtype=torch.float64, requires_grad=True),
            B=torch.ones((1, 1), dtype=torch.float64),
            C=torch.ones((1, 1), dtype=torch.float64),
            D=torch.full((1, 1), 0.1, dtype=torch.float64),
            representation="Y",
            port_order=("load",),
        )
        message = "direct trainable state-space matrices"
    bridge = object.__new__(_FDTDGradientBridge)
    solver = SimpleNamespace(
        scene=SimpleNamespace(
            networks=(SimpleNamespace(delay_seconds=None, model=model),)
        )
    )
    with pytest.raises(NotImplementedError, match=message):
        bridge._validate_supported_configuration(solver)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_network_adjoint_rejects_trainable_poles() -> None:
    block, _residues, _direct = _rational_network(
        residue_value=2.0e8,
        direct_value=1.0e-2,
        trainable=False,
    )
    model = mw.RationalModel(
        poles=block.model.poles.detach().clone().requires_grad_(True),
        residues=block.model.residues,
        direct=block.model.direct,
        representation="Y",
    )
    unsupported = mw.NetworkBlock(
        name=block.name,
        network=block.network,
        connections=block.connections,
        fit=False,
        model=model,
    )
    with pytest.raises(NotImplementedError, match="residues and direct terms only"):
        mw.Simulation.fdtd(
            _fdtd_scene(network=unsupported),
            frequency=2.75e9,
            run_time=mw.TimeConfig(time_steps=8),
        )
