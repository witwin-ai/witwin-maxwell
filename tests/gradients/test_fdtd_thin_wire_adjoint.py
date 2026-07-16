from __future__ import annotations

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge
from witwin.maxwell.fdtd.adjoint.dispatch import validate_native_adjoint_preparation
from witwin.maxwell.fdtd.wire import _target_masses, reverse_wire_step


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD requires CUDA")

_FREQUENCY = 2.0e9


def _wire_simulation(
    radius,
    *,
    density=None,
    time_steps=48,
    frequency=_FREQUENCY,
    boundary=None,
    checkpoint_stride=4,
):
    uses_pml = boundary is not None
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.20, 0.20),) * 3 if uses_pml else ((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none() if boundary is None else boundary,
        device="cuda",
    )
    if density is not None:
        scene.add_material_region(
            mw.MaterialRegion(
                name="host",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.48, 0.48, 0.48)),
                density=density,
                eps_bounds=(1.0, 3.0),
                mu_bounds=(1.0, 1.0),
            )
        )
    scene.add_thin_wire(
        mw.ThinWire(
            name="wire",
            points=((0.0, 0.0, -0.08), (0.0, 0.0, 0.08)),
            radius=radius,
            conductor=mw.WireConductor.pec(),
        )
    )
    scene.add_source(
        mw.PointDipole(
            name="drive",
            position=(0.04, 0.0, 0.0),
            polarization="Ez",
            width=0.04,
            source_time=mw.CW(frequency=frequency, amplitude=10.0),
        )
    )
    scene.add_monitor(
        mw.WireMonitor(
            name="wire_state",
            wire="wire",
            frequencies=(frequency,),
            quantities=("current", "charge"),
        )
    )
    simulation = mw.Simulation.fdtd(
        scene,
        frequencies=(frequency,),
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )
    simulation.config.adjoint_checkpoint_stride = checkpoint_stride
    return simulation


def _branch_solver():
    junction = mw.WireEnd.node("J")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    for name, points, endpoints in (
        (
            "a_top",
            ((0.0, 0.0, 0.0), (0.0, 0.08, 0.0)),
            (junction, mw.WireEnd.open()),
        ),
        (
            "b_left",
            ((-0.08, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (mw.WireEnd.open(), junction),
        ),
        (
            "c_right",
            ((0.0, 0.0, 0.0), (0.08, 0.0, 0.0)),
            (junction, mw.WireEnd.open()),
        ),
    ):
        scene.add_thin_wire(
            mw.ThinWire(
                name=name,
                points=points,
                radius=2.0e-3,
                conductor=mw.WireConductor.pec(),
                endpoints=endpoints,
            )
        )
    return mw.Simulation.fdtd(
        scene,
        frequencies=(_FREQUENCY,),
        run_time=mw.TimeConfig(time_steps=2),
    ).prepare().solver


def _promote_wire_reverse_fixture_to_float64(solver):
    for name in (
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
        "eps_Ex",
        "eps_Ey",
        "eps_Ez",
    ):
        setattr(solver, name, getattr(solver, name).to(torch.float64))
    runtime = solver._wire_runtime
    runtime.current = runtime.current.to(torch.float64)
    runtime.charge = runtime.charge.to(torch.float64)
    runtime.emf = runtime.emf.to(torch.float64)
    for name, value in tuple(runtime.coefficients.items()):
        if value.is_floating_point():
            runtime.coefficients[name] = value.to(torch.float64)
    coeff = runtime.coefficients
    target_masses = _target_masses(
        solver, coeff["target_components"], coeff["target_offsets"]
    )
    target_groups = torch.arange(
        coeff["target_offsets"].numel(), device="cuda", dtype=torch.int64
    ).repeat_interleave(
        coeff["edge_group_offsets"][1:] - coeff["edge_group_offsets"][:-1]
    )
    coeff["contribution_scales"] = (
        float(solver.dt)
        * coeff["contribution_weights"]
        / target_masses.index_select(0, target_groups)
    )
    sample_masses = _target_masses(
        solver, coeff["edge_components"], coeff["edge_offsets"]
    )
    coeff["sample_deposition_scales"] = (
        float(solver.dt) * coeff["weights"] / sample_masses
    )
    return solver


def _wire_objective(radius):
    result = _wire_simulation(radius).run()
    data = result.monitor("wire_state")
    objective = data.current.abs().square().sum() + 1.0e16 * data.charge.abs().square().sum()
    return result, data, objective


def _wire_material_objective(density):
    result = _wire_simulation(2.0e-3, density=density).run()
    data = result.monitor("wire_state")
    objective = data.current.abs().square().sum() + 1.0e16 * data.charge.abs().square().sum()
    return result, data, objective


def test_wire_monitor_backpropagates_radius_through_native_wire_reverse():
    radius = torch.tensor(2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True)
    result, data, objective = _wire_objective(radius)

    assert data.current.requires_grad
    assert data.charge.requires_grad
    objective.backward()

    assert radius.grad is not None
    assert torch.isfinite(radius.grad)
    assert abs(float(radius.grad)) > 0.0
    assert validate_native_adjoint_preparation(result.solver).name == "WIRE_STANDARD"


def test_wire_monitor_backpropagates_radius_through_native_cpml_reverse():
    radius = torch.tensor(2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True)
    result = _wire_simulation(
        radius,
        time_steps=16,
        boundary=mw.BoundarySpec.pml(num_layers=2),
    ).run()
    data = result.monitor("wire_state")
    objective = data.current.abs().square().sum() + 1.0e16 * data.charge.abs().square().sum()

    assert validate_native_adjoint_preparation(result.solver).name == "WIRE_CPML"
    objective.backward()
    assert radius.grad is not None
    assert torch.isfinite(radius.grad)
    assert abs(float(radius.grad)) > 0.0


def test_trainable_wire_rejects_automatic_joint_cfl_adjustment():
    radius = torch.tensor(2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True)
    with pytest.raises(NotImplementedError, match="automatic dt adjustment"):
        _wire_simulation(radius, time_steps=2, frequency=1.0e6).run()


def test_wire_dispatch_never_falls_back_to_a_non_wire_reverse_backend():
    radius = torch.tensor(2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True)
    solver = _wire_simulation(radius, time_steps=2).prepare().solver
    solver.tfsf_enabled = True
    with pytest.raises(RuntimeError, match="no native CUDA adjoint variant"):
        validate_native_adjoint_preparation(solver)


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("_lumped_element_runtimes", (object(),)),
        ("tfsf_enabled", True),
        ("magnetic_dispersive_enabled", True),
    ),
)
def test_wire_bridge_rejects_state_couplings_without_a_wire_aware_transpose(
    attribute, value
):
    radius = torch.tensor(2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True)
    solver = _wire_simulation(radius, time_steps=2).prepare().solver
    setattr(solver, attribute, value)
    with pytest.raises(NotImplementedError, match="wire-aware reverse"):
        _FDTDGradientBridge._validate_supported_configuration(None, solver)


def test_branch_wire_reverse_matches_independent_dense_autograd_vjp():
    torch.manual_seed(207)
    solver = _promote_wire_reverse_fixture_to_float64(_branch_solver())
    coeff = solver._wire_runtime.coefficients
    field_templates = (solver.Ex, solver.Ey, solver.Ez)
    sizes = tuple(field.numel() for field in field_templates)
    bases = torch.tensor(
        (0, sizes[0], sizes[0] + sizes[1]), device="cuda", dtype=torch.int64
    )
    field = torch.cat(
        tuple(1.0e-3 * torch.randn_like(value).reshape(-1) for value in field_templates)
    ).requires_grad_()
    inductance = coeff["inductance"].detach().clone().requires_grad_()
    capacitance = coeff["node_capacitance"].detach().clone().requires_grad_()
    current0 = (1.0e-3 * torch.randn_like(inductance)).requires_grad_()
    charge0 = (
        capacitance.detach() * torch.randn_like(capacitance)
    ).requires_grad_()
    eps0 = torch.cat(
        (solver.eps_Ex.reshape(-1), solver.eps_Ey.reshape(-1), solver.eps_Ez.reshape(-1))
    ).detach()
    eps = eps0.clone().requires_grad_()

    segment_ids = torch.arange(
        inductance.numel(), device="cuda", dtype=torch.int64
    ).repeat_interleave(coeff["segment_offsets"][1:] - coeff["segment_offsets"][:-1])
    sample_indices = coeff["edge_offsets"] + bases.index_select(
        0, coeff["edge_components"].to(torch.int64)
    )
    sample = torch.zeros(
        (inductance.numel(), field.numel()), device="cuda", dtype=field.dtype
    )
    sample.index_put_((segment_ids, sample_indices), coeff["weights"], accumulate=True)

    incidence = torch.zeros(
        (capacitance.numel(), inductance.numel()), device="cuda", dtype=field.dtype
    )
    segment_range = torch.arange(inductance.numel(), device="cuda")
    incidence.index_put_(
        (coeff["tail"], segment_range), torch.ones_like(inductance), accumulate=True
    )
    incidence.index_put_(
        (coeff["head"], segment_range), -torch.ones_like(inductance), accumulate=True
    )

    target_indices = coeff["target_offsets"] + bases.index_select(
        0, coeff["target_components"].to(torch.int64)
    )
    target_groups = torch.arange(
        target_indices.numel(), device="cuda", dtype=torch.int64
    ).repeat_interleave(
        coeff["edge_group_offsets"][1:] - coeff["edge_group_offsets"][:-1]
    )
    deposit0 = torch.zeros(
        (field.numel(), inductance.numel()), device="cuda", dtype=field.dtype
    )
    deposit0.index_put_(
        (
            target_indices.index_select(0, target_groups),
            coeff["contribution_segments"],
        ),
        coeff["contribution_scales"],
        accumulate=True,
    )
    deposit = deposit0 * (eps0 / eps).unsqueeze(1)

    dt = float(solver.dt)
    potential = torch.where(
        coeff["grounded"], torch.zeros_like(charge0), charge0 / capacitance
    )
    current1 = current0 + dt * (
        sample @ field + incidence.transpose(0, 1) @ potential
    ) / inductance
    charge1 = torch.where(
        coeff["grounded"],
        torch.zeros_like(charge0),
        charge0 - dt * (incidence @ current1),
    )
    field1 = field - deposit @ current1
    field_seed = torch.randn_like(field1)
    current_seed = torch.randn_like(current1)
    charge_seed = torch.randn_like(charge1)
    objective = (
        torch.sum(field_seed * field1)
        + torch.sum(current_seed * current1)
        + torch.sum(charge_seed * charge1)
    )
    reference = torch.autograd.grad(
        objective, (field, current0, charge0, inductance, capacitance, eps)
    )

    field_parts = torch.split(field.detach(), sizes)
    seed_parts = torch.split(field_seed, sizes)
    eps_parts = torch.split(eps0, sizes)
    reverse = reverse_wire_step(
        solver,
        {
            **{
                name: value.reshape_as(template)
                for name, value, template in zip(
                    ("Ex", "Ey", "Ez"), field_parts, field_templates
                )
            },
            "wire_current": current0.detach(),
            "wire_charge": charge0.detach(),
        },
        {
            **{
                name: value.reshape_as(template)
                for name, value, template in zip(
                    ("Ex", "Ey", "Ez"), seed_parts, field_templates
                )
            },
            "wire_current": current_seed,
            "wire_charge": charge_seed,
        },
        eps_by_field={
            name: value.reshape_as(template)
            for name, value, template in zip(
                ("Ex", "Ey", "Ez"), eps_parts, field_templates
            )
        },
    )
    repeated = reverse_wire_step(
        solver,
        {
            **{
                name: value.reshape_as(template)
                for name, value, template in zip(
                    ("Ex", "Ey", "Ez"), field_parts, field_templates
                )
            },
            "wire_current": current0.detach(),
            "wire_charge": charge0.detach(),
        },
        {
            **{
                name: value.reshape_as(template)
                for name, value, template in zip(
                    ("Ex", "Ey", "Ez"), seed_parts, field_templates
                )
            },
            "wire_current": current_seed,
            "wire_charge": charge_seed,
        },
        eps_by_field={
            name: value.reshape_as(template)
            for name, value, template in zip(
                ("Ex", "Ey", "Ez"), eps_parts, field_templates
            )
        },
    )
    field_wire_adjoint = torch.cat(
        tuple(reverse.field_adjoint[name].reshape(-1) for name in ("Ex", "Ey", "Ez"))
    )
    eps_adjoint = torch.cat(
        tuple(reverse.grad_eps[name].reshape(-1) for name in ("Ex", "Ey", "Ez"))
    )
    actual = (
        field_seed + field_wire_adjoint,
        reverse.pre_current,
        reverse.pre_charge,
        reverse.grad_inductance,
        reverse.grad_node_capacitance,
        eps_adjoint,
    )
    for computed, expected in zip(actual, reference):
        assert computed.is_cuda
        torch.testing.assert_close(computed, expected, rtol=1.0e-10, atol=1.0e-12)
    torch.testing.assert_close(
        repeated.pre_current, reverse.pre_current, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        repeated.pre_charge, reverse.pre_charge, rtol=0.0, atol=0.0
    )
    for name in ("Ex", "Ey", "Ez"):
        torch.testing.assert_close(
            repeated.field_adjoint[name], reverse.field_adjoint[name], rtol=0.0, atol=0.0
        )


def test_wire_adjoint_checkpoint_stride_replay_is_invariant():
    gradients = []
    objectives = []
    for stride in (1, 7, None):
        radius = torch.tensor(
            2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True
        )
        result = _wire_simulation(
            radius, time_steps=24, checkpoint_stride=stride
        ).run()
        data = result.monitor("wire_state")
        objective = (
            data.current.abs().square().sum()
            + 1.0e16 * data.charge.abs().square().sum()
        )
        objective.backward()
        objectives.append(objective.detach())
        gradients.append(radius.grad.detach())

    for objective in objectives[1:]:
        torch.testing.assert_close(objective, objectives[0], rtol=0.0, atol=0.0)
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, gradients[0], rtol=1.0e-5, atol=1.0e-12)


def test_wire_radius_adjoint_matches_centered_finite_difference():
    radius = torch.tensor(2.0e-3, device="cuda", dtype=torch.float64, requires_grad=True)
    _result, _data, objective = _wire_objective(radius)
    objective.backward()
    adjoint = float(radius.grad)

    errors = []
    for step in (8.0e-5, 4.0e-5, 2.0e-5):
        plus = float(_wire_objective(2.0e-3 + step)[2])
        minus = float(_wire_objective(2.0e-3 - step)[2])
        finite_difference = (plus - minus) / (2.0 * step)
        errors.append(
            abs(adjoint - finite_difference) / max(abs(finite_difference), 1.0e-20)
        )
    assert min(errors) < 2.0e-2
    assert errors[-1] < errors[0]


def test_wire_host_material_adjoint_matches_centered_finite_difference():
    density = torch.tensor(
        [[[0.4]]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    _result, data, objective = _wire_material_objective(density)
    assert data.current.requires_grad and data.charge.requires_grad
    objective.backward()
    adjoint = float(density.grad)

    errors = []
    for step in (2.0e-2, 1.0e-2, 5.0e-3):
        plus_density = torch.tensor([[[0.4 + step]]], device="cuda")
        minus_density = torch.tensor([[[0.4 - step]]], device="cuda")
        plus = float(_wire_material_objective(plus_density)[2])
        minus = float(_wire_material_objective(minus_density)[2])
        finite_difference = (plus - minus) / (2.0 * step)
        errors.append(
            abs(adjoint - finite_difference) / max(abs(finite_difference), 1.0e-20)
        )
    assert min(errors) < 2.0e-2
    assert errors[-1] < errors[0]
