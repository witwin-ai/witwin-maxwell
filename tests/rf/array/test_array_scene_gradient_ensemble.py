"""Ensemble 2-GPU aggregation of the array scene-gradient VJP (plan 06 Phase 4, F3b).

``aggregate_scene_gradient_vjp`` reduces per-column scene-gradient VJPs that may be
computed on different devices; ``ensemble_scene_gradient_vjp`` distributes the
per-column forwards over the ensemble device pool and then reduces. The headline
gate is 1-GPU-vs-2-GPU aggregated-gradient parity: the same combined objective's
scene gradient must be invariant to whether each column ran on one GPU or was
split across two, because the per-column VJPs are bit-identical on homogeneous
GPUs and the reduction order is fixed.

CPU tests cover the aggregation math exactly (consistency with the single-device
``scene_gradient_vjp``, per-column replica reduction, field-cotangent parity,
deterministic order, fail-closed contracts). The CUDA tests close the 2-GPU loop
with a fast synthetic per-column map (measured bitwise/floor parity) and a real
two-column FDTD array split across the two GPUs.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell import (
    aggregate_scene_gradient_vjp,
    ensemble_scene_gradient_vjp,
)
from witwin.maxwell.execution import MultiGPUExecution

_TWO_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two CUDA devices for ensemble aggregation parity",
)


def _basis(*, port_count, points_theta, points_phi, freqs, device="cpu"):
    dtype, cdtype = torch.float64, torch.complex128
    frequencies = torch.tensor(freqs, device=device, dtype=dtype)
    frequency_count = len(freqs)
    scattering = torch.zeros((frequency_count, port_count, port_count), device=device, dtype=cdtype)
    for index in range(port_count):
        scattering[:, index, index] = 0.05 - 0.02j
    z0 = torch.full((port_count,), 50.0, device=device, dtype=cdtype)
    network = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=z0,
        port_names=tuple(f"p{index + 1}" for index in range(port_count)),
    )
    theta_vector = torch.linspace(0.0, math.pi, points_theta, device=device, dtype=dtype)
    phi_vector = torch.linspace(0.0, 2.0 * math.pi, points_phi, device=device, dtype=dtype)
    theta, phi = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    base = torch.sin(theta).to(cdtype)
    e_theta = torch.stack([base * (index + 1) for index in range(port_count)], dim=0)[None]
    e_theta = e_theta.expand(frequency_count, -1, -1, -1).clone()
    e_phi = 0.3j * e_theta
    patterns = mw.EmbeddedElementPatternData(
        frequencies=frequencies,
        port_names=network.port_names,
        theta=theta,
        phi=phi,
        e_theta=e_theta,
        e_phi=e_phi,
        phase_center=torch.zeros(3, device=device, dtype=dtype),
        frame=torch.eye(3, device=device, dtype=dtype),
    )
    return mw.ArrayBasisData(network=network, embedded_patterns=patterns, fingerprint="scene-grad-ens")


def _column_map(param, base_e_theta, base_e_phi, index):
    """Nonlinear per-column far field as a differentiable function of ``param``.

    ``base_e_*`` are the detached ``[F, T, P]`` template columns on ``param``'s
    device. A distinct per-element coefficient guarantees every element of
    ``param`` receives a non-trivial gradient.
    """

    complex_param = param.to(base_e_theta.dtype)
    coefficients = torch.arange(1, param.numel() + 1, device=param.device, dtype=param.dtype)
    reduced = (complex_param * coefficients.to(complex_param.dtype)).sum()
    scale = 1.0 + 0.2 * reduced
    e_theta = base_e_theta * torch.exp(1j * (index + 1) * reduced) * scale
    e_phi = base_e_phi * torch.exp(1j * (0.5 * index + 1.0) * reduced)
    return e_theta, e_phi


def _objective(e_theta, e_phi):
    return (
        (e_theta * e_theta.conj()).real.sum()
        + 0.4 * (e_phi * e_phi.conj()).real.sum()
        + 0.1 * e_theta.real.sum()
    )


def _weights(basis):
    frequency_count, port_count, _ = basis.network.s.shape
    generator = torch.Generator().manual_seed(11)
    real = torch.rand((frequency_count, port_count), generator=generator, dtype=torch.float64) - 0.5
    imag = torch.rand((frequency_count, port_count), generator=generator, dtype=torch.float64) - 0.5
    return torch.complex(real, imag)


def _shared_param_columns(param, basis):
    """One shared leaf feeding every column (single-device F3a scenario)."""

    port_count = basis.network.s.shape[1]
    columns = []
    for index in range(port_count):
        base_theta = basis.embedded_patterns.e_theta[:, index].detach()
        base_phi = basis.embedded_patterns.e_phi[:, index].detach()
        columns.append(_column_map(param, base_theta, base_phi, index))
    return columns


# --------------------------------------------------------------------------- #
# CPU aggregation math.
# --------------------------------------------------------------------------- #


def test_aggregate_matches_single_device_scene_gradient_vjp():
    """Aggregating shared-leaf columns equals the single-device VJP bit-for-bit."""

    basis = _basis(port_count=3, points_theta=5, points_phi=4, freqs=[1.0e9, 1.5e9])
    weights = _weights(basis)

    param = torch.tensor([0.31, -0.12], dtype=torch.float64, requires_grad=True)
    columns = _shared_param_columns(param, basis)
    reference = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=param, objective=_objective
    )

    param_a = torch.tensor([0.31, -0.12], dtype=torch.float64, requires_grad=True)
    columns_a = _shared_param_columns(param_a, basis)
    aggregated = aggregate_scene_gradient_vjp(
        basis,
        columns=columns_a,
        parameters=[param_a] * 3,
        weights=weights,
        objective=_objective,
    )
    assert aggregated.reduction_order == (0, 1, 2)
    assert aggregated.column_devices == ("cpu", "cpu", "cpu")
    assert aggregated.port_names == basis.port_names
    assert torch.equal(aggregated.gradient, reference)


def test_aggregate_per_column_replicas_equal_shared_leaf():
    """Independent per-column replicas reduce to the shared-leaf gradient.

    This is the CPU stand-in for the 2-GPU data-parallel case: each column owns a
    replica of the same design, and the summed per-column VJPs equal the gradient
    of the single shared leaf.
    """

    basis = _basis(port_count=3, points_theta=6, points_phi=5, freqs=[2.0e9])
    weights = _weights(basis)

    shared = torch.tensor([0.2, 0.5, -0.3], dtype=torch.float64, requires_grad=True)
    shared_columns = _shared_param_columns(shared, basis)
    reference = basis.scene_gradient_vjp(
        columns=shared_columns, weights=weights, parameters=shared, objective=_objective
    )

    replicas = [
        torch.tensor([0.2, 0.5, -0.3], dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    replica_columns = [
        _column_map(
            replicas[index],
            basis.embedded_patterns.e_theta[:, index].detach(),
            basis.embedded_patterns.e_phi[:, index].detach(),
            index,
        )
        for index in range(3)
    ]
    aggregated = aggregate_scene_gradient_vjp(
        basis,
        columns=replica_columns,
        parameters=replicas,
        weights=weights,
        objective=_objective,
    )
    torch.testing.assert_close(aggregated.gradient, reference, rtol=1.0e-12, atol=1.0e-12)


def test_aggregate_field_cotangents_matches_objective():
    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)

    param = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    columns = _shared_param_columns(param, basis)
    via_objective = aggregate_scene_gradient_vjp(
        basis, columns=columns, parameters=[param] * 2, weights=weights, objective=_objective
    )

    param_c = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    columns_c = _shared_param_columns(param_c, basis)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns_c[n][0] for n in range(2))
    e_phi = sum(column_weight[:, n] * columns_c[n][1] for n in range(2))
    cotangents = torch.autograd.grad(_objective(e_theta, e_phi), (e_theta, e_phi), retain_graph=True)
    via_cotangents = aggregate_scene_gradient_vjp(
        basis,
        columns=columns_c,
        parameters=[param_c] * 2,
        weights=weights,
        field_cotangents=cotangents,
    )
    assert torch.equal(via_objective.gradient, via_cotangents.gradient)


def test_aggregate_reduction_order_is_deterministic():
    basis = _basis(port_count=3, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)

    param = torch.tensor([0.29, 0.11], dtype=torch.float64, requires_grad=True)
    default = aggregate_scene_gradient_vjp(
        basis,
        columns=_shared_param_columns(param, basis),
        parameters=[param] * 3,
        weights=weights,
        objective=_objective,
    )
    param_p = torch.tensor([0.29, 0.11], dtype=torch.float64, requires_grad=True)
    permuted = aggregate_scene_gradient_vjp(
        basis,
        columns=_shared_param_columns(param_p, basis),
        parameters=[param_p] * 3,
        weights=weights,
        objective=_objective,
        reduction_order=(2, 0, 1),
    )
    assert permuted.reduction_order == (2, 0, 1)
    torch.testing.assert_close(default.gradient, permuted.gradient, rtol=1.0e-12, atol=1.0e-12)


def test_aggregate_multi_leaf_parameters():
    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)

    param_a = torch.tensor([0.31, -0.1], dtype=torch.float64, requires_grad=True)
    param_b = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)

    def columns_for(a, b):
        result = []
        for index in range(2):
            base_theta = basis.embedded_patterns.e_theta[:, index].detach()
            base_phi = basis.embedded_patterns.e_phi[:, index].detach()
            e_theta, e_phi = _column_map(a, base_theta, base_phi, index)
            e_theta = e_theta * (1.0 + 0.1 * b.to(torch.complex128).sum())
            result.append((e_theta, e_phi))
        return result

    aggregated = aggregate_scene_gradient_vjp(
        basis,
        columns=columns_for(param_a, param_b),
        parameters=[(param_a, param_b), (param_a, param_b)],
        weights=weights,
        objective=_objective,
    )
    grad_a, grad_b = aggregated.gradient

    param_a2 = torch.tensor([0.31, -0.1], dtype=torch.float64, requires_grad=True)
    param_b2 = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
    columns_e = columns_for(param_a2, param_b2)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns_e[n][0] for n in range(2))
    e_phi = sum(column_weight[:, n] * columns_e[n][1] for n in range(2))
    ref_a, ref_b = torch.autograd.grad(_objective(e_theta, e_phi), (param_a2, param_b2))
    torch.testing.assert_close(grad_a, ref_a, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(grad_b, ref_b, rtol=1.0e-12, atol=1.0e-12)


# --------------------------------------------------------------------------- #
# CPU fail-closed contracts.
# --------------------------------------------------------------------------- #


def test_aggregate_requires_exactly_one_seed_source():
    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)
    param = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    columns = _shared_param_columns(param, basis)
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_scene_gradient_vjp(
            basis, columns=columns, parameters=[param] * 2, weights=weights
        )


def test_aggregate_rejects_wrong_parameter_count():
    basis = _basis(port_count=3, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)
    param = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    columns = _shared_param_columns(param, basis)
    with pytest.raises(ValueError, match="one entry per port"):
        aggregate_scene_gradient_vjp(
            basis, columns=columns, parameters=[param, param], weights=weights, objective=_objective
        )


def test_aggregate_rejects_inconsistent_parameter_structure():
    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)
    param = torch.tensor([0.3, 0.1], dtype=torch.float64, requires_grad=True)
    other = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    columns = _shared_param_columns(param, basis)
    with pytest.raises(ValueError, match="every column to share"):
        aggregate_scene_gradient_vjp(
            basis,
            columns=columns,
            parameters=[param, other],
            weights=weights,
            objective=_objective,
        )


def test_aggregate_detached_columns_fail_closed():
    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)
    param = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    columns = [
        (e_theta.detach(), e_phi.detach())
        for e_theta, e_phi in _shared_param_columns(param, basis)
    ]
    with pytest.raises(ValueError, match="stores detached patterns"):
        aggregate_scene_gradient_vjp(
            basis, columns=columns, parameters=[param] * 2, weights=weights, objective=_objective
        )


def test_ensemble_requires_multigpu_execution():
    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[1.0e9])
    weights = _weights(basis)
    with pytest.raises(TypeError, match="MultiGPUExecution"):
        ensemble_scene_gradient_vjp(
            basis,
            column_forward=lambda index, device: None,
            weights=weights,
            execution=object(),
            objective=_objective,
        )


# --------------------------------------------------------------------------- #
# CUDA 2-GPU parity.
# --------------------------------------------------------------------------- #


def _synthetic_column_forward(base_param, basis_on_device):
    patterns = basis_on_device.embedded_patterns

    def column_forward(index, device):
        param = base_param.detach().to(device).requires_grad_(True)
        base_theta = patterns.e_theta[:, index].detach().to(device)
        base_phi = patterns.e_phi[:, index].detach().to(device)
        e_theta, e_phi = _column_map(param, base_theta, base_phi, index)
        return e_theta, e_phi, param

    return column_forward


@_TWO_GPU
def test_ensemble_one_vs_two_gpu_parity_synthetic():
    """1-GPU vs 2-GPU aggregated scene gradient parity on a fast synthetic map.

    Same combined objective, same fixed reduction order; the only difference is
    that the 2-GPU run splits the columns across cuda:0/cuda:1. On homogeneous
    A6000s the per-column VJPs are bit-identical, so the reduced gradient must be
    invariant to placement.
    """

    basis = _basis(port_count=4, points_theta=7, points_phi=5, freqs=[1.0e9, 2.0e9], device="cuda:0")
    weights = _weights(basis).to("cuda:0")
    base_param = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)

    forward = _synthetic_column_forward(base_param, basis)

    single = ensemble_scene_gradient_vjp(
        basis,
        column_forward=forward,
        weights=weights,
        execution=MultiGPUExecution.ensemble(devices=("cuda:0",)),
        objective=_objective,
        reduction_device="cuda:0",
    )
    dual = ensemble_scene_gradient_vjp(
        basis,
        column_forward=forward,
        weights=weights,
        execution=MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1")),
        objective=_objective,
        reduction_device="cuda:0",
    )

    # The 2-GPU run genuinely used both devices.
    assert set(dual.column_devices) == {"cuda:0", "cuda:1"}
    assert set(single.column_devices) == {"cuda:0"}
    assert single.gradient.device.type == "cuda" and single.gradient.get_device() == 0
    assert dual.gradient.get_device() == 0

    difference = float((single.gradient - dual.gradient).abs().max())
    # The fixed public-port reduction order makes the sum associative-invariant, and
    # each per-column analytic float64 VJP is bit-identical on the homogeneous A6000s,
    # so placement does not perturb a single bit. Measured maxabsdiff == 0 (bitwise).
    assert torch.equal(single.gradient, dual.gradient), (
        f"1-vs-2-GPU parity is not bitwise; maxabsdiff {difference:.3e}"
    )


@_TWO_GPU
def test_ensemble_two_gpu_matches_central_difference_synthetic():
    """The 2-GPU aggregated gradient matches central differences of the objective."""

    basis = _basis(port_count=3, points_theta=6, points_phi=5, freqs=[1.5e9], device="cuda:0")
    weights = _weights(basis).to("cuda:0")
    values = [0.15, -0.25]
    base_param = torch.tensor(values, dtype=torch.float64)

    aggregated = ensemble_scene_gradient_vjp(
        basis,
        column_forward=_synthetic_column_forward(base_param, basis),
        weights=weights,
        execution=MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1")),
        objective=_objective,
        reduction_device="cuda:0",
    )
    gradient = aggregated.gradient.detach().cpu()

    def loss(point):
        detached = torch.tensor(point, dtype=torch.float64, device="cuda:0")
        cols = []
        for index in range(3):
            base_theta = basis.embedded_patterns.e_theta[:, index].detach()
            base_phi = basis.embedded_patterns.e_phi[:, index].detach()
            cols.append(_column_map(detached, base_theta, base_phi, index))
        column_weight = weights[:, :, None, None]
        e_theta = sum(column_weight[:, n] * cols[n][0] for n in range(3))
        e_phi = sum(column_weight[:, n] * cols[n][1] for n in range(3))
        return float(_objective(e_theta, e_phi).detach())

    step = 1.0e-6
    for slot in range(len(values)):
        plus = list(values)
        minus = list(values)
        plus[slot] += step
        minus[slot] -= step
        finite_difference = (loss(plus) - loss(minus)) / (2.0 * step)
        analytic = float(gradient[slot])
        assert abs(analytic - finite_difference) / (abs(finite_difference) + 1.0e-30) < 1.0e-6


# --------------------------------------------------------------------------- #
# Real FDTD two-column array split across the two GPUs.
# --------------------------------------------------------------------------- #


class _SinglePortColumn(mw.SceneModule):
    """One single-port FDTD array column with a per-device trainable density.

    Mirrors the tested F3a SceneModule path (differentiable ``MaterialRegion``
    density through the adjoint bridge) but drives a single selectable port so the
    per-column forwards can be distributed one per GPU.
    """

    def __init__(self, base_logits, device, position):
        super().__init__()
        self.position = float(position)
        self.logits = torch.nn.Parameter(base_logits.detach().to(device))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.04, 0.04),) * 3),
            grid=mw.GridSpec.uniform(0.005),
            boundary=mw.BoundarySpec.pml(num_layers=4),
            device=str(self.logits.device),
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.01, 0.005, 0.005)),
                density=density,
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(self.position, 0.0, 0.0),
                polarization="Ez",
                width=0.004,
                source_time=mw.GaussianPulse(
                    frequency=2.5e9, fwidth=1.0e9, delay=0.3e-9, amplitude=1.0
                ),
            )
        )
        scene.add_monitor(
            mw.ClosedSurfaceMonitor.box(
                "nf", position=(0.0, 0.0, 0.0), size=(0.05, 0.05, 0.05), frequencies=(2.5e9,)
            )
        )
        return scene


def _fdtd_column_forward(base_logits, port_positions, *, theta, phi, time_steps):
    from witwin.maxwell.postprocess.antenna import _far_fields_from_result

    def column_forward(index, device):
        module = _SinglePortColumn(base_logits, device, port_positions[index])
        result = mw.Simulation.fdtd(
            module,
            frequency=2.5e9,
            run_time=mw.TimeConfig(time_steps=time_steps),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        ).run()
        frequencies = torch.tensor([2.5e9], device=device, dtype=torch.float64)
        transformed = _far_fields_from_result(
            result,
            surface="nf",
            frequencies=frequencies,
            theta=theta.to(device),
            phi=phi.to(device),
            radius=1.0,
            phase_center=torch.zeros(3, device=device, dtype=torch.float64),
            frame=torch.eye(3, device=device, dtype=torch.float64),
        )
        return transformed["e_theta"], transformed["e_phi"], module.logits

    return column_forward


@_TWO_GPU
def test_ensemble_fdtd_array_one_vs_two_gpu_parity():
    """Real two-column FDTD array: 1-GPU vs 2-GPU aggregated scene-gradient parity.

    Each column is a single-port FDTD forward whose NF2FF far field depends on a
    shared trainable ``MaterialRegion`` density (replicated per device). The
    aggregated scene gradient of a combined-field objective must agree whether the
    two columns ran on one GPU or one per GPU.
    """

    time_steps = 160
    theta = torch.linspace(0.0, math.pi, 5, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 4, dtype=torch.float64)
    port_positions = (-0.01, 0.01)
    weights = torch.tensor([0.8 + 0.2j, -0.3 + 0.6j], dtype=torch.complex128)

    basis = _basis(port_count=2, points_theta=5, points_phi=4, freqs=[2.5e9], device="cuda:0")
    weights = weights.to("cuda:0")
    base_logits = torch.full((2, 1, 1), 0.3, dtype=torch.float64)

    def objective(e_theta, e_phi):
        return (e_theta * e_theta.conj()).real.sum() + (e_phi * e_phi.conj()).real.sum()

    forward = _fdtd_column_forward(
        base_logits, port_positions, theta=theta, phi=phi, time_steps=time_steps
    )

    single = ensemble_scene_gradient_vjp(
        basis,
        column_forward=forward,
        weights=weights,
        execution=MultiGPUExecution.ensemble(devices=("cuda:0",)),
        objective=objective,
        reduction_device="cuda:0",
    )
    dual = ensemble_scene_gradient_vjp(
        basis,
        column_forward=forward,
        weights=weights,
        execution=MultiGPUExecution.ensemble(devices=("cuda:0", "cuda:1")),
        objective=objective,
        reduction_device="cuda:0",
    )

    assert set(dual.column_devices) == {"cuda:0", "cuda:1"}
    assert torch.isfinite(dual.gradient).all()
    assert float(dual.gradient.abs().max()) > 0.0

    single_grad = single.gradient.detach().cpu()
    dual_grad = dual.gradient.detach().cpu()
    denominator = float(single_grad.abs().max()) + 1.0e-30
    relative = float((single_grad - dual_grad).abs().max()) / denominator
    # FDTD runs float32 internally; the NF2FF transform promotes to float64. On
    # homogeneous GPUs the split is expected at or near bitwise; gate at the FDTD
    # float32 adjoint floor and record the measured value.
    assert relative < 1.0e-4, f"1-vs-2-GPU FDTD parity relative difference {relative:.3e}"
