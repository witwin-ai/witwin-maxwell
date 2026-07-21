"""Scene-gradient VJP through the array embedded-pattern basis (plan 06 Phase 4).

``ArrayBasisData.scene_gradient_vjp`` aggregates the per-column adjoints of the
linear beam combine ``E = sum_n w_n e_n`` back onto the scene parameters. The
retained basis stores the columns detached, so scene gradients require re-running
the per-column forwards under autograd and passing the resulting *live* columns
here; the method then applies the derived per-column seed ``conj(w_n) * cot_E``
and sums the VJPs in a deterministic order.

The CPU gates cover the aggregation math exactly (equivalence to end-to-end
autograd, central-difference agreement, weight conjugation load-bearing,
deterministic reduction order, fail-closed contracts). The CUDA gate closes the
loop through a genuine FDTD array: N single-port forwards produce live NF2FF
far-field columns as a differentiable function of a trainable ``MaterialRegion``
density, and the aggregated scene gradient is checked against central
differences of the same combined objective.
"""

import math

import pytest
import torch

import witwin.maxwell as mw

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")


def _placeholder_basis(*, port_count, points_theta, points_phi, freqs, device="cpu"):
    """A shape/port/device-valid basis whose detached columns are never used.

    ``scene_gradient_vjp`` reads the basis only for port count, frequency count,
    angular shape, device, and dtype validation; the gradient flows through the
    *live* columns the caller supplies, never through these placeholder patterns.
    """

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
    return mw.ArrayBasisData(network=network, embedded_patterns=patterns, fingerprint="scene-grad")


def _synthetic_columns(param, basis, *, extra=None):
    """Per-column live far fields as differentiable functions of ``param``.

    A distinct nonlinear per-column map guarantees every column contributes and
    that the combine is not degenerate.
    """

    port_count = basis.network.s.shape[1]
    complex_param = param.to(torch.complex128)
    columns = []
    for index in range(port_count):
        scale = 1.0 + 0.2 * complex_param
        if extra is not None:
            scale = scale + 0.15 * extra.to(torch.complex128)
        e_theta = basis.embedded_patterns.e_theta[:, index] * torch.exp(
            1j * (index + 1) * complex_param
        ) * scale
        e_phi = basis.embedded_patterns.e_phi[:, index] * torch.exp(
            1j * (0.5 * index + 1.0) * complex_param
        )
        columns.append((e_theta, e_phi))
    return columns


def _objective(e_theta, e_phi):
    return (
        (e_theta * e_theta.conj()).real.sum()
        + 0.4 * (e_phi * e_phi.conj()).real.sum()
        + 0.1 * e_theta.real.sum()
    )


def _weights(basis):
    frequency_count, port_count, _ = basis.network.s.shape
    generator = torch.Generator().manual_seed(7)
    real = torch.rand((frequency_count, port_count), generator=generator, dtype=torch.float64) - 0.5
    imag = torch.rand((frequency_count, port_count), generator=generator, dtype=torch.float64) - 0.5
    return torch.complex(real, imag)


def _basis_2c():
    return _placeholder_basis(port_count=3, points_theta=5, points_phi=4, freqs=[1.0e9, 1.5e9])


def test_scene_gradient_vjp_matches_end_to_end_autograd():
    """The per-column seeded VJP sum is bit-identical to end-to-end autograd.

    This is the structural proof of the weight-conjugation convention: the same
    combined objective differentiated by ``autograd.grad`` and by the method's
    ``conj(w_n) * cot`` seeds must coincide exactly.
    """

    basis = _basis_2c()
    weights = _weights(basis)

    param = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    method_grad = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=param, objective=_objective
    )

    param_e = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    columns_e = _synthetic_columns(param_e, basis)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns_e[n][0] for n in range(3))
    e_phi = sum(column_weight[:, n] * columns_e[n][1] for n in range(3))
    (reference_grad,) = torch.autograd.grad(_objective(e_theta, e_phi), param_e)

    assert torch.equal(method_grad, reference_grad)


def test_scene_gradient_vjp_matches_central_difference():
    basis = _basis_2c()
    weights = _weights(basis)
    value = 0.37

    param = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    gradient = float(
        basis.scene_gradient_vjp(
            columns=columns, weights=weights, parameters=param, objective=_objective
        )
    )
    assert abs(gradient) > 1.0e-6

    def loss(point):
        detached = torch.tensor(point, dtype=torch.float64)
        cols = _synthetic_columns(detached, basis)
        column_weight = weights[:, :, None, None]
        e_theta = sum(column_weight[:, n] * cols[n][0] for n in range(3))
        e_phi = sum(column_weight[:, n] * cols[n][1] for n in range(3))
        return float(_objective(e_theta, e_phi))

    step = 1.0e-5
    finite_difference = (loss(value + step) - loss(value - step)) / (2.0 * step)
    assert abs(gradient - finite_difference) / abs(finite_difference) < 1.0e-6


def test_scene_gradient_vjp_weight_conjugation_is_load_bearing():
    """Dropping the ``conj`` on the weight seed breaks agreement with autograd.

    A manual non-conjugated seed (``w_n`` instead of ``conj(w_n)``) yields a
    materially different gradient, so the conjugation the method applies is not
    incidental.
    """

    basis = _basis_2c()
    weights = _weights(basis)

    param = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    method_grad = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=param, objective=_objective
    )

    # Reconstruct the seed path with the WRONG (non-conjugated) weight.
    param_w = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    columns_w = _synthetic_columns(param_w, basis)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns_w[n][0] for n in range(3))
    e_phi = sum(column_weight[:, n] * columns_w[n][1] for n in range(3))
    cot_theta, cot_phi = torch.autograd.grad(
        _objective(e_theta, e_phi), (e_theta, e_phi), retain_graph=True
    )
    wrong = torch.zeros_like(param_w)
    for n in range(3):
        seed_scale = weights[:, n, None, None]  # NOT conjugated
        (grad_theta,) = torch.autograd.grad(
            columns_w[n][0], param_w, grad_outputs=seed_scale * cot_theta, retain_graph=True
        )
        (grad_phi,) = torch.autograd.grad(
            columns_w[n][1], param_w, grad_outputs=seed_scale * cot_phi, retain_graph=True
        )
        wrong = wrong + grad_theta + grad_phi

    assert abs(float(method_grad) - float(wrong)) > 1.0e-3 * abs(float(method_grad))


def test_scene_gradient_vjp_reduction_order_invariant():
    basis = _basis_2c()
    weights = _weights(basis)

    param = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    default = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=param, objective=_objective
    )

    param_p = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    columns_p = _synthetic_columns(param_p, basis)
    permuted = basis.scene_gradient_vjp(
        columns=columns_p,
        weights=weights,
        parameters=param_p,
        objective=_objective,
        reduction_order=(2, 0, 1),
    )
    # A fixed reduction order is bitwise deterministic; two different orders agree
    # only up to floating-point non-associativity of the per-column sum.
    torch.testing.assert_close(default, permuted, rtol=1.0e-12, atol=1.0e-12)


def test_scene_gradient_vjp_field_cotangents_matches_objective():
    basis = _basis_2c()
    weights = _weights(basis)

    param = torch.tensor(0.42, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    via_objective = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=param, objective=_objective
    )

    param_c = torch.tensor(0.42, dtype=torch.float64, requires_grad=True)
    columns_c = _synthetic_columns(param_c, basis)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns_c[n][0] for n in range(3))
    e_phi = sum(column_weight[:, n] * columns_c[n][1] for n in range(3))
    cotangents = torch.autograd.grad(_objective(e_theta, e_phi), (e_theta, e_phi), retain_graph=True)
    via_cotangents = basis.scene_gradient_vjp(
        columns=columns_c, weights=weights, parameters=param_c, field_cotangents=cotangents
    )
    assert torch.equal(via_objective, via_cotangents)


def test_scene_gradient_vjp_supports_multiple_parameters():
    basis = _basis_2c()
    weights = _weights(basis)

    param_a = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    param_b = torch.tensor(-0.12, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param_a, basis, extra=param_b)
    grad_a, grad_b = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=(param_a, param_b), objective=_objective
    )

    param_a2 = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    param_b2 = torch.tensor(-0.12, dtype=torch.float64, requires_grad=True)
    columns_e = _synthetic_columns(param_a2, basis, extra=param_b2)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns_e[n][0] for n in range(3))
    e_phi = sum(column_weight[:, n] * columns_e[n][1] for n in range(3))
    ref_a, ref_b = torch.autograd.grad(_objective(e_theta, e_phi), (param_a2, param_b2))
    # Per-column accumulation order differs from autograd's internal reduction for a
    # shared parameter, so equality holds up to floating-point round-off.
    torch.testing.assert_close(grad_a, ref_a, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(grad_b, ref_b, rtol=1.0e-12, atol=1.0e-12)


def test_scene_gradient_vjp_frequency_flat_weights_broadcast():
    basis = _basis_2c()
    flat = torch.tensor([0.6 + 0.2j, -0.3 + 0.7j, 0.1 - 0.5j], dtype=torch.complex128)
    param = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    grad_flat = basis.scene_gradient_vjp(
        columns=columns, weights=flat, parameters=param, objective=_objective
    )

    param_e = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    columns_e = _synthetic_columns(param_e, basis)
    explicit = flat[None, :].expand(2, 3).contiguous()
    grad_explicit = basis.scene_gradient_vjp(
        columns=columns_e, weights=explicit, parameters=param_e, objective=_objective
    )
    assert torch.equal(grad_flat, grad_explicit)


# --- fail-closed / validation contracts -------------------------------------


def test_scene_gradient_vjp_detached_columns_fail_closed():
    """Detached columns (the retained-basis default) cannot back-propagate."""

    basis = _basis_2c()
    weights = _weights(basis)
    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = [
        (e_theta.detach(), e_phi.detach())
        for e_theta, e_phi in _synthetic_columns(param, basis)
    ]
    with pytest.raises(ValueError, match="stores detached patterns"):
        basis.scene_gradient_vjp(
            columns=columns, weights=weights, parameters=param, objective=_objective
        )


def test_scene_gradient_vjp_requires_exactly_one_seed_source():
    basis = _basis_2c()
    weights = _weights(basis)
    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    with pytest.raises(ValueError, match="exactly one"):
        basis.scene_gradient_vjp(columns=columns, weights=weights, parameters=param)
    column_weight = weights[:, :, None, None]
    e_theta = sum(column_weight[:, n] * columns[n][0] for n in range(3))
    e_phi = sum(column_weight[:, n] * columns[n][1] for n in range(3))
    cotangents = torch.autograd.grad(_objective(e_theta, e_phi), (e_theta, e_phi), retain_graph=True)
    with pytest.raises(ValueError, match="exactly one"):
        basis.scene_gradient_vjp(
            columns=_synthetic_columns(param, basis),
            weights=weights,
            parameters=param,
            objective=_objective,
            field_cotangents=cotangents,
        )


def test_scene_gradient_vjp_rejects_wrong_column_count():
    basis = _basis_2c()
    weights = _weights(basis)
    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)[:2]
    with pytest.raises(ValueError, match="one .* pair per port"):
        basis.scene_gradient_vjp(
            columns=columns, weights=weights, parameters=param, objective=_objective
        )


def test_scene_gradient_vjp_rejects_batched_weights():
    basis = _basis_2c()
    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    batched = torch.ones((2, 2, 3), dtype=torch.complex128)
    with pytest.raises(ValueError, match=r"\[N\] or \[F, N\]"):
        basis.scene_gradient_vjp(
            columns=columns, weights=batched, parameters=param, objective=_objective
        )


def test_scene_gradient_vjp_rejects_shape_mismatched_columns():
    basis = _basis_2c()
    weights = _weights(basis)
    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    bad_theta = columns[0][0][:, :-1, :]  # drop one theta sample
    columns[0] = (bad_theta, columns[0][1])
    with pytest.raises(ValueError, match=r"\[F, T, P\]"):
        basis.scene_gradient_vjp(
            columns=columns, weights=weights, parameters=param, objective=_objective
        )


def test_scene_gradient_vjp_rejects_non_scalar_objective():
    basis = _basis_2c()
    weights = _weights(basis)
    param = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    columns = _synthetic_columns(param, basis)
    with pytest.raises(TypeError, match="real scalar"):
        basis.scene_gradient_vjp(
            columns=columns,
            weights=weights,
            parameters=param,
            objective=lambda e_theta, e_phi: (e_theta * e_theta.conj()).real.sum(dim=0),
        )


# --- regression: combine() weight gradients are unchanged --------------------


def test_combine_weight_gradient_unchanged_by_scene_gradient_addition():
    """The pre-existing weight-gradient path through combine() still holds.

    ``scene_gradient_vjp`` adds scene-parameter gradients without touching the
    weight-gradient contract: differentiating a combined-field objective w.r.t.
    the incident weights through ``combine`` must still match finite differences.
    """

    basis = _basis_2c()
    real = torch.tensor([[0.6, -0.3, 0.2], [0.1, 0.4, -0.5]], dtype=torch.float64, requires_grad=True)
    imag = torch.tensor([[0.2, 0.5, -0.1], [-0.4, 0.3, 0.2]], dtype=torch.float64, requires_grad=True)

    def objective_from_weights(real_part, imag_part):
        weights = torch.complex(real_part, imag_part)
        beam = basis.combine(weights)
        return beam.far_field.e_theta.abs().square().sum() + beam.antenna.realized_gain.sum()

    loss = objective_from_weights(real, imag)
    grad_real, grad_imag = torch.autograd.grad(loss, (real, imag))
    assert torch.isfinite(grad_real).all() and torch.isfinite(grad_imag).all()

    step = 1.0e-6
    flat = grad_real.flatten()
    index = int(flat.abs().argmax())
    base_real = real.detach().clone()
    perturbed = base_real.flatten()
    perturbed[index] += step
    plus = float(objective_from_weights(perturbed.reshape(base_real.shape), imag.detach()))
    perturbed[index] -= 2.0 * step
    minus = float(objective_from_weights(perturbed.reshape(base_real.shape), imag.detach()))
    finite_difference = (plus - minus) / (2.0 * step)
    assert abs(float(flat[index]) - finite_difference) / abs(finite_difference) < 1.0e-4


# --- CUDA end-to-end FDTD array gate ----------------------------------------


class _ArrayColumnModel(mw.SceneModule):
    """N single-port array whose active drive column is selectable per run.

    All columns share the one trainable ``logits`` leaf, so the per-column NF2FF
    far fields are differentiable functions of the same design density.
    """

    def __init__(self, *, port_positions, shape=(2, 1, 1), init=0.0):
        super().__init__()
        self.port_positions = tuple(port_positions)
        self.active_index = 0
        self.logits = torch.nn.Parameter(torch.full(shape, float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.04, 0.04),) * 3),
            grid=mw.GridSpec.uniform(0.005),
            boundary=mw.BoundarySpec.pml(num_layers=4),
            device="cuda",
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
        position = self.port_positions[self.active_index]
        scene.add_source(
            mw.PointDipole(
                position=(position, 0.0, 0.0),
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


def _fdtd_columns(model, *, theta, phi, time_steps):
    """Run one forward per drive column and return live NF2FF far-field columns."""

    from witwin.maxwell.postprocess.antenna import _far_fields_from_result

    frequencies = torch.tensor([2.5e9], device="cuda", dtype=torch.float64)
    columns = []
    for index in range(len(model.port_positions)):
        model.active_index = index
        result = mw.Simulation.fdtd(
            model,
            frequency=2.5e9,
            run_time=mw.TimeConfig(time_steps=time_steps),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        ).run()
        transformed = _far_fields_from_result(
            result,
            surface="nf",
            frequencies=frequencies,
            theta=theta,
            phi=phi,
            radius=1.0,
            phase_center=torch.zeros(3, device="cuda", dtype=torch.float64),
            frame=torch.eye(3, device="cuda", dtype=torch.float64),
        )
        columns.append((transformed["e_theta"], transformed["e_phi"]))
    return columns


@_CUDA
def test_scene_gradient_vjp_matches_fd_on_fdtd_array():
    """End-to-end: aggregated scene gradient vs central differences on real FDTD.

    Two single-port FDTD forwards produce live NF2FF far-field columns that
    depend on a shared trainable ``MaterialRegion`` density. The combined-field
    objective's scene gradient from ``scene_gradient_vjp`` must match a per-voxel
    central difference of the same objective. FDTD runs float32 internally and
    the NF2FF transform promotes to float64, so the dominant voxel is checked at
    the FDTD-adjoint tolerance floor.
    """

    torch.manual_seed(0)
    time_steps = 192
    theta = torch.linspace(0.0, math.pi, 5, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 4, device="cuda", dtype=torch.float64)
    port_positions = (-0.01, 0.01)
    weights = torch.tensor([0.8 + 0.2j, -0.3 + 0.6j], device="cuda", dtype=torch.complex128)

    basis = _placeholder_basis(
        port_count=2, points_theta=5, points_phi=4, freqs=[2.5e9], device="cuda"
    )

    model = _ArrayColumnModel(port_positions=port_positions, shape=(2, 1, 1), init=0.3).cuda()

    def objective(e_theta, e_phi):
        return (e_theta * e_theta.conj()).real.sum() + (e_phi * e_phi.conj()).real.sum()

    columns = _fdtd_columns(model, theta=theta, phi=phi, time_steps=time_steps)
    scene_grad = basis.scene_gradient_vjp(
        columns=columns, weights=weights, parameters=model.logits, objective=objective
    )
    assert torch.isfinite(scene_grad).all()
    assert float(scene_grad.abs().max()) > 0.0

    def loss_value():
        cols = _fdtd_columns(model, theta=theta, phi=phi, time_steps=time_steps)
        column_weight = weights[:, None, None]
        e_theta = sum(column_weight[n] * cols[n][0] for n in range(2))
        e_phi = sum(column_weight[n] * cols[n][1] for n in range(2))
        return float(objective(e_theta, e_phi).detach())

    flat = model.logits.detach().flatten()
    dominant = int(scene_grad.abs().flatten().argmax())
    step = 2.0e-2
    saved = float(flat[dominant])
    with torch.no_grad():
        flat[dominant] = saved + step
        model.logits.copy_(flat.reshape(model.logits.shape))
    loss_plus = loss_value()
    with torch.no_grad():
        flat[dominant] = saved - step
        model.logits.copy_(flat.reshape(model.logits.shape))
    loss_minus = loss_value()
    with torch.no_grad():
        flat[dominant] = saved
        model.logits.copy_(flat.reshape(model.logits.shape))
    finite_difference = (loss_plus - loss_minus) / (2.0 * step)

    analytic = float(scene_grad.flatten()[dominant])
    relative_error = abs(analytic - finite_difference) / abs(finite_difference)
    assert relative_error < 3.0e-2, (
        f"dominant-voxel scene-gradient relative error {relative_error:.3e} "
        f"(analytic {analytic:.4e} vs FD {finite_difference:.4e})"
    )


@_CUDA
def test_scene_gradient_vjp_second_column_shifts_gradient():
    """Zeroing one column's weight changes the aggregated scene gradient.

    Confirms both columns genuinely feed the aggregate, ruling out a single-column
    fallback masquerading as the full per-column sum.
    """

    time_steps = 160
    theta = torch.linspace(0.0, math.pi, 5, device="cuda", dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 4, device="cuda", dtype=torch.float64)
    port_positions = (-0.01, 0.01)
    basis = _placeholder_basis(
        port_count=2, points_theta=5, points_phi=4, freqs=[2.5e9], device="cuda"
    )
    model = _ArrayColumnModel(port_positions=port_positions, shape=(2, 1, 1), init=0.3).cuda()

    def objective(e_theta, e_phi):
        return (e_theta * e_theta.conj()).real.sum() + (e_phi * e_phi.conj()).real.sum()

    both = torch.tensor([0.8 + 0.2j, -0.3 + 0.6j], device="cuda", dtype=torch.complex128)
    only_first = torch.tensor([0.8 + 0.2j, 0.0 + 0.0j], device="cuda", dtype=torch.complex128)

    columns = _fdtd_columns(model, theta=theta, phi=phi, time_steps=time_steps)
    grad_both = basis.scene_gradient_vjp(
        columns=columns, weights=both, parameters=model.logits, objective=objective
    )
    columns_first = _fdtd_columns(model, theta=theta, phi=phi, time_steps=time_steps)
    grad_first = basis.scene_gradient_vjp(
        columns=columns_first, weights=only_first, parameters=model.logits, objective=objective
    )
    relative_change = float((grad_both - grad_first).abs().max()) / (
        float(grad_both.abs().max()) + 1.0e-30
    )
    assert relative_change > 0.1, f"second column barely moves the gradient (rel {relative_change:.3e})"
