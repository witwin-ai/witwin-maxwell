from types import SimpleNamespace

import pytest
import torch
from witwin.core.material import VACUUM_PERMITTIVITY

import witwin.maxwell as mw
import witwin.maxwell.fdtd.excitation.modes as mode_solver
from witwin.maxwell.compiler.materials import evaluate_material_permittivity
from witwin.maxwell.compiler.sources import _compile_mode_source
from witwin.maxwell.fdtd.excitation.modes import sample_mode_source_component, solve_mode_source_profile
from witwin.maxwell.scene import prepare_scene


def _mode_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(
        # A rectangular, z-offset core: with a symmetric core the Ez profile on
        # the z half-grid has an even interior window whose two central |peak|
        # entries tie within float64 noise, making the peak-normalization
        # argmax (and hence the implicit sparse gradient) a knife-edge function
        # of the grid-derived transverse spacing.
        mw.Box(position=(0.0, 0.0, 0.02), size=(1.28, 0.24, 0.32)).with_material(
            mw.Material(eps_r=12.0),
            name="core",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(0.0, 0.0, 0.0),
            # Aperture edges land exactly on grid nodes (+-0.24); a 0.56 span
            # would place them exactly midway between nodes, where the nearest-
            # node resolution is an ill-conditioned floating-point tie.
            size=(0.0, 0.48, 0.48),
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="port0",
        )
    )
    return scene


def _multimode_scene(mode_index=1):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(1.6, 0.48, 0.48)).with_material(
            mw.Material(eps_r=12.0),
            name="core",
        )
    )
    scene.add_source(
        mw.ModeSource(
            position=(0.0, 0.0, 0.0),
            size=(0.0, 0.8, 0.8),
            mode_index=mode_index,
            polarization="Ez",
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name=f"port{mode_index}",
        )
    )
    return scene


def _mode_context(scene, *, rebuild_from_fields: bool, requires_grad: bool, use_compiled_material_model: bool | None = None):
    prepared_scene = prepare_scene(scene)
    model = prepared_scene.compile_materials()
    eps_r = evaluate_material_permittivity(model, 1.0e9).to(dtype=torch.float32)
    eps_ex = (0.5 * (eps_r[:-1, :, :] + eps_r[1:, :, :]) * VACUUM_PERMITTIVITY).contiguous()
    eps_ey = (0.5 * (eps_r[:, :-1, :] + eps_r[:, 1:, :]) * VACUUM_PERMITTIVITY).contiguous()
    eps_ez = (0.5 * (eps_r[:, :, :-1] + eps_r[:, :, 1:]) * VACUUM_PERMITTIVITY).contiguous()
    if requires_grad:
        eps_ez = eps_ez.detach().clone().requires_grad_(True)

    if use_compiled_material_model is None:
        use_compiled_material_model = not rebuild_from_fields

    return SimpleNamespace(
        scene=prepared_scene,
        dx=prepared_scene.dx,
        dy=prepared_scene.dy,
        dz=prepared_scene.dz,
        Ex=torch.empty((1,), device=prepared_scene.device, dtype=torch.float32),
        c=299792458.0,
        eps0=VACUUM_PERMITTIVITY,
        boundary_kind=prepared_scene.boundary.kind,
        _compiled_material_model=model if use_compiled_material_model else None,
        _mode_source_rebuild_from_fields=rebuild_from_fields,
        eps_Ex=eps_ex,
        eps_Ey=eps_ey,
        eps_Ez=eps_ez,
    )


def test_torch_mode_solver_matches_scipy_forward_profile_and_beta():
    scene = _mode_scene()
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)

    reference = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=False),
        compiled_source,
    )
    differentiable = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=True, requires_grad=True),
        compiled_source,
    )

    assert differentiable["beta_tensor"] is not None
    assert differentiable["effective_index_tensor"] is not None
    assert differentiable["beta"] == pytest.approx(reference["beta"], rel=1e-5, abs=1e-6)
    assert differentiable["effective_index"] == pytest.approx(reference["effective_index"], rel=1e-5, abs=1e-6)
    assert torch.allclose(differentiable["profile"], reference["profile"], rtol=1e-4, atol=1e-4)


def test_torch_mode_solver_backpropagates_through_beta():
    scene = _mode_scene()
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)
    context = _mode_context(scene, rebuild_from_fields=True, requires_grad=True)

    mode_data = solve_mode_source_profile(context, compiled_source)
    loss = mode_data["beta_tensor"]
    assert loss is not None

    loss.backward()

    assert context.eps_Ez.grad is not None
    assert torch.isfinite(context.eps_Ez.grad).all()
    assert float(torch.max(torch.abs(context.eps_Ez.grad)).item()) > 0.0


def test_sparse_lobpcg_mode_solver_matches_dense_forward_baseline(monkeypatch):
    scene = _mode_scene()
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)

    reference = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=False),
        compiled_source,
    )

    sparse_calls = {"count": 0}
    original_sparse_solver = mode_solver._solve_mode_eigenpair_torch_sparse

    def wrapped_sparse_solver(*args, **kwargs):
        sparse_calls["count"] += 1
        return original_sparse_solver(*args, **kwargs)

    monkeypatch.setattr(mode_solver, "_DENSE_EIGEN_LIMIT", 4)
    monkeypatch.setattr(mode_solver, "_solve_mode_eigenpair_torch_sparse", wrapped_sparse_solver)

    sparse_result = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=False),
        compiled_source,
    )

    assert sparse_calls["count"] == 1
    assert sparse_result["beta"] == pytest.approx(reference["beta"], rel=1e-4, abs=1e-6)
    assert sparse_result["effective_index"] == pytest.approx(reference["effective_index"], rel=1e-4, abs=1e-6)
    assert torch.allclose(sparse_result["profile"], reference["profile"], rtol=5e-4, atol=5e-4)


def _profile_loss(mode_data):
    weight = torch.linspace(
        0.2,
        1.1,
        steps=mode_data["profile"].numel(),
        device=mode_data["profile"].device,
        dtype=mode_data["profile"].dtype,
    ).reshape(mode_data["profile"].shape)
    return mode_data["beta_tensor"] + torch.sum(mode_data["profile"] * weight)


def test_differentiable_sparse_mode_solver_matches_dense_forward_and_gradient_baseline(monkeypatch):
    scene = _mode_scene()
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)

    dense_context = _mode_context(scene, rebuild_from_fields=True, requires_grad=True)
    dense_result = solve_mode_source_profile(dense_context, compiled_source)
    dense_loss = _profile_loss(dense_result)
    dense_loss.backward()
    dense_grad = dense_context.eps_Ez.grad.detach().clone()

    sparse_calls = {"count": 0}
    original_sparse_solver = mode_solver._solve_mode_eigenpair_torch_sparse_implicit

    def wrapped_sparse_solver(*args, **kwargs):
        sparse_calls["count"] += 1
        return original_sparse_solver(*args, **kwargs)

    sparse_context = _mode_context(scene, rebuild_from_fields=True, requires_grad=True)
    monkeypatch.setattr(mode_solver, "_DENSE_EIGEN_LIMIT", 4)
    monkeypatch.setattr(mode_solver, "_solve_mode_eigenpair_torch_sparse_implicit", wrapped_sparse_solver)
    sparse_result = solve_mode_source_profile(sparse_context, compiled_source)
    sparse_loss = _profile_loss(sparse_result)
    sparse_loss.backward()

    assert sparse_calls["count"] == 1
    assert sparse_result["beta"] == pytest.approx(dense_result["beta"], rel=1e-4, abs=1e-6)
    assert sparse_result["effective_index"] == pytest.approx(dense_result["effective_index"], rel=1e-4, abs=1e-6)
    assert torch.allclose(sparse_result["profile"], dense_result["profile"], rtol=5e-4, atol=5e-4)
    assert torch.allclose(sparse_context.eps_Ez.grad, dense_grad, rtol=5e-3, atol=5e-5)


def test_differentiable_sparse_mode_solver_supports_higher_order_modes(monkeypatch):
    scene = _multimode_scene(mode_index=1)
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)

    dense_context = _mode_context(scene, rebuild_from_fields=True, requires_grad=True)
    dense_result = solve_mode_source_profile(dense_context, compiled_source)
    dense_loss = _profile_loss(dense_result)
    dense_loss.backward()
    dense_grad = dense_context.eps_Ez.grad.detach().clone()

    sparse_context = _mode_context(scene, rebuild_from_fields=True, requires_grad=True)
    monkeypatch.setattr(mode_solver, "_DENSE_EIGEN_LIMIT", 4)
    sparse_result = solve_mode_source_profile(sparse_context, compiled_source)
    sparse_loss = _profile_loss(sparse_result)
    sparse_loss.backward()

    assert sparse_result["beta"] == pytest.approx(dense_result["beta"], rel=1e-4, abs=1e-6)
    assert sparse_result["effective_index"] == pytest.approx(dense_result["effective_index"], rel=1e-4, abs=1e-6)
    assert torch.allclose(sparse_result["profile"], dense_result["profile"], rtol=7e-4, atol=7e-4)
    assert torch.allclose(sparse_context.eps_Ez.grad, dense_grad, rtol=1e-2, atol=1e-4)


def test_full_vector_mode_solver_matches_scalar_beta_on_rectangular_waveguide():
    scene = _mode_scene()
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)

    scalar_result = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=False),
        compiled_source,
    )
    vector_result = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=True),
        compiled_source,
    )

    assert vector_result["mode_solver_kind"] == "vector_dense"
    assert vector_result["beta"] == pytest.approx(scalar_result["beta"], rel=5e-2, abs=1e-4)
    dominant = torch.max(torch.abs(vector_result["component_profiles"]["Ez"]))
    secondary = torch.max(torch.abs(vector_result["component_profiles"].get("Ey", torch.zeros_like(vector_result["profile"]))))
    assert float(dominant.item()) > 0.0
    assert float(secondary.item()) < float(dominant.item())


def test_full_vector_mode_solver_supports_mu_not_unity():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, 0.0), size=(1.28, 0.24, 0.24)).with_material(
            mw.Material(eps_r=12.0, mu_r=2.0),
            name="core",
        )
    )
    source = mw.ModeSource(
        position=(0.0, 0.0, 0.0),
        size=(0.0, 0.56, 0.56),
        polarization="Ez",
        source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
        name="port0",
    )
    compiled_source = _compile_mode_source(source, default_frequency=1.0e9)
    result = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=True),
        compiled_source,
    )

    assert result["mode_solver_kind"] == "vector_dense"
    assert result["beta"] > 0.0
    assert torch.max(torch.abs(result["component_profiles"]["Ez"])).item() > 0.0
    assert torch.max(torch.abs(result["component_profiles"]["Hy"])).item() > 0.0


def test_full_vector_sparse_mode_solver_matches_dense_forward(monkeypatch):
    scene = _mode_scene()
    compiled_source = _compile_mode_source(scene.sources[0], default_frequency=1.0e9)

    dense_result = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=True),
        compiled_source,
    )

    monkeypatch.setattr(mode_solver, "_FULL_VECTOR_DENSE_LIMIT", 4)
    sparse_result = solve_mode_source_profile(
        _mode_context(scene, rebuild_from_fields=False, requires_grad=False, use_compiled_material_model=True),
        compiled_source,
    )

    assert sparse_result["mode_solver_kind"] == "vector_sparse"
    assert sparse_result["beta"] == pytest.approx(dense_result["beta"], rel=5e-3, abs=1e-5)
    assert torch.allclose(
        sparse_result["component_profiles"]["Ez"],
        dense_result["component_profiles"]["Ez"],
        rtol=5e-2,
        atol=5e-3,
    )
