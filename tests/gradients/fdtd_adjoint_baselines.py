from __future__ import annotations

import torch

import witwin.maxwell.fdtd.adjoint as fdtd_adjoint
from witwin.maxwell.fdtd.adjoint import reference as fdtd_adjoint_reference
from witwin.maxwell.fdtd.material_pullback import node_gradient_from_yee_permittivity
from witwin.maxwell.scene import MaterialRegion, prepare_scene


def expected_cpml_reverse_backend() -> str:
    """The reverse-backend label ``auto`` mode records per step for a CUDA CPML scene.

    ``auto`` prefers the fused native CUDA CPML reverse step when its runner is
    registered (P6 native adjoint) and falls back to the analytic Torch reference
    label otherwise. The FDTD gradient bridges in these tests run on CUDA, where a
    registered native CPML runner qualifies, so ``native_cpml`` is what the
    backward profile records; this helper keeps the ``uses the explicit CPML
    reverse backend, never the torch_vjp fallback`` assertions correct in both
    configurations.
    """
    from witwin.maxwell.fdtd.adjoint.dispatch import (
        _NATIVE_REVERSE_LABELS,
        _ReverseBackend,
        _native_backend_available,
    )

    if _native_backend_available(_ReverseBackend.PYTHON_CPML):
        return _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_CPML]
    return "python_reference_cpml"


def expected_bloch_reverse_backend() -> str:
    """The reverse-backend label ``auto`` mode records per step for a CUDA Bloch scene.

    ``auto`` prefers the fused native CUDA complex Bloch reverse step when its
    runner is registered (P6 native adjoint) and falls back to the analytic Torch
    reference label otherwise. The FDTD gradient bridges in these tests run on CUDA,
    where a registered native Bloch runner qualifies, so ``native_bloch`` is what
    the backward profile records; this helper keeps the ``uses the explicit Bloch
    reverse backend, never the torch_vjp fallback`` assertion correct in both
    configurations.
    """
    from witwin.maxwell.fdtd.adjoint.dispatch import (
        _NATIVE_REVERSE_LABELS,
        _ReverseBackend,
        _native_backend_available,
    )

    if _native_backend_available(_ReverseBackend.PYTHON_BLOCH):
        return _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_BLOCH]
    return "python_reference_bloch"


def expected_dispersive_reverse_backend() -> str:
    """The reverse-backend label ``auto`` mode records per step for a CUDA electric-dispersive scene.

    ``auto`` prefers the fused native CUDA dispersive reverse step when its runner
    is registered (P6 native adjoint) and falls back to the analytic Torch
    reference label otherwise. The FDTD gradient bridges in these tests run on CUDA
    with electric-only dispersion, where the native dispersive runner qualifies, so
    ``native_dispersive`` is what the backward profile records; this helper keeps
    the ``uses the explicit dispersive reverse backend, never the torch_vjp
    fallback`` assertion correct in both configurations.
    """
    from witwin.maxwell.fdtd.adjoint.dispatch import (
        _NATIVE_REVERSE_LABELS,
        _ReverseBackend,
        _native_backend_available,
    )

    if _native_backend_available(_ReverseBackend.PYTHON_DISPERSIVE):
        return _NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_DISPERSIVE]
    return "python_reference_dispersive_cpml"


def expected_tfsf_reverse_backend() -> str:
    """The reverse-backend label ``auto`` mode records per step for a CUDA TFSF scene.

    ``auto`` prefers the fused native CUDA TFSF reverse step (the native standard /
    CPML base reverse plus the native 1D auxiliary reverse and sample-adjoint
    kernels) when its runner is registered (P6 native adjoint) and falls back to the
    analytic Torch reference label otherwise. The FDTD gradient bridges in these
    tests run on CUDA where the native TFSF runner qualifies, so ``native_tfsf`` is
    what the backward profile records; this helper keeps the ``uses the explicit
    TFSF reverse backend, never the torch_vjp fallback`` assertion correct in both
    configurations. The reference fallback label matches the absorbing-boundary
    (CPML) TFSF box the profile scenes use.
    """
    from witwin.maxwell.fdtd.adjoint.dispatch import (
        _NATIVE_REVERSE_LABELS,
        _ReverseBackend,
        _native_backend_available,
    )

    if _native_backend_available(_ReverseBackend.TFSF):
        return _NATIVE_REVERSE_LABELS[_ReverseBackend.TFSF]
    return "python_reference_tfsf_cpml"


def reverse_step_torch_vjp(solver, forward_state, adjoint_state, *, time_value, eps_ex, eps_ey, eps_ez):
    return fdtd_adjoint_reference.reverse_step_torch_vjp(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )


def reverse_step_standard_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_standard_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_cpml_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_cpml_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_conductive_cpml_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_conductive_cpml_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_kerr_cpml_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_kerr_cpml_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_full_aniso_cpml_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_full_aniso_cpml_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_bloch_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_bloch_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_dispersive_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    return fdtd_adjoint_reference.reverse_step_dispersive_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def reverse_step_tfsf_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    profiler=None,
):
    return fdtd_adjoint_reference.reverse_step_tfsf(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        profiler=profiler,
    )


def accumulate_source_term_gradients(
    step_result,
    *,
    solver,
    adjoint_state,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
):
    return fdtd_adjoint._accumulate_source_term_gradients(
        step_result,
        solver=solver,
        adjoint_state=adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez):
    return fdtd_adjoint._resolved_source_term_lists(solver, eps_ex, eps_ey, eps_ez)


def _clone_material_region(region, density):
    return MaterialRegion(
        name=region.name,
        geometry=region.geometry,
        density=density,
        basis=region.basis,
        bounds=region.bounds,
        eps_bounds=region.eps_bounds,
        mu_bounds=region.mu_bounds,
        filter_radius=region.filter_radius,
        projection_beta=region.projection_beta,
        symmetry=region.symmetry,
    )


def _clone_scene_with_densities(scene, density_tensors, trainable_region_indices):
    density_iter = iter(density_tensors)
    region_indices = set(trainable_region_indices)
    material_regions = []
    for index, region in enumerate(scene.material_regions):
        if index in region_indices:
            material_regions.append(_clone_material_region(region, next(density_iter)))
        else:
            material_regions.append(region)
    return scene.clone(material_regions=material_regions)


def material_pullback_autograd(
    scene,
    *,
    density_tensors,
    trainable_region_indices,
    grad_eps_ex: torch.Tensor,
    grad_eps_ey: torch.Tensor,
    grad_eps_ez: torch.Tensor,
    eps0: float,
) -> tuple[torch.Tensor, ...]:
    if not density_tensors:
        return ()
    prepared_scene = prepare_scene(scene)

    grad_eps_r = node_gradient_from_yee_permittivity(
        prepared_scene,
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        eps0=eps0,
    )

    with torch.enable_grad():
        density_inputs = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in density_tensors
        )
        cloned_scene = _clone_scene_with_densities(
            scene,
            density_inputs,
            trainable_region_indices,
        )
        model = prepare_scene(cloned_scene).compile_materials()
        scalar = torch.sum(model["eps_r"] * grad_eps_r.to(dtype=model["eps_r"].dtype))
        grads = torch.autograd.grad(
            scalar,
            density_inputs,
            allow_unused=True,
        )

    return tuple(
        torch.zeros_like(density)
        if grad is None
        else grad.to(device=density.device, dtype=density.dtype)
        for density, grad in zip(density_tensors, grads)
    )
