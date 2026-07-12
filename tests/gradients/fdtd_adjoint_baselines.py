from __future__ import annotations

import torch

import witwin.maxwell.fdtd.adjoint as fdtd_adjoint
from witwin.maxwell.fdtd.material_pullback import node_gradient_from_yee_permittivity
from witwin.maxwell.scene import MaterialRegion, prepare_scene

def expected_cpml_reverse_backend() -> str:
    return "native_cpml"

def expected_bloch_reverse_backend() -> str:
    return "native_bloch"

def expected_dispersive_reverse_backend() -> str:
    return "native_dispersive"

def expected_tfsf_reverse_backend() -> str:
    return "native_tfsf"

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
