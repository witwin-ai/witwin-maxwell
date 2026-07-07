"""Shared helpers for resolving trainable material inputs of a scene.

Used by both the FDTD and FDFD gradient bridges to find the leaf tensors
(SceneModule parameters, structure geometry tensors, material-region
densities) that actually contribute to the compiled material tensors.
"""

from __future__ import annotations

import torch

from .scene import prepare_scene


def unique_trainable_tensors(candidates) -> tuple[torch.Tensor, ...]:
    unique = []
    seen = set()
    for tensor in candidates:
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            continue
        key = tensor.data_ptr() if tensor.data_ptr() != 0 else id(tensor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tensor)
    return tuple(unique)


def scene_trainable_material_tensors(scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for structure in scene.structures:
        geometry = getattr(structure, "geometry", None)
        if geometry is not None:
            candidates.extend(
                value for value in vars(geometry).values() if isinstance(value, torch.Tensor) and value.requires_grad
            )
    for region in getattr(scene, "material_regions", ()):
        density = getattr(region, "density", None)
        if isinstance(density, torch.Tensor) and density.requires_grad:
            candidates.append(density)
    return unique_trainable_tensors(candidates)


def material_dependent_inputs(scene, candidates) -> tuple[torch.Tensor, ...]:
    inputs = unique_trainable_tensors(candidates)
    if not inputs:
        return ()
    dependencies = ()
    prepared_scene = prepare_scene(scene)
    try:
        with torch.enable_grad():
            eps_r, _mu_r = prepared_scene.compile_material_tensors()
            if eps_r.requires_grad:
                dependencies = torch.autograd.grad(
                    eps_r.sum(),
                    inputs,
                    allow_unused=True,
                    retain_graph=False,
                )
    finally:
        prepared_scene.release_meshgrid()
    return tuple(
        tensor
        for tensor, dependency in zip(inputs, dependencies)
        if dependency is not None
    )
