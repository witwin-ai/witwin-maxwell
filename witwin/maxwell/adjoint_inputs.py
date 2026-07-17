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


def ancestry_safe_trainable_tensors(candidates) -> tuple[torch.Tensor, ...]:
    """Keep only candidate tensors that are not derived from another candidate.

    A custom autograd function must not receive both ``p`` and ``q = f(p)`` as
    independent inputs: returning the semantic pullback for both makes the
    contribution through ``q`` reach ``p`` twice.  A derived tensor remains a
    valid input when its ancestor is not itself present in the candidate set.
    """

    tensors = unique_trainable_tensors(candidates)
    roots = []
    for index, tensor in enumerate(tensors):
        other_tensors = tensors[:index] + tensors[index + 1 :]
        if not other_tensors or tensor.grad_fn is None:
            roots.append(tensor)
            continue
        probe = tensor.real.sum()
        if tensor.is_complex():
            probe = probe + tensor.imag.sum()
        dependencies = torch.autograd.grad(
            probe,
            other_tensors,
            allow_unused=True,
            retain_graph=True,
        )
        if not any(dependency is not None for dependency in dependencies):
            roots.append(tensor)
    return tuple(roots)


def scene_trainable_material_tensors(scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for structure in scene.structures:
        geometry = getattr(structure, "geometry", None)
        if geometry is not None:
            candidates.extend(
                value for value in vars(geometry).values() if isinstance(value, torch.Tensor) and value.requires_grad
            )
        material = getattr(structure, "material", None)
        perturbation = getattr(material, "perturbation", None)
        if isinstance(perturbation, torch.Tensor) and perturbation.requires_grad:
            candidates.append(perturbation)
    for region in getattr(scene, "material_regions", ()):
        density = getattr(region, "density", None)
        if isinstance(density, torch.Tensor) and density.requires_grad:
            candidates.append(density)
    return unique_trainable_tensors(candidates)


def scene_trainable_rf_tensors(scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for port in getattr(scene, "ports", ()):
        termination = getattr(port, "termination", None)
        if termination is not None:
            candidates.extend(
                getattr(termination, name, None)
                for name in ("r", "l", "c")
            )
        candidates.append(getattr(port, "reference_impedance", None))
    for element in getattr(scene, "lumped_elements", ()):
        candidates.append(getattr(element, "value", None))
    for circuit in getattr(scene, "circuits", ()):
        candidates.extend(getattr(circuit, "parameters", {}).values())
        candidates.extend(
            value
            for value, _constraint in getattr(circuit, "initial_conditions", {}).values()
        )
        for device in getattr(circuit, "devices", ()):
            for value in getattr(device, "parameters", {}).values():
                if isinstance(value, torch.Tensor):
                    candidates.append(value)
                elif value is not None and hasattr(value, "__dict__"):
                    candidates.extend(vars(value).values())
    for block in getattr(scene, "networks", ()):
        model = getattr(block, "model", None)
        for name in (
            "poles",
            "residues",
            "direct",
            "proportional",
            "A",
            "B",
            "C",
            "D",
        ):
            candidates.append(getattr(model, name, None))
    return unique_trainable_tensors(candidates)


def scene_trainable_wire_tensors(scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for wire in getattr(scene, "thin_wires", ()):
        candidates.extend((getattr(wire, "points", None), getattr(wire, "radius", None)))
    return unique_trainable_tensors(candidates)


def wire_dependent_inputs(scene, candidates) -> tuple[torch.Tensor, ...]:
    inputs = unique_trainable_tensors(candidates)
    if not inputs or not getattr(scene, "thin_wires", ()):
        return ()
    prepared_scene = prepare_scene(scene)
    try:
        with torch.enable_grad():
            network = prepared_scene.compile_thin_wires()
            outputs = tuple(
                value
                for value in (network.inductance, network.node_capacitance)
                if value.requires_grad
            )
            dependencies = (
                torch.autograd.grad(
                    outputs,
                    inputs,
                    grad_outputs=tuple(torch.ones_like(value) for value in outputs),
                    allow_unused=True,
                )
                if outputs
                else tuple(None for _input in inputs)
            )
    finally:
        prepared_scene.release_meshgrid()
    return tuple(
        tensor
        for tensor, dependency in zip(inputs, dependencies)
        if dependency is not None
    )


def rf_dependent_inputs(scene, candidates) -> tuple[torch.Tensor, ...]:
    inputs = ancestry_safe_trainable_tensors(candidates)
    if not inputs:
        return ()
    outputs = scene_trainable_rf_tensors(scene)
    if not outputs:
        return ()
    probe = sum(
        value.real.sum() + (value.imag.sum() if value.is_complex() else 0.0)
        for value in outputs
    )
    dependencies = torch.autograd.grad(
        probe,
        inputs,
        allow_unused=True,
        # The bridge subsequently enumerates semantic outputs on this same
        # direct Scene graph.  Keep ancestry available for that census.
        retain_graph=True,
    )
    return tuple(
        tensor
        for tensor, dependency in zip(inputs, dependencies)
        if dependency is not None
    )


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
                    # Direct scenes reuse their geometry graph during semantic
                    # capability validation and the final material pullback.
                    retain_graph=True,
                )
    finally:
        prepared_scene.release_meshgrid()
    return tuple(
        tensor
        for tensor, dependency in zip(inputs, dependencies)
        if dependency is not None
    )
