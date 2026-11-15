"""Shared helpers for resolving trainable material inputs of a scene.

Used by both the FDTD and FDFD gradient bridges to find the leaf tensors
(SceneModule parameters, structure geometry tensors, material-region
densities) that actually contribute to the compiled material tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
import torch

from .scene import prepare_scene
from .compiler.materials import compile_materials
from .compiler.thin_wire import compile_thin_wires


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


def named_trainable_tensors(value, path="") -> tuple[tuple[str, torch.Tensor], ...]:
    """Every grad-carrying tensor reachable from ``value``, with its attribute path.

    The walk covers the whole declaration graph (``vars``, mappings and
    sequences), not a fixed list of channels, so a tensor no gradient channel
    knows about is still found and can be named in a refusal.
    """
    found = []
    seen = set()

    def visit(item, where):
        if torch.is_tensor(item):
            if item.requires_grad:
                found.append((where, item))
            return
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, Mapping):
            children = ((f"{where}[{key!r}]", child) for key, child in item.items())
        elif isinstance(item, (list, tuple)):
            children = ((f"{where}[{index}]", child) for index, child in enumerate(item))
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            children = ((f"{where}.{key}" if where else key, child) for key, child in vars(item).items())
        else:
            return
        for child_path, child in children:
            visit(child, child_path)

    visit(value, path)
    return tuple(found)


def scene_trainable_material_tensors(scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for structure in scene.structures:
        candidates.extend(geometry_tensors(getattr(structure, "geometry", None)))
        material = getattr(structure, "material", None)
        candidates.extend(getattr(material, name, None) for name in ("eps_r", "sigma_e", "thickness", "conductivity"))
        candidates.extend(material_pole_tensors(material))
        perturbation = getattr(material, "perturbation", None)
        if isinstance(perturbation, torch.Tensor) and perturbation.requires_grad:
            candidates.append(perturbation)
    for region in scene.material_regions:
        density = getattr(region, "density", None)
        if isinstance(density, torch.Tensor) and density.requires_grad:
            candidates.append(density)
        candidates.extend(geometry_tensors(region.geometry))
    return unique_trainable_tensors(candidates)


def geometry_tensors(geometry) -> tuple[torch.Tensor, ...]:
    """The grad-carrying tensor attributes of one geometry (boolean trees expose their children's)."""
    if geometry is None:
        return ()
    return tuple(value for value in vars(geometry).values() if torch.is_tensor(value) and value.requires_grad)


# Declaration families of a Scene that can hold physical inputs. Metadata and
# prepared caches are not inputs.
_SCENE_FAMILIES = (
    "structures", "material_regions", "background", "sources", "monitors", "ports", "networks",
    "lumped_elements", "circuits", "thin_wires", "internal_absorbers", "electrostatic_terminals",
    "charge_densities", "domain", "grid", "boundary", "subpixel", "symmetry",
)


def named_scene_tensors(scene) -> tuple[tuple[str, torch.Tensor], ...]:
    """Every grad-carrying tensor of a Scene declaration, named by its attribute path."""
    return tuple(
        item
        for family in _SCENE_FAMILIES
        for item in named_trainable_tensors(getattr(scene, family), f"scene.{family}")
    )


def tensor_depends_on(tensor, inputs) -> bool:
    """Whether the autograd graph of ``tensor`` reaches any of ``inputs`` (a structural test)."""
    if not inputs or tensor.grad_fn is None:
        return False
    probe = tensor.real.sum()
    if tensor.is_complex():
        probe = probe + tensor.imag.sum()
    # The caller keeps using this scene graph for its remaining admission census.
    reached = torch.autograd.grad(probe, inputs, allow_unused=True, retain_graph=True)
    return any(gradient is not None for gradient in reached)


def unclaimed_trainable_inputs(scene, declared, inputs) -> tuple[str, ...]:
    """Name the declared trainable scene tensors that no bridge input accounts for.

    ``declared`` are the grad-carrying scene tensors of the trainable census and
    ``inputs`` the tensors a gradient bridge differentiates. A declared tensor is
    claimed when it is an input or its graph reaches one (a SceneModule output
    reaches its parameters). Any other declared tensor changes the run without a
    gradient channel, so the bridge would leave its gradient at ``None``.
    """
    input_ids = {id(tensor) for tensor in inputs}
    unclaimed = [
        tensor for tensor in unique_trainable_tensors(declared)
        if id(tensor) not in input_ids and not tensor_depends_on(tensor, inputs)
    ]
    if not unclaimed:
        return ()
    names = {}
    for path, tensor in named_scene_tensors(scene):
        names.setdefault(id(tensor), path)
    return tuple(names.get(id(tensor), f"unnamed scene tensor of shape {tuple(tensor.shape)}") for tensor in unclaimed)


def _magnetic_signature(material):
    """The permeability-side constitutive values one material blends into the grid."""
    mu_tensor = getattr(material, "mu_tensor", None)
    return (
        float(getattr(material, "mu_r", 1.0)),
        float(getattr(material, "sigma_m", 0.0)),
        None if mu_tensor is None else repr(mu_tensor),
        tuple(repr(pole) for kind in ("debye", "drude", "lorentz")
              for pole in getattr(material, f"mu_{kind}_poles", ())),
    )


def ungradable_material_channels(scene, *, trainable_geometry_indices=()) -> tuple[str, ...]:
    """Name the trainable inputs whose FDTD effect has no material gradient channel.

    The FDTD material pullback differentiates the electric material tensors
    (permittivity, electric conductivity and dispersive pole weights). Permeability,
    magnetic conductivity and the PEC/PMC staircase masks carry no gradient. A
    trainable input that moves any of them would silently lose that part of its
    gradient, so it is named here and refused. The checks are structural: a moving
    boundary in a scene whose materials differ in permeability is refused even
    where the two never touch.
    """
    trainable_geometry_indices = frozenset(trainable_geometry_indices)
    structures = tuple(scene.structures)
    regions = tuple(scene.material_regions)
    moving = [
        index for index, structure in enumerate(structures)
        if index in trainable_geometry_indices or geometry_tensors(getattr(structure, "geometry", None))
    ]
    moving_regions = [index for index, region in enumerate(regions) if geometry_tensors(region.geometry)]
    items = [
        f"scene.structures[{index}] ({structure.name!r}): trainable permeability mu_r"
        for index, structure in enumerate(structures)
        if torch.is_tensor(mu := getattr(structure.material, "mu_r", None)) and mu.requires_grad
    ]
    for index in moving:
        material = structures[index].material
        for flag, label in (("is_pec", "PEC"), ("is_pmc", "PMC")):
            if getattr(material, flag, False):
                items.append(
                    f"scene.structures[{index}] ({structures[index].name!r}): trainable geometry of a {label} "
                    "structure, whose fixed staircase mask has no occupancy gradient"
                )
    for index, region in enumerate(regions):
        if (region.density.requires_grad or index in moving_regions) and region.mu_bounds[0] != region.mu_bounds[1]:
            items.append(
                f"scene.material_regions[{index}] ({region.name!r}): trainable density or geometry maps "
                f"mu_bounds={region.mu_bounds} onto the permeability"
            )
    if moving or moving_regions:
        background = _magnetic_signature(scene.background)
        magnetic = [
            f"scene.structures[{index}] ({structure.name!r})"
            for index, structure in enumerate(structures)
            if _magnetic_signature(structure.material) != background
        ]
        magnetic += [
            f"scene.material_regions[{index}] ({region.name!r})"
            for index, region in enumerate(regions)
            if any((float(bound), 0.0, None, ()) != background for bound in region.mu_bounds)
        ]
        if magnetic:
            movers = [f"scene.structures[{index}]" for index in moving]
            movers += [f"scene.material_regions[{index}]" for index in moving_regions]
            items.append(
                f"trainable geometry of {', '.join(movers)} moves material boundaries in a scene whose "
                f"permeability or magnetic conductivity differs from the background at {', '.join(magnetic)}"
            )
    return tuple(items)


def material_channel_message(items) -> str:
    return (
        "FDTD adjoint differentiates the electric material tensors only; permeability, magnetic conductivity "
        "and PEC/PMC occupancy have no gradient channel, and these trainable inputs move them:\n"
        + "\n".join(f"- {item}" for item in items)
        + "\nKeep these inputs fixed (requires_grad=False or .detach())."
    )


def require_material_gradient_channels(scene, *, trainable_geometry_indices=()) -> None:
    """Raise by name when a trainable input moves a material channel without a pullback."""
    from .errors import UnsupportedFeatureError

    items = ungradable_material_channels(scene, trainable_geometry_indices=trainable_geometry_indices)
    if items:
        raise UnsupportedFeatureError(
            material_channel_message(items),
            feature="gradient.material_channel",
            method="fdtd",
            alternative="keep magnetic, PEC and PMC declarations fixed",
            items=items,
        )


def require_claimed_trainable_inputs(scene, declared, inputs) -> None:
    """Raise by name when a declared trainable scene tensor reaches no bridge input."""
    from .errors import UnsupportedFeatureError

    unclaimed = unclaimed_trainable_inputs(scene, declared, inputs)
    if unclaimed:
        raise UnsupportedFeatureError(
            f"FDTD adjoint cannot differentiate {len(unclaimed)} trainable scene input(s): they change the run "
            "but reach none of its gradient channels (electric material tensors, RF and circuit values, "
            "thin-wire coefficients, port amplitudes), so their gradient would stay None:\n"
            + "\n".join(f"- {name}" for name in unclaimed)
            + "\nKeep these inputs fixed (requires_grad=False or .detach()).",
            feature="gradient.unclaimed_tensor",
            method="fdtd",
            alternative="keep the listed inputs fixed",
            items=unclaimed,
        )


def material_pole_tensors(material):
    return tuple(value for prefix in ("", "mu_") for kind in ("debye", "drude", "lorentz")
                 for pole in getattr(material, prefix + kind + "_poles", ())
                 for value in vars(pole).values() if torch.is_tensor(value))


def scene_trainable_rf_tensors(scene) -> tuple[torch.Tensor, ...]:
    candidates = []
    for port in scene.ports:
        termination = getattr(port, "termination", None)
        if termination is not None:
            candidates.extend(
                getattr(termination, name, None)
                for name in ("r", "l", "c")
            )
        candidates.append(getattr(port, "reference_impedance", None))
    for element in scene.lumped_elements:
        candidates.append(getattr(element, "value", None))
    for circuit in scene.circuits:
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
    for block in scene.networks:
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
    for wire in scene.thin_wires:
        candidates.extend((getattr(wire, "points", None), getattr(wire, "radius", None)))
    return unique_trainable_tensors(candidates)


def wire_dependent_inputs(scene, candidates, *, device=None) -> tuple[torch.Tensor, ...]:
    inputs = unique_trainable_tensors(candidates)
    if not inputs or not scene.thin_wires:
        return ()
    prepared_scene = prepare_scene(scene, device=device)
    try:
        with torch.enable_grad():
            network = compile_thin_wires(prepared_scene)
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


def material_dependent_inputs(scene, candidates, *, device=None) -> tuple[torch.Tensor, ...]:
    inputs = unique_trainable_tensors(candidates)
    if not inputs:
        return ()
    dependencies = ()
    prepared_scene = prepare_scene(scene, device=device)
    try:
        with torch.enable_grad():
            model = compile_materials(prepared_scene)
            outputs = [model["eps_components"][axis] for axis in ("x", "y", "z")]
            outputs += [model["sigma_e_components"][axis] for axis in ("x", "y", "z")]
            for prefix in ("", "mu_"):
                for kind in ("debye", "drude", "lorentz"):
                    for entry in model[prefix+kind+"_poles"]:
                        outputs.extend(entry.get("edge_weights", {}).values())
                        outputs.extend(v for v in vars(entry["pole"]).values() if torch.is_tensor(v))
            from .compiler.copper_foil import compile_copper_response
            for structure in prepared_scene.scene.structures:
                if getattr(structure.material, "is_copper_foil", False):
                    response = compile_copper_response(structure.material)
                    outputs.extend((response.rates, response.state_space.B, response.state_space.D))
            probe = sum(value.sum() for value in outputs if value.requires_grad)
            if isinstance(probe, torch.Tensor):
                dependencies = torch.autograd.grad(
                    probe,
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
