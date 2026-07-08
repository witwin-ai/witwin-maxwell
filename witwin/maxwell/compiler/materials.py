from __future__ import annotations

from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from witwin.core import Box
from witwin.core.material import VACUUM_PERMITTIVITY

from ..media import DiagonalTensor3, Tensor3x3

_AXES = ("x", "y", "z")


def _scalar_tensor(value, *, device, dtype=torch.float32):
    return torch.as_tensor(value, device=device, dtype=dtype)


def _complex_scalar_tensor(value, *, device):
    return torch.as_tensor(value, device=device, dtype=torch.complex64)


def _sorted_structures(scene):
    indexed = [
        (index, structure)
        for index, structure in enumerate(scene.structures)
        if getattr(structure, "enabled", True)
    ]
    indexed.sort(key=lambda item: (int(getattr(item[1], "priority", 0)), item[0]))
    return [structure for _, structure in indexed]


def _structure_material(structure):
    return getattr(structure, "material", None)


def _scene_has_anisotropic_material(scene) -> bool:
    for structure in _sorted_structures(scene):
        material = _structure_material(structure)
        if material is not None and bool(getattr(material, "is_anisotropic", False)):
            return True
    return False


def _scene_has_dispersive_material(scene) -> bool:
    for structure in _sorted_structures(scene):
        material = _structure_material(structure)
        if material is not None and bool(getattr(material, "is_dispersive", False)):
            return True
    return False


def _scene_has_kerr_material(scene) -> bool:
    for structure in _sorted_structures(scene):
        material = _structure_material(structure)
        if material is not None and bool(getattr(material, "is_nonlinear", False)):
            return True
    return False


def _validate_scene_material_combinations(scene):
    if not _scene_has_kerr_material(scene):
        return
    if _scene_has_dispersive_material(scene):
        raise NotImplementedError("Kerr media cannot be combined with dispersive materials elsewhere in the same Scene in v1.")
    if _scene_has_anisotropic_material(scene):
        raise NotImplementedError("Kerr media cannot be combined with anisotropic materials elsewhere in the same Scene in v1.")


def _component_average(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return (components["x"] + components["y"] + components["z"]) / 3.0


def _refresh_model_summary_aliases(model):
    model["eps_r"] = _component_average(model["eps_components"])
    model["mu_r"] = _component_average(model["mu_components"])
    model["sigma_e"] = _component_average(model["sigma_e_components"])
    return model


def _new_component_field(shape, *, fill_value, device, dtype=torch.float32):
    return {
        axis: torch.full(shape, float(fill_value), device=device, dtype=dtype)
        for axis in _AXES
    }


def _component_values_from_sample(value, *, name: str):
    if isinstance(value, DiagonalTensor3):
        return {"x": float(value.xx), "y": float(value.yy), "z": float(value.zz)}
    if isinstance(value, Tensor3x3):
        raise NotImplementedError(
            f"The Maxwell material compiler currently supports DiagonalTensor3 only for {name}; Tensor3x3 is not implemented yet."
        )
    scalar = float(value)
    return {axis: scalar for axis in _AXES}


def _static_structure_material(structure):
    material = structure.material
    sample = material.evaluate_static()
    try:
        eps_components = _component_values_from_sample(sample.eps_r, name="eps_r")
        mu_components = _component_values_from_sample(sample.mu_r, name="mu_r")
        sigma_components = _component_values_from_sample(getattr(sample, "sigma_e", 0.0), name="sigma_e")
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "The Maxwell material compiler currently supports scalar isotropic or axis-aligned DiagonalTensor3 material samples only."
        ) from exc
    return (
        material,
        eps_components,
        mu_components,
        sigma_components,
        float(getattr(material, "kerr_chi3", 0.0) or 0.0),
    )


def _new_material_model(scene, layout, *, eps_fill, mu_fill):
    shape = (scene.Nx, scene.Ny, scene.Nz)
    device = scene.device
    model = {
        "eps_components": _new_component_field(shape, fill_value=eps_fill, device=device),
        "mu_components": _new_component_field(shape, fill_value=mu_fill, device=device),
        "sigma_e_components": _new_component_field(shape, fill_value=0.0, device=device),
        "debye_poles": [
            {"pole": entry["pole"], "weight": torch.zeros(shape, device=device, dtype=torch.float32)}
            for entry in layout["debye_poles"]
        ],
        "drude_poles": [
            {"pole": entry["pole"], "weight": torch.zeros(shape, device=device, dtype=torch.float32)}
            for entry in layout["drude_poles"]
        ],
        "lorentz_poles": [
            {"pole": entry["pole"], "weight": torch.zeros(shape, device=device, dtype=torch.float32)}
            for entry in layout["lorentz_poles"]
        ],
        "mu_debye_poles": [
            {"pole": entry["pole"], "weight": torch.zeros(shape, device=device, dtype=torch.float32)}
            for entry in layout["mu_debye_poles"]
        ],
        "mu_drude_poles": [
            {"pole": entry["pole"], "weight": torch.zeros(shape, device=device, dtype=torch.float32)}
            for entry in layout["mu_drude_poles"]
        ],
        "mu_lorentz_poles": [
            {"pole": entry["pole"], "weight": torch.zeros(shape, device=device, dtype=torch.float32)}
            for entry in layout["mu_lorentz_poles"]
        ],
        "kerr_chi3": torch.zeros(shape, device=device, dtype=torch.float32),
    }
    return _refresh_model_summary_aliases(model)


def _material_model_has_dispersion(model) -> bool:
    return bool(
        model["debye_poles"]
        or model["drude_poles"]
        or model["lorentz_poles"]
        or model["mu_debye_poles"]
        or model["mu_drude_poles"]
        or model["mu_lorentz_poles"]
    )


def _material_model_has_electric_dispersion(model) -> bool:
    return bool(model["debye_poles"] or model["drude_poles"] or model["lorentz_poles"])


def _material_model_has_magnetic_dispersion(model) -> bool:
    return bool(model["mu_debye_poles"] or model["mu_drude_poles"] or model["mu_lorentz_poles"])


def _material_model_has_conductivity(model) -> bool:
    return any(torch.any(model["sigma_e_components"][axis] != 0).item() for axis in _AXES)


def _material_model_has_kerr(model) -> bool:
    return bool(torch.any(model["kerr_chi3"] != 0).item())


def _blend_material(tensor, occupancy, *, value):
    value_tensor = _scalar_tensor(value, device=tensor.device, dtype=tensor.dtype)
    return (1.0 - occupancy) * tensor + occupancy * value_tensor


def _iter_weight_entries(model):
    yield from model["debye_poles"]
    yield from model["drude_poles"]
    yield from model["lorentz_poles"]
    yield from model["mu_debye_poles"]
    yield from model["mu_drude_poles"]
    yield from model["mu_lorentz_poles"]


def _clear_dispersive_region(model, region):
    keep = 1.0 - region
    for entry in _iter_weight_entries(model):
        entry["weight"] = entry["weight"] * keep


def _assign_structure_weights(model, structure_slots, region):
    for slot_index in structure_slots["debye"]:
        model["debye_poles"][slot_index]["weight"] = model["debye_poles"][slot_index]["weight"] + region
    for slot_index in structure_slots["drude"]:
        model["drude_poles"][slot_index]["weight"] = model["drude_poles"][slot_index]["weight"] + region
    for slot_index in structure_slots["lorentz"]:
        model["lorentz_poles"][slot_index]["weight"] = model["lorentz_poles"][slot_index]["weight"] + region
    for slot_index in structure_slots["mu_debye"]:
        model["mu_debye_poles"][slot_index]["weight"] = model["mu_debye_poles"][slot_index]["weight"] + region
    for slot_index in structure_slots["mu_drude"]:
        model["mu_drude_poles"][slot_index]["weight"] = model["mu_drude_poles"][slot_index]["weight"] + region
    for slot_index in structure_slots["mu_lorentz"]:
        model["mu_lorentz_poles"][slot_index]["weight"] = model["mu_lorentz_poles"][slot_index]["weight"] + region


def _coordinate_grids(scene, sample_offset):
    if sample_offset == (0.0, 0.0, 0.0):
        return scene.X, scene.Y, scene.Z
    x = scene.x + float(sample_offset[0])
    y = scene.y + float(sample_offset[1])
    z = scene.z + float(sample_offset[2])
    return torch.meshgrid(x, y, z, indexing="ij")


def _geometry_beta(scene) -> float:
    return 0.05 * min(scene.grid.min_spacing)


def _geometry_occupancy(scene, geometry, coords=None):
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    return geometry.to_mask(xx, yy, zz, offset=0.0, beta=_geometry_beta(scene))


def _build_dispersive_layout(scene):
    layout = {
        "debye_poles": [],
        "drude_poles": [],
        "lorentz_poles": [],
        "mu_debye_poles": [],
        "mu_drude_poles": [],
        "mu_lorentz_poles": [],
        "structure_slots": [],
    }
    for structure in _sorted_structures(scene):
        material, _, _, _, _ = _static_structure_material(structure)
        slots = {
            "debye": [],
            "drude": [],
            "lorentz": [],
            "mu_debye": [],
            "mu_drude": [],
            "mu_lorentz": [],
        }
        for pole in getattr(material, "debye_poles", ()):
            slots["debye"].append(len(layout["debye_poles"]))
            layout["debye_poles"].append({"pole": pole})
        for pole in getattr(material, "drude_poles", ()):
            slots["drude"].append(len(layout["drude_poles"]))
            layout["drude_poles"].append({"pole": pole})
        for pole in getattr(material, "lorentz_poles", ()):
            slots["lorentz"].append(len(layout["lorentz_poles"]))
            layout["lorentz_poles"].append({"pole": pole})
        for pole in getattr(material, "mu_debye_poles", ()):
            slots["mu_debye"].append(len(layout["mu_debye_poles"]))
            layout["mu_debye_poles"].append({"pole": pole})
        for pole in getattr(material, "mu_drude_poles", ()):
            slots["mu_drude"].append(len(layout["mu_drude_poles"]))
            layout["mu_drude_poles"].append({"pole": pole})
        for pole in getattr(material, "mu_lorentz_poles", ()):
            slots["mu_lorentz"].append(len(layout["mu_lorentz_poles"]))
            layout["mu_lorentz_poles"].append({"pole": pole})
        layout["structure_slots"].append(slots)
    return layout


def _apply_structure_material(
    scene,
    model,
    structure,
    structure_slots,
    *,
    coords=None,
    eps_background=1.0,
    mu_background=1.0,
):
    _, eps_components, mu_components, sigma_components, kerr_chi3 = _static_structure_material(structure)
    occupancy = _geometry_occupancy(scene, structure.geometry, coords=coords)

    for axis in _AXES:
        model["eps_components"][axis] = _blend_material(
            model["eps_components"][axis],
            occupancy,
            value=eps_components[axis] * float(eps_background),
        )
        model["mu_components"][axis] = _blend_material(
            model["mu_components"][axis],
            occupancy,
            value=mu_components[axis] * float(mu_background),
        )
        model["sigma_e_components"][axis] = _blend_material(
            model["sigma_e_components"][axis],
            occupancy,
            value=sigma_components[axis],
        )
    model["kerr_chi3"] = _blend_material(model["kerr_chi3"], occupancy, value=kerr_chi3)
    _clear_dispersive_region(model, occupancy)
    _assign_structure_weights(model, structure_slots, occupancy)
    return _refresh_model_summary_aliases(model)


def _compile_material_sample(
    scene,
    layout,
    *,
    eps_background,
    mu_background,
    sample_offset=(0.0, 0.0, 0.0),
):
    model = _new_material_model(scene, layout, eps_fill=eps_background, mu_fill=mu_background)
    coords = _coordinate_grids(scene, sample_offset)

    for structure, structure_slots in zip(_sorted_structures(scene), layout["structure_slots"]):
        model = _apply_structure_material(
            scene,
            model,
            structure,
            structure_slots,
            coords=coords,
            eps_background=eps_background,
            mu_background=mu_background,
        )
    return model


def _sample_offsets(scene, subpixel_samples):
    def axis_offsets(step, count):
        if count == 1:
            return [0.0]
        return [((index + 0.5) / count - 0.5) * step for index in range(count)]

    x_offsets = axis_offsets(scene.dx, int(subpixel_samples[0]))
    y_offsets = axis_offsets(scene.dy, int(subpixel_samples[1]))
    z_offsets = axis_offsets(scene.dz, int(subpixel_samples[2]))
    return list(product(x_offsets, y_offsets, z_offsets))


def _region_axis_slice(nodes64: np.ndarray, lower: float, upper: float) -> slice | None:
    # Cells are indexed by their low-side node. The window is lower-inclusive
    # and upper-exclusive so a box whose faces land exactly on grid nodes
    # covers exactly size/spacing cells. The tolerance absorbs float32
    # rounding of the geometry bounds (positions are stored as float32).
    span = float(nodes64[-1] - nodes64[0])
    tol = 1e-6 * span
    start = int(np.searchsorted(nodes64, lower - tol, side="left"))
    stop = int(np.searchsorted(nodes64, upper - tol, side="left"))
    if stop <= start:
        return None
    return slice(start, stop)


def _density_kernel_size(scene, filter_radius: float) -> tuple[int, int, int]:
    kernel = []
    for spacing in scene.grid.min_spacing:
        cells = int(np.ceil(float(filter_radius) / float(spacing)))
        kernel.append(max(1, 2 * cells + 1))
    return tuple(kernel)


def _filter_region_density(scene, region, density: torch.Tensor) -> torch.Tensor:
    if region.filter_radius is None:
        return density
    kernel = _density_kernel_size(scene, region.filter_radius)
    kernel = tuple(
        min(size, shape if shape % 2 == 1 else max(1, shape - 1))
        for size, shape in zip(kernel, density.shape)
    )
    if kernel == (1, 1, 1):
        return density
    filtered = F.avg_pool3d(
        density[None, None, ...],
        kernel_size=kernel,
        stride=1,
        padding=tuple(size // 2 for size in kernel),
    )
    return filtered[0, 0]


def _project_region_density(region, density: torch.Tensor) -> torch.Tensor:
    if region.projection_beta is None:
        return density
    beta = float(region.projection_beta)
    midpoint = density.new_tensor(0.5)
    numerator = torch.tanh(beta * midpoint) + torch.tanh(beta * (density - midpoint))
    denominator = torch.tanh(beta * midpoint) + torch.tanh(beta * (1.0 - midpoint))
    return numerator / denominator


def _region_density_field(scene, region, reference: torch.Tensor):
    geometry = region.geometry
    if not isinstance(geometry, Box):
        raise ValueError(
            f"MaterialRegion currently supports Box geometry only, got {type(geometry).__name__}."
        )

    pos = geometry.position
    sz_vec = geometry.size
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    sx, sy, sz = float(sz_vec[0]), float(sz_vec[1]), float(sz_vec[2])
    x_slice = _region_axis_slice(scene.x_nodes64, cx - sx / 2.0, cx + sx / 2.0)
    y_slice = _region_axis_slice(scene.y_nodes64, cy - sy / 2.0, cy + sy / 2.0)
    z_slice = _region_axis_slice(scene.z_nodes64, cz - sz / 2.0, cz + sz / 2.0)
    if x_slice is None or y_slice is None or z_slice is None:
        empty = torch.zeros_like(reference)
        return empty, empty.to(dtype=torch.bool)

    density = region.density.to(device=scene.device, dtype=reference.real.dtype)
    lower, upper = region.bounds
    if not np.isclose(lower, 0.0) or not np.isclose(upper, 1.0):
        density = (density - lower) / max(upper - lower, 1e-12)
    density = density.clamp(0.0, 1.0)
    density = _filter_region_density(scene, region, density)
    density = _project_region_density(region, density).clamp(0.0, 1.0)

    target_shape = (
        x_slice.stop - x_slice.start,
        y_slice.stop - y_slice.start,
        z_slice.stop - z_slice.start,
    )
    if density.shape != target_shape:
        density = F.interpolate(
            density[None, None, ...],
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    field = torch.zeros_like(reference)
    mask = torch.zeros(reference.shape, dtype=torch.bool, device=reference.device)
    field[x_slice, y_slice, z_slice] = density.to(dtype=reference.dtype)
    mask[x_slice, y_slice, z_slice] = True
    return field, mask


def _apply_material_regions(scene, model):
    eps_base = model["eps_r"].clone()
    mu_base = model["mu_r"].clone()
    eps_design = torch.zeros_like(eps_base)
    mu_design = torch.zeros_like(mu_base)
    design_mask = torch.zeros(eps_base.shape, dtype=torch.bool, device=eps_base.device)

    for region in getattr(scene, "material_regions", ()):
        density_field, region_mask = _region_density_field(scene, region, eps_base)
        if not torch.any(region_mask):
            continue
        eps_lo, eps_hi = region.eps_bounds
        mu_lo, mu_hi = region.mu_bounds
        eps_region = eps_lo + density_field * (eps_hi - eps_lo)
        mu_region = mu_lo + density_field * (mu_hi - mu_lo)
        eps_design = torch.where(region_mask, eps_region - eps_base, eps_design)
        mu_design = torch.where(region_mask, mu_region - mu_base, mu_design)
        design_mask = torch.where(region_mask, torch.ones_like(design_mask), design_mask)

    model["eps_r_base"] = eps_base
    model["mu_r_base"] = mu_base
    model["eps_r_design"] = eps_design
    model["mu_r_design"] = mu_design
    model["design_mask"] = design_mask
    for axis in _AXES:
        model["eps_components"][axis] = model["eps_components"][axis] + eps_design
        model["mu_components"][axis] = model["mu_components"][axis] + mu_design
    return _refresh_model_summary_aliases(model)


def compile_material_model(
    scene,
    eps_background=1.0,
    mu_background=1.0,
    subpixel_samples=(1, 1, 1),
):
    _validate_scene_material_combinations(scene)
    samples = tuple(int(v) for v in subpixel_samples)
    layout = _build_dispersive_layout(scene)
    if samples == (1, 1, 1):
        model = _compile_material_sample(
            scene,
            layout,
            eps_background=eps_background,
            mu_background=mu_background,
        )
        return _apply_material_regions(scene, model)

    accum = _new_material_model(scene, layout, eps_fill=0.0, mu_fill=0.0)
    sample_offsets = _sample_offsets(scene, samples)
    for sample_offset in sample_offsets:
        sample = _compile_material_sample(
            scene,
            layout,
            eps_background=eps_background,
            mu_background=mu_background,
            sample_offset=sample_offset,
        )
        for axis in _AXES:
            accum["eps_components"][axis] += sample["eps_components"][axis]
            accum["mu_components"][axis] += sample["mu_components"][axis]
            accum["sigma_e_components"][axis] += sample["sigma_e_components"][axis]
        accum["kerr_chi3"] += sample["kerr_chi3"]
        for slot_index, entry in enumerate(sample["debye_poles"]):
            accum["debye_poles"][slot_index]["weight"] += entry["weight"]
        for slot_index, entry in enumerate(sample["drude_poles"]):
            accum["drude_poles"][slot_index]["weight"] += entry["weight"]
        for slot_index, entry in enumerate(sample["lorentz_poles"]):
            accum["lorentz_poles"][slot_index]["weight"] += entry["weight"]
        for slot_index, entry in enumerate(sample["mu_debye_poles"]):
            accum["mu_debye_poles"][slot_index]["weight"] += entry["weight"]
        for slot_index, entry in enumerate(sample["mu_drude_poles"]):
            accum["mu_drude_poles"][slot_index]["weight"] += entry["weight"]
        for slot_index, entry in enumerate(sample["mu_lorentz_poles"]):
            accum["mu_lorentz_poles"][slot_index]["weight"] += entry["weight"]

    scale = 1.0 / float(len(sample_offsets))
    for axis in _AXES:
        accum["eps_components"][axis] *= scale
        accum["mu_components"][axis] *= scale
        accum["sigma_e_components"][axis] *= scale
    accum["kerr_chi3"] *= scale
    for entry in accum["debye_poles"]:
        entry["weight"] *= scale
    for entry in accum["drude_poles"]:
        entry["weight"] *= scale
    for entry in accum["lorentz_poles"]:
        entry["weight"] *= scale
    for entry in accum["mu_debye_poles"]:
        entry["weight"] *= scale
    for entry in accum["mu_drude_poles"]:
        entry["weight"] *= scale
    for entry in accum["mu_lorentz_poles"]:
        entry["weight"] *= scale
    return _apply_material_regions(scene, _refresh_model_summary_aliases(accum))


def _evaluate_electric_components(model, frequency: float | None):
    if frequency is None:
        return model["eps_components"]

    resolved_frequency = float(frequency)
    if resolved_frequency < 0.0:
        raise ValueError("frequency must be >= 0.")
    if resolved_frequency == 0.0:
        if model["drude_poles"]:
            raise ValueError("Drude media require frequency > 0.")
        if _material_model_has_conductivity(model):
            raise ValueError("Conductive materials require frequency > 0.")

    if not _material_model_has_electric_dispersion(model) and not _material_model_has_conductivity(model):
        return model["eps_components"]

    angular_frequency = 2.0 * np.pi * resolved_frequency
    epsilon = {
        axis: model["eps_components"][axis].to(dtype=torch.complex64)
        for axis in _AXES
    }
    if _material_model_has_conductivity(model):
        for axis in _AXES:
            epsilon[axis] = epsilon[axis] - 1j * model["sigma_e_components"][axis].to(dtype=torch.complex64) / (
                angular_frequency * VACUUM_PERMITTIVITY
            )

    for entries in (model["debye_poles"], model["drude_poles"], model["lorentz_poles"]):
        for entry in entries:
            susceptibility = _complex_scalar_tensor(
                entry["pole"].susceptibility(angular_frequency),
                device=entry["weight"].device,
            )
            for axis in _AXES:
                epsilon[axis] = epsilon[axis] + entry["weight"].to(dtype=torch.complex64) * susceptibility

    return epsilon


def _evaluate_magnetic_components(model, frequency: float | None):
    if frequency is None:
        return model["mu_components"]

    resolved_frequency = float(frequency)
    if resolved_frequency < 0.0:
        raise ValueError("frequency must be >= 0.")
    if resolved_frequency == 0.0 and model["mu_drude_poles"]:
        raise ValueError("Magnetic Drude media require frequency > 0.")

    if not _material_model_has_magnetic_dispersion(model):
        return model["mu_components"]

    angular_frequency = 2.0 * np.pi * resolved_frequency
    permeability = {
        axis: model["mu_components"][axis].to(dtype=torch.complex64)
        for axis in _AXES
    }

    for entries in (model["mu_debye_poles"], model["mu_drude_poles"], model["mu_lorentz_poles"]):
        for entry in entries:
            susceptibility = _complex_scalar_tensor(
                entry["pole"].susceptibility(angular_frequency),
                device=entry["weight"].device,
            )
            for axis in _AXES:
                permeability[axis] = permeability[axis] + entry["weight"].to(dtype=torch.complex64) * susceptibility

    return permeability


def evaluate_material_components(model, frequency: float | None):
    return _evaluate_electric_components(model, frequency), _evaluate_magnetic_components(model, frequency)


def evaluate_material_permittivity(model, frequency: float | None):
    return _component_average(_evaluate_electric_components(model, frequency))


def evaluate_material_permeability(model, frequency: float | None):
    return _component_average(_evaluate_magnetic_components(model, frequency))


def compile_material_tensors(
    scene,
    eps_background=1.0,
    mu_background=1.0,
    subpixel_samples=(1, 1, 1),
    frequency: float | None = None,
):
    model = compile_material_model(
        scene,
        eps_background=eps_background,
        mu_background=mu_background,
        subpixel_samples=subpixel_samples,
    )
    eps_components, mu_components = evaluate_material_components(model, frequency)
    return _component_average(eps_components), _component_average(mu_components)
