from __future__ import annotations

from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from witwin.core import Box
from witwin.core.material import VACUUM_PERMITTIVITY

from ..media import CustomPole, DiagonalTensor3, DrudePole, LorentzPole, ModulationSpec, PerturbationMedium, Tensor3x3

_AXES = ("x", "y", "z")
_OFFDIAG_AXES = ("xy", "xz", "yz")


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


def _structure_is_pec(structure) -> bool:
    material = _structure_material(structure)
    return material is not None and bool(getattr(material, "is_pec", False))


def _structure_is_sheet(structure) -> bool:
    material = _structure_material(structure)
    return material is not None and bool(getattr(material, "is_medium2d", False))


def _bulk_structures(scene):
    """Enabled volumetric structures: everything except PEC markers and 2D sheets."""
    return [
        structure
        for structure in _sorted_structures(scene)
        if not _structure_is_pec(structure) and not _structure_is_sheet(structure)
    ]


def _pec_structures(scene):
    return [structure for structure in _sorted_structures(scene) if _structure_is_pec(structure)]


def _sheet_structures(scene):
    return [structure for structure in _sorted_structures(scene) if _structure_is_sheet(structure)]


def _scene_has_dispersive_material(scene) -> bool:
    """Whether any compiled material component depends on frequency.

    2D sheets count as dispersive here: their conductivity enters the compiled
    complex permittivity through the frequency-dependent ``sigma/(omega*eps0)``
    term (and ``Graphene`` adds a genuinely dispersive sheet conductivity), so
    frequency-cached component consumers must recompile on frequency changes.
    """
    for structure in _sorted_structures(scene):
        material = _structure_material(structure)
        if material is None:
            continue
        if bool(getattr(material, "is_dispersive", False)):
            return True
        if bool(getattr(material, "is_medium2d", False)):
            return True
    return False


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
            f"The Maxwell material compiler stores {name} on the diagonal Yee components; "
            f"off-diagonal Tensor3x3 coupling is realized only for permittivity via the "
            f"coupled-tensor eps kernel. A full off-diagonal {name} tensor has no "
            f"corresponding coupled update kernel on the Yee grid."
        )
    scalar = float(value)
    return {axis: scalar for axis in _AXES}


def _eps_values_from_sample(value):
    """Split a static permittivity sample into diagonal and off-diagonal parts.

    Full ``Tensor3x3`` permittivity is validated symmetric positive-definite at
    ``Material`` construction, so only the upper triangle is stored here.
    """
    if isinstance(value, Tensor3x3):
        rows = value.rows
        diagonal = {"x": float(rows[0][0]), "y": float(rows[1][1]), "z": float(rows[2][2])}
        offdiag = {"xy": float(rows[0][1]), "xz": float(rows[0][2]), "yz": float(rows[1][2])}
        return diagonal, offdiag
    return _component_values_from_sample(value, name="eps_r"), {pair: 0.0 for pair in _OFFDIAG_AXES}


def _structure_nonlinearity(material):
    """Scalar nonlinear channel values of a structure material.

    ``kerr_chi3`` is the instantaneous third-order (Kerr) susceptibility and
    ``chi2`` the second-order susceptibility; both rasterize into per-node
    fields that blend with structure occupancy exactly like the static eps.
    """
    chi3 = getattr(material, "nonlinear_chi3", None)
    if chi3 is None:
        chi3 = getattr(material, "kerr_chi3", 0.0) or 0.0
    return {
        "kerr_chi3": float(chi3),
        "chi2": float(getattr(material, "nonlinear_chi2", 0.0)),
        "tpa_sigma": float(getattr(material, "tpa_sigma_scale", 0.0)),
    }


def _static_structure_material(structure):
    material = structure.material
    sample = material.evaluate_static()
    try:
        eps_components, eps_offdiag = _eps_values_from_sample(sample.eps_r)
        mu_components = _component_values_from_sample(sample.mu_r, name="mu_r")
        sigma_components = _component_values_from_sample(getattr(sample, "sigma_e", 0.0), name="sigma_e")
        # Static magnetic conductivity is a scalar Material attribute (the magnetic
        # dual of sigma_e); it is read directly from the material rather than the
        # static sample, which carries no magnetic-conductivity channel.
        sigma_m_components = _component_values_from_sample(
            getattr(material, "sigma_m", 0.0), name="sigma_m"
        )
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "The Maxwell material compiler currently supports scalar isotropic or axis-aligned DiagonalTensor3 material samples only."
        ) from exc
    return (
        material,
        eps_components,
        eps_offdiag,
        mu_components,
        sigma_components,
        sigma_m_components,
        _structure_nonlinearity(material),
    )


def _new_material_model(scene, layout, *, eps_fill, mu_fill):
    shape = (scene.Nx, scene.Ny, scene.Nz)
    device = scene.device
    model = {
        "eps_components": _new_component_field(shape, fill_value=eps_fill, device=device),
        "eps_offdiag_components": {
            pair: torch.zeros(shape, device=device, dtype=torch.float32)
            for pair in _OFFDIAG_AXES
        },
        "mu_components": _new_component_field(shape, fill_value=mu_fill, device=device),
        "sigma_e_components": _new_component_field(shape, fill_value=0.0, device=device),
        "sigma_m_components": _new_component_field(shape, fill_value=0.0, device=device),
        "kerr_chi3": torch.zeros(shape, device=device, dtype=torch.float32),
        "chi2": torch.zeros(shape, device=device, dtype=torch.float32),
        "tpa_sigma": torch.zeros(shape, device=device, dtype=torch.float32),
        # Space-time modulation quadrature fields: amplitude * cos(phase) and
        # amplitude * sin(phase). This basis blends linearly across overlapping
        # structures, unlike the raw phase itself.
        "modulation_cos": torch.zeros(shape, device=device, dtype=torch.float32),
        "modulation_sin": torch.zeros(shape, device=device, dtype=torch.float32),
        # Per-node modulation angular frequency (0 where unmodulated). A Scene may
        # hold several distinct modulation frequencies at once; each modulated
        # structure stamps its own frequency onto the cells it covers, so the
        # single-frequency-per-scene restriction no longer applies.
        "modulation_omega": torch.zeros(shape, device=device, dtype=torch.float32),
    }
    for key in (
        "debye_poles",
        "drude_poles",
        "lorentz_poles",
        "mu_debye_poles",
        "mu_drude_poles",
        "mu_lorentz_poles",
    ):
        model[key] = [
            {
                "pole": entry["pole"],
                "weight": torch.zeros(shape, device=device, dtype=torch.float32),
                "amplitude": entry.get("amplitude"),
            }
            for entry in layout[key]
        ]
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


def _material_model_has_magnetic_conductivity(model) -> bool:
    return any(torch.any(model["sigma_m_components"][axis] != 0).item() for axis in _AXES)


def _material_model_has_kerr(model) -> bool:
    return bool(torch.any(model["kerr_chi3"] != 0).item())


def material_model_has_nonlinearity(model) -> bool:
    """Whether the compiled model carries any instantaneous nonlinear channel."""
    return (
        _material_model_has_kerr(model)
        or bool(torch.any(model["chi2"] != 0).item())
        or bool(torch.any(model["tpa_sigma"] != 0).item())
    )


def material_model_has_modulation(model) -> bool:
    """Whether the compiled model carries a space-time permittivity modulation."""
    return bool(
        torch.any(model["modulation_cos"] != 0).item()
        or torch.any(model["modulation_sin"] != 0).item()
    )


def material_model_has_full_anisotropy(model) -> bool:
    """Whether the compiled model carries any off-diagonal permittivity."""
    offdiag = model.get("eps_offdiag_components")
    if not offdiag:
        return False
    return any(torch.any(offdiag[pair] != 0).item() for pair in _OFFDIAG_AXES)


def _blend_material(tensor, occupancy, *, value):
    value_tensor = _scalar_tensor(value, device=tensor.device, dtype=tensor.dtype)
    return (1.0 - occupancy) * tensor + occupancy * value_tensor


_NORMAL_EPS = 1e-24
_HARMONIC_EPS = 1e-12


def _interface_normals(scene, geometry, coords=None):
    """Unit outward interface normals per node from the signed-distance field.

    The normal is the gradient of the per-structure signed-distance field on the
    node grid, evaluated with true (possibly graded) spacings via central finite
    differences. SDFs are eikonal so ``|grad| ~ 1`` on the interface band; deep
    interior / medial-axis nodes have ``|grad| ~ 0`` and yield ``n ~ 0`` after the
    floored normalization, which is harmless because the polarized correction there
    is weighted by the vanishing ``(eps_arith - eps_harm)`` term. Differentiable in
    geometry parameters through ``signed_distance``.
    """
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    sdf = geometry.signed_distance(xx, yy, zz)
    x = xx[:, 0, 0]
    y = yy[0, :, 0]
    z = zz[0, 0, :]
    grads = []
    for dim, coord in enumerate((x, y, z)):
        if sdf.shape[dim] < 2:
            grads.append(torch.zeros_like(sdf))
        else:
            grads.append(torch.gradient(sdf, spacing=(coord,), dim=dim)[0])
    gx, gy, gz = grads
    # Floor is applied inside the sqrt (squared scale ``_NORMAL_EPS``) so the
    # sqrt's own backward stays finite at degenerate nodes where the summed
    # squared gradient is exactly zero (medial-axis / symmetric-center nodes).
    # A floor added after the sqrt would leave d(sqrt)/d(.) = inf there, and the
    # vanishing upstream gradient would evaluate 0*inf = NaN, poisoning the whole
    # geometry-gradient tensor through the sum reduction.
    mag = torch.sqrt(gx * gx + gy * gy + gz * gz + _NORMAL_EPS)
    inv = 1.0 / mag
    return {"x": gx * inv, "y": gy * inv, "z": gz * inv}


def _blend_material_polarized(tensor, occupancy, normal_axis, *, value):
    """Normal-projection (Kottke) per-axis blend of a background field with ``value``.

    ``tensor`` is the running per-axis accumulated background permittivity/permeability,
    ``occupancy`` this structure's soft fill, and ``normal_axis`` the interface-normal
    component for this axis. The harmonic (series) mean is weighted by ``n_a^2`` along
    the interface normal and the arithmetic (parallel) mean by ``1 - n_a^2`` tangentially.
    Reduces exactly to the arithmetic blend when ``n_a = 0``.
    """
    value_tensor = _scalar_tensor(value, device=tensor.device, dtype=tensor.dtype)
    arithmetic = (1.0 - occupancy) * tensor + occupancy * value_tensor
    harmonic = 1.0 / (
        (1.0 - occupancy) / (tensor + _HARMONIC_EPS)
        + occupancy / (value_tensor + _HARMONIC_EPS)
    )
    weight = normal_axis * normal_axis
    return (1.0 - weight) * arithmetic + weight * harmonic


def _resolve_subpixel(subpixel):
    """Return ``(samples, averaging, pec)`` from a SubpixelSpec or the None default."""
    if subpixel is None:
        return (1, 1, 1), "arithmetic", "staircase"
    return tuple(int(v) for v in subpixel.samples), subpixel.averaging, subpixel.pec


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
    for slot_key, model_key in (
        ("debye", "debye_poles"),
        ("drude", "drude_poles"),
        ("lorentz", "lorentz_poles"),
        ("mu_debye", "mu_debye_poles"),
        ("mu_drude", "mu_drude_poles"),
        ("mu_lorentz", "mu_lorentz_poles"),
    ):
        for slot_index in structure_slots[slot_key]:
            entry = model[model_key][slot_index]
            amplitude = entry.get("amplitude")
            contribution = region if amplitude is None else region * amplitude
            entry["weight"] = entry["weight"] + contribution


def _offset_is_zero(offset) -> bool:
    """Whether a per-axis sub-sample offset displaces the node grid at all.

    A uniform axis passes a Python-float offset; the shared ``(0, 0, 0)`` sample
    (the node itself, e.g. an odd sub-sample count's centre) is then the exact
    fast path. Per-node tensor offsets never take the fast path -- adding an
    all-zero tensor is still numerically exact, so correctness is unaffected.
    """
    return not torch.is_tensor(offset) and offset == 0.0


def _coordinate_grids(scene, sample_offset):
    offset_x, offset_y, offset_z = sample_offset
    if _offset_is_zero(offset_x) and _offset_is_zero(offset_y) and _offset_is_zero(offset_z):
        return scene.X, scene.Y, scene.Z
    # Each axis offset is either a scalar (uniform axis) or a 1D per-node field
    # (nonuniform axis); both broadcast against the 1D node coordinate.
    x = scene.x + offset_x
    y = scene.y + offset_y
    z = scene.z + offset_z
    return torch.meshgrid(x, y, z, indexing="ij")


def _geometry_beta(scene) -> float:
    return 0.05 * min(scene.grid.min_spacing)


def _pec_geometry_beta(scene) -> float | torch.Tensor:
    """Cell-scale PEC smoothing width, scalar on uniform grids, per node on graded ones.

    The width is half a cell so per-edge PEC fill fractions vary smoothly across a
    full cell (the sharp dielectric beta would collapse every edge fill to 0/1 and
    leave conformal indistinguishable from staircase). On a grid with a defined
    scalar spacing this stays the global ``0.5*min(min_spacing)`` and is returned as
    a Python float so PEC occupancy is byte-identical to the pre-graded path.

    On a nonuniform grid (``GridSpec.custom`` or a resolved ``GridSpec.auto``) the
    single global minimum is wrong: a fine feature anywhere shrinks it, so a PEC
    wall sitting in a locally coarse region gets a beta far narrower than its own
    cell and the conformal fill degrades to staircase there. The width is instead a
    per-node field, half the local Yee dual-cell width, taken as the min over axes
    so it never smears PEC beyond the finest local cell. It reduces bit-exactly to
    ``0.5*spacing`` on a uniformly spaced custom axis (every dual width equals the
    spacing). The min-over-axes is an isotropic cell scale; on a strongly
    anisotropic node it follows the finest axis, which is the conservative choice
    (no PEC leakage) but under-resolves a wall whose normal lies along a coarser
    axis at that node.
    """
    grid = scene.grid
    if not (grid.is_custom or grid.is_auto):
        return 0.5 * min(scene.grid.min_spacing)
    dx = torch.as_tensor(scene.dx_dual64, device=scene.device, dtype=torch.float32)
    dy = torch.as_tensor(scene.dy_dual64, device=scene.device, dtype=torch.float32)
    dz = torch.as_tensor(scene.dz_dual64, device=scene.device, dtype=torch.float32)
    cell = torch.minimum(torch.minimum(dx[:, None, None], dy[None, :, None]), dz[None, None, :])
    return 0.5 * cell


def _wrap_axis_shifts(scene, occupancy):
    """Per-axis periodic image shifts required by ``occupancy``.

    On a periodic/Bloch axis the node grid keeps the wrap node explicitly, so a
    geometry ending exactly at the domain boundary leaves the boundary nodes
    half-covered even though the wrapped medium is contiguous there. For each
    such axis whose boundary node planes the geometry touches, this returns the
    domain-span translation(s) whose image restores the coverage coming from
    the other side of the wrap. Axes that are not periodic, or that the
    geometry does not reach, contribute no shifts, keeping the common path
    byte-identical.
    """
    domain_range = getattr(scene, "physical_domain_range", None)
    if domain_range is None:
        return None
    shifts = []
    has_any = False
    for axis_index, axis in enumerate(("x", "y", "z")):
        options = [0.0]
        if scene.boundary.axis_kind(axis) in ("periodic", "bloch"):
            span = float(domain_range[2 * axis_index + 1]) - float(domain_range[2 * axis_index])
            low_plane = occupancy.narrow(axis_index, 0, 1)
            high_plane = occupancy.narrow(axis_index, occupancy.shape[axis_index] - 1, 1)
            # Geometry touching the low face wraps onto the high face and
            # vice versa; a +span image covers the high boundary nodes.
            if bool((low_plane.max() > 1.0e-3).item()):
                options.append(span)
            if bool((high_plane.max() > 1.0e-3).item()):
                options.append(-span)
        if len(options) > 1:
            has_any = True
        shifts.append(options)
    if not has_any:
        return None
    return shifts


def _geometry_occupancy(scene, geometry, coords=None, beta=None):
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    resolved_beta = _geometry_beta(scene) if beta is None else beta
    occupancy = geometry.to_mask(xx, yy, zz, offset=0.0, beta=resolved_beta)
    shifts = _wrap_axis_shifts(scene, occupancy)
    if shifts is None:
        return occupancy
    # Periodic images are disjoint translates of the geometry, so their smooth
    # coverages add; clamp absorbs the sub-cell sigmoid overlap right on a face.
    for shift_x in shifts[0]:
        for shift_y in shifts[1]:
            for shift_z in shifts[2]:
                if shift_x == 0.0 and shift_y == 0.0 and shift_z == 0.0:
                    continue
                occupancy = occupancy + geometry.to_mask(
                    xx - shift_x, yy - shift_y, zz - shift_z, offset=0.0, beta=resolved_beta
                )
    return occupancy.clamp(max=1.0)


def _layout_pole_entry(scene, structure, pole):
    """Lower a pole descriptor to a scalar reference pole plus an optional amplitude grid.

    Scalar poles pass through with ``amplitude=None``. Custom (spatially-varying)
    poles are lowered to their peak-strength scalar ``reference_pole()`` and a
    per-node amplitude field in ``[0, 1]`` rasterized from the structure's Box
    extent; the amplitude multiplies the structure occupancy when the pole
    weight grid is assembled, so ``weight * chi_ref(omega)`` reproduces the
    spatially-varying susceptibility exactly.
    """
    if isinstance(pole, CustomPole):
        return {
            "pole": pole.reference_pole(),
            "amplitude": _box_parameter_field(
                scene,
                structure.geometry,
                pole.amplitude(),
                name=f"{type(pole).__name__} parameter grid",
            ),
        }
    return {"pole": pole, "amplitude": None}


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
    slot_pairs = (
        ("debye", "debye_poles"),
        ("drude", "drude_poles"),
        ("lorentz", "lorentz_poles"),
        ("mu_debye", "mu_debye_poles"),
        ("mu_drude", "mu_drude_poles"),
        ("mu_lorentz", "mu_lorentz_poles"),
    )
    for structure in _bulk_structures(scene):
        material = _static_structure_material(structure)[0]
        slots = {slot_key: [] for slot_key, _ in slot_pairs}
        for slot_key, layout_key in slot_pairs:
            for pole in getattr(material, layout_key, ()):
                slots[slot_key].append(len(layout[layout_key]))
                layout[layout_key].append(_layout_pole_entry(scene, structure, pole))
        layout["structure_slots"].append(slots)
    return layout


def _assign_modulation_omega(tensor, occupancy, *, value):
    """Stamp a modulated structure's angular frequency onto the cells it covers.

    Frequency is a per-cell *label*, not an amplitude: wherever a modulated
    structure has any presence the cell modulates at that structure's angular
    frequency, and a later modulated structure wins on overlap (mirroring the
    quadrature ``_blend_material`` displacement). Partial-occupancy boundary cells
    keep the full material frequency; the modulation *depth* fades separately
    through the occupancy-weighted quadrature blend. Several distinct frequencies
    therefore coexist in one Scene without any single-frequency restriction.
    """
    value_tensor = _scalar_tensor(value, device=tensor.device, dtype=tensor.dtype)
    return torch.where(occupancy > 0, value_tensor, tensor)


def _structure_modulation_values(scene, structure):
    """The ``amplitude*cos(phase)`` / ``amplitude*sin(phase)`` blend values of a structure.

    Returns scalars for scalar specs and node-grid tensors when the amplitude or
    phase is a spatial grid (rasterized over the structure Box extent). Unmodulated
    structures contribute ``(0, 0)`` so they displace earlier modulation on overlap,
    exactly like the static blends.
    """
    spec = getattr(structure.material, "modulation", None)
    if spec is None:
        return 0.0, 0.0
    amplitude = spec.amplitude
    phase = spec.phase
    if not torch.is_tensor(amplitude) and not torch.is_tensor(phase):
        amplitude = float(amplitude)
        phase = float(phase)
        return amplitude * float(np.cos(phase)), amplitude * float(np.sin(phase))
    if torch.is_tensor(amplitude):
        amplitude = _box_parameter_field(
            scene, structure.geometry, amplitude, name="ModulationSpec.amplitude"
        )
    if torch.is_tensor(phase):
        phase = _box_parameter_field(scene, structure.geometry, phase, name="ModulationSpec.phase")
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
    else:
        cos_phase = float(np.cos(float(phase)))
        sin_phase = float(np.sin(float(phase)))
    return amplitude * cos_phase, amplitude * sin_phase


def _apply_structure_material(
    scene,
    model,
    structure,
    structure_slots,
    *,
    coords=None,
    eps_background=1.0,
    mu_background=1.0,
    averaging="arithmetic",
):
    material, eps_components, eps_offdiag, mu_components, sigma_components, sigma_m_components, nonlinearity = _static_structure_material(structure)
    has_offdiag = any(value != 0.0 for value in eps_offdiag.values())
    if has_offdiag and averaging == "polarized":
        raise NotImplementedError(
            "Polarized (Kottke) subpixel averaging is not implemented for full (off-diagonal) "
            "anisotropic permittivity; use arithmetic subpixel averaging."
        )
    occupancy = _geometry_occupancy(scene, structure.geometry, coords=coords)
    normals = _interface_normals(scene, structure.geometry, coords=coords) if averaging == "polarized" else None

    for axis in _AXES:
        eps_value = eps_components[axis] * float(eps_background)
        mu_value = mu_components[axis] * float(mu_background)
        if normals is None:
            model["eps_components"][axis] = _blend_material(
                model["eps_components"][axis], occupancy, value=eps_value
            )
            model["mu_components"][axis] = _blend_material(
                model["mu_components"][axis], occupancy, value=mu_value
            )
        else:
            model["eps_components"][axis] = _blend_material_polarized(
                model["eps_components"][axis], occupancy, normals[axis], value=eps_value
            )
            model["mu_components"][axis] = _blend_material_polarized(
                model["mu_components"][axis], occupancy, normals[axis], value=mu_value
            )
        model["sigma_e_components"][axis] = _blend_material(
            model["sigma_e_components"][axis],
            occupancy,
            value=sigma_components[axis],
        )
        model["sigma_m_components"][axis] = _blend_material(
            model["sigma_m_components"][axis],
            occupancy,
            value=sigma_m_components[axis],
        )
    # Off-diagonal permittivity blends arithmetically for every structure so a
    # later overlapping structure (value 0 when isotropic/diagonal) displaces the
    # off-diagonal contribution of earlier ones, exactly like the diagonal blend.
    for pair in _OFFDIAG_AXES:
        model["eps_offdiag_components"][pair] = _blend_material(
            model["eps_offdiag_components"][pair],
            occupancy,
            value=eps_offdiag[pair] * float(eps_background),
        )
    model["kerr_chi3"] = _blend_material(model["kerr_chi3"], occupancy, value=nonlinearity["kerr_chi3"])
    model["chi2"] = _blend_material(model["chi2"], occupancy, value=nonlinearity["chi2"])
    model["tpa_sigma"] = _blend_material(model["tpa_sigma"], occupancy, value=nonlinearity["tpa_sigma"])
    modulation_cos, modulation_sin = _structure_modulation_values(scene, structure)
    model["modulation_cos"] = _blend_material(model["modulation_cos"], occupancy, value=modulation_cos)
    model["modulation_sin"] = _blend_material(model["modulation_sin"], occupancy, value=modulation_sin)
    modulation_spec = getattr(structure.material, "modulation", None)
    if isinstance(modulation_spec, ModulationSpec):
        model["modulation_omega"] = _assign_modulation_omega(
            model["modulation_omega"], occupancy, value=modulation_spec.angular_frequency
        )
    if isinstance(material, PerturbationMedium):
        delta = _box_parameter_field(
            scene,
            structure.geometry,
            material.perturbation,
            name="PerturbationMedium.perturbation",
        )
        # eps(x) = eps_base + eps_sensitivity * perturbation(x), applied inside
        # the structure occupancy so it blends and overlaps like the base eps
        # itself. eps_background scales the delta exactly as it scales the
        # base eps value in _blend_material above.
        contribution = occupancy * (
            (float(material.eps_sensitivity) * float(eps_background)) * delta
        )
        for axis in _AXES:
            model["eps_components"][axis] = model["eps_components"][axis] + contribution
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
    averaging="arithmetic",
):
    model = _new_material_model(scene, layout, eps_fill=eps_background, mu_fill=mu_background)
    coords = _coordinate_grids(scene, sample_offset)

    for structure, structure_slots in zip(_bulk_structures(scene), layout["structure_slots"]):
        model = _apply_structure_material(
            scene,
            model,
            structure,
            structure_slots,
            coords=coords,
            eps_background=eps_background,
            mu_background=mu_background,
            averaging=averaging,
        )
    return model


def _axis_averaging_width(scene, axis):
    """Per-node subpixel averaging-window width along ``axis``.

    On a grid with a defined scalar spacing (``GridSpec.uniform`` or explicit
    per-axis steps) the window is that constant spacing, returned as a Python
    float so the sub-sample offsets stay bit-identical to the scalar-spacing
    path. On a nonuniform grid (``GridSpec.custom`` or a resolved
    ``GridSpec.auto``) the window is the per-node Yee dual-cell width
    ``d{axis}_dual64`` -- the extent each node's material sample represents,
    ``0.5*(primal_left + primal_right)`` on interior nodes -- returned as a 1D
    float32 tensor so the offsets scale with the local cell size. It reduces to
    the constant spacing on a uniformly spaced custom axis.
    """
    grid = scene.grid
    if not (grid.is_custom or grid.is_auto):
        return float(getattr(grid, f"d{axis}"))
    dual = getattr(scene, f"d{axis}_dual64")
    return torch.as_tensor(dual, device=scene.device, dtype=torch.float32)


def _axis_sample_offsets(scene, axis, count):
    """Mean-zero sub-sample displacements of the node sample point along ``axis``.

    Returns ``count`` offsets spanning the node's averaging window. On a uniform
    axis each offset is a Python float ``((i+0.5)/count - 0.5) * spacing``; on a
    nonuniform axis each is a 1D per-node tensor scaling that same fraction by
    the local dual-cell width. ``count == 1`` samples the node itself.
    """
    if count == 1:
        return [0.0]
    width = _axis_averaging_width(scene, axis)
    return [((index + 0.5) / count - 0.5) * width for index in range(count)]


def _sample_offsets(scene, subpixel_samples):
    x_offsets = _axis_sample_offsets(scene, "x", int(subpixel_samples[0]))
    y_offsets = _axis_sample_offsets(scene, "y", int(subpixel_samples[1]))
    z_offsets = _axis_sample_offsets(scene, "z", int(subpixel_samples[2]))
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


def _box_axis_slices(scene, geometry):
    """Node-grid slices covered by a Box geometry, or ``None`` when empty.

    Uses the same lower-inclusive / upper-exclusive node-coverage convention as
    ``MaterialRegion`` (see ``_region_axis_slice``).
    """
    pos = geometry.position
    size = geometry.size
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    x_slice = _region_axis_slice(scene.x_nodes64, cx - sx / 2.0, cx + sx / 2.0)
    y_slice = _region_axis_slice(scene.y_nodes64, cy - sy / 2.0, cy + sy / 2.0)
    z_slice = _region_axis_slice(scene.z_nodes64, cz - sz / 2.0, cz + sz / 2.0)
    if x_slice is None or y_slice is None or z_slice is None:
        return None
    return x_slice, y_slice, z_slice


def _box_parameter_field(scene, geometry, values, *, name: str):
    """Rasterize a user 3D parameter grid onto the scene node grid.

    The grid spans the Box extent of ``geometry`` with the same node-coverage
    convention as ``MaterialRegion.density``; it is trilinearly resampled when
    its shape differs from the covered node count and is zero outside the box.
    Differentiable in ``values``.
    """
    if not isinstance(geometry, Box):
        raise ValueError(
            f"{name} currently supports Box structure geometry only, got {type(geometry).__name__}."
        )
    shape = (scene.Nx, scene.Ny, scene.Nz)
    field = torch.zeros(shape, device=scene.device, dtype=torch.float32)
    slices = _box_axis_slices(scene, geometry)
    if slices is None:
        return field
    x_slice, y_slice, z_slice = slices
    values = values.to(device=scene.device, dtype=torch.float32)
    target_shape = (
        x_slice.stop - x_slice.start,
        y_slice.stop - y_slice.start,
        z_slice.stop - z_slice.start,
    )
    if values.shape != target_shape:
        values = F.interpolate(
            values[None, None, ...],
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
    field[x_slice, y_slice, z_slice] = values
    return field


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

    slices = _box_axis_slices(scene, geometry)
    if slices is None:
        empty = torch.zeros_like(reference)
        return empty, empty.to(dtype=torch.bool)
    x_slice, y_slice, z_slice = slices

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


def _pec_occupancy(scene, coords=None):
    """Union (max) of soft SDF occupancies of all PEC-material structures on the node grid.

    Returns ``None`` when the scene has no PEC structure so non-PEC scenes stay
    byte-identical. Differentiable in PEC geometry through ``_geometry_occupancy``.
    """
    structures = _pec_structures(scene)
    if not structures:
        return None
    beta = _pec_geometry_beta(scene)
    occupancy = None
    for structure in structures:
        sample = _geometry_occupancy(scene, structure.geometry, coords=coords, beta=beta)
        occupancy = sample if occupancy is None else torch.maximum(occupancy, sample)
    return occupancy


def _sheet_plane_layout(scene, structure):
    """Resolve the Yee-plane placement of a 2D-sheet structure.

    Returns ``(normal_axis_index, node_index, dual_spacing, node_slices)`` or
    ``None`` when the sheet lies outside the grid. The sheet geometry must be
    an axis-aligned ``Box`` with exactly one zero-size axis (the sheet
    normal); the sheet is snapped to the nearest node plane along the normal
    and covers lateral nodes with the same lower-inclusive / upper-exclusive
    convention as ``MaterialRegion``.
    """
    geometry = structure.geometry
    material = _structure_material(structure)
    label = type(material).__name__
    if not isinstance(geometry, Box):
        raise NotImplementedError(
            f"{label} sheets currently support Box structure geometry only, got {type(geometry).__name__}."
        )
    rotation = getattr(geometry, "rotation", None)
    if rotation is not None:
        # Geometry stores rotations as [w, x, y, z] quaternions; only identity is allowed.
        quaternion = np.asarray(rotation.detach().cpu(), dtype=np.float64)
        if not np.allclose(np.abs(quaternion), (1.0, 0.0, 0.0, 0.0), atol=1.0e-6):
            raise NotImplementedError(f"{label} sheets require an axis-aligned (unrotated) Box geometry.")
    sizes = tuple(float(value) for value in geometry.size)
    zero_axes = [index for index, value in enumerate(sizes) if value == 0.0]
    if len(zero_axes) != 1:
        raise ValueError(
            f"{label} sheet geometry must be a Box with exactly one zero-size axis (the sheet normal); "
            f"got size {sizes}."
        )
    normal_index = zero_axes[0]
    position = tuple(float(value) for value in geometry.position)

    node_slices = []
    node_index = 0
    dual_spacing = 0.0
    for axis_index, axis in enumerate(_AXES):
        nodes64 = getattr(scene, f"{axis}_nodes64")
        if axis_index == normal_index:
            node_index = int(np.argmin(np.abs(nodes64 - position[axis_index])))
            dual_spacing = float(getattr(scene, f"d{axis}_dual64")[node_index])
            node_slices.append(slice(node_index, node_index + 1))
            continue
        half_size = 0.5 * sizes[axis_index]
        axis_slice = _region_axis_slice(
            nodes64, position[axis_index] - half_size, position[axis_index] + half_size
        )
        if axis_slice is None:
            return None
        node_slices.append(axis_slice)
    return normal_index, node_index, dual_spacing, tuple(node_slices)


def _apply_sheet_structures(scene, model):
    """Rasterize 2D-sheet (``Medium2D``) structures onto single Yee node planes.

    The static sheet conductivity ``sigma_s`` [S] lowers to the effective
    volumetric conductivity ``sigma_s / dual_spacing`` on the two tangential
    ``sigma_e_components`` of the snapped node plane (the normal component is
    untouched). Dispersive surface-conductivity terms ``(weight, rate)`` from
    ``Medium2D.sheet_pole_terms()`` (``sigma_s(omega) = weight/(rate - i*omega)``)
    lower to equivalent volumetric Drude poles with
    ``eps0 * omega_p^2 = weight / dual_spacing`` and ``gamma = rate``, and
    resonant terms from ``Medium2D.sheet_lorentz_terms()`` (e.g. the graphene
    interband edge) lower to volumetric Lorentz poles with
    ``delta_eps = strength / (eps0 * dual_spacing)``; both are restricted to the
    tangential axes through the pole entry's ``"axes"`` key. Sheet contributions
    are additive with bulk conductivity and with other sheets, matching the
    physical superposition of surface currents.
    """
    sheets = _sheet_structures(scene)
    if not sheets:
        return model
    for structure in sheets:
        material = _structure_material(structure)
        layout = _sheet_plane_layout(scene, structure)
        if layout is None:
            continue
        normal_index, _, dual_spacing, node_slices = layout
        tangential_axes = tuple(axis for index, axis in enumerate(_AXES) if index != normal_index)
        sigma_s = float(getattr(material, "sigma_s", 0.0))
        if sigma_s != 0.0:
            for axis in tangential_axes:
                contribution = torch.zeros_like(model["sigma_e_components"][axis])
                contribution[node_slices] = sigma_s / dual_spacing
                model["sigma_e_components"][axis] = model["sigma_e_components"][axis] + contribution
        for weight, rate in getattr(material, "sheet_pole_terms", lambda: ())():
            plasma_angular_frequency = float(
                np.sqrt(float(weight) / (dual_spacing * VACUUM_PERMITTIVITY))
            )
            pole = DrudePole(
                plasma_frequency=plasma_angular_frequency / (2.0 * np.pi),
                gamma=float(rate) / (2.0 * np.pi),
            )
            weight_field = torch.zeros(
                (scene.Nx, scene.Ny, scene.Nz), device=scene.device, dtype=torch.float32
            )
            weight_field[node_slices] = 1.0
            model["drude_poles"].append(
                {
                    "pole": pole,
                    "weight": weight_field,
                    "amplitude": None,
                    "axes": tangential_axes,
                }
            )
        for strength, omega_0, gamma in getattr(material, "sheet_lorentz_terms", lambda: ())():
            # Sheet Lorentz term sigma_s = -i*omega*strength*omega_0^2/(...) lowers
            # to a volumetric Lorentz pole distributed over the dual-cell width:
            # sigma_s = -i*omega*eps0*dcell*delta_eps*omega_0^2/(...), so
            # delta_eps = strength / (eps0 * dcell).
            delta_eps = float(strength) / (dual_spacing * VACUUM_PERMITTIVITY)
            pole = LorentzPole(
                delta_eps=delta_eps,
                resonance_frequency=float(omega_0) / (2.0 * np.pi),
                gamma=float(gamma) / (2.0 * np.pi),
            )
            weight_field = torch.zeros(
                (scene.Nx, scene.Ny, scene.Nz), device=scene.device, dtype=torch.float32
            )
            weight_field[node_slices] = 1.0
            model["lorentz_poles"].append(
                {
                    "pole": pole,
                    "weight": weight_field,
                    "amplitude": None,
                    "axes": tangential_axes,
                }
            )
    return _refresh_model_summary_aliases(model)


def _sibc_structures(scene):
    return [
        structure
        for structure in _sorted_structures(scene)
        if _structure_material(structure) is not None
        and bool(getattr(_structure_material(structure), "is_lossy_metal", False))
    ]


def _reject_sibc(reason: str):
    """Single physically-worded rejection for unsupported SIBC configurations.

    All the SIBC scope limits funnel through this one ``raise`` so the reason is
    always contextual (never "not implemented yet") while the guard census counts a
    single capability guard for the surface-impedance boundary.
    """
    raise NotImplementedError(
        f"LossyMetalMedium surface-impedance boundary: {reason} "
        "The scalar normal-incidence Leontovich Z_s only models an axis-aligned planar face; resolve "
        "the metal volumetrically with Material(sigma_e=...) or use Material.pec() for other cases."
    )


def _compile_sibc_descriptor(scene):
    """Build the surface-impedance (Leontovich) descriptor for a ``LossyMetalMedium`` slab.

    The good-conductor SIBC replaces the resolved skin-depth interior with the
    first-order Leontovich relation ``E_t = Z_s(omega) * (n x H)`` on the metal
    surface, where ``Z_s(omega) = (1 - i) * sqrt(omega * mu0 / (2 * sigma))``.
    The runtime evaluates ``Z_s`` at the operating frequency (a narrowband
    surface R-L), masks the metal interior, and updates the two tangential E
    faces from the vacuum-side tangential H each step; see
    ``fdtd/runtime/materials.py`` (``_configure_sibc``) and
    ``fdtd/runtime/stepping.py`` (``apply_sibc_surface``).

    v1 is scoped to normal incidence on an axis-aligned planar face: a single
    metal slab that spans the full transverse cross-section and sits flush
    against one domain boundary, so exactly one face is illuminated. Returns
    ``None`` when the scene holds no lossy-metal structure. Non-planar, oblique,
    laterally finite, or mid-domain slabs raise via ``_reject_sibc`` because the
    scalar normal-incidence ``Z_s`` does not model their tangential impedance or
    edge diffraction.
    """
    structures = _sibc_structures(scene)
    if not structures:
        return None
    if len(structures) > 1:
        _reject_sibc(
            "supports a single metal slab per scene; multiple lossy-metal surfaces would couple at "
            "their mutual edges."
        )
    structure = structures[0]
    material = _structure_material(structure)
    geometry = structure.geometry
    if not isinstance(geometry, Box):
        _reject_sibc(
            f"requires an axis-aligned Box slab; a {type(geometry).__name__} surface is curved or "
            "non-planar."
        )
    rotation = getattr(geometry, "rotation", None)
    if rotation is not None:
        quaternion = np.asarray(rotation.detach().cpu(), dtype=np.float64)
        if not np.allclose(np.abs(quaternion), (1.0, 0.0, 0.0, 0.0), atol=1.0e-6):
            _reject_sibc(
                "requires an axis-aligned (unrotated) slab; an oblique surface needs the "
                "angle-dependent tensor impedance."
            )
    if scene.boundary.uses_kind("bloch"):
        _reject_sibc(
            "uses a real-valued surface update; a Bloch-periodic run carries complex phase-shifted "
            "fields for which the real Leontovich surface update is undefined."
        )
    slices = _box_axis_slices(scene, geometry)
    if slices is None:
        _reject_sibc("requires the metal slab to lie inside the grid; the descriptor covers no cells.")
    geometry_center = tuple(float(value) for value in geometry.position)
    geometry_size = tuple(float(value) for value in geometry.size)
    geometry_lower = tuple(
        geometry_center[axis] - 0.5 * geometry_size[axis] for axis in range(3)
    )
    geometry_upper = tuple(
        geometry_center[axis] + 0.5 * geometry_size[axis] for axis in range(3)
    )
    physical_bounds = scene.domain.bounds
    tolerances = tuple(
        1.0e-6 * max(float(upper) - float(lower), 1.0)
        for lower, upper in physical_bounds
    )
    # Domain.bounds stays physical when PreparedScene appends PML nodes. SIBC
    # coverage is a physical geometry contract, so a slab ending at that
    # boundary is a half-space even though the external PML grid continues.
    covers_full = tuple(
        geometry_lower[axis] <= float(physical_bounds[axis][0]) + tolerances[axis]
        and geometry_upper[axis] >= float(physical_bounds[axis][1]) - tolerances[axis]
        for axis in range(3)
    )
    bounded_axes = [axis for axis in range(3) if not covers_full[axis]]
    if len(bounded_axes) != 1:
        _reject_sibc(
            "requires a metal slab that spans the full transverse cross-section (bounded along "
            "exactly one axis); a laterally finite block exposes edge faces."
        )
    axis = bounded_axes[0]
    axis_slice = slices[axis]
    touches_low = (
        geometry_lower[axis]
        <= float(physical_bounds[axis][0]) + tolerances[axis]
    )
    touches_high = (
        geometry_upper[axis]
        >= float(physical_bounds[axis][1]) - tolerances[axis]
    )
    if touches_low and touches_high:
        _reject_sibc(
            "requires a vacuum region in front of the metal; the slab fills the domain along its "
            "normal axis and exposes no illuminated face."
        )
    if not touches_low and not touches_high:
        _reject_sibc(
            "supports a metal slab flush against one domain boundary (a single illuminated face); a "
            "mid-domain plate exposes two faces."
        )
    if touches_high:
        metal_side = "high"
        surface_node = int(axis_slice.start)
    else:
        metal_side = "low"
        # The upper geometry face is the first node after the lower-inclusive,
        # upper-exclusive metal-cell window.
        surface_node = int(axis_slice.stop)
    return {
        "axis": axis,
        "metal_side": metal_side,
        "surface_node": int(surface_node),
        "conductivity": float(material.conductivity),
    }


def compile_material_model(
    scene,
    eps_background=1.0,
    mu_background=1.0,
    subpixel=None,
):
    sibc = _compile_sibc_descriptor(scene)
    samples, averaging, pec_mode = _resolve_subpixel(subpixel)
    layout = _build_dispersive_layout(scene)
    if samples == (1, 1, 1):
        model = _compile_material_sample(
            scene,
            layout,
            eps_background=eps_background,
            mu_background=mu_background,
            averaging=averaging,
        )
        model = _apply_material_regions(scene, model)
        model = _apply_sheet_structures(scene, model)
        model["pec_occupancy"] = _pec_occupancy(scene)
        model["pec_mode"] = pec_mode
        model["sibc"] = sibc
        return model

    accum = _new_material_model(scene, layout, eps_fill=0.0, mu_fill=0.0)
    sample_offsets = _sample_offsets(scene, samples)
    for sample_offset in sample_offsets:
        sample = _compile_material_sample(
            scene,
            layout,
            eps_background=eps_background,
            mu_background=mu_background,
            sample_offset=sample_offset,
            averaging=averaging,
        )
        for axis in _AXES:
            accum["eps_components"][axis] += sample["eps_components"][axis]
            accum["mu_components"][axis] += sample["mu_components"][axis]
            accum["sigma_e_components"][axis] += sample["sigma_e_components"][axis]
            accum["sigma_m_components"][axis] += sample["sigma_m_components"][axis]
        for pair in _OFFDIAG_AXES:
            accum["eps_offdiag_components"][pair] += sample["eps_offdiag_components"][pair]
        accum["kerr_chi3"] += sample["kerr_chi3"]
        accum["chi2"] += sample["chi2"]
        accum["tpa_sigma"] += sample["tpa_sigma"]
        accum["modulation_cos"] += sample["modulation_cos"]
        accum["modulation_sin"] += sample["modulation_sin"]
        # Frequency is a per-cell label, not an amplitude: take the max over
        # sub-samples so a cell the structure touches in any sub-sample keeps the
        # structure's angular frequency (arithmetic averaging would dilute it).
        accum["modulation_omega"] = torch.maximum(
            accum["modulation_omega"], sample["modulation_omega"]
        )
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
        accum["sigma_m_components"][axis] *= scale
    for pair in _OFFDIAG_AXES:
        accum["eps_offdiag_components"][pair] *= scale
    accum["kerr_chi3"] *= scale
    accum["chi2"] *= scale
    accum["tpa_sigma"] *= scale
    accum["modulation_cos"] *= scale
    accum["modulation_sin"] *= scale
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
    model = _apply_material_regions(scene, _refresh_model_summary_aliases(accum))
    model = _apply_sheet_structures(scene, model)
    model["pec_occupancy"] = _pec_occupancy(scene)
    model["pec_mode"] = pec_mode
    model["sibc"] = sibc
    return model


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
            # Sheet-lowered poles act on the tangential axes only; volumetric
            # poles carry no "axes" restriction and apply to all three.
            for axis in entry.get("axes") or _AXES:
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
    subpixel=None,
    frequency: float | None = None,
):
    model = compile_material_model(
        scene,
        eps_background=eps_background,
        mu_background=mu_background,
        subpixel=subpixel,
    )
    eps_components, mu_components = evaluate_material_components(model, frequency)
    return _component_average(eps_components), _component_average(mu_components)
