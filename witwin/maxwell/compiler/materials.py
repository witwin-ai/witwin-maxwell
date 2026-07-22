from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from witwin.core import Box
from witwin.core.material import VACUUM_PERMITTIVITY

from ..media import CustomPole, DiagonalTensor3, DrudePole, LorentzPole, ModulationSpec, PerturbationMedium, Tensor3x3
from .structures import pec_structures
from .subpixel import (
    _HARMONIC_EPS,
    _NORMAL_EPS,
    _blend_material_polarized,
    _field_gradients,
    _interface_normals,
    _reconstruct_sampled_polarized_components,
)

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
    # A GyromagneticFerrite compiles here only as its diagonal background
    # (eps_r, mu_infinity, sigma_e); the non-reciprocal off-diagonal permeability
    # is produced separately by the local magnetization-ADE runtime
    # (compile_gyromagnetic_layout + the FDTD gyromagnetic forward hooks), never by
    # widening mu_tensor. See docs/reference/ferrite-physics-contract.md.
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


def _material_model_has_electric_dispersion(model) -> bool:
    return bool(model["debye_poles"] or model["drude_poles"] or model["lorentz_poles"])


def _material_model_has_magnetic_dispersion(model) -> bool:
    return bool(model["mu_debye_poles"] or model["mu_drude_poles"] or model["mu_lorentz_poles"])


def _material_model_has_conductivity(model) -> bool:
    return any(torch.any(model["sigma_e_components"][axis] != 0).item() for axis in _AXES)


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


def _resolve_subpixel(subpixel):
    """Return ``(samples, averaging, pec)`` from a SubpixelSpec or the None default."""
    if subpixel is None:
        return (1, 1, 1), "arithmetic", "staircase"
    return tuple(int(v) for v in subpixel.samples), subpixel.averaging, subpixel.pec


def _validate_polarized_materials(scene):
    for structure in _bulk_structures(scene):
        eps_offdiag = _static_structure_material(structure)[2]
        if any(value != 0.0 for value in eps_offdiag.values()):
            raise NotImplementedError(
                "Polarized (Kottke) subpixel averaging is not implemented for full "
                "(off-diagonal) anisotropic permittivity; use arithmetic subpixel averaging."
            )


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


def _wrap_periodic_boundary_planes(scene, occupancy, midpoint_occupancy, skip_axes=()):
    """Make duplicate periodic endpoint planes carry their union occupancy.

    Periodic axes retain both endpoint nodes even though they denote one physical
    plane. Only those endpoint planes need wrap composition. Adding translated
    geometry through the whole volume is incorrect when a structure spans or
    exceeds one period because the base and image overlap away from the seam.

    ``skip_axes`` lists axis indices sampled on the staggered (Yee edge/face)
    grid: those carry the primal-cell midpoints, which have no duplicated endpoint
    plane, so the seam composition (and its >=2-node requirement) does not apply --
    the periodic-image union in :func:`_geometry_occupancy` already resolves any
    seam-crossing geometry there.
    """
    wrapped = occupancy
    for axis_index, axis in enumerate(("x", "y", "z")):
        if axis_index in skip_axes:
            continue
        if scene.boundary.axis_kind(axis) not in ("periodic", "bloch"):
            continue
        axis_size = wrapped.shape[axis_index]
        if axis_size < 2:
            raise ValueError("Periodic material occupancy requires at least two axis nodes.")
        low_plane = wrapped.narrow(axis_index, 0, 1)
        high_plane = wrapped.narrow(axis_index, axis_size - 1, 1)
        if axis_size == 2:
            bridge = midpoint_occupancy(axis_index)
        else:
            low_interior = wrapped.narrow(axis_index, 1, 1)
            high_interior = wrapped.narrow(axis_index, axis_size - 2, 1)
            bridge = torch.minimum(low_interior, high_interior)
        # Add the two endpoint coverages only where the geometry continues on
        # both sides of the seam. An orthogonal interface gives the same partial
        # occupancy on the adjacent interior planes and must remain partial.
        seam = torch.maximum(
            torch.maximum(low_plane, high_plane),
            torch.minimum((low_plane + high_plane).clamp(max=1.0), bridge),
        )
        interior = wrapped.narrow(axis_index, 1, wrapped.shape[axis_index] - 2)
        wrapped = torch.cat((seam, interior, seam), dim=axis_index)
    return wrapped


def _contains_trainable_tensor(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.requires_grad)
    if isinstance(value, dict):
        return any(_contains_trainable_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_trainable_tensor(item) for item in value)
    return False


def _static_periodic_shift_options(scene, geometry):
    """Limit periodic images from static world-space geometry bounds when possible."""
    options = [(0.0,), (0.0,), (0.0,)]
    periodic_axes = [
        axis_index
        for axis_index, axis in enumerate(("x", "y", "z"))
        if scene.boundary.axis_kind(axis) in ("periodic", "bloch")
    ]
    if not periodic_axes:
        return tuple(options)
    if any(_contains_trainable_tensor(value) for value in vars(geometry).values()):
        return None
    try:
        vertices, _ = geometry.to_mesh()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        return None
    vertices = torch.as_tensor(vertices)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        return None
    lower = vertices.amin(dim=0).detach().cpu().numpy()
    upper = vertices.amax(dim=0).detach().cpu().numpy()

    for axis_index in periodic_axes:
        axis = "xyz"[axis_index]
        domain_low, domain_high = scene.domain.bounds[axis_index]
        span = float(domain_high) - float(domain_low)
        nodes = np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64)
        padding = 0.5 * float(np.max(np.diff(nodes))) + _geometry_beta(scene)
        axis_options = [0.0]
        if float(lower[axis_index]) <= float(domain_low) + padding:
            axis_options.append(span)
        if float(upper[axis_index]) >= float(domain_high) - padding:
            axis_options.append(-span)
        options[axis_index] = tuple(axis_options)
    return tuple(options)


def _geometry_occupancy(
    scene,
    geometry,
    coords=None,
    beta=None,
    *,
    half_weight_boundary=False,
    periodic_shift_options=None,
    wrap_skip_axes=(),
):
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    resolved_beta = _geometry_beta(scene) if beta is None else beta

    def _sample(sample_x, sample_y, sample_z):
        occupancy = geometry.to_mask(
            sample_x,
            sample_y,
            sample_z,
            offset=0.0,
            beta=resolved_beta,
        )
        if half_weight_boundary:
            tolerance = 1.0e-5 * min(scene.grid.min_spacing)
            signed_distance = geometry.signed_distance(sample_x, sample_y, sample_z)
            half_occupancy = occupancy + (0.5 - occupancy).detach()
            occupancy = torch.where(
                torch.abs(signed_distance) <= tolerance,
                half_occupancy,
                occupancy,
            )
        return occupancy

    shift_options = periodic_shift_options
    if shift_options is None:
        domain_range = getattr(scene, "physical_domain_range", None)
        shift_options = []
        for axis_index, axis in enumerate(("x", "y", "z")):
            if domain_range is not None and scene.boundary.axis_kind(axis) in (
                "periodic",
                "bloch",
            ):
                span = float(domain_range[2 * axis_index + 1]) - float(
                    domain_range[2 * axis_index]
                )
                shift_options.append((-span, 0.0, span))
            else:
                shift_options.append((0.0,))

    def _periodic_union(sample_coords):
        union = None
        for shift_x, shift_y, shift_z in product(*shift_options):
            shifted = _sample(
                sample_coords[0] - shift_x,
                sample_coords[1] - shift_y,
                sample_coords[2] - shift_z,
            )
            union = shifted if union is None else torch.maximum(union, shifted)
        return union

    coords = (xx, yy, zz)
    occupancy = _periodic_union(coords)

    def _midpoint_occupancy(axis_index):
        midpoint_coords = list(coords)
        reduced_axes = [axis_index]
        reduced_axes.extend(
            index
            for index, axis in enumerate(("x", "y", "z"))
            if index != axis_index
            and scene.boundary.axis_kind(axis) in ("periodic", "bloch")
            and occupancy.shape[index] == 2
        )
        for reduced_axis in reduced_axes:
            for coord_index, coord in enumerate(midpoint_coords):
                low = coord.narrow(reduced_axis, 0, 1)
                high = coord.narrow(reduced_axis, coord.shape[reduced_axis] - 1, 1)
                midpoint_coords[coord_index] = (
                    0.5 * (low + high) if coord_index == reduced_axis else low
                )
        bridge = _periodic_union(tuple(midpoint_coords))
        bridge_shape = list(occupancy.shape)
        bridge_shape[axis_index] = 1
        return bridge.expand(bridge_shape)

    return _wrap_periodic_boundary_planes(
        scene, occupancy, _midpoint_occupancy, skip_axes=wrap_skip_axes
    )


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
    half_weight_boundary=False,
    periodic_shift_options=None,
    inverse_components=None,
):
    material, eps_components, eps_offdiag, mu_components, sigma_components, sigma_m_components, nonlinearity = _static_structure_material(structure)
    has_offdiag = any(value != 0.0 for value in eps_offdiag.values())
    if has_offdiag and averaging == "polarized":
        raise NotImplementedError(
            "Polarized (Kottke) subpixel averaging is not implemented for full (off-diagonal) "
            "anisotropic permittivity; use arithmetic subpixel averaging."
        )
    occupancy = _geometry_occupancy(
        scene,
        structure.geometry,
        coords=coords,
        half_weight_boundary=half_weight_boundary,
        periodic_shift_options=periodic_shift_options,
    )
    normals = _interface_normals(scene, structure.geometry, coords=coords) if averaging == "polarized" else None
    perturbation_delta = None
    if isinstance(material, PerturbationMedium):
        perturbation_delta = (
            float(material.eps_sensitivity)
            * float(eps_background)
            * _box_parameter_field(
                scene,
                structure.geometry,
                material.perturbation,
                name="PerturbationMedium.perturbation",
            )
        )

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
        if inverse_components is not None:
            effective_eps = eps_value if perturbation_delta is None else eps_value + perturbation_delta
            inverse_components["eps_components"][axis] = _blend_material(
                inverse_components["eps_components"][axis],
                occupancy,
                value=1.0 / (effective_eps + _HARMONIC_EPS),
            )
            inverse_components["mu_components"][axis] = _blend_material(
                inverse_components["mu_components"][axis],
                occupancy,
                value=1.0 / (mu_value + _HARMONIC_EPS),
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
        # eps(x) = eps_base + eps_sensitivity * perturbation(x), applied inside
        # the structure occupancy so it blends and overlaps like the base eps
        # itself. eps_background scales the delta exactly as it scales the
        # base eps value in _blend_material above.
        contribution = occupancy * perturbation_delta
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
    region_densities=(),
    sample_offset=(0.0, 0.0, 0.0),
    averaging="arithmetic",
    half_weight_boundary=False,
    geometry_periodic_shifts=None,
    collect_inverse=False,
):
    model = _new_material_model(scene, layout, eps_fill=eps_background, mu_fill=mu_background)
    inverse_components = None
    if collect_inverse:
        shape = (scene.Nx, scene.Ny, scene.Nz)
        inverse_components = {
            "eps_components": _new_component_field(
                shape,
                fill_value=1.0 / (float(eps_background) + _HARMONIC_EPS),
                device=scene.device,
            ),
            "mu_components": _new_component_field(
                shape,
                fill_value=1.0 / (float(mu_background) + _HARMONIC_EPS),
                device=scene.device,
            ),
        }
    coords = _coordinate_grids(scene, sample_offset)
    geometry_periodic_shifts = geometry_periodic_shifts or {}

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
            half_weight_boundary=half_weight_boundary,
            periodic_shift_options=geometry_periodic_shifts.get(id(structure.geometry)),
            inverse_components=inverse_components,
        )
    if region_densities:
        model = _apply_material_regions(
            scene,
            model,
            region_densities,
            coords=coords,
            averaging=averaging,
            half_weight_boundary=half_weight_boundary,
            geometry_periodic_shifts=geometry_periodic_shifts,
            inverse_components=inverse_components,
        )
    if collect_inverse:
        return model, inverse_components
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


# --- Edge-native (per-Yee-component) material sampling -----------------------
#
# Each Yee field component lives at its own staggered location, not at the node
# centre. The legacy path Kottke-blended every material at the node grid and then
# arithmetically averaged the node values onto the Yee edges/faces (a node->edge
# "smear"): the interface operator was applied at the wrong place and then
# linearly interpolated, which does not match a reference solver that evaluates
# the subpixel blend natively at each staggered location. The helpers below
# evaluate the diagonal background permittivity / permeability and the static
# conductivities directly at each component's Yee location -- the SDF occupancy,
# the interface normal, and the region density are all sampled there -- so the
# polarized (Kottke) or arithmetic subpixel blend is edge-native with no
# node->edge average. The node model is still produced as the canonical
# representation consumed by the summaries, monitors, mode solver, SAR and mass
# models; only the FDTD update-coefficient materials switch to the edge fields.

# The set of grid axes each Yee component is half-shifted along, and the material
# polarization axis it carries.
_EDGE_STAGGER_AXES = {
    "Ex": (0,),
    "Ey": (1,),
    "Ez": (2,),
    "Hx": (1, 2),
    "Hy": (0, 2),
    "Hz": (0, 1),
}
_EDGE_POLARIZATION_AXIS = {
    "Ex": "x",
    "Ey": "y",
    "Ez": "z",
    "Hx": "x",
    "Hy": "y",
    "Hz": "z",
}
_ELECTRIC_EDGE_COMPONENTS = ("Ex", "Ey", "Ez")
_MAGNETIC_EDGE_COMPONENTS = ("Hx", "Hy", "Hz")


def _edge_axis_base_1d(scene, axis, staggered):
    """Per-axis 1D query coordinates for one Yee component along ``axis``.

    On a staggered axis the samples sit at the primal-cell midpoints
    ``0.5*(node_i + node_{i+1})`` (length ``N-1``, the Yee edge/face count); on an
    unstaggered axis they stay on the node coordinates (length ``N``).
    """
    coord = getattr(scene, axis)
    if staggered:
        return 0.5 * (coord[:-1] + coord[1:])
    return coord


def _edge_axis_offsets(scene, axis, count, staggered):
    """Mean-zero subpixel offsets spanning one sample's averaging window on ``axis``.

    Mirrors :func:`_axis_sample_offsets` but scales by the primal-cell width on a
    staggered axis (the extent each Yee edge/face sample represents) and by the
    node dual-cell width on an unstaggered axis, so the offsets broadcast against
    the matching 1D base coordinates on both uniform and graded grids.
    """
    if count == 1:
        return [0.0]
    grid = scene.grid
    if not (grid.is_custom or grid.is_auto):
        width = float(getattr(grid, f"d{axis}"))
    elif staggered:
        coord = getattr(scene, axis)
        width = (coord[1:] - coord[:-1]).to(dtype=torch.float32)
    else:
        width = torch.as_tensor(
            getattr(scene, f"d{axis}_dual64"), device=scene.device, dtype=torch.float32
        )
    return [((index + 0.5) / count - 0.5) * width for index in range(count)]


def _edge_component_coords(bases, offsets):
    return torch.meshgrid(
        bases["x"] + offsets[0],
        bases["y"] + offsets[1],
        bases["z"] + offsets[2],
        indexing="ij",
    )


def _blend_edge_component(acc, inverse_acc, occupancy, normal_axis, value, *, polarized_direct):
    """Blend one structure/region into an edge accumulator.

    ``polarized_direct`` applies the Kottke normal-projection blend in place (the
    no-subpixel polarized path). Otherwise an arithmetic blend is used and, when
    ``inverse_acc`` is not None (the polarized-with-subpixel path), the reciprocal
    is accumulated arithmetically so the harmonic (series) mean can be
    reconstructed after the subpixel average.
    """
    if polarized_direct:
        return _blend_material_polarized(acc, occupancy, normal_axis, value=value), inverse_acc
    acc = _blend_material(acc, occupancy, value=value)
    if inverse_acc is not None:
        inverse_acc = _blend_material(inverse_acc, occupancy, value=1.0 / (value + _HARMONIC_EPS))
    return acc, inverse_acc


def _reconstruct_edge_polarized(scene, coords, arithmetic, inverse, axis):
    """Normal-projection reconstruction of one edge component from subcell means.

    The harmonic (series) mean ``1/inverse`` is blended toward the arithmetic
    (parallel) mean by the squared interface-normal component along ``axis``,
    estimated from the gradient of the arithmetic permittivity/permeability at the
    Yee location. This is the per-component edge-native form of
    :func:`_reconstruct_sampled_polarized_components`; for an isotropic medium it
    reproduces that node reconstruction exactly (the three per-axis components are
    identical there, so a single component's gradient carries the same direction).
    """
    grads = _field_gradients(scene, arithmetic, coords=coords)
    squared = {a: grads[a] * grads[a] for a in _AXES}
    norm = squared["x"] + squared["y"] + squared["z"] + _NORMAL_EPS
    weight = squared[axis] / norm
    harmonic = 1.0 / (inverse + _HARMONIC_EPS)
    return (1.0 - weight) * arithmetic + weight * harmonic


def _sample_box_parameter_field(scene, geometry, values, coords):
    """Trilinearly sample a Box-extent parameter grid at physical query coordinates.

    The edge-native counterpart of :func:`_box_parameter_field`: the grid spans the
    Box extent with the same convention, and is zero outside the box (border-padded
    values there are multiplied by a zero occupancy so they never contribute).
    Differentiable in ``values``.
    """
    xx, yy, zz = coords
    center = torch.as_tensor(geometry.position, device=scene.device, dtype=torch.float32)
    size = torch.as_tensor(geometry.size, device=scene.device, dtype=torch.float32)
    normalized = (
        2.0 * (xx - center[0]) / size[0],
        2.0 * (yy - center[1]) / size[1],
        2.0 * (zz - center[2]) / size[2],
    )
    query_grid = torch.stack(normalized, dim=-1)[None, ...]
    texture = values.to(device=scene.device, dtype=torch.float32).permute(2, 1, 0)[None, None, ...]
    return F.grid_sample(
        texture,
        query_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, 0]


def _edge_structure_perturbation_delta(scene, structure, material, coords, eps_background):
    """Edge-sampled ``eps_sensitivity * eps_background * perturbation`` or ``None``."""
    if not isinstance(material, PerturbationMedium):
        return None
    return (
        float(material.eps_sensitivity)
        * float(eps_background)
        * _sample_box_parameter_field(scene, structure.geometry, material.perturbation, coords)
    )


def _compile_edge_component(
    scene,
    component,
    *,
    eps_background,
    mu_background,
    averaging,
    samples,
    region_densities,
    geometry_periodic_shifts,
):
    """Edge-native diagonal permittivity/permeability and conductivity for one component.

    Returns ``(material_relative, sigma)`` where ``material_relative`` is the
    relative eps (electric components) or mu (magnetic components) sampled at the
    component's Yee location and ``sigma`` is the matching static electric /
    magnetic conductivity there. Differentiable in geometry / density parameters.
    """
    axis = _EDGE_POLARIZATION_AXIS[component]
    stagger = _EDGE_STAGGER_AXES[component]
    electric = component[0] == "E"
    background = float(eps_background if electric else mu_background)

    bases = {}
    axis_offsets = {}
    for axis_index, grid_axis in enumerate(_AXES):
        staggered = axis_index in stagger
        bases[grid_axis] = _edge_axis_base_1d(scene, grid_axis, staggered)
        axis_offsets[grid_axis] = _edge_axis_offsets(
            scene, grid_axis, int(samples[axis_index]), staggered
        )
    offset_combos = list(product(axis_offsets["x"], axis_offsets["y"], axis_offsets["z"]))
    half_weight_boundary = len(offset_combos) > 1
    polarized = averaging == "polarized"
    polarized_direct = polarized and len(offset_combos) == 1

    structures = list(_bulk_structures(scene))
    regions = list(zip(getattr(scene, "material_regions", ()), region_densities))

    material_sum = None
    sigma_sum = None
    inverse_sum = None
    for offsets in offset_combos:
        coords = _edge_component_coords(bases, offsets)
        shape = coords[0].shape
        material_acc = torch.full(shape, background, device=scene.device, dtype=torch.float32)
        sigma_acc = torch.zeros(shape, device=scene.device, dtype=torch.float32)
        inverse_acc = (
            torch.full(shape, 1.0 / (background + _HARMONIC_EPS), device=scene.device, dtype=torch.float32)
            if (polarized and not polarized_direct)
            else None
        )
        for structure in structures:
            parts = _static_structure_material(structure)
            eps_components = parts[1]
            mu_components = parts[3]
            sigma_components = parts[4]
            sigma_m_components = parts[5]
            occupancy = _geometry_occupancy(
                scene,
                structure.geometry,
                coords=coords,
                half_weight_boundary=half_weight_boundary,
                periodic_shift_options=geometry_periodic_shifts.get(id(structure.geometry)),
                wrap_skip_axes=stagger,
            )
            normal_axis = (
                _interface_normals(scene, structure.geometry, coords=coords)[axis]
                if polarized_direct
                else None
            )
            if electric:
                value = eps_components[axis] * float(eps_background)
                # PerturbationMedium: eps(x) = eps_base + eps_sensitivity *
                # eps_background * perturbation(x), sampled at this Yee location so
                # the zero-perturbation limit reduces exactly to the base material.
                delta = _edge_structure_perturbation_delta(
                    scene, structure, parts[0], coords, eps_background
                )
                if delta is not None:
                    value = value + delta
                sigma_value = sigma_components[axis]
            else:
                value = mu_components[axis] * float(mu_background)
                sigma_value = sigma_m_components[axis]
            material_acc, inverse_acc = _blend_edge_component(
                material_acc, inverse_acc, occupancy, normal_axis, value, polarized_direct=polarized_direct
            )
            sigma_acc = _blend_material(sigma_acc, occupancy, value=sigma_value)
        for region, density in regions:
            occupancy = _geometry_occupancy(
                scene,
                region.geometry,
                coords=coords,
                half_weight_boundary=half_weight_boundary,
                periodic_shift_options=geometry_periodic_shifts.get(id(region.geometry)),
                wrap_skip_axes=stagger,
            )
            normal_axis = (
                _interface_normals(scene, region.geometry, coords=coords)[axis]
                if polarized_direct
                else None
            )
            density_field = _sample_material_region_density(scene, region, density, coords=coords)
            if electric:
                lo, hi = region.eps_bounds
            else:
                lo, hi = region.mu_bounds
            value = lo + density_field * (hi - lo)
            material_acc, inverse_acc = _blend_edge_component(
                material_acc, inverse_acc, occupancy, normal_axis, value, polarized_direct=polarized_direct
            )
            # Regions carry no conductivity (they displace it toward zero).
            sigma_acc = _blend_material(sigma_acc, occupancy, value=0.0)
        material_sum = material_acc if material_sum is None else material_sum + material_acc
        sigma_sum = sigma_acc if sigma_sum is None else sigma_sum + sigma_acc
        if inverse_acc is not None:
            inverse_sum = inverse_acc if inverse_sum is None else inverse_sum + inverse_acc

    scale = 1.0 / float(len(offset_combos))
    material_mean = material_sum * scale
    sigma_mean = sigma_sum * scale
    if inverse_sum is not None:
        base_coords = _edge_component_coords(bases, (0.0, 0.0, 0.0))
        material_mean = _reconstruct_edge_polarized(
            scene, base_coords, material_mean, inverse_sum * scale, axis
        )
    return material_mean.contiguous(), sigma_mean.contiguous()


def _edge_native_eligible(scene, model, surface_layout):
    """Whether the scene's material families all support edge-native staggering.

    Fails closed (returns False, keeping the node->edge path) for families whose
    per-Yee-component sampling has not been validated in this step: full
    off-diagonal anisotropy (handled by the runtime's own per-edge inverse-tensor
    path), 2D sheets (node-plane conductivity rasterization), and surface-impedance
    metals (interior-masked good conductors). The dominant curved/misaligned
    dielectric and diagonal-anisotropic geometry cluster -- including
    PerturbationMedium, whose eps offset is sampled at the Yee edge -- stays
    edge-native.
    """
    if material_model_has_full_anisotropy(model):
        return False
    if _sheet_structures(scene):
        return False
    if surface_layout is not None and getattr(surface_layout, "metals", ()):
        return False
    return True


def compile_edge_material_components(
    scene,
    model,
    surface_layout,
    *,
    eps_background,
    mu_background,
    samples,
    averaging,
    region_densities,
    geometry_periodic_shifts,
):
    """Edge-native diagonal material fields for the six Yee components, or ``None``.

    Returns a dict with per-component relative permittivity (``eps``), relative
    permeability (``mu``) and static conductivities (``sigma_e`` / ``sigma_m``)
    sampled at each component's own Yee location, or ``None`` when the scene
    contains a material family that is not yet edge-native (see
    :func:`_edge_native_eligible`), in which case the runtime keeps the node->edge
    average path.
    """
    if not _edge_native_eligible(scene, model, surface_layout):
        return None
    eps = {}
    sigma_e = {}
    for component in _ELECTRIC_EDGE_COMPONENTS:
        material, sigma = _compile_edge_component(
            scene,
            component,
            eps_background=eps_background,
            mu_background=mu_background,
            averaging=averaging,
            samples=samples,
            region_densities=region_densities,
            geometry_periodic_shifts=geometry_periodic_shifts,
        )
        eps[component] = material
        sigma_e[component] = sigma
    mu = {}
    sigma_m = {}
    for component in _MAGNETIC_EDGE_COMPONENTS:
        material, sigma = _compile_edge_component(
            scene,
            component,
            eps_background=eps_background,
            mu_background=mu_background,
            averaging=averaging,
            samples=samples,
            region_densities=region_densities,
            geometry_periodic_shifts=geometry_periodic_shifts,
        )
        mu[component] = material
        sigma_m[component] = sigma
    return {"eps": eps, "sigma_e": sigma_e, "mu": mu, "sigma_m": sigma_m}


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


def _prepare_material_region_density(scene, region) -> torch.Tensor:
    """Normalize, filter, and project a region density on its native design grid."""
    if not isinstance(region.geometry, Box):
        raise ValueError(
            "MaterialRegion currently supports Box geometry only, "
            f"got {type(region.geometry).__name__}."
        )
    density = region.density.to(device=scene.device, dtype=torch.float32)
    lower, upper = region.bounds
    if not np.isclose(lower, 0.0) or not np.isclose(upper, 1.0):
        density = (density - lower) / max(upper - lower, 1e-12)
    density = density.clamp(0.0, 1.0)
    density = _filter_region_density(scene, region, density)
    return _project_region_density(region, density).clamp(0.0, 1.0)


def _sample_material_region_density(
    scene,
    region,
    density: torch.Tensor,
    *,
    coords,
) -> torch.Tensor:
    """Trilinearly sample a region density texture at physical query coordinates."""
    xx, yy, zz = coords
    center = torch.as_tensor(region.geometry.position, device=scene.device, dtype=torch.float32)
    size = torch.as_tensor(region.geometry.size, device=scene.device, dtype=torch.float32)
    normalized = (
        2.0 * (xx - center[0]) / size[0],
        2.0 * (yy - center[1]) / size[1],
        2.0 * (zz - center[2]) / size[2],
    )
    query_grid = torch.stack(normalized, dim=-1)[None, ...]
    # grid_sample uses input order (depth=z, height=y, width=x) and query
    # components (x, y, z), while public density tensors are stored (x, y, z).
    texture = density.permute(2, 1, 0)[None, None, ...]
    return F.grid_sample(
        texture,
        query_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, 0]


def _material_region_design_mask(scene, reference: torch.Tensor) -> torch.Tensor:
    """Return the nominal hard Box support retained by the public design metadata."""
    mask = torch.zeros(reference.shape, dtype=torch.bool, device=reference.device)
    for region in getattr(scene, "material_regions", ()):
        slices = _box_axis_slices(scene, region.geometry)
        if slices is not None:
            mask[slices] = True
    return mask


def _apply_material_regions(
    scene,
    model,
    region_densities,
    *,
    coords,
    averaging="arithmetic",
    half_weight_boundary=False,
    geometry_periodic_shifts=None,
    inverse_components=None,
):
    """Blend design regions through the same occupancy path as ordinary structures."""
    eps_base = {axis: model["eps_components"][axis].clone() for axis in _AXES}
    mu_base = {axis: model["mu_components"][axis].clone() for axis in _AXES}

    for region, density in zip(scene.material_regions, region_densities):
        occupancy = _geometry_occupancy(
            scene,
            region.geometry,
            coords=coords,
            half_weight_boundary=half_weight_boundary,
            periodic_shift_options=geometry_periodic_shifts.get(id(region.geometry)),
        )
        normals = (
            _interface_normals(scene, region.geometry, coords=coords)
            if averaging == "polarized"
            else None
        )
        density_field = _sample_material_region_density(
            scene,
            region,
            density,
            coords=coords,
        )
        eps_lo, eps_hi = region.eps_bounds
        mu_lo, mu_hi = region.mu_bounds
        eps_region = eps_lo + density_field * (eps_hi - eps_lo)
        mu_region = mu_lo + density_field * (mu_hi - mu_lo)
        for axis in _AXES:
            if normals is None:
                model["eps_components"][axis] = _blend_material(
                    model["eps_components"][axis], occupancy, value=eps_region
                )
                model["mu_components"][axis] = _blend_material(
                    model["mu_components"][axis], occupancy, value=mu_region
                )
            else:
                model["eps_components"][axis] = _blend_material_polarized(
                    model["eps_components"][axis],
                    occupancy,
                    normals[axis],
                    value=eps_region,
                )
                model["mu_components"][axis] = _blend_material_polarized(
                    model["mu_components"][axis],
                    occupancy,
                    normals[axis],
                    value=mu_region,
                )
            if inverse_components is not None:
                inverse_components["eps_components"][axis] = _blend_material(
                    inverse_components["eps_components"][axis],
                    occupancy,
                    value=1.0 / (eps_region + _HARMONIC_EPS),
                )
                inverse_components["mu_components"][axis] = _blend_material(
                    inverse_components["mu_components"][axis],
                    occupancy,
                    value=1.0 / (mu_region + _HARMONIC_EPS),
                )
            model["sigma_e_components"][axis] = _blend_material(
                model["sigma_e_components"][axis], occupancy, value=0.0
            )
            model["sigma_m_components"][axis] = _blend_material(
                model["sigma_m_components"][axis], occupancy, value=0.0
            )
        for pair in _OFFDIAG_AXES:
            model["eps_offdiag_components"][pair] = _blend_material(
                model["eps_offdiag_components"][pair], occupancy, value=0.0
            )
        for key in ("kerr_chi3", "chi2", "tpa_sigma", "modulation_cos", "modulation_sin"):
            model[key] = _blend_material(model[key], occupancy, value=0.0)
        _clear_dispersive_region(model, occupancy)

    model = _refresh_model_summary_aliases(model)
    model["eps_r_base"] = _component_average(eps_base)
    model["mu_r_base"] = _component_average(mu_base)
    model["eps_r_design"] = model["eps_r"] - model["eps_r_base"]
    model["mu_r_design"] = model["mu_r"] - model["mu_r_base"]
    return model


def _pec_occupancy(scene, coords=None):
    """Union (max) of soft SDF occupancies of all PEC-material structures on the node grid.

    Returns ``None`` when the scene has no PEC structure so non-PEC scenes stay
    byte-identical. Differentiable in PEC geometry through ``_geometry_occupancy``.
    """
    structures = pec_structures(scene)
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


def _reject_surface_impedance(reason: str, *, phase: str):
    """Single physically-worded rejection for unsupported surface-impedance configs.

    Every surface-impedance scope limit funnels through this one ``raise`` so the
    reason is always contextual -- it states the physical or mathematical reason the
    case is unsupported and names the phase that lifts it -- while the guard census
    counts a single capability guard for the surface-impedance boundary. Axis-aligned
    finite blocks, mid-domain double-sided plates, multiple metals, multiple
    orientations, and the generic rational surface impedance are wired into the runtime;
    the oblique/curved (conformal) and Bloch-periodic cases are converged in later
    phases.
    """
    raise NotImplementedError(
        f"Surface-impedance boundary: {reason} "
        f"The runtime is generalized to this case in {phase}; until then resolve the "
        "metal volumetrically with Material(sigma_e=...) or use Material.pec() for a "
        "lossless conductor."
    )


def _geometries_coincide(first, second) -> bool:
    """Whether two axis-aligned ``Box`` geometries occupy the same extent.

    A Phase 0 skeleton for surface-ownership overlap detection: it recognizes only the
    unambiguous case of two boxes sharing a position and size. A PEC or 2D sheet that
    partially overlaps the metal's illuminated face (different extent, same interface
    plane) is NOT detected here and passes silently; per-face conformal / partial-overlap
    ownership resolution is deferred to the Phase 1 surface layout.
    """
    if not isinstance(first, Box) or not isinstance(second, Box):
        return False
    first_center = tuple(float(value) for value in first.position)
    second_center = tuple(float(value) for value in second.position)
    first_size = tuple(float(value) for value in first.size)
    second_size = tuple(float(value) for value in second.size)
    # Scale the coincidence tolerance to the geometry's own extents (sizes and
    # positions) rather than clamping at a 1-metre absolute floor: a 1e-9-metre floor
    # is ~1e-3 relative for micron-scale photonics, which would falsely merge two
    # genuinely distinct sub-nanometre-offset boxes. Falling back to 1.0 only for the
    # fully degenerate all-zero geometry keeps the tolerance well defined.
    extent = max(
        max(first_size),
        max(second_size),
        max(abs(value) for value in first_center + second_center),
    )
    scale = extent if extent > 0.0 else 1.0
    tolerance = 1.0e-9 * scale
    return all(
        abs(first_center[axis] - second_center[axis]) <= tolerance
        and abs(first_size[axis] - second_size[axis]) <= tolerance
        for axis in range(3)
    )


def _reject_overlapping_surface_ownership(scene, surface_structures):
    """Fail closed when a surface impedance and a PEC / 2D sheet claim one interface.

    The tangential-E write on a shared interface has exactly one physical owner; a PEC
    (E_t = 0) and a surface impedance (E_t = Z_s (n x H)) on the same face are two
    contradictory owners of the same degree of freedom.
    """
    others = (*pec_structures(scene), *_sheet_structures(scene))
    if not others:
        return
    for surface in surface_structures:
        surface_geometry = getattr(surface, "geometry", None)
        for other in others:
            if _geometries_coincide(surface_geometry, getattr(other, "geometry", None)):
                _reject_surface_impedance(
                    "the same interface is claimed by both a surface impedance and a PEC "
                    "or 2D sheet, so the tangential-E write on that interface has two "
                    "contradictory owners.",
                    phase="Phase 1",
                )


@dataclass(frozen=True)
class CompiledSurfaceMetal:
    """One compiled axis-aligned metal volume carrying a surface-impedance boundary.

    ``structure_index`` is the global scene structure index. Metals are enumerated in
    ascending ``structure_index`` order, so a face's ``metal_index`` (the final key of
    :attr:`CompiledSurfaceFace.owner_rank`) realizes the deterministic owner tie-break
    for edges shared by two faces: the minimum-structure-index owner wins, matching the
    plan-07 minimum-global-edge owner discipline and the reference oracle's
    ``assemble_surface_dissipation``. ``cell_slices`` is the box's lower-inclusive /
    upper-exclusive node window per axis (the metal interior masked to a
    good-conductor termination). ``touches_lower`` / ``touches_upper`` flag whether the
    box is flush against the physical domain boundary on each axis-side (a flush face is
    a half-space against the PML, never an illuminated surface). ``material`` is the
    public surface material; ``conductivity`` is the good-conductor bulk conductivity
    for the narrowband order-0 path, or ``None`` for a generic rational model.
    """

    structure_index: int
    cell_slices: tuple[slice, slice, slice]
    touches_lower: tuple[bool, bool, bool]
    touches_upper: tuple[bool, bool, bool]
    material: object
    conductivity: float | None
    # Staircased (voxelized) metals carry a boolean node-occupancy mask instead of a
    # rectangular ``cell_slices`` window: the runtime zeroes the E-update coefficients on
    # every edge the occupancy fills (node->edge fill >= 0.5), so an arbitrary voxelized
    # conductor (a curved cylinder/sphere) terminates as a good conductor. ``None`` for an
    # axis-aligned Box metal (which uses the analytic ``cell_slices`` interior mask).
    interior_node_mask: object = None


@dataclass(frozen=True)
class CompiledSurfaceFace:
    """One axis-aligned exposed metal face (a tangential surface-impedance plane).

    ``metal_index`` indexes into :attr:`CompiledSurfaceImpedanceLayout.metals`.
    ``axis`` is the outward-normal axis; ``metal_side`` is ``"high"`` when the metal
    fills the node indices at or above ``surface_node`` (outward normal ``-axis``) and
    ``"low"`` when it fills below (outward normal ``+axis``). ``surface_node`` is the
    node plane where the tangential E is written and ``magnetic_index`` the
    vacuum-side tangential H plane one cell out. ``transverse_slices`` is the face's
    node window in the two non-normal axes (``(b_slice, c_slice)`` with
    ``b = (axis + 1) % 3``, ``c = (axis + 2) % 3``); ``full_plane`` is true when it
    spans the whole transverse cross-section (the fused-kernel fast path). ``area`` is
    the face's physical area in m^2 (the per-face dual-area sum).
    """

    metal_index: int
    axis: int
    metal_side: str
    surface_node: int
    magnetic_index: int
    transverse_slices: tuple[slice, slice]
    full_plane: bool
    area: float
    # Staircased faces carry a boolean node-plane mask over the ``(b, c)`` transverse
    # dimensions (``b = (axis + 1) % 3``, ``c = (axis + 2) % 3``) selecting exactly the
    # exposed voxel faces at ``surface_node``; the runtime reduces it to each tangential
    # E component's edge grid and writes the Leontovich value only there. ``None`` for an
    # axis-aligned Box face (which owns a rectangular ``transverse_slices`` window).
    transverse_mask: object = None

    @property
    def owner_rank(self) -> tuple[int, int, int, int, int]:
        """Deterministic total order for shared-edge ownership (lower wins).

        ``metal_index`` is the final tie-break so two coincident same-material faces
        from distinct abutting metals never tie: because ``metal_index`` is assigned in
        ascending scene-structure order and ``faces`` is sorted descending (the minimum
        owner sorts last and last-writer-wins), the minimum-structure-index metal owns a
        shared edge, matching the plan-07 minimum-global-edge owner discipline instead of
        relying on enumeration order.
        """
        b = (self.axis + 1) % 3
        return (
            self.axis,
            self.surface_node,
            b,
            int(self.transverse_slices[0].start),
            self.metal_index,
        )


@dataclass(frozen=True)
class CompiledSurfaceImpedanceLayout:
    """Axis-aligned exposed-face layout for the surface-impedance boundary.

    Replaces the single-plane v1 SIBC descriptor: it enumerates every illuminated
    axis-aligned face of every surface-impedance metal in the scene (finite blocks,
    mid-domain double-sided plates, multiple metals, and multiple orientations), with a
    deterministic ownership order for edges shared at box corners and overlap rejection
    for contradictory owners on one interface. ``faces`` is sorted by
    :attr:`CompiledSurfaceFace.owner_rank`; ``total_area`` is the summed face area.
    """

    metals: tuple[CompiledSurfaceMetal, ...]
    faces: tuple[CompiledSurfaceFace, ...]
    total_area: float

    def __bool__(self) -> bool:
        return bool(self.faces)


def _surface_metal_conductivity(material) -> float | None:
    """Good-conductor bulk conductivity for the narrowband order-0 path, else ``None``.

    A ``LossyMetalMedium`` without a broadband ``frequency_range`` is realized as an
    order-0 (pure-resistance) surface evaluated at the operating frequency; a generic
    ``SurfaceImpedanceMedium`` (or a future broadband metal) uses the rational ADE and
    reports ``None`` here.
    """
    if bool(getattr(material, "is_lossy_metal", False)) and getattr(
        material, "frequency_range", None
    ) is None:
        return float(material.conductivity)
    return None


def _surface_impedance_structures(scene):
    """Enabled structures carrying any surface-impedance material (metal or rational)."""
    structures = []
    for index, structure in enumerate(scene.structures):
        if not getattr(structure, "enabled", True):
            continue
        material = _structure_material(structure)
        if material is None:
            continue
        if bool(getattr(material, "is_lossy_metal", False)) or bool(
            getattr(material, "is_surface_impedance", False)
        ):
            structures.append((index, structure))
    return structures


def _axis_span(nodes, axis_slice: slice) -> float:
    """Physical extent (m) spanned by a cell-node window, clamped to the node grid.

    A box larger than the domain yields a slice stop equal to the node count, so the
    span is clamped to the physical grid ends (the face never extends past the domain).
    """
    last = len(nodes) - 1
    start = max(0, min(int(axis_slice.start), last))
    stop = max(0, min(int(axis_slice.stop), last))
    return float(nodes[stop] - nodes[start])


def _axis_face_area(scene, b_axis: int, b_slice: slice, c_axis: int, c_slice: slice) -> float:
    """Physical area (m^2) of an axis-aligned face over its transverse node window."""
    nodes = (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64)
    return _axis_span(nodes[b_axis], b_slice) * _axis_span(nodes[c_axis], c_slice)


def _transverse_full_plane(scene, b_axis: int, b_slice: slice, c_axis: int, c_slice: slice) -> bool:
    """Whether a face spans the entire transverse cross-section (fused-kernel path).

    A metal that covers every transverse cell owns the whole surface E plane, so the
    fused whole-plane kernel is exact. Node counts are ``scene.N*``; the cell count is
    one fewer, and a box exactly filling the domain (or overflowing it) reaches it.
    """
    node_counts = (scene.Nx, scene.Ny, scene.Nz)
    b_cells = node_counts[b_axis] - 1
    c_cells = node_counts[c_axis] - 1
    return (
        b_slice.start == 0
        and b_slice.stop >= b_cells
        and c_slice.start == 0
        and c_slice.stop >= c_cells
    )


def _is_axis_aligned_box(geometry) -> bool:
    """Whether a geometry is an unrotated (axis-aligned) Box."""
    if not isinstance(geometry, Box):
        return False
    rotation = getattr(geometry, "rotation", None)
    if rotation is None:
        return True
    quaternion = np.asarray(rotation.detach().cpu(), dtype=np.float64)
    return bool(np.allclose(np.abs(quaternion), (1.0, 0.0, 0.0, 0.0), atol=1.0e-6))


def _select_node_plane(occupancy, axis: int, index: int):
    """The 2D node plane of ``occupancy`` at ``index`` along ``axis`` (ascending order)."""
    return occupancy[_axis_index_tuple(3, axis, index)]


def _axis_index_tuple(ndim, axis, index_or_slice):
    selector = [slice(None)] * ndim
    selector[axis] = index_or_slice
    return tuple(selector)


def _mean_node_spacing(nodes) -> float:
    array = np.asarray(nodes, dtype=np.float64)
    if array.size < 2:
        return 0.0
    return float((array[-1] - array[0]) / (array.size - 1))


def _compile_voxel_surface_metal(
    scene, structure_index, geometry, material, metal_index, physical_bounds, tolerances
):
    """Staircase a voxelized (possibly curved) good-conductor into exposed faces.

    A node belongs to the metal when its center is inside the geometry
    (``signed_distance <= 0``): the staircase approximation of an arbitrary conductor.
    Every axis-aligned voxel face on the metal/vacuum boundary is an exposed
    surface-impedance face; faces of a given orientation at one node plane are grouped
    into a boolean transverse node mask. A voxel face is illuminated only when its
    vacuum-side node lies inside the physical domain (a face backing onto the PML is a
    half-space, never illuminated), matching the Box path's flush-face exclusion. Only the
    narrowband good-conductor (order-0 resistance) surface is wired for the staircase; the
    per-edge rational ADE stays Box-only.
    """
    conductivity = _surface_metal_conductivity(material)
    if conductivity is None:
        _reject_surface_impedance(
            "requires a narrowband good-conductor metal for a staircased (voxelized) "
            "curved surface; the per-edge rational surface ADE is wired only for "
            "axis-aligned Box faces.",
            phase="Phase 2",
        )
    occupancy = geometry.signed_distance(scene.X, scene.Y, scene.Z) <= 0.0
    if not bool(occupancy.any().item()):
        _reject_surface_impedance(
            "requires the metal to lie inside the grid; the surface covers no cells.",
            phase="Phase 1",
        )
    occ = occupancy.detach().to("cpu")
    nodes = (scene.x_nodes64, scene.y_nodes64, scene.z_nodes64)
    physical_node = []
    for axis in range(3):
        coords = np.asarray(nodes[axis], dtype=np.float64)
        lo = float(physical_bounds[axis][0]) - tolerances[axis]
        hi = float(physical_bounds[axis][1]) + tolerances[axis]
        physical_node.append((coords >= lo) & (coords <= hi))

    faces: list[CompiledSurfaceFace] = []
    for axis in range(3):
        b = (axis + 1) % 3
        c = (axis + 2) % 3
        naxes = tuple(a for a in range(3) if a != axis)
        cell_area = _mean_node_spacing(nodes[naxes[0]]) * _mean_node_spacing(nodes[naxes[1]])
        n_along = occ.shape[axis]
        # Half-cell surface-node placement convention (documented asymmetry, not a bug).
        # The physical metal/vacuum boundary between nodes p-1 and p is a half-cell wide
        # region on the staggered Yee grid; the tangential-E surface node is written at
        # node index p in BOTH orientations, but that index sits on OPPOSITE sides of the
        # boundary depending on the outward normal:
        #   * -axis-normal (low-side) face: metal at p, vacuum at p-1 -> E written at the
        #     first metal node p (surface_node=p, paired H at p-1).
        #   * +axis-normal (high-side) face: metal at p-1, vacuum at p -> E written at the
        #     first vacuum node p (surface_node=p, paired H at p).
        # So a face whose normal points toward +axis places the effective surface a half
        # cell farther into the vacuum than a -axis-facing face at the same physical
        # interface. On a flat axis-aligned plate this is exact (Yee symmetry); on a
        # staircased CURVED conductor the two step orientations are offset by up to one
        # cell relative to the true surface, which is part of the grid- and
        # R/delta-independent ~18% absorbed-power under-prediction documented in
        # docs/assessments/g4-sibc-oblique-acceptance-2026-07-21.md (a first-order
        # boundary-on-a-staircased-curve systematic, not an implementation error; a
        # curvature-corrected surface impedance is the future refinement).
        for p in range(1, n_along):
            plane_metal = _select_node_plane(occ, axis, p)
            plane_vacuum = _select_node_plane(occ, axis, p - 1)
            # Low-side face: node p is metal, node p-1 is vacuum (outward normal -axis).
            if physical_node[axis][p - 1]:
                mask = plane_metal & (~plane_vacuum)
                if bool(mask.any().item()):
                    faces.append(
                        _make_voxel_face(
                            metal_index, axis, "high", p, p - 1, mask, naxes, cell_area
                        )
                    )
            # High-side face: node p-1 is metal, node p is vacuum (outward normal +axis).
            # The paired vacuum-side H sits at index p on the axis-reduced H grid, so the
            # face is only valid when that index exists (p <= n_along - 2); the boundary
            # node is otherwise a PML node and is excluded by the physical-node test.
            if p <= n_along - 2 and physical_node[axis][p]:
                mask = plane_vacuum & (~plane_metal)
                if bool(mask.any().item()):
                    faces.append(
                        _make_voxel_face(
                            metal_index, axis, "low", p, p, mask, naxes, cell_area
                        )
                    )
    if not faces:
        _reject_surface_impedance(
            "requires a vacuum region in front of the metal; the voxelized conductor "
            "exposes no illuminated face inside the physical domain.",
            phase="Phase 1",
        )
    metal = CompiledSurfaceMetal(
        structure_index=int(structure_index),
        cell_slices=tuple(slice(0, occ.shape[axis]) for axis in range(3)),
        touches_lower=(False, False, False),
        touches_upper=(False, False, False),
        material=material,
        conductivity=conductivity,
        interior_node_mask=occupancy.to(torch.float32),
    )
    return metal, faces


def _make_voxel_face(metal_index, axis, side, surface_node, magnetic_index, mask, naxes, cell_area):
    """Build one staircased face from a boolean node-plane mask in ``naxes`` order.

    ``mask`` is indexed by the two non-normal axes in ascending order (``naxes``), the same
    layout as a field tensor's transverse plane, so the runtime reduces it to each
    tangential E component's edge grid without any axis reordering. ``transverse_slices`` is
    a full-plane placeholder used only by the deterministic owner ordering; the boolean
    ``transverse_mask`` is the true footprint.
    """
    mask = mask.contiguous()
    nb, nc = int(mask.shape[0]), int(mask.shape[1])
    area = float(mask.sum().item()) * float(cell_area)
    return CompiledSurfaceFace(
        metal_index=metal_index,
        axis=axis,
        metal_side=side,
        surface_node=int(surface_node),
        magnetic_index=int(magnetic_index),
        transverse_slices=(slice(0, nb), slice(0, nc)),
        full_plane=False,
        area=area,
        transverse_mask=mask,
    )


def compile_surface_impedance_layout(scene):
    """Extract every axis-aligned exposed surface-impedance face in the scene.

    The surface-impedance boundary replaces the resolved skin-depth interior of a metal
    with the first-order tangential relation ``E_t = Z_s(omega) * (n x H)`` on the
    metal's illuminated faces. This walks every enabled surface-impedance metal, rejects
    the cases a later phase owns (rotated/curved geometry, Bloch runs) through the single
    ``_reject_surface_impedance`` funnel, masks each metal interior to a good-conductor
    termination, and enumerates the exposed faces: a face on the box's low/high side of
    an axis is illuminated when the box does not touch the physical domain boundary on
    that side (a flush face backs onto the PML and is not illuminated). Finite blocks,
    mid-domain double-sided plates, multiple metals, and multiple orientations are all
    supported; the runtime consumes the returned :class:`CompiledSurfaceImpedanceLayout`
    (``fdtd/runtime/materials.py`` and ``fdtd/runtime/stepping.py``). Returns an empty
    layout when the scene holds no surface-impedance structure.
    """
    indexed = _surface_impedance_structures(scene)
    if not indexed:
        return CompiledSurfaceImpedanceLayout(metals=(), faces=(), total_area=0.0)

    surface_structures = [structure for _, structure in indexed]
    _reject_overlapping_surface_ownership(scene, surface_structures)

    physical_bounds = scene.domain.bounds
    tolerances = tuple(
        1.0e-6 * max(float(upper) - float(lower), 1.0)
        for lower, upper in physical_bounds
    )

    metals: list[CompiledSurfaceMetal] = []
    faces: list[CompiledSurfaceFace] = []
    for structure_index, structure in indexed:
        material = _structure_material(structure)
        geometry = structure.geometry
        if scene.boundary.uses_kind("bloch"):
            _reject_surface_impedance(
                "uses a real-valued surface update; a Bloch-periodic run carries complex "
                "phase-shifted fields for which the real surface update is undefined.",
                phase="Phase 3",
            )
        if not isinstance(geometry, Box):
            # Any non-Box conductor (a curved cylinder/sphere) is staircased: the
            # occupancy grid supplies the axis-aligned exposed faces directly. True
            # oblique/conformal (non-staircase) SIBC remains the deferred gap.
            metal, voxel_faces = _compile_voxel_surface_metal(
                scene, structure_index, geometry, material, len(metals),
                physical_bounds, tolerances,
            )
            metals.append(metal)
            faces.extend(voxel_faces)
            continue
        if not _is_axis_aligned_box(geometry):
            _reject_surface_impedance(
                "requires an axis-aligned (unrotated) surface; a rotated Box presents an "
                "oblique face whose local normal is not grid-aligned, which is the deferred "
                "conformal/oblique case (staircasing a rotated Box is not wired).",
                phase="Phase 2",
            )
        slices = _box_axis_slices(scene, geometry)
        if slices is None:
            _reject_surface_impedance(
                "requires the metal to lie inside the grid; the surface covers no cells.",
                phase="Phase 1",
            )
        center = tuple(float(value) for value in geometry.position)
        size = tuple(float(value) for value in geometry.size)
        lower = tuple(center[axis] - 0.5 * size[axis] for axis in range(3))
        upper = tuple(center[axis] + 0.5 * size[axis] for axis in range(3))
        # Domain.bounds stays physical when PreparedScene appends PML nodes. Surface
        # coverage is a physical geometry contract, so a box face flush against the
        # physical boundary backs onto the PML half-space and is never illuminated.
        touches_lower = tuple(
            lower[axis] <= float(physical_bounds[axis][0]) + tolerances[axis]
            for axis in range(3)
        )
        touches_upper = tuple(
            upper[axis] >= float(physical_bounds[axis][1]) - tolerances[axis]
            for axis in range(3)
        )
        fills_every_axis = all(
            touches_lower[axis] and touches_upper[axis] for axis in range(3)
        )
        if fills_every_axis:
            _reject_surface_impedance(
                "requires a vacuum region in front of the metal; the box fills the domain "
                "on every axis and exposes no illuminated face.",
                phase="Phase 1",
            )
        conductivity = _surface_metal_conductivity(material)
        if conductivity is None:
            impedance = getattr(material, "impedance", None)
            if impedance is not None and int(getattr(impedance, "port_count", 1)) != 1:
                _reject_surface_impedance(
                    "carries a tangential 2x2 surface response; the per-edge ADE steps a "
                    "scalar impedance, and the cross-polarized 2x2 tangential coupling is "
                    "a distinct local update.",
                    phase="Phase 2",
                )
        metal_index = len(metals)
        metals.append(
            CompiledSurfaceMetal(
                structure_index=int(structure_index),
                cell_slices=slices,
                touches_lower=touches_lower,
                touches_upper=touches_upper,
                material=material,
                conductivity=conductivity,
            )
        )
        for axis in range(3):
            b = (axis + 1) % 3
            c = (axis + 2) % 3
            b_slice = slices[b]
            c_slice = slices[c]
            area = _axis_face_area(scene, b, b_slice, c, c_slice)
            full_plane = _transverse_full_plane(scene, b, b_slice, c, c_slice)
            # Low face (metal fills at/above surface_node): illuminated iff the box does
            # not sit flush against the physical low boundary on this axis.
            if not touches_lower[axis]:
                start = int(slices[axis].start)
                faces.append(
                    CompiledSurfaceFace(
                        metal_index=metal_index,
                        axis=axis,
                        metal_side="high",
                        surface_node=start,
                        magnetic_index=start - 1,
                        transverse_slices=(b_slice, c_slice),
                        full_plane=full_plane,
                        area=area,
                    )
                )
            # High face (metal fills below surface_node): illuminated iff not flush
            # against the physical high boundary on this axis.
            if not touches_upper[axis]:
                stop = int(slices[axis].stop)
                faces.append(
                    CompiledSurfaceFace(
                        metal_index=metal_index,
                        axis=axis,
                        metal_side="low",
                        surface_node=stop,
                        magnetic_index=stop,
                        transverse_slices=(b_slice, c_slice),
                        full_plane=full_plane,
                        area=area,
                    )
                )

    _reject_conflicting_surface_faces(metals, faces)
    # Sort by deterministic owner rank so a shared corner edge has a single owner: the
    # minimum-rank face writes last (last-write-wins), making its Leontovich value the
    # deterministic owner of the shared tangential edge.
    faces.sort(key=lambda face: face.owner_rank, reverse=True)
    total_area = float(sum(face.area for face in faces))
    return CompiledSurfaceImpedanceLayout(
        metals=tuple(metals), faces=tuple(faces), total_area=total_area
    )


def _reject_conflicting_surface_faces(metals, faces):
    """Fail closed when two metals with different materials claim one interface plane.

    Two coincident faces (same axis and surface node, overlapping transverse window)
    from different metals with different surface materials are two contradictory owners
    of the same tangential-E degree of freedom. Identical materials on a shared corner
    edge are resolved by the deterministic owner rank, not rejected.
    """
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            first, second = faces[i], faces[j]
            if first.axis != second.axis or first.surface_node != second.surface_node:
                continue
            if not _faces_transverse_overlap(first, second):
                continue
            mat_first = metals[first.metal_index].material
            mat_second = metals[second.metal_index].material
            if mat_first is not mat_second:
                _reject_surface_impedance(
                    "the same interface plane is claimed by two surface-impedance "
                    "metals with different materials, so its tangential-E write has "
                    "two contradictory owners.",
                    phase="Phase 2",
                )


def _faces_transverse_overlap(first, second) -> bool:
    """Whether two coincident-plane faces share any transverse footprint.

    Box faces overlap by rectangular slice intersection; staircased faces by boolean
    mask intersection (both masks are in the same ascending non-normal-axis node order).
    A mixed Box/staircase pair uses the mask restricted to the Box's slice window, which is
    conservative (it never misses a genuine overlap) so the fail-closed owner guard holds.
    """
    mask_first = getattr(first, "transverse_mask", None)
    mask_second = getattr(second, "transverse_mask", None)
    if mask_first is not None and mask_second is not None:
        return bool((mask_first & mask_second).any().item())
    if mask_first is None and mask_second is None:
        b_first, c_first = first.transverse_slices
        b_second, c_second = second.transverse_slices
        return (
            b_first.start < b_second.stop
            and b_second.start < b_first.stop
            and c_first.start < c_second.stop
            and c_second.start < c_first.stop
        )
    mask_face, slice_face = (first, second) if mask_first is not None else (second, first)
    b_slice, c_slice = slice_face.transverse_slices
    return bool(mask_face.transverse_mask[b_slice, c_slice].any().item())


def compile_material_model(
    scene,
    eps_background=1.0,
    mu_background=1.0,
    subpixel=None,
):
    surface_layout = compile_surface_impedance_layout(scene)
    samples, averaging, pec_mode = _resolve_subpixel(subpixel)
    layout = _build_dispersive_layout(scene)
    region_densities = tuple(
        _prepare_material_region_density(scene, region)
        for region in getattr(scene, "material_regions", ())
    )
    geometries = [structure.geometry for structure in _bulk_structures(scene)]
    geometries.extend(region.geometry for region in getattr(scene, "material_regions", ()))
    geometry_periodic_shifts = {
        id(geometry): _static_periodic_shift_options(scene, geometry)
        for geometry in geometries
    }
    if samples == (1, 1, 1):
        model = _compile_material_sample(
            scene,
            layout,
            eps_background=eps_background,
            mu_background=mu_background,
            region_densities=region_densities,
            averaging=averaging,
            geometry_periodic_shifts=geometry_periodic_shifts,
        )
        if region_densities:
            model["design_mask"] = _material_region_design_mask(scene, model["eps_r"])
        model = _apply_sheet_structures(scene, model)
        model["pec_occupancy"] = _pec_occupancy(scene)
        model["pec_mode"] = pec_mode
        model["surface_impedance"] = surface_layout
        model["edge_components"] = compile_edge_material_components(
            scene,
            model,
            surface_layout,
            eps_background=eps_background,
            mu_background=mu_background,
            samples=samples,
            averaging=averaging,
            region_densities=region_densities,
            geometry_periodic_shifts=geometry_periodic_shifts,
        )
        return model

    collect_inverse = averaging == "polarized"
    if collect_inverse:
        _validate_polarized_materials(scene)
    accum = _new_material_model(scene, layout, eps_fill=0.0, mu_fill=0.0)
    inverse_accum = None
    if collect_inverse:
        shape = (scene.Nx, scene.Ny, scene.Nz)
        inverse_accum = {
            "eps_components": _new_component_field(shape, fill_value=0.0, device=scene.device),
            "mu_components": _new_component_field(shape, fill_value=0.0, device=scene.device),
        }
    if region_densities:
        accum["eps_r_base"] = torch.zeros_like(accum["eps_r"])
        accum["mu_r_base"] = torch.zeros_like(accum["mu_r"])
        accum["eps_r_design"] = torch.zeros_like(accum["eps_r"])
        accum["mu_r_design"] = torch.zeros_like(accum["mu_r"])
    sample_offsets = _sample_offsets(scene, samples)
    for sample_offset in sample_offsets:
        compiled_sample = _compile_material_sample(
            scene,
            layout,
            eps_background=eps_background,
            mu_background=mu_background,
            region_densities=region_densities,
            sample_offset=sample_offset,
            averaging="arithmetic" if collect_inverse else averaging,
            half_weight_boundary=True,
            geometry_periodic_shifts=geometry_periodic_shifts,
            collect_inverse=collect_inverse,
        )
        if collect_inverse:
            sample, inverse_sample = compiled_sample
        else:
            sample = compiled_sample
        for axis in _AXES:
            accum["eps_components"][axis] += sample["eps_components"][axis]
            accum["mu_components"][axis] += sample["mu_components"][axis]
            accum["sigma_e_components"][axis] += sample["sigma_e_components"][axis]
            accum["sigma_m_components"][axis] += sample["sigma_m_components"][axis]
            if inverse_accum is not None:
                inverse_accum["eps_components"][axis] += inverse_sample["eps_components"][axis]
                inverse_accum["mu_components"][axis] += inverse_sample["mu_components"][axis]
        for pair in _OFFDIAG_AXES:
            accum["eps_offdiag_components"][pair] += sample["eps_offdiag_components"][pair]
        accum["kerr_chi3"] += sample["kerr_chi3"]
        accum["chi2"] += sample["chi2"]
        accum["tpa_sigma"] += sample["tpa_sigma"]
        accum["modulation_cos"] += sample["modulation_cos"]
        accum["modulation_sin"] += sample["modulation_sin"]
        if region_densities:
            accum["eps_r_base"] += sample["eps_r_base"]
            accum["mu_r_base"] += sample["mu_r_base"]
            accum["eps_r_design"] += sample["eps_r_design"]
            accum["mu_r_design"] += sample["mu_r_design"]
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
        if inverse_accum is not None:
            inverse_accum["eps_components"][axis] *= scale
            inverse_accum["mu_components"][axis] *= scale
    if inverse_accum is not None:
        accum["eps_components"] = _reconstruct_sampled_polarized_components(
            scene,
            accum["eps_components"],
            inverse_accum["eps_components"],
        )
        accum["mu_components"] = _reconstruct_sampled_polarized_components(
            scene,
            accum["mu_components"],
            inverse_accum["mu_components"],
        )
    for pair in _OFFDIAG_AXES:
        accum["eps_offdiag_components"][pair] *= scale
    accum["kerr_chi3"] *= scale
    accum["chi2"] *= scale
    accum["tpa_sigma"] *= scale
    accum["modulation_cos"] *= scale
    accum["modulation_sin"] *= scale
    if region_densities:
        accum["eps_r_base"] *= scale
        accum["mu_r_base"] *= scale
        accum["eps_r_design"] *= scale
        accum["mu_r_design"] *= scale
        accum["design_mask"] = _material_region_design_mask(scene, accum["eps_r"])
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
    model = _refresh_model_summary_aliases(accum)
    if region_densities:
        model["eps_r_design"] = model["eps_r"] - model["eps_r_base"]
        model["mu_r_design"] = model["mu_r"] - model["mu_r_base"]
    model = _apply_sheet_structures(scene, model)
    model["pec_occupancy"] = _pec_occupancy(scene)
    model["pec_mode"] = pec_mode
    model["surface_impedance"] = surface_layout
    model["edge_components"] = compile_edge_material_components(
        scene,
        model,
        surface_layout,
        eps_background=eps_background,
        mu_background=mu_background,
        samples=samples,
        averaging=averaging,
        region_densities=region_densities,
        geometry_periodic_shifts=geometry_periodic_shifts,
    )
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
