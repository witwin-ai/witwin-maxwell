from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import copy
from enum import Enum
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from witwin.core import Box, Sphere, Structure
from witwin.core.material import VACUUM_PERMITTIVITY

from ..media import CustomPole, DiagonalTensor3, DrudePole, LorentzPole, ModulationSpec, PerturbationMedium, Tensor3x3
from .structures import pec_structures, pmc_structures
from .trilinear import _sample_trilinear_border
from .planar_pec import attach_planar_pec, compile_planar_pec, validate_legacy_thin_pec, volume_pec_structures
from .subpixel import (
    _HARMONIC_EPS,
    _NORMAL_EPS,
    _blend_material_polarized,
    _field_gradients,
    SubsampleFillGeometry,
    _interface_normals,
    _mesh_fill_cache,
    _inverse_diagonal_blend,
    _kottke_edge_diagonal,
    _reconstruct_sampled_kottke_tensor,
    _reconstruct_sampled_polarized_inverse,
    _subsample_spacing,
    _subsample_fill_fraction,
    _uniform_axis_spacing,
)

_AXES = ("x", "y", "z")
_OFFDIAG_AXES = ("xy", "xz", "yz")


def _material_dtype(scene):
    return getattr(scene, "_material_dtype", torch.float32)


def _scalar_tensor(value, *, device, dtype=torch.float32):
    from ._scalar_cache import scalar_tensor
    return scalar_tensor(value, device=device, dtype=dtype)


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


def _structure_is_pmc(structure) -> bool:
    material = _structure_material(structure)
    return material is not None and bool(getattr(material, "is_pmc", False))


def _structure_is_sheet(structure) -> bool:
    material = _structure_material(structure)
    return material is not None and bool(getattr(material, "is_medium2d", False))


def _bulk_structures(scene):
    """Enabled volumetric structures: everything except PEC markers and 2D sheets."""
    return [
        structure
        for structure in _sorted_structures(scene)
        if not _structure_is_pec(structure) and not _structure_is_sheet(structure)
        and not bool(getattr(_structure_material(structure), "is_copper_foil", False))
        and not _structure_is_pmc(structure)
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
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor.")
        scalar = value
    else:
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
    if getattr(material,'has_tensor_nonlinearity',False):
        raise NotImplementedError('Tensor nonlinear susceptibility requires a tensor constitutive runtime')
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


def _constant_zero_field(shape, device):
    """Read-only broadcast storage until a material blend activates this channel."""
    value = torch.zeros((), device=device, dtype=torch.float32).expand(shape)
    value._material_constant_zero_version = None if torch.is_inference(value) else value._version
    return value


def _is_constant_zero(value):
    version = getattr(value, "_material_constant_zero_version", None)
    return version is not None and not torch.is_inference(value) and value._version == version


def _add_material_samples(first, second):
    if _is_constant_zero(first) and _is_constant_zero(second):
        return first
    return first + second


def _scale_material_sample(value, scale):
    return value if _is_constant_zero(value) else value * scale


def _new_material_model(scene, layout, *, eps_fill, mu_fill):
    shape = (scene.Nx, scene.Ny, scene.Nz)
    device = scene.device
    model = {
        "eps_components": _new_component_field(shape, fill_value=eps_fill, device=device, dtype=_material_dtype(scene)),
        "eps_offdiag_components": {
            pair: _constant_zero_field(shape, device)
            for pair in _OFFDIAG_AXES
        },
        "mu_components": _new_component_field(shape, fill_value=mu_fill, device=device, dtype=_material_dtype(scene)),
        "sigma_e_components": _new_component_field(shape, fill_value=0.0, device=device, dtype=_material_dtype(scene)),
        "sigma_m_components": _new_component_field(shape, fill_value=0.0, device=device, dtype=_material_dtype(scene)),
        "kerr_chi3": _constant_zero_field(shape, device),
        "chi2": _constant_zero_field(shape, device),
        "tpa_sigma": _constant_zero_field(shape, device),
        # Space-time modulation quadrature fields: amplitude * cos(phase) and
        # amplitude * sin(phase). This basis blends linearly across overlapping
        # structures, unlike the raw phase itself.
        "modulation_cos": _constant_zero_field(shape, device),
        "modulation_sin": _constant_zero_field(shape, device),
        # Per-node modulation angular frequency (0 where unmodulated). A Scene may
        # hold several distinct modulation frequencies at once; each modulated
        # structure stamps its own frequency onto the cells it covers, so the
        # single-frequency-per-scene restriction no longer applies.
        "modulation_omega": _constant_zero_field(shape, device),
    }
    if any(getattr(s.material, "modulation", None) is not None for s in scene.structures):
        # Positive IDs identify a single fully occupied material; -1 marks a
        # mixed window. Ownership is used only to retain interior bit identity.
        model["modulation_owner"] = torch.zeros(shape, device=device, dtype=torch.int64)
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
                "weight": torch.zeros(shape, device=device, dtype=_material_dtype(scene)),
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
    offdiag = model.get("pole_tensor_nodes", {}).get("offdiag", model.get("eps_offdiag_components"))
    if not offdiag:
        return False
    return any(torch.any(offdiag[pair] != 0).item() for pair in _OFFDIAG_AXES)


def _blend_material(tensor, occupancy, *, value):
    # Preserve the ordinary graph when occupancy is trainable, even when the
    # derivative happens to be zero. Active channels always materialize normally.
    if (_is_constant_zero(tensor) and isinstance(value, (int, float)) and value == 0
            and not occupancy.requires_grad):
        return tensor
    value_tensor = _scalar_tensor(value, device=tensor.device, dtype=tensor.dtype)
    return (1.0 - occupancy) * tensor + occupancy * value_tensor


def _resolve_subpixel(subpixel):
    """Return ``(samples, averaging, pec)`` from a SubpixelSpec or the None default."""
    if subpixel is None:
        from ..scene import SubpixelSpec
        subpixel = SubpixelSpec()
    return tuple(int(v) for v in subpixel.samples), subpixel.averaging, subpixel.pec


def _validate_polarized_materials(scene):
    for structure in _bulk_structures(scene):
        eps_offdiag = _static_structure_material(structure)[2]
        if any(value != 0.0 for value in eps_offdiag.values()):
            raise NotImplementedError(
                "Polarized (Kottke) subpixel averaging is not implemented for full "
                "(off-diagonal) anisotropic permittivity; use arithmetic subpixel averaging."
            )


def _anisotropic_epsilon(eps, offdiag):
    """Whether a constituent needs the existing diagonal Kottke treatment."""
    isotropic = (eps["x"] is eps["y"] is eps["z"]) or (eps["x"] == eps["y"] == eps["z"])
    return not isotropic or any(v != 0.0 for v in offdiag.values())


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
    dx = torch.as_tensor(scene.dx_dual64, device=scene.device, dtype=_material_dtype(scene))
    dy = torch.as_tensor(scene.dy_dual64, device=scene.device, dtype=_material_dtype(scene))
    dz = torch.as_tensor(scene.dz_dual64, device=scene.device, dtype=_material_dtype(scene))
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
    fill_spacing=None,
    surface_inclusive=False,
):
    if fill_spacing is not None:
        # Multi-sample dielectric fill: exact coverage ramp per sub-sample.
        geometry = SubsampleFillGeometry(geometry, fill_spacing)
        half_weight_boundary = False
    if coords is None and surface_inclusive:
        coords = torch.meshgrid(*(torch.tensor(getattr(scene, f"{a}_nodes64"),
            device=scene.device, dtype=torch.float64) for a in _AXES), indexing="ij")
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    resolved_beta = _geometry_beta(scene) if beta is None else beta
    closed_box_bounds = None
    if surface_inclusive and isinstance(geometry, Box):
        rotation = geometry.rotation
        if torch.equal(rotation.abs(), rotation.new_tensor((1., 0., 0., 0.))):
            # Form the declared bounds in the geometry's own dtype, then compare
            # in float64. Subtracting the centre first can lose a boundary ulp.
            closed_box_bounds = []
            half_size = geometry.size / 2
            for bound in (geometry.position - half_size,
                          geometry.position + half_size):
                declared = bound.to(device=xx.device, dtype=torch.float64)
                if bound.dtype == torch.float32:
                    # For f = fl32(p +/- fl32(s/2)), declaration rounding
                    # contributes ULP32(p)/2 + ULP32(s)/4. Division by two
                    # is exact except at subnormal underflow; include its
                    # actual residual |fl32(s/2) - s/2|. Final +/- rounding
                    # contributes ULP32(f)/2. Use the larger adjacent spacing
                    # at binade boundaries, and sum in float64 so half an ULP
                    # cannot underflow. Snap only within this propagated bound.
                    values = torch.stack((geometry.position, geometry.size, bound))
                    ulp = torch.maximum(
                        torch.nextafter(values, torch.full_like(values, float("inf"))) - values,
                        values - torch.nextafter(values, torch.full_like(values, -float("inf"))),
                    ).to(device=xx.device, dtype=torch.float64)
                    division_error = (half_size.to(device=xx.device, dtype=torch.float64)
                                      - geometry.size.to(device=xx.device, dtype=torch.float64) / 2).abs()
                    roundoff = ulp[0] / 2 + ulp[1] / 4 + division_error + ulp[2] / 2
                    snapped = []
                    for i, axis in enumerate(_AXES):
                        nodes = torch.tensor(getattr(scene, f"{axis}_nodes64"),
                                             device=xx.device, dtype=torch.float64)
                        nodes = torch.cat((nodes, .5 * (nodes[:-1] + nodes[1:]))).sort().values
                        node = nodes[(nodes - declared[i]).abs().argmin()]
                        snapped.append(torch.where((node - declared[i]).abs() <= roundoff[i], node, declared[i]))
                    declared = torch.stack(snapped)
                closed_box_bounds.append(declared)

    def _sample(sample_x, sample_y, sample_z):
        if surface_inclusive:
            if closed_box_bounds is not None:
                lower, upper = closed_box_bounds
                inside = ((sample_x >= lower[0]) & (sample_x <= upper[0])
                          & (sample_y >= lower[1]) & (sample_y <= upper[1])
                          & (sample_z >= lower[2]) & (sample_z <= upper[2]))
            else:
                inside = geometry.signed_distance(sample_x, sample_y, sample_z) <= 0
            return inside.to(_material_dtype(scene))
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

    Scalar poles and uniform fixed custom grids pass through with ``amplitude=None``.
    Spatially varying poles are lowered to their peak-strength scalar ``reference_pole()`` and a
    per-node amplitude field in ``[0, 1]`` rasterized from the structure's Box
    extent; the amplitude multiplies the structure occupancy when the pole
    weight grid is assembled, so ``weight * chi_ref(omega)`` reproduces the
    spatially-varying susceptibility exactly.
    """
    if isinstance(pole, CustomPole):
        reference = pole.reference_pole()
        strength = pole.plasma_frequency if isinstance(reference, DrudePole) else pole.delta_eps
        uniform = bool((strength == strength.flatten()[0]).all())
        # Uniform fixed grids must lower exactly like their scalar equivalents.
        # Keep trainable amplitudes connected to the material-value pullback.
        field = None
        uniform_amplitude = None
        if not uniform or strength.requires_grad:
            amplitude = pole.amplitude()
            field = _box_parameter_field(
                scene,
                structure.geometry,
                amplitude,
                name=f"{type(pole).__name__} parameter grid",
            )
            if uniform:
                uniform_amplitude = amplitude
        elif not isinstance(structure.geometry, Box):
            raise ValueError(f"{type(pole).__name__} parameter grid currently supports Box structure geometry only.")
        return {
            "pole": reference,
            "amplitude": field,
            "graded": not uniform,
            "uniform_amplitude": uniform_amplitude,
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


def _structure_modulation_values(scene, structure, coords=None):
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
        amplitude = (_box_parameter_field(scene, structure.geometry, amplitude, name="ModulationSpec.amplitude")
                     if coords is None else _sample_box_parameter_field(scene, structure.geometry, amplitude, coords))
    if torch.is_tensor(phase):
        phase = (_box_parameter_field(scene, structure.geometry, phase, name="ModulationSpec.phase")
                 if coords is None else _sample_box_parameter_field(scene, structure.geometry, phase, coords))
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
    else:
        cos_phase = float(np.cos(float(phase)))
        sin_phase = float(np.sin(float(phase)))
    return amplitude * cos_phase, amplitude * sin_phase


def _modulated_material_value(scene, structure, value, coords, quadrature):
    """Seed δε = ε a cos(phi) or ε a sin(phi) before constitutive averaging.

    For a harmonic mean H = 1 / sum(f / ε), forward AD gives
    δH = H² sum(f δε / ε²). Tensor reconstructions use their own full derivative,
    including the normal estimated by the static rule. Reverse-mode gradients
    through these tangent fields remain native tensor operations.
    """
    if quadrature is None or getattr(structure.material, "modulation", None) is None:
        return value
    q = _structure_modulation_values(scene, structure, coords)[quadrature]
    value = torch.as_tensor(value, device=scene.device, dtype=_material_dtype(scene))
    value, delta = torch.broadcast_tensors(value, value * q)
    return torch.autograd.forward_ad.make_dual(value.clone(memory_format=torch.contiguous_format), delta)


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
    fill_spacing=None,
    full_tensor=False,
    modulation_quadrature=None,
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
        fill_spacing=fill_spacing,
    )
    if "modulation_owner" in model:
        owner = (next(index for index, item in enumerate(_bulk_structures(scene), 1) if item is structure)
                 if getattr(structure.material, "modulation", None) is not None else 0)
        model["modulation_owner"] = torch.where(
            occupancy == 1, owner,
            torch.where(occupancy == 0, model["modulation_owner"], -1),
        )
    if full_tensor:
        previous = inverse_components.get("anisotropic_eps")
        if previous is not None:
            previous = previous & (occupancy < 1)
        if _anisotropic_epsilon(eps_components, eps_offdiag):
            occupied = occupancy > 0
            previous = occupied if previous is None else previous | occupied
        if previous is not None:
            inverse_components["anisotropic_eps"] = previous
    capture = getattr(scene, "_conductivity_capture", None)
    if capture is not None:
        capture.record(structure.name, occupancy)
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
        if perturbation_delta is not None and modulation_quadrature is not None:
            eps_value = (_modulated_material_value(
                scene, structure, eps_value + perturbation_delta, coords, modulation_quadrature
            ) - perturbation_delta)
        else:
            eps_value = _modulated_material_value(scene, structure, eps_value, coords, modulation_quadrature)
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
    fill_spacing=None,
    full_tensor=False,
    modulation_quadrature=None,
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
                dtype=_material_dtype(scene)),
            "mu_components": _new_component_field(
                shape,
                fill_value=1.0 / (float(mu_background) + _HARMONIC_EPS),
                device=scene.device,
                dtype=_material_dtype(scene)),
        }
    coords = _coordinate_grids(scene, sample_offset)
    capture = getattr(scene, "_conductivity_capture", None)
    if capture is not None:
        capture.begin("nodes", coords[0].shape)
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
            fill_spacing=fill_spacing,
            full_tensor=full_tensor,
            modulation_quadrature=modulation_quadrature,
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
            fill_spacing=fill_spacing,
        )
    if collect_inverse:
        return model, inverse_components
    return model


def _axis_averaging_width(scene, axis):
    """Per-node subpixel averaging-window width along ``axis``.

    On a grid with a defined scalar spacing (``GridSpec.uniform`` or explicit
    per-axis steps) the window is the resolved Yee spacing, returned as a Python
    float so the sub-sample offsets stay bit-identical to the scalar-spacing
    path (identical to ``grid.d*`` when the span is an exact multiple). On a nonuniform grid (``GridSpec.custom`` or a resolved
    ``GridSpec.auto``) the window is the per-node Yee dual-cell width
    ``d{axis}_dual64`` -- the extent each node's material sample represents,
    ``0.5*(primal_left + primal_right)`` on interior nodes -- returned as a 1D
    float32 tensor so the offsets scale with the local cell size. It reduces to
    the constant spacing on a uniformly spaced custom axis.
    """
    grid = scene.grid
    if not (grid.is_custom or grid.is_auto):
        return _uniform_axis_spacing(scene, axis)
    dual = getattr(scene, f"d{axis}_dual64")
    return torch.as_tensor(dual, device=scene.device, dtype=_material_dtype(scene))


def _axis_sample_offsets(scene, axis, count):
    """Sub-sample displacements spanning the node's physical dual cell.

    Returns ``count`` offsets spanning the node's averaging window. On a uniform
    axis each offset is a Python float ``((i+0.5)/count - 0.5) * spacing``; on a
    nonuniform axis the interior cell center is displaced from its node by
    one quarter of the right-minus-left primal width. Omitting that shift
    double counts thin layers between coarse air cells, even with infinitely
    many subsamples. ``count == 1`` retains point sampling at the node.
    """
    if count == 1:
        return [0.0]
    width = _axis_averaging_width(scene, axis)
    center = _dual_cell_center_offset(scene, axis)
    return [center + ((index + 0.5) / count - 0.5) * width for index in range(count)]


def _dual_cell_center_offset(scene, axis):
    """Interior Yee dual-cell centers relative to their native nodes.

    The existing outer/periodic endpoint window convention is retained. Primal
    widths are rounded at sampling precision before subtraction so represented
    uniform axes retain exactly zero displacement.
    """
    if not (scene.grid.is_custom or scene.grid.is_auto):
        return 0.0
    primal = torch.as_tensor(getattr(scene, f"d{axis}_primal64"), device=scene.device, dtype=_material_dtype(scene))
    offset = primal.new_zeros(primal.numel()+1)
    offset[1:-1] = .25*(primal[1:]-primal[:-1])
    return offset


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
# models; both FDTD update coefficients and FDFD operators use the edge fields.

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


def _edge_axis_base_1d(scene, axis, staggered, *, dtype=None):
    """Per-axis 1D query coordinates for one Yee component along ``axis``.

    On a staggered axis the samples sit at the primal-cell midpoints
    ``0.5*(node_i + node_{i+1})`` (length ``N-1``, the Yee edge/face count); on an
    unstaggered axis they stay on the node coordinates (length ``N``).
    An explicit dtype reads the scene's float64 axes before forming midpoints.
    """
    coord = (getattr(scene, axis) if dtype is None else torch.tensor(
        getattr(scene, f"{axis}_nodes64"), device=scene.device, dtype=dtype))
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
        width = _uniform_axis_spacing(scene, axis)
    elif staggered:
        coord = getattr(scene, axis)
        width = (coord[1:] - coord[:-1]).to(dtype=_material_dtype(scene))
    else:
        width = torch.as_tensor(
            getattr(scene, f"d{axis}_dual64"), device=scene.device, dtype=_material_dtype(scene)
        )
    center = 0.0 if staggered else _dual_cell_center_offset(scene, axis)
    return [center + ((index + 0.5) / count - 0.5) * width for index in range(count)]


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

    Returns the diagonal of the inverse Kottke tensor
    (``n_a^2 <1/eps> + (1 - n_a^2) / <eps>``, see ``_inverse_diagonal_blend``) with
    the squared interface-normal component along ``axis`` estimated from the
    gradient of the arithmetic permittivity/permeability at the Yee location.
    This is the per-component edge-native form of
    :func:`_reconstruct_sampled_polarized_inverse`; for an isotropic medium it
    reproduces that node reconstruction exactly (the three per-axis components are
    identical there, so a single component's gradient carries the same direction).
    """
    grads = _field_gradients(scene, arithmetic, coords=coords)
    squared = {a: grads[a] * grads[a] for a in _AXES}
    norm = squared["x"] + squared["y"] + squared["z"] + _NORMAL_EPS
    weight = squared[axis] / norm
    return _inverse_diagonal_blend(arithmetic, inverse, weight)


def _sample_box_parameter_field(scene, geometry, values, coords):
    """Trilinearly sample a Box-extent parameter grid at physical query coordinates.

    The edge-native counterpart of :func:`_box_parameter_field`: the grid spans the
    Box extent with the same convention, and is zero outside the box (border-padded
    values there are multiplied by a zero occupancy so they never contribute).
    Differentiable in ``values``.
    """
    xx, yy, zz = coords
    center = torch.as_tensor(geometry.position, device=scene.device, dtype=_material_dtype(scene))
    size = torch.as_tensor(geometry.size, device=scene.device, dtype=_material_dtype(scene))
    normalized = (
        2.0 * (xx - center[0]) / size[0],
        2.0 * (yy - center[1]) / size[1],
        2.0 * (zz - center[2]) / size[2],
    )
    query_grid = torch.stack(normalized, dim=-1)
    texture = values.to(device=scene.device, dtype=_material_dtype(scene)).permute(2, 1, 0)
    return _sample_trilinear_border(texture, query_grid)


def _edge_structure_perturbation_delta(scene, structure, material, coords, eps_background):
    """Edge-sampled ``eps_sensitivity * eps_background * perturbation`` or ``None``."""
    if not isinstance(material, PerturbationMedium):
        return None
    return (
        float(material.eps_sensitivity)
        * float(eps_background)
        * _sample_box_parameter_field(scene, structure.geometry, material.perturbation, coords)
    )


def _mapped_dual_face_weights(scene, component, y_base64, y_offsets):
    """Positive normalized material quadrature on the physical electric dual face.

    For the orthogonal bend h_s=1-kappa*u, only E_z's dual face spans
    the varying s scale factor. E_s has area du*dz; E_u's own u is
    fixed on its dual face, so its constant h_s cancels from the mean.
    Magnetic material uses its physical own-edge length; that length's
    normalized weights are constant for this chart. The metric Hodge is
    still evaluated at the component center by attach_coordinate_metric.
    """
    chart=getattr(scene,"coordinates",None)
    if chart is None or chart.curvature==0 or component!="Ez" or len(y_offsets)==1:
        return None
    offsets=torch.stack([torch.as_tensor(offset,device=scene.device,dtype=torch.float64).expand_as(y_base64)
                         for offset in y_offsets])
    weights=1-chart.curvature*(y_base64[None]+offsets)
    if not bool(torch.isfinite(weights).all() and (weights>0).all()):
        raise ValueError("Mapped material quadrature requires positive finite dual-face metric weights.")
    return weights


def _compile_kottke_edge_component(scene, component, bases, samples, background,
                                   region_densities, geometry_periodic_shifts, modulation_quadrature=None, *, smooth=False):
    """Local-moment quadrature in interface windows, with later-object priority.

    Fixed solid meshes prune center queries by padded world-space bounds. Binary
    samples query each structure only in its straddling windows; smooth samples
    use curvature-corrected fractional coverage. Packed chunks bound temporary
    sample storage independently of the scene size.
    """
    from witwin.core import Mesh

    electric = component[0] == "E"
    axis = _EDGE_POLARIZATION_AXIS[component]
    stagger = _EDGE_STAGGER_AXES[component]
    coords = _edge_component_coords(bases, (0., 0., 0.))
    shape = coords[0].shape
    points = torch.stack(coords, -1).reshape(-1, 3)
    widths = []
    for index, a in enumerate(_AXES):
        if not (scene.grid.is_custom or scene.grid.is_auto):
            width = points.new_full((len(bases[a]),), _uniform_axis_spacing(scene, a))
        else:
            if index in stagger:
                width = getattr(scene, a).diff().to(points.dtype)
            else:
                width = torch.as_tensor(getattr(scene, f"d{a}_dual64"), device=scene.device, dtype=points.dtype)
            # Global-axis widths: a partial selection is a contiguous subset of the global Yee axis.
            base = _edge_axis_base_1d(scene, a, index in stagger)
            position = torch.searchsorted(base, bases[a])
            if not bool((position < len(base)).all() and torch.equal(base[position.clamp(max=len(base) - 1)], bases[a])):
                raise ValueError(f"Edge-sample coordinates along {a} are not a subset of the global Yee axis.")
            width = width[position]
        view = [1, 1, 1]
        view[index] = -1
        widths.append(width.reshape(view).expand(shape).reshape(-1))
    widths = torch.stack(widths, -1)
    band = .5 * torch.linalg.vector_norm(widths, dim=-1) + 1e-6
    lattice = [(torch.arange(n, device=scene.device, dtype=points.dtype) + .5) / n - .5 for n in samples]
    offsets = torch.stack(torch.meshgrid(*lattice, indexing="ij"), -1).reshape(-1, 3)
    entries = []
    for structure in _bulk_structures(scene):
        parts = _static_structure_material(structure)
        value = parts[1 if electric else 3][axis] * background
        entries.append((structure.geometry, value, structure, None))
    for region, density in zip(scene.material_regions, region_densities):
        entries.append((region.geometry, None, region, density))
    if modulation_quadrature is None and all(value is not None and not _contains_trainable_tensor(value) and bool(value == background) and
           not (electric and isinstance(owner.material, PerturbationMedium))
           for _, value, owner, _ in entries):
        return points.new_full(shape, background)

    def distance(geometry, query, padding):
        result = query.new_full((len(query),), float("inf"))
        lower = upper = None
        if isinstance(geometry, Mesh) and geometry.fill_mode != "surface":
            vertices, faces, _, _ = geometry._sdf_query_data(device=query.device, dtype=query.dtype)
            edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).long()
            forward = edges[:, 0] * len(vertices) + edges[:, 1]
            reverse = edges[:, 1] * len(vertices) + edges[:, 0]
            if geometry.is_watertight or torch.equal(forward.sort().values, reverse.sort().values):
                lower, upper = vertices.amin(0), vertices.amax(0)
        shifts = geometry_periodic_shifts[id(geometry)]
        if shifts is None:
            shifts = tuple((-float(hi-lo), 0., float(hi-lo))
                           if scene.boundary.axis_kind(a) in ("periodic", "bloch") else (0.,)
                           for a, (lo, hi) in zip(_AXES, scene.domain.bounds))
        for shift in product(*shifts):
            shifted = query - query.new_tensor(shift)
            active = torch.ones(len(query), device=query.device, dtype=torch.bool)
            if lower is not None:
                active = ((shifted >= lower - padding[:, None]) &
                          (shifted <= upper + padding[:, None])).all(-1)
            if bool(active.any()):
                packed = shifted[active]
                sdf = geometry.signed_distance(*(packed[:, i:i+1, None] for i in range(3))).reshape(-1)
                updated = result.clone()
                updated[active] = torch.minimum(result[active], sdf)
                result = updated
        return result

    def values(entry, query):
        geometry, value, owner, density = entry
        packed = tuple(query[:, i:i+1, None] for i in range(3))
        if density is not None:
            lo, hi = owner.eps_bounds if electric else owner.mu_bounds
            value = lo + _sample_material_region_density(scene, owner, density, coords=packed).reshape(-1) * (hi-lo)
        elif electric:
            delta = _edge_structure_perturbation_delta(scene, owner, owner.material, packed, background)
            if delta is not None:
                value = value + delta.reshape(-1)
            value = _modulated_material_value(scene, owner, value, packed, modulation_quadrature)
            if torch.is_tensor(value) and value.ndim:
                value = value.reshape(-1)
        return torch.as_tensor(value, device=query.device, dtype=query.dtype)

    def fractional_fill(entry, query, spacing):
        shifts = geometry_periodic_shifts[id(entry[0])]
        if shifts is None:
            shifts = tuple((-float(hi-lo), 0., float(hi-lo))
                           if scene.boundary.axis_kind(a) in ("periodic", "bloch") else (0.,)
                           for a, (lo, hi) in zip(_AXES, scene.domain.bounds))
        fill = query.new_zeros(len(query))
        for shift in product(*shifts):
            shifted = query - query.new_tensor(shift)
            fill = torch.maximum(fill, _subsample_fill_fraction(
                entry[0], *(shifted[:, i:i+1, None] for i in range(3)), spacing).reshape(-1))
        return fill

    distances = [distance(entry[0], points, band) for entry in entries]
    material = points.new_full((len(points),), background)
    near = torch.zeros(len(points), device=scene.device, dtype=torch.bool)
    for entry, sdf in zip(entries, distances):
        if smooth:
            # Only saturated bulk survives; interface windows are overwritten below.
            fill = torch.clamp(.5 - sdf / (2 * band), 0., 1.)
            material = fill * values(entry, points) + (1 - fill) * material
        else:
            material = torch.where(sdf < 0, values(entry, points), material)
        near |= sdf.abs() < band
    indices = near.nonzero()[:, 0]
    if indices.numel() == 0:
        return material.reshape(shape)
    for selected in indices.split(4096):
        local_offsets = offsets[None] * widths[selected, None]
        query = (points[selected, None] + local_offsets).reshape(-1, 3)
        sample = query.new_full((len(query),), background)
        for entry, sdf in zip(entries, distances):
            if smooth:
                spacing = tuple(float(widths[selected, i].min()) / samples[i] for i in range(3))
                fill = fractional_fill(entry, query, spacing)
                sample = fill * values(entry, query) + (1 - fill) * sample
            else:
                inside = (sdf[selected] < 0)[:, None].expand(-1, len(offsets)).clone()
                straddle = sdf[selected].abs() < band[selected]
                if bool(straddle.any()):
                    q = query.reshape(len(selected), -1, 3)[straddle].reshape(-1, 3)
                    inside[straddle] = (distance(entry[0], q, q.new_zeros(len(q))) < 0).reshape(-1, len(offsets))
                sample = torch.where(inside.reshape(-1), values(entry, query), sample)
        material[selected] = _kottke_edge_diagonal(sample.reshape(len(selected), -1), local_offsets, _AXES.index(axis))
    return material.reshape(shape)


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
    selection=None,
    modulation_quadrature=None,
):
    """Edge-native diagonal permittivity/permeability and conductivity for one component.

    Returns ``(material_relative, sigma)`` where ``material_relative`` is the
    relative eps (electric components) or mu (magnetic components) sampled at the
    component's Yee location and ``sigma`` is the matching static electric /
    magnetic conductivity there. Differentiable in geometry / density parameters.

    ``selection`` samples a component-index patch through the same formulas,
    retaining a one-cell differentiation halo. It does not prove that changing
    a geometry or density has compact support outside that patch. Partial
    periodic-axis selections require a separate opposite-face halo plan.
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
    metric_y_base=torch.tensor(np.asarray(scene.y_nodes64),device=scene.device,dtype=torch.float64)
    crop = None
    windows = {}
    if selection is not None:
        if len(selection) != 3:
            raise ValueError("An edge material selection must contain three slices.")
        crop = []
        for grid_axis, selected in zip(_AXES, selection):
            size = len(bases[grid_axis])
            if not isinstance(selected, slice) or selected.step not in (None, 1):
                raise ValueError("Edge material selections require contiguous slices.")
            start = 0 if selected.start is None else selected.start
            stop = size if selected.stop is None else selected.stop
            if (isinstance(start, bool) or isinstance(stop, bool)
                    or not isinstance(start, int) or not isinstance(stop, int)
                    or not 0 <= start < stop <= size):
                raise ValueError(f"Invalid edge material selection on {grid_axis}: {selected} for size {size}.")
            if scene.boundary.axis_kind(grid_axis) in ("periodic", "bloch") and (start != 0 or stop != size):
                raise NotImplementedError(
                    "Partial sampling along periodic axes requires opposite-face halo routing."
                )
            # Both signed-distance normals and subpixel normal reconstruction
            # use a radius-one finite difference. Retain global-edge one-sided
            # differences, and discard the artificial local boundary after sampling.
            lower, upper = max(0, start - 1), min(size, stop + 1)
            windows[grid_axis] = slice(lower, upper)
            bases[grid_axis] = bases[grid_axis][lower:upper]
            if grid_axis=="y":metric_y_base=metric_y_base[lower:upper]
            axis_offsets[grid_axis] = [offset[lower:upper] if torch.is_tensor(offset) and offset.ndim else offset
                                       for offset in axis_offsets[grid_axis]]
            crop.append(slice(start - lower, stop - lower))
        crop = tuple(crop)
    area_weights=_mapped_dual_face_weights(scene,component,metric_y_base,axis_offsets["y"])
    offset_combos = list(product(axis_offsets["x"], axis_offsets["y"], axis_offsets["z"]))
    half_weight_boundary = len(offset_combos) > 1
    fill_spacing = _subsample_spacing(scene, samples, stagger) if half_weight_boundary else None
    if fill_spacing is not None and windows:
        # Graded per-line spacings follow the same halo-extended patch window.
        fill_spacing = tuple(
            value[windows[grid_axis]] if torch.is_tensor(value) else value
            for grid_axis, value in zip(_AXES, fill_spacing)
        )
    kottke_edge = averaging in {"kottke_edge", "kottke_smooth"}
    polarized = averaging in {"polarized", "polarized_tensor", "kottke_edge", "kottke_smooth"}
    polarized_direct = polarized and len(offset_combos) == 1

    structures = list(_bulk_structures(scene))
    regions = list(zip(getattr(scene, "material_regions", ()), region_densities))
    if kottke_edge:
        # Only structures can carry conductivity here: MaterialRegion has no sigma
        # (eps/mu bounds only) and PreparedScene rejects background declarations.
        parts = [_static_structure_material(structure) for structure in structures]
        scalar = all(not _anisotropic_epsilon(part[1 if electric else 3], part[2] if electric else {})
                     for part in parts)
        lossless = all(not _contains_trainable_tensor(part[4 if electric else 5][axis]) and
                       bool(part[4 if electric else 5][axis] == 0) for part in parts)
        if scalar and lossless:
            material = _compile_kottke_edge_component(
                scene, component, bases, samples, background, region_densities, geometry_periodic_shifts,
                modulation_quadrature, smooth=averaging == "kottke_smooth")
            sigma = torch.zeros_like(material)
            return (material[crop].contiguous(), sigma[crop].contiguous()) if crop is not None else (material.contiguous(), sigma)

    material_sum = None
    sigma_sum = None
    inverse_sum = None
    anisotropic = None
    weight_sum = None
    for sample_index,offsets in enumerate(offset_combos):
        coords = _edge_component_coords(bases, offsets)
        shape = coords[0].shape
        capture = getattr(scene, "_conductivity_capture", None) if electric else None
        if capture is not None:
            capture.begin(component, shape)
        material_acc = torch.full(shape, background, device=scene.device, dtype=_material_dtype(scene))
        sigma_acc = torch.zeros(shape, device=scene.device, dtype=_material_dtype(scene))
        inverse_acc = (
            torch.full(shape, 1.0 / (background + _HARMONIC_EPS), device=scene.device, dtype=_material_dtype(scene))
            if (polarized and not polarized_direct)
            else None
        )
        anisotropic_sample = None
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
                fill_spacing=fill_spacing,
            )
            if kottke_edge:
                if anisotropic_sample is not None:
                    anisotropic_sample = anisotropic_sample & (occupancy < 1)
                components = eps_components if electric else mu_components
                offdiag = parts[2] if electric else {}
                if _anisotropic_epsilon(components, offdiag):
                    occupied = occupancy > 0
                    anisotropic_sample = occupied if anisotropic_sample is None else anisotropic_sample | occupied
            if capture is not None:
                capture.record(structure.name, occupancy)
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
                value = _modulated_material_value(scene, structure, value, coords, modulation_quadrature)
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
                fill_spacing=fill_spacing,
            )
            if anisotropic_sample is not None:
                anisotropic_sample = anisotropic_sample & (occupancy < 1)
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
        if area_weights is not None:
            y_index=(sample_index//len(axis_offsets["z"]))%len(axis_offsets["y"])
            weight=area_weights[y_index][None,:,None]
            material_acc=material_acc.double()*weight
            sigma_acc=sigma_acc.double()*weight
            if inverse_acc is not None:inverse_acc=inverse_acc.double()*weight
            weight_sum=weight if weight_sum is None else weight_sum+weight
        material_sum = material_acc if material_sum is None else material_sum + material_acc
        if anisotropic_sample is not None:
            anisotropic = anisotropic_sample if anisotropic is None else anisotropic | anisotropic_sample
        sigma_sum = sigma_acc if sigma_sum is None else sigma_sum + sigma_acc
        if inverse_acc is not None:
            inverse_sum = inverse_acc if inverse_sum is None else inverse_sum + inverse_acc

    scale = 1.0 / float(len(offset_combos))
    if weight_sum is None:
        material_mean = material_sum * scale
        sigma_mean = sigma_sum * scale
        inverse_mean=None if inverse_sum is None else inverse_sum*scale
    else:
        material_mean=(material_sum/weight_sum).to(_material_dtype(scene))
        sigma_mean=(sigma_sum/weight_sum).to(_material_dtype(scene))
        inverse_mean=None if inverse_sum is None else (inverse_sum/weight_sum).to(_material_dtype(scene))
    if inverse_sum is not None:
        base_coords = _edge_component_coords(bases, (0.0, 0.0, 0.0))
        material_mean = _reconstruct_edge_polarized(
            scene, base_coords, material_mean, inverse_mean, axis
        )
    if kottke_edge:
        window = _compile_kottke_edge_component(
            scene, component, bases, samples, background, region_densities, geometry_periodic_shifts,
            modulation_quadrature, smooth=averaging == "kottke_smooth"
        )
        material_mean = window if anisotropic is None else torch.where(anisotropic, material_mean, window)
    if crop is not None:
        material_mean = material_mean[crop]
        sigma_mean = sigma_mean[crop]
    return material_mean.contiguous(), sigma_mean.contiguous()


class _PCBConductivityReplay:
    """Retain static PCB compilation and replay its scalar conductivity arithmetic.

    Used by the declared-band campaign for static isotropic dielectric boxes,
    PEC and copper foil, in Cartesian coordinates. Occupancies are captured from
    the ordinary compiler, not resampled or inferred from compiled conductivity.
    Only their nonzero bounding boxes are retained. Sample order and float32
    blends/additions remain identical, including overlapping dielectrics.
    """

    def __init__(self, scene):
        if scene.subpixel.averaging == "kottke_smooth":
            raise NotImplementedError("PCB conductivity replay does not support kottke_smooth averaging.")
        if scene.subpixel.averaging == "kottke_edge":
            raise NotImplementedError("PCB conductivity replay does not support kottke_edge averaging.")
        if scene.subpixel.averaging == "polarized_tensor":
            raise NotImplementedError("PCB conductivity replay does not support polarized_tensor averaging.")
        self.samples, self.shapes = {}, {}
        scene._conductivity_capture = self
        self.model = scene.compile_materials()
        del scene._conductivity_capture
        self.selections = {}
        for component, samples in self.samples.items():
            boxes = [selection for sample in samples for _, selection, _ in sample]
            self.selections[component] = tuple(
                slice(min(box[axis].start for box in boxes), max(box[axis].stop for box in boxes))
                for axis in range(3))

    def begin(self, component, shape):
        self.shapes[component] = shape
        self.current = []
        self.samples.setdefault(component, []).append(self.current)

    def record(self, name, occupancy):
        indices = occupancy.nonzero()
        if indices.numel():
            bounds = torch.stack((indices.amin(0), indices.amax(0) + 1)).tolist()
            selection = tuple(slice(low, high) for low, high in zip(*bounds))
            self.current.append((name, selection, occupancy[selection].clone()))

    def compile(self, conductivities):
        model = dict(self.model)
        fields = {}
        reference = self.model["sigma_e_components"]["x"]
        conductivities = {name: reference.new_tensor(value) for name, value in conductivities.items()}
        for component, samples in self.samples.items():
            selection = self.selections[component]
            shape = tuple(axis.stop - axis.start for axis in selection)
            total = reference.new_zeros(shape)
            for sample in samples:
                value = reference.new_zeros(shape)
                for name, box, occupancy in sample:
                    local = tuple(slice(axis.start - outer.start, axis.stop - outer.start)
                                  for axis, outer in zip(box, selection))
                    value[local] = _blend_material(value[local], occupancy, value=conductivities[name])
                total += value
            total *= 1.0 / len(samples)
            field = reference.new_zeros(self.shapes[component])
            field[selection] = total
            fields[component] = field
        # Isotropic PCB conductivity has the same node samples on all axes.
        model["sigma_e_components"] = dict.fromkeys(_AXES, fields["nodes"])
        model["sigma_e"] = _component_average(model["sigma_e_components"])
        if model["edge_components"] is not None:
            model["edge_components"] = dict(model["edge_components"],
                sigma_e={component: fields[component] for component in _ELECTRIC_EDGE_COMPONENTS})
        return model


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
    if material_model_has_full_anisotropy(model) and not model.get('subpixel_full_tensor', False):
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
    enabled=True,
    modulation_quadrature=None,
):
    """Edge-native diagonal material fields for the six Yee components, or ``None``.

    Returns a dict with per-component relative permittivity (``eps``), relative
    permeability (``mu``) and static conductivities (``sigma_e`` / ``sigma_m``)
    sampled at each component's own Yee location, or ``None`` when the scene
    contains a material family that is not yet edge-native (see
    :func:`_edge_native_eligible`), in which case the runtime keeps the node->edge
    average path.
    """
    if not enabled or not _edge_native_eligible(scene, model, surface_layout):
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
            modulation_quadrature=modulation_quadrature,
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


def _compile_binary_pole_edges(scene, model, layout, *, samples, eps_background,
                               mu_background, region_densities, geometry_periodic_shifts):
    """Select bulk pole constituents at the Yee centre (factorial arm a).

    A window touched by a pole constituent uses the centre constituent's complete
    background and poles, including the nondispersive side of an interface.
    Windows containing only nondispersive constituents retain their subpixel rule.
    Spatially uniform pole fields use the same binary assignment as scalar poles.
    Any window touching a spatially varying pole retains the previous fractional
    weights and background for every constituent, including scalar neighbours.
    Sheet poles retain their separate surface-conductivity discretization.
    """
    from .accel.dispatch import UNCHANGED, setup_enabled
    if setup_enabled.get():
        from .accel.poles import try_compile_binary_pole_edges
        if try_compile_binary_pole_edges(scene, model, layout, samples=samples,
                eps_background=eps_background, mu_background=mu_background,
                region_densities=region_densities,
                geometry_periodic_shifts=geometry_periodic_shifts) is not UNCHANGED:
            return

    from ..fdtd.runtime.materials import average_node_to_component, average_node_to_magnetic_component

    structures = _bulk_structures(scene)
    parts = [_static_structure_material(s) for s in structures]
    graded = [any(layout[key + "_poles"][slot].get("graded", False)
                  for key, slots in structure_slots.items() for slot in slots)
              for structure_slots in layout["structure_slots"]]
    tensor_background = material_model_has_full_anisotropy(model)
    graded_masks = {}
    for prefix, components, channel, background in (
        ("", _ELECTRIC_EDGE_COMPONENTS, "eps", eps_background),
        ("mu_", _MAGNETIC_EDGE_COMPONENTS, "mu", mu_background),
    ):
        keys = tuple(prefix + kind + "_poles" for kind in ("debye", "drude", "lorentz"))
        if not any(layout[key] for key in keys):
            continue
        carriers = [any(getattr(part[0], key, ()) for key in keys) for part in parts]
        if not any(carrier and not custom for carrier, custom in zip(carriers, graded)):
            continue
        modulated = prefix == "" and material_model_has_modulation(model)
        for key in keys:
            for entry in model[key][:len(layout[key])]:
                entry["edge_weights"] = {}
        for component in components:
            stagger = _EDGE_STAGGER_AXES[component]
            axis = component[-1].lower()
            bases = {a: _edge_axis_base_1d(scene, a, i in stagger) for i, a in enumerate(_AXES)}
            coords = _edge_component_coords(bases, (0., 0., 0.))
            bases64 = {a: _edge_axis_base_1d(scene, a, i in stagger, dtype=torch.float64)
                       for i, a in enumerate(_AXES)}
            coords64 = _edge_component_coords(bases64, (0., 0., 0.))
            owner = torch.full_like(coords[0], -1, dtype=torch.int64)
            value = torch.full_like(coords[0], float(background))
            touch = torch.zeros_like(owner, dtype=torch.bool)
            graded_touch = torch.zeros_like(touch)
            average = average_node_to_component if prefix == "" else average_node_to_magnetic_component
            for custom, slots in zip(graded, layout["structure_slots"]):
                if custom:
                    for kind, key in zip(("debye", "drude", "lorentz"), keys):
                        for slot in slots[prefix + kind]:
                            graded_touch |= average(None, model[key][slot]["weight"], component) != 0
            if model["edge_components"] is None or tensor_background:
                for key in keys:
                    for entry in model[key][:len(layout[key])]:
                        touch |= average(None, entry["weight"], component) > 0
            modulation = {name: torch.zeros_like(value) for name in ("cos", "sin", "omega")} if modulated else None
            offsets = [_edge_axis_offsets(scene, a, n, i in stagger)
                       for i, (a, n) in enumerate(zip(_AXES, samples))]
            fill_spacing = _subsample_spacing(scene, samples, stagger) if any(n > 1 for n in samples) else None
            for index, (structure, part, carrier) in enumerate(zip(structures, parts, carriers)):
                def occupancy(query, fill_spacing=None):
                    return _geometry_occupancy(
                        scene, structure.geometry, coords=query,
                        periodic_shift_options=geometry_periodic_shifts[id(structure.geometry)],
                        wrap_skip_axes=stagger,
                        fill_spacing=fill_spacing,
                        surface_inclusive=fill_spacing is None,
                    )
                inside = occupancy(coords64) >= 0.5
                owner = torch.where(inside, index, owner)
                material_value = part[1 if prefix == "" else 3][axis] * float(background)
                if prefix == "":
                    delta = _edge_structure_perturbation_delta(scene, structure, part[0], coords, background)
                    if delta is not None:
                        material_value = material_value + delta
                    material_value = _modulated_material_value(scene, structure, material_value, coords, None)
                value = torch.where(inside, material_value, value)
                if modulation is not None:
                    for name, quadrature in zip(("cos", "sin"), _structure_modulation_values(scene, structure, coords)):
                        modulation[name] = torch.where(inside, quadrature, modulation[name])
                    spec = getattr(structure.material, "modulation", None)
                    omega = 0. if spec is None else 2*np.pi*spec.frequency
                    modulation["omega"] = torch.where(inside, omega, modulation["omega"])
                if carrier or graded[index]:
                    if not graded[index] and any(_contains_trainable_tensor(v) for v in vars(structure.geometry).values()):
                        raise ValueError("Binary pole assignment requires fixed pole geometry.")
                    support = inside.clone()
                    query_bases = bases if fill_spacing is not None else bases64
                    for offset in product(*offsets):
                        support |= occupancy(_edge_component_coords(query_bases, offset), fill_spacing) > 0
                    if graded[index]:
                        graded_touch |= support
                    else:
                        touch |= support
            touch &= ~graded_touch
            for region, density in zip(scene.material_regions, region_densities):
                inside = _geometry_occupancy(
                    scene, region.geometry, coords=coords64,
                    periodic_shift_options=geometry_periodic_shifts[id(region.geometry)],
                    wrap_skip_axes=stagger,
                    surface_inclusive=True,
                ) >= 0.5
                owner = torch.where(inside, -1, owner)
                lo, hi = region.eps_bounds if prefix == "" else region.mu_bounds
                density_value = _sample_material_region_density(scene, region, density, coords=coords)
                value = torch.where(inside, lo + density_value * (hi - lo), value)
                if modulation is not None:
                    modulation = {name: torch.where(inside, 0., field) for name, field in modulation.items()}
            if modulation is not None:
                model.setdefault("pole_edge_modulation", {})[component] = dict(modulation, mask=touch)
            if prefix == "" and tensor_background:
                model.setdefault("pole_edge_masks", {})[component] = touch
                graded_masks[component] = graded_touch
                # A physical full tensor needs its selected inverse diagonal,
                # rather than the reciprocal of its selected diagonal epsilon.
                if any(any(bool(v != 0) for v in part[2].values()) for part in parts):
                    tensor = {}
                    for pair in ("xx", "yy", "zz", "xy", "xz", "yz"):
                        field = torch.full_like(value, float(background) if pair[0] == pair[1] else 0.)
                        for index, part in enumerate(parts):
                            selected = (part[1][pair[0]] if pair[0] == pair[1] else part[2][pair]) * float(background)
                            if pair[0] == pair[1]:
                                delta = _edge_structure_perturbation_delta(scene, structures[index], part[0], coords, background)
                                if delta is not None:
                                    selected = selected + delta
                            field = torch.where(owner == index, selected, field)
                        # Density regions are scalar and carry no poles.
                        if pair[0] == pair[1]:
                            field = torch.where(owner == -1, value, field)
                        tensor[pair] = field
                    for a in _AXES:
                        tensor["sigma_"+a] = average_node_to_component(None, model["sigma_e_components"][a], component)
                    model.setdefault("pole_edge_tensors", {})[component] = tensor
            for index, (structure, slots) in enumerate(zip(structures, layout["structure_slots"])):
                for kind, key in zip(("debye", "drude", "lorentz"), keys):
                    for slot in slots[prefix + kind]:
                        entry = model[key][slot]
                        fractional = average(None, entry["weight"], component)
                        binary = (owner == index).to(value.dtype)
                        amplitude = layout[key][slot].get("uniform_amplitude")
                        if amplitude is not None:
                            # Interpolate the zero deviation to keep uniform weights
                            # exactly binary while preserving strength derivatives.
                            binary = binary * (1 + _sample_box_parameter_field(
                                scene, structure.geometry, amplitude - 1, coords))
                        weight = torch.where(graded_touch, fractional, binary)
                        model[key][slot]["edge_weights"][component] = weight.contiguous()
            edge = model["edge_components"]
            if edge is None:
                old = average(None, model[channel + "_components"][axis], component)
                target = model.setdefault("pole_edge_background", {})
                target[component] = torch.where(touch, value, old).contiguous()
            else:
                edge[channel][component] = torch.where(touch, value, edge[channel][component]).contiguous()
    if tensor_background and model.get("pole_edge_masks"):
        _compile_pole_tensor_nodes(scene, model, eps_background, region_densities,
                                   geometry_periodic_shifts, graded_masks)


def _compile_pole_tensor_nodes(scene, model, background, region_densities, shifts, graded_masks):
    """Remove subpixel tensor mixing on the support of centre-owned pole edges.

    The existing reciprocal tensor gather/scatter uses node tensors. Retain that
    assembly but select complete constituent tensors in the pole-interface band;
    its diagonal edge entries are selected separately at the actual Yee centres.
    Node-facing material diagnostics remain unchanged.
    """
    reference = model["eps_components"]["x"]
    mask = torch.zeros_like(reference, dtype=torch.bool)
    graded_nodes = torch.zeros_like(mask)
    for i, component in enumerate(_ELECTRIC_EDGE_COMPONENTS):
        lower = [slice(None)] * 3
        upper = lower.copy()
        lower[i], upper[i] = slice(None, -1), slice(1, None)
        mask[tuple(lower)] |= model["pole_edge_masks"][component]
        mask[tuple(upper)] |= model["pole_edge_masks"][component]
        graded_nodes[tuple(lower)] |= graded_masks[component]
        graded_nodes[tuple(upper)] |= graded_masks[component]
    mask &= ~graded_nodes
    selected = {pair: torch.full_like(reference, float(background) if pair[0] == pair[1] else 0.)
                for pair in ("xx", "yy", "zz", "xy", "xz", "yz")}
    for structure in _bulk_structures(scene):
        part = _static_structure_material(structure)
        inside = _geometry_occupancy(scene, structure.geometry,
            periodic_shift_options=shifts[id(structure.geometry)], surface_inclusive=True) >= .5
        for pair in selected:
            value = part[1][pair[0]] if pair[0] == pair[1] else part[2][pair]
            if pair[0] == pair[1]:
                delta = _edge_structure_perturbation_delta(
                    scene, structure, part[0], (scene.X, scene.Y, scene.Z), background)
                value = value * float(background) + (0. if delta is None else delta)
            else:
                value = value * float(background)
            selected[pair] = torch.where(inside, value, selected[pair])
    for region, density in zip(scene.material_regions, region_densities):
        inside = _geometry_occupancy(scene, region.geometry,
            periodic_shift_options=shifts[id(region.geometry)], surface_inclusive=True) >= .5
        lo, hi = region.eps_bounds
        value = lo + _sample_material_region_density(scene, region, density) * (hi-lo)
        selected = {pair: torch.where(inside, value if pair[0] == pair[1] else 0., field)
                    for pair, field in selected.items()}
    diagonal = {a: torch.where(mask, selected[a+a], model["eps_components"][a]) for a in _AXES}
    offdiag = {pair: torch.where(mask, selected[pair], model["eps_offdiag_components"][pair]) for pair in _OFFDIAG_AXES}
    model["pole_tensor_nodes"] = {"diagonal": diagonal, "offdiag": offdiag}


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
    field = torch.zeros(shape, device=scene.device, dtype=_material_dtype(scene))
    slices = _box_axis_slices(scene, geometry)
    if slices is None:
        return field
    x_slice, y_slice, z_slice = slices
    values = values.to(device=scene.device, dtype=_material_dtype(scene))
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
    density = region.density.to(device=scene.device, dtype=_material_dtype(scene))
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
    geometry = getattr(region, "_pml_density_geometry", region.geometry)
    center = torch.as_tensor(geometry.position, device=scene.device, dtype=_material_dtype(scene))
    size = torch.as_tensor(geometry.size, device=scene.device, dtype=_material_dtype(scene))
    normalized = (
        2.0 * (xx - center[0]) / size[0],
        2.0 * (yy - center[1]) / size[1],
        2.0 * (zz - center[2]) / size[2],
    )
    query_grid = torch.stack(normalized, dim=-1)
    # grid_sample uses input order (depth=z, height=y, width=x) and query
    # components (x, y, z), while public density tensors are stored (x, y, z).
    texture = density.permute(2, 1, 0)
    return _sample_trilinear_border(texture, query_grid)


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
    fill_spacing=None,
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
            fill_spacing=fill_spacing,
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
        if "modulation_owner" in model:
            model["modulation_owner"] = torch.where(
                occupancy == 1, 0,
                torch.where(occupancy == 0, model["modulation_owner"], -1),
            )
        if inverse_components is not None and "anisotropic_eps" in inverse_components:
            inverse_components["anisotropic_eps"] = inverse_components["anisotropic_eps"] & (occupancy < 1)
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
    structures = volume_pec_structures(scene)
    if not structures:
        return None
    beta = _pec_geometry_beta(scene)
    occupancy = None
    for structure in structures:
        sample = _geometry_occupancy(
            scene, structure.geometry, coords=coords, beta=beta, half_weight_boundary=True
        )
        occupancy = sample if occupancy is None else torch.maximum(occupancy, sample)
    return occupancy


def _pmc_occupancy(scene):
    """Union (max) of soft SDF occupancies of all PMC-material structures on the node grid.

    The magnetic dual of :func:`_pec_occupancy`. Returns ``None`` when the scene has
    no PMC structure so non-PMC scenes stay byte-identical. The FDTD runtime
    averages it onto each H dual edge and holds the covered H components at zero.
    """
    from ..geometry.planar_pec import PlanarPEC
    from ..geometry.surface_mesh import SurfaceMesh

    structures = pmc_structures(scene)
    if not structures:
        return None
    beta = _pec_geometry_beta(scene)
    occupancy = None
    for structure in structures:
        if isinstance(structure.geometry, (PlanarPEC, SurfaceMesh)):
            raise NotImplementedError(
                "A PMC material requires a volumetric geometry; PlanarPEC and SurfaceMesh "
                "describe zero-thickness electric conductors with no dual magnetic-sheet support."
            )
        sample = _geometry_occupancy(
            scene, structure.geometry, beta=beta, half_weight_boundary=True
        )
        occupancy = sample if occupancy is None else torch.maximum(occupancy, sample)
    return occupancy.detach()


def _pec_periodic_shift_options(scene):
    """Per-axis translation options that bring periodic images of PEC geometry into view."""
    domain_range = getattr(scene, "physical_domain_range", None)
    options = []
    for axis_index, axis in enumerate(("x", "y", "z")):
        if domain_range is not None and scene.boundary.axis_kind(axis) in ("periodic", "bloch"):
            span = float(domain_range[2 * axis_index + 1]) - float(domain_range[2 * axis_index])
            options.append((-span, 0.0, span))
        else:
            options.append((0.0,))
    return tuple(options)


def _pec_signed_distance(scene, coords=None):
    """Union signed distance (min over PEC structures and periodic images) on the node grid.

    Returns ``None`` when the scene has no PEC structure. Differentiable in PEC
    geometry through ``signed_distance``.
    """
    structures = volume_pec_structures(scene)
    if not structures:
        return None
    xx, yy, zz = (scene.X, scene.Y, scene.Z) if coords is None else coords
    shift_options = _pec_periodic_shift_options(scene)
    union = None
    for structure in structures:
        for shift_x, shift_y, shift_z in product(*shift_options):
            sample = structure.geometry.signed_distance(
                xx - shift_x, yy - shift_y, zz - shift_z
            )
            # Float32 geometry arithmetic can put an exactly aligned face a few
            # ulps outside its node plane. Match the existing material-boundary
            # convention; otherwise a one-cell foil loses both tangential faces.
            tolerance = 1.0e-5 * min(scene.grid.min_spacing)
            exact_boundary = sample - sample.detach()
            sample = torch.where(sample.abs() <= tolerance, exact_boundary, sample)
            union = sample if union is None else torch.minimum(union, sample)
    return union


def _edge_line_coverage(low, high):
    """Fraction of a Yee edge covered by the conductor, from its endpoint signed distances.

    The signed distance is interpolated linearly along the edge (an SDF is
    eikonal, so this is exact for a planar cut and second-order for a curved one)
    and the covered fraction is the part of the segment with ``sd <= 0``. The
    result has compact support by construction: an edge with both endpoints
    outside is exactly ``0`` and an edge with both endpoints inside is exactly
    ``1``, with no tail. That is what makes the conformal treatment collapse onto
    the exact staircase treatment wherever the conductor surface does not cut the
    edge -- in particular for every tangential edge of a grid-aligned face.
    """
    lower = torch.minimum(low, high)
    upper = torch.maximum(low, high)
    span = upper - lower
    cut = span > 0
    safe_span = torch.where(cut, span, torch.ones_like(span))
    fraction = torch.clamp(-lower / safe_span, 0.0, 1.0)
    # ``span == 0`` means the SDF is constant along the edge (a face parallel to
    # the edge): the edge is either wholly conductor or wholly open.
    degenerate = (upper <= 0.0).to(low.dtype)
    return torch.where(cut, fraction, degenerate)


def _pec_edge_fill(scene, coords=None):
    """Conformal volume coverage plus exact hard tangential planar support."""
    validate_legacy_thin_pec(scene)
    signed_distance = _pec_signed_distance(scene, coords=coords)
    layout = compile_planar_pec(scene)
    if signed_distance is None:
        return None if layout is None else layout["edges"]
    fills = {
        "Ex": _edge_line_coverage(signed_distance[:-1, :, :], signed_distance[1:, :, :]),
        "Ey": _edge_line_coverage(signed_distance[:, :-1, :], signed_distance[:, 1:, :]),
        "Ez": _edge_line_coverage(signed_distance[:, :, :-1], signed_distance[:, :, 1:]),
    }
    if layout is not None:
        fills = {name: torch.maximum(fill, layout["edges"][name]) for name, fill in fills.items()}
    return fills


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
                (scene.Nx, scene.Ny, scene.Nz), device=scene.device, dtype=_material_dtype(scene)
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
                (scene.Nx, scene.Ny, scene.Nz), device=scene.device, dtype=_material_dtype(scene)
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


def _declared_geometry(scene, structure):
    """The geometry a structure was declared with, before PML-interface extrusion.

    Preparation replaces a PML-contacting structure by a copy whose geometry continues
    through the PML. Ownership guards compare the declared physical geometry, so an
    extruded copy cannot hide a contradictory owner of the same interface.
    """
    public = getattr(scene, "_public_scene", None)
    for record in getattr(scene, "pml_extrusions", ()) if public is not None else ():
        index = record.get("structure_index")
        if index is not None and record["kind"] in ("box", "wrapper") and scene.structures[index] is structure:
            return public.structures[index].geometry
    return getattr(structure, "geometry", None)


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
        surface_geometry = _declared_geometry(scene, surface)
        for other in others:
            if _geometries_coincide(surface_geometry, _declared_geometry(scene, other)):
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
        interior_node_mask=occupancy.to(_material_dtype(scene)),
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


from ._scalar_cache import with_compile_scalar_cache


@with_compile_scalar_cache
def compile_material_model(
    scene,
    eps_background=1.0,
    mu_background=1.0,
    subpixel=None,
    include_edges=True,
):
    chart = getattr(scene, "coordinates", None)
    if chart is not None:
        bounds = tuple((float(getattr(scene, axis + "_nodes64")[0]), float(getattr(scene, axis + "_nodes64")[-1])) for axis in "xyz")
        chart.validate_bounds(bounds)
    if subpixel is None:
        subpixel = scene.subpixel
    samples, averaging, _ = _resolve_subpixel(subpixel)
    if averaging == "kottke_edge" and any(
        _contains_trainable_tensor(value)
        for item in (*scene.structures, *scene.material_regions)
        for value in vars(item.geometry).values()
    ):
        raise ValueError("kottke_edge (the default subpixel averaging) is not differentiable in geometry; set SubpixelSpec(averaging='kottke_smooth') or 'polarized' for shape gradients.")
    if averaging == "kottke_edge" and any(
        _contains_trainable_tensor(value)
        for region in scene.material_regions
        for value in vars(region).values()
    ):
        raise ValueError("kottke_edge (the default subpixel averaging) does not support gradients through MaterialRegion design densities; set SubpixelSpec(samples=1, averaging='arithmetic') for density/topology optimization.")
    with _mesh_fill_cache(scene, samples, include_edges):
        from .accel.dispatch import UNCHANGED, try_compile_material_model
        model = try_compile_material_model(scene, eps_background, mu_background, subpixel, include_edges)
        if model is UNCHANGED:
            model = _compile_material_model(scene, eps_background, mu_background, subpixel, include_edges)
        if material_model_has_modulation(model):
            # Replay only the electric constitutive map at preparation time. Forward
            # AD differentiates the actual arithmetic/harmonic/tensor rule, including
            # its reconstructed normals, without another implementation of that rule.
            # Constitutive derivatives are needed even during inference; keep the
            # caller's reverse-mode setting while evaluating the forward Jacobian.
            grad_enabled = torch.is_grad_enabled()
            with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled):
                region_densities = tuple(_prepare_material_region_density(scene, region)
                                         for region in scene.material_regions)
                geometries = [s.geometry for s in _bulk_structures(scene)]
                geometries.extend(region.geometry for region in scene.material_regions)
                shifts = {id(g): _static_periodic_shift_options(scene, g) for g in geometries}
                absolute = {}
                for index, name in enumerate(("cos", "sin")):
                    with torch.autograd.forward_ad.dual_level():
                        if model["edge_components"] is not None:
                            fields = {component: _compile_edge_component(
                                scene, component, eps_background=eps_background, mu_background=mu_background,
                                averaging=averaging, samples=samples, region_densities=region_densities,
                                geometry_periodic_shifts=shifts, modulation_quadrature=index,
                            )[0] for component in _ELECTRIC_EDGE_COMPONENTS}
                        else:
                            fields = _compile_material_model(
                                scene, eps_background, mu_background, subpixel, False,
                                modulation_quadrature=index,
                            )["eps_components"]
                        absolute[name] = {}
                        for component, field in fields.items():
                            primal, tangent = torch.autograd.forward_ad.unpack_dual(field)
                            absolute[name][component] = torch.zeros_like(primal) if tangent is None else tangent
                model["modulation_absolute"] = absolute
        return model


def _compile_material_model(scene, eps_background, mu_background, subpixel, include_edges,
                            modulation_quadrature=None):
    validate_legacy_thin_pec(scene)
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
            modulation_quadrature=modulation_quadrature,
        )
        if region_densities:
            model["design_mask"] = _material_region_design_mask(scene, model["eps_r"])
        model = _apply_sheet_structures(scene, model)
        model["pec_occupancy"] = _pec_occupancy(scene)
        model["pec_mode"] = pec_mode
        model["pec_edge_fill"] = _pec_edge_fill(scene) if pec_mode == "conformal" else None
        attach_planar_pec(scene, model)
        model["pmc_occupancy"] = _pmc_occupancy(scene)
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
            enabled=include_edges,
            modulation_quadrature=modulation_quadrature,
        )
        from .coordinates import attach_coordinate_metric
        if include_edges and averaging == "kottke_edge":
            _compile_binary_pole_edges(scene, model, layout, samples=samples,
                eps_background=eps_background, mu_background=mu_background,
                region_densities=region_densities, geometry_periodic_shifts=geometry_periodic_shifts)
        return attach_coordinate_metric(scene,model)

    kottke_edge = averaging in {"kottke_edge", "kottke_smooth"}
    full_tensor = averaging == "polarized_tensor" or kottke_edge
    collect_inverse = averaging == "polarized" or full_tensor
    if averaging == "polarized":
        _validate_polarized_materials(scene)
    accum = _new_material_model(scene, layout, eps_fill=0.0, mu_fill=0.0)
    inverse_accum = None
    if collect_inverse:
        shape = (scene.Nx, scene.Ny, scene.Nz)
        inverse_accum = {
            "eps_components": _new_component_field(shape, fill_value=0.0, device=scene.device, dtype=_material_dtype(scene)),
            "mu_components": _new_component_field(shape, fill_value=0.0, device=scene.device, dtype=_material_dtype(scene)),
        }
    if region_densities:
        accum["eps_r_base"] = torch.zeros_like(accum["eps_r"])
        accum["mu_r_base"] = torch.zeros_like(accum["mu_r"])
        accum["eps_r_design"] = torch.zeros_like(accum["eps_r"])
        accum["mu_r_design"] = torch.zeros_like(accum["mu_r"])
    sample_offsets = _sample_offsets(scene, samples)
    anisotropic = None
    modulation_owner = None
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
            fill_spacing=_subsample_spacing(scene, samples),
            full_tensor=full_tensor,
            modulation_quadrature=modulation_quadrature,
        )
        if collect_inverse:
            sample, inverse_sample = compiled_sample
            occupied = inverse_sample.get("anisotropic_eps")
            if occupied is not None:
                anisotropic = occupied if anisotropic is None else anisotropic | occupied
        else:
            sample = compiled_sample
        if "modulation_owner" in sample:
            owner = sample["modulation_owner"]
            modulation_owner = (owner if modulation_owner is None else
                                torch.where(modulation_owner == owner, owner, -1))
        for axis in _AXES:
            accum["eps_components"][axis] += sample["eps_components"][axis]
            accum["mu_components"][axis] += sample["mu_components"][axis]
            accum["sigma_e_components"][axis] += sample["sigma_e_components"][axis]
            accum["sigma_m_components"][axis] += sample["sigma_m_components"][axis]
            if inverse_accum is not None:
                inverse_accum["eps_components"][axis] += inverse_sample["eps_components"][axis]
                inverse_accum["mu_components"][axis] += inverse_sample["mu_components"][axis]
        for pair in _OFFDIAG_AXES:
            accum["eps_offdiag_components"][pair] = _add_material_samples(
                accum["eps_offdiag_components"][pair], sample["eps_offdiag_components"][pair])
        for key in ("kerr_chi3", "chi2", "tpa_sigma", "modulation_cos", "modulation_sin"):
            accum[key] = _add_material_samples(accum[key], sample[key])
        if region_densities:
            accum["eps_r_base"] += sample["eps_r_base"]
            accum["mu_r_base"] += sample["mu_r_base"]
            accum["eps_r_design"] += sample["eps_r_design"]
            accum["mu_r_design"] += sample["mu_r_design"]
        # Frequency is a per-cell label, not an amplitude: take the max over
        # sub-samples so a cell the structure touches in any sub-sample keeps the
        # structure's angular frequency (arithmetic averaging would dilute it).
        if not (_is_constant_zero(accum["modulation_omega"]) and _is_constant_zero(sample["modulation_omega"])):
            accum["modulation_omega"] = torch.maximum(accum["modulation_omega"], sample["modulation_omega"])
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
        if full_tensor and not kottke_edge:
            diagonal, offdiagonal = _reconstruct_sampled_kottke_tensor(
                scene, accum["eps_components"], inverse_accum["eps_components"])
            if anisotropic is not None:
                fallback = _reconstruct_sampled_polarized_inverse(
                    scene, accum["eps_components"], inverse_accum["eps_components"])
                diagonal = {axis: torch.where(anisotropic, fallback[axis], diagonal[axis]) for axis in _AXES}
                offdiagonal = {pair: torch.where(anisotropic,
                    accum["eps_offdiag_components"][pair] * scale, offdiagonal[pair]) for pair in _OFFDIAG_AXES}
            accum["eps_components"], accum["eps_offdiag_components"] = diagonal, offdiagonal
            accum["subpixel_full_tensor"] = True
        else:
            accum["eps_components"] = _reconstruct_sampled_polarized_inverse(
                scene, accum["eps_components"], inverse_accum["eps_components"])
        accum["mu_components"] = _reconstruct_sampled_polarized_inverse(
            scene,
            accum["mu_components"],
            inverse_accum["mu_components"],
        )
    if not accum.get('subpixel_full_tensor', False):
        for pair in _OFFDIAG_AXES:
            accum["eps_offdiag_components"][pair] = _scale_material_sample(accum["eps_offdiag_components"][pair], scale)
    for key in ("kerr_chi3", "chi2", "tpa_sigma", "modulation_cos", "modulation_sin"):
        accum[key] = _scale_material_sample(accum[key], scale)
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
    if modulation_owner is not None:
        model["modulation_owner"] = modulation_owner
    if region_densities:
        model["eps_r_design"] = model["eps_r"] - model["eps_r_base"]
        model["mu_r_design"] = model["mu_r"] - model["mu_r_base"]
    model = _apply_sheet_structures(scene, model)
    model["pec_occupancy"] = _pec_occupancy(scene)
    model["pec_mode"] = pec_mode
    model["pec_edge_fill"] = _pec_edge_fill(scene) if pec_mode == "conformal" else None
    attach_planar_pec(scene, model)
    model["pmc_occupancy"] = _pmc_occupancy(scene)
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
        enabled=include_edges,
        modulation_quadrature=modulation_quadrature,
    )
    from .coordinates import attach_coordinate_metric
    if include_edges and averaging == "kottke_edge":
        _compile_binary_pole_edges(scene, model, layout, samples=samples,
            eps_background=eps_background, mu_background=mu_background,
            region_densities=region_densities, geometry_periodic_shifts=geometry_periodic_shifts)
    return attach_coordinate_metric(scene,model)


def _material_definition(value, *, changing=(), structures=(), snapshot=False):
    """Retain actual fixed material values, not declaration-only shape hashes."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, type, Enum)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().clone() if snapshot else value
    if isinstance(value, np.ndarray):
        return value.copy() if snapshot else value
    if isinstance(value, (tuple, list)):
        return tuple(_material_definition(v, changing=changing, structures=structures, snapshot=snapshot) for v in value)
    if isinstance(value, dict):
        return {k: _material_definition(v, changing=changing, structures=structures, snapshot=snapshot) for k, v in value.items()}
    if is_dataclass(value) or hasattr(value, "__dict__"):
        names = [f.name for f in fields(value)] if is_dataclass(value) else vars(value)
        return {"type": type(value), **{name: _material_definition(getattr(value, name), snapshot=snapshot)
            for name in names if not (name == "density" and getattr(value, "name", None) in changing)
            # Declared design replacements acquire new core identity tokens;
            # these do not change the material field outside the proven envelope.
            and not (isinstance(value, Structure) and name in (
                "geometry", "material", "structure_id", "material_id", "assignment_id", "surface_id")
                     and getattr(value, "name", None) in structures)}}
    return value


def _same_material_definition(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.device == b.device and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_material_definition(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same_material_definition(x, y) for x, y in zip(a, b))
    return a == b


def _crop_material_scene(scene, selection):
    """Sample a node patch with global coordinates and global averaging widths."""
    local = copy.copy(scene)
    local._xx = local._yy = local._zz = None
    for axis, selected in zip(_AXES, selection):
        setattr(local, axis, getattr(scene, axis)[selected])
        setattr(local, f"{axis}_nodes64", getattr(scene, f"{axis}_nodes64")[selected])
        setattr(local, f"d{axis}_dual64", getattr(scene, f"d{axis}_dual64")[selected])
        half = slice(selected.start, selected.stop - 1)
        for name in (f"{axis}_half", f"{axis}_half64", f"d{axis}_primal64"):
            setattr(local, name, getattr(scene, name)[half])
        setattr(local, f"N{axis}", selected.stop - selected.start)
    local.N_total = local.N_total_nodes = local.Nx * local.Ny * local.Nz
    return local


_PLANAR_PEC_MODEL_KEYS = ("pec_volume_occupancy", "pec_sheet_edges", "pec_sheet_owners", "pec_sheet_audit")


def _local_structure_kind(structure):
    """How a change of ``structure`` is recompiled locally, or ``None`` without a proof.

    ``"inert"`` (disabled, contributes nothing), ``"planar"`` (a PEC PlanarPEC sheet,
    recomposed from per-sheet ownership) or ``"bulk"`` (a Box/Sphere volume with a
    static edge-native material, whose influence ends inside its circumsphere plus
    the occupancy and Yee halos). The material admission mirrors the Batch envelope
    guard in :meth:`LocalMaterialCache._validate_structures`.
    """
    from ..geometry.planar_pec import PlanarPEC

    if not getattr(structure, "enabled", True):
        return "inert"
    geometry, material = structure.geometry, structure.material
    if any(isinstance(value, torch.Tensor) and value.requires_grad for value in vars(geometry).values()):
        return None
    if type(geometry) is PlanarPEC:
        return "planar" if bool(getattr(material, "is_pec", False)) else None
    if (type(geometry) not in (Box, Sphere)
            or isinstance(material, PerturbationMedium)
            or any(getattr(material, flag, False) for flag in (
                "is_nonlinear", "is_electric_dispersive", "is_magnetic_dispersive", "is_medium2d",
                "is_surface_impedance", "is_lossy_metal", "is_pec", "is_pmc", "is_copper_foil"))
            or getattr(material, "modulation", None) is not None
            or isinstance(getattr(material, "epsilon_tensor", None), Tensor3x3)):
        return None
    return "bulk"


def _circumsphere_bounds(geometry):
    """Per-axis bounds of a Box/Sphere circumsphere, conservative for every rotation."""
    radius = (float(torch.linalg.vector_norm(geometry.size).item()) / 2
              if type(geometry) is Box else float(geometry.radius))
    position = geometry.position.detach().cpu().numpy().astype(np.float64)
    return tuple((float(p) - radius, float(p) + radius) for p in position)


def _planar_pec_points(geometry):
    """World-space outer-loop vertices of an axis-aligned PlanarPEC sheet."""
    normal = _AXES.index(geometry.normal_axis)
    transverse = [dim for dim in range(3) if dim != normal]
    vertices = geometry.vertices.detach().to(torch.float64)
    points = vertices.new_full((vertices.shape[0], 3), float(geometry.coordinate))
    points[:, transverse[0]] = vertices[:, 0]
    points[:, transverse[1]] = vertices[:, 1]
    return points


class LocalMaterialCache:
    """One immutable material bank plus proven local patch compilation.

    Fixed geometry permits finite support in the float32 occupancy evaluation.
    Periodic axes remain whole so opposite endpoint composition is unchanged.
    Full dense tensors are materialized only where the runtime requires them;
    unchanged channels retain the immutable bank's storage.

    Two admissions share the patch machinery. Batch declares changing density
    regions and named structures with fixed influence envelopes (``compile``).
    ``base=`` adopts an already compiled model of ``scene`` and ``recompile``
    patches it for a scene whose unchanged structures the caller has proven,
    sizing the volume patch from the old and new bounds of the changed volumes.
    PEC PlanarPEC sheets are recomposed from per-sheet ownership: only changed
    sheets are resampled, unchanged ones are reused, and the composition runs
    through the same ``attach_planar_pec`` as the full compile.
    """

    def __init__(self, scene, changing_regions=(), *, structure_envelopes=None, base=None):
        from ..scene import _place_scene_value
        self.changing = tuple(changing_regions)
        self.structure_envelopes = _place_scene_value(dict(structure_envelopes or {}), scene.device, {})
        self._envelope_definition = _material_definition(self.structure_envelopes, snapshot=True)
        self.full_compilations = 1 if base is None else 0
        self.patch_compilations = 0
        self.base = scene.compile_materials() if base is None else base
        if (self.base.get("edge_components") is None or scene._has_trainable_geometry()
                or any(r.density.requires_grad for r in scene.material_regions)
                or any(v.requires_grad for r in scene.material_regions for v in vars(r.geometry).values()
                       if isinstance(v, torch.Tensor))
                or material_model_has_nonlinearity(self.base) or material_model_has_modulation(self.base)
                or _material_model_has_electric_dispersion(self.base) or _material_model_has_magnetic_dispersion(self.base)):
            raise NotImplementedError("Local material reuse requires fixed static edge-native materials.")
        if any(isinstance(s.material, PerturbationMedium) for s in scene.structures):
            raise NotImplementedError("Local node sampling for perturbation grids requires a physical-coordinate sampler.")
        # Per-sheet PlanarPEC ownership of the base scene, keyed by base structure
        # index and filled on first reuse.
        self._planar_sheets = {}
        self._base_kinds = tuple(_local_structure_kind(s) for s in scene.structures)
        if base is not None:
            if self.changing or self.structure_envelopes:
                raise ValueError("An adopted material bank is patched by recompile, not by declared Batch targets.")
            # Snapshot the old bounds now: geometry tensors may be edited in place later.
            self._base_bounds = {i: _circumsphere_bounds(s.geometry)
                                 for i, (s, kind) in enumerate(zip(scene.structures, self._base_kinds)) if kind == "bulk"}
            self.definition = None
            return
        self.definition = self._definition(scene, snapshot=True)
        geometries = [r.geometry for r in scene.material_regions if r.name in self.changing]
        if len(geometries) != len(self.changing) or any(type(g) is not Box for g in geometries):
            raise ValueError("Local density compilation requires named Box material regions.")
        self._validate_structures(scene)
        geometries.extend(self.structure_envelopes.values())
        if not geometries:
            raise ValueError("Local compilation requires at least one changing region or structure.")
        # Tanh is exactly saturated far outside this conservative float32 band.
        # The box circumsphere covers every rotation; subpixel offsets and a
        # further reconstruction halo are retained on each coordinate axis.
        bounds = [(g.position.detach().cpu().numpy(), float(torch.linalg.vector_norm(g.size).item()) / 2)
                  for g in geometries]
        margin = 64 * _geometry_beta(scene) + max(float(np.max(getattr(scene, f"d{a}_dual64"))) for a in _AXES)
        self.bounds = tuple((min(p[i] - r for p, r in bounds) - margin,
                             max(p[i] + r for p, r in bounds) + margin) for i in range(3))
        self.node_selection = self._selection(scene, self._host_axes(scene))
        self.edge_selections = self._edge_selections(scene, self.bounds)

    def _definition(self, scene, *, snapshot=False):
        return _material_definition((scene.domain, scene.grid, scene.boundary, scene.subpixel,
            scene.structures, scene.material_regions), changing=self.changing,
            structures=self.structure_envelopes, snapshot=snapshot)

    def _validate_structures(self, scene):
        from ..geometry.planar_pec import PlanarPEC

        if not _same_material_definition(self._envelope_definition, _material_definition(self.structure_envelopes)):
            raise ValueError("Batch influence envelopes must remain unchanged while their material bank is retained.")
        for name, envelope in self.structure_envelopes.items():
            matches = [s for s in scene.structures if s.name == name]
            if len(matches) != 1 or type(envelope) is not Box:
                raise ValueError("Local structure reuse requires unique named structures and Box envelopes.")
            structure = matches[0]
            geometry = structure.geometry
            if type(geometry) is PlanarPEC:
                # A sheet owns hard tangential edges in its own node plane only;
                # its envelope bounds the declared outline for every variant.
                if _local_structure_kind(structure) != "planar":
                    raise ValueError(f"PlanarPEC structure {name!r} requires Material.pec() and fixed geometry.")
                distance = torch.linalg.vector_norm(
                    _planar_pec_points(geometry) - envelope.position.detach().to(geometry.vertices.device, torch.float64),
                    dim=1).max()
                if not bool(torch.isfinite(distance)) or float(distance) > float(envelope.size.min()) / 2:
                    raise ValueError(f"Structure {name!r} exceeds its conservative Batch influence envelope.")
                continue
            if type(geometry) not in (Box, Sphere):
                raise NotImplementedError("Local structure support is proven for Box and Sphere geometry.")
            # A circumsphere of the design must fit inside the envelope's
            # inscribed sphere. This proof is conservative for every rotation.
            radius = (float(torch.linalg.vector_norm(geometry.size).item()) / 2
                      if type(geometry) is Box else float(geometry.radius))
            distance = float(torch.linalg.vector_norm(geometry.position.detach().to(envelope.position.device)
                                                       - envelope.position).item())
            if not np.isfinite(radius + distance) or radius < 0 or radius + distance > float(envelope.size.min()) / 2:
                raise ValueError(f"Structure {name!r} exceeds its conservative Batch influence envelope.")
            material = structure.material
            if (isinstance(material, PerturbationMedium)
                    or any(getattr(material, flag, False) for flag in (
                        "is_nonlinear", "is_electric_dispersive", "is_magnetic_dispersive",
                        "is_medium2d", "is_surface_impedance", "is_lossy_metal", "is_pec", "is_pmc"))
                    or getattr(material, "modulation", None) is not None
                    or isinstance(getattr(material, "epsilon_tensor", None), Tensor3x3)):
                raise NotImplementedError("Local structure changes require static edge-native volume materials.")

    def _selection(self, scene, axes, bounds=None):
        result = []
        for axis, coordinates, (lo, hi) in zip(_AXES, axes, self.bounds if bounds is None else bounds):
            if scene.boundary.axis_kind(axis) in ("periodic", "bloch"):
                result.append(slice(0, len(coordinates)))
                continue
            values = np.asarray(coordinates)
            selected = np.flatnonzero((values >= lo) & (values <= hi))
            if not selected.size:
                # An exterior design still gets a bounded empty-influence probe.
                start = min(len(values) - 1, int(np.searchsorted(values, lo)))
                stop = start + 1
            else:
                start, stop = int(selected[0]), int(selected[-1]) + 1
            result.append(slice(max(0, start - 2), min(len(coordinates), stop + 2)))
        return tuple(result)

    @staticmethod
    def _host_axes(scene, stagger=()):
        """Host node (or primal-midpoint, on staggered axes) coordinates per axis.

        A selection only has to contain the conservative bounds, which carry a
        cell-scale margin and a two-index pad, so the float64 host nodes stand in
        for the device sampling coordinates without a device round trip.
        """
        axes = []
        for index, axis in enumerate(_AXES):
            nodes = np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64)
            axes.append(0.5 * (nodes[:-1] + nodes[1:]) if index in stagger else nodes)
        return tuple(axes)

    def _edge_selections(self, scene, bounds):
        return {c: self._selection(scene, self._host_axes(scene, _EDGE_STAGGER_AXES[c]), bounds)
                for c in (*_ELECTRIC_EDGE_COMPONENTS, *_MAGNETIC_EDGE_COMPONENTS)}

    def compile(self, scene):
        if self.definition is None:
            raise ValueError("An adopted material bank is patched by recompile, not by declared Batch targets.")
        self._validate_structures(scene)
        if not _same_material_definition(self.definition, self._definition(scene)):
            raise ValueError("Fixed material definition changed during local Batch compilation.")
        if any(not bool(torch.isfinite(r.density).all()) for r in scene.material_regions):
            raise ValueError("Local material compilation requires finite density values.")
        model = dict(self.base)
        named = {i: s for i, s in enumerate(scene.structures) if s.name in self.structure_envelopes}
        kinds = {self._base_kinds[i] for i in named} | {_local_structure_kind(s) for s in named.values()}
        if self.changing or kinds != {"planar"}:
            self._patch_volumes(scene, model, self.node_selection, self.edge_selections)
        if "planar" in kinds:
            # Batch definitions fix every unnamed structure in place.
            model = self._attach_planar(scene, model, {i: i for i in range(len(scene.structures)) if i not in named})
        self.patch_compilations += 1
        return model

    def recompile(self, scene, unchanged):
        """Material model of ``scene`` patched from the adopted base, or ``None``.

        ``unchanged`` maps structure indices of ``scene`` to base structure
        indices; the caller proves those structures identical in every compiled
        input, and the domain, grid, boundary, subpixel specification and device
        identical to the base scene. Every other structure of either scene is a
        change. ``None`` means locality is not proven and the caller compiles the
        full model: a change outside Box/Sphere static volumes and PEC PlanarPEC
        sheets, a volume change in a scene with other volume primitives, a
        reordered mapping, material regions, a mapped coordinate chart, or a
        PlanarPEC change under conformal PEC.
        """
        if self.definition is not None:
            raise ValueError("Declared Batch targets are patched by compile.")
        if getattr(scene, "material_regions", ()) or getattr(scene, "coordinates", None) is not None:
            return None
        pairs = sorted(unchanged.items())
        if any(first[1] >= second[1] for first, second in zip(pairs, pairs[1:])):
            return None
        kept = set(unchanged.values())
        new = {i: _local_structure_kind(s) for i, s in enumerate(scene.structures) if i not in unchanged}
        old = {i: kind for i, kind in enumerate(self._base_kinds) if i not in kept}
        kinds = set(new.values()) | set(old.values())
        if None in kinds or ("planar" in kinds and self.base.get("pec_mode") == "conformal"):
            return None
        model = dict(self.base)
        bounds = [self._base_bounds[i] for i, kind in old.items() if kind == "bulk"]
        bounds += [_circumsphere_bounds(scene.structures[i].geometry) for i, kind in new.items() if kind == "bulk"]
        if bounds:
            # The cropped compile resamples every volume there; only primitives
            # with pointwise occupancy are proven to sample identically on a crop.
            if any(_local_structure_kind(s) != "bulk" for s in _bulk_structures(scene)):
                return None
            margin = 64 * _geometry_beta(scene) + max(float(np.max(getattr(scene, f"d{a}_dual64"))) for a in _AXES)
            box = tuple((min(b[i][0] for b in bounds) - margin, max(b[i][1] for b in bounds) + margin)
                        for i in range(3))
            self._patch_volumes(scene, model, self._selection(scene, self._host_axes(scene), box),
                                self._edge_selections(scene, box))
        # Unchanged sheets that moved in the structure list change owner ids.
        if "planar" in kinds or any(new_index != old_index and self._base_kinds[old_index] == "planar"
                                    for new_index, old_index in pairs):
            model = self._attach_planar(scene, model, unchanged)
        self.patch_compilations += 1
        return model

    @with_compile_scalar_cache
    def _patch_volumes(self, scene, model, node_selection, edge_selections):
        """Resample volume materials inside ``node_selection`` / ``edge_selections``.

        Only bulk structures and material regions enter the cropped compile: PEC,
        PMC, sheet and surface families never write the volume channels.
        """
        selected = node_selection
        halo = tuple(slice(max(0, s.start - 2), min(n, s.stop + 2))
                     for s, n in zip(selected, (scene.Nx, scene.Ny, scene.Nz)))
        crop = tuple(slice(s.start - h.start, s.stop - h.start) for s, h in zip(selected, halo))
        local = _crop_material_scene(scene, halo)
        local.structures = _bulk_structures(scene)
        patch = compile_material_model(local, subpixel=scene.subpixel, include_edges=False)
        # (container, key, bank tensor, patch, selection); unchanged channels keep bank storage.
        writes = []
        for key in ("eps_components", "mu_components", "sigma_e_components", "sigma_m_components"):
            model[key] = dict(self.base[key])
            writes.extend((model[key], a, self.base[key][a], patch[key][a][crop], selected) for a in _AXES)
        if self.base.get('subpixel_full_tensor', False):
            key = 'eps_offdiag_components'
            model[key] = dict(self.base[key])
            writes.extend((model[key], a, self.base[key][a], patch[key][a][crop], selected) for a in _OFFDIAG_AXES)
        for key in ("eps_r", "mu_r", "sigma_e", "eps_r_base", "mu_r_base", "eps_r_design", "mu_r_design"):
            if key in patch:
                writes.append((model, key, self.base[key], patch[key][crop], selected))
        densities = tuple(_prepare_material_region_density(scene, r) for r in scene.material_regions)
        geometries = [s.geometry for s in _bulk_structures(scene)] + [r.geometry for r in scene.material_regions]
        shifts = {id(g): _static_periodic_shift_options(scene, g) for g in geometries}
        model["edge_components"] = {k: dict(v) for k, v in self.base["edge_components"].items()}
        for component, selection in edge_selections.items():
            material, sigma = _compile_edge_component(scene, component, eps_background=1., mu_background=1.,
                samples=scene.subpixel.samples, averaging=scene.subpixel.averaging,
                region_densities=densities, geometry_periodic_shifts=shifts, selection=selection)
            for key, value in (("eps" if component[0] == "E" else "mu", material),
                               ("sigma_e" if component[0] == "E" else "sigma_m", sigma)):
                writes.append((model["edge_components"][key], component,
                               self.base["edge_components"][key][component], value, selection))
        # One host synchronization decides every channel (torch.equal semantics).
        same = torch.stack([(bank[selection] == value).all() for _, _, bank, value, selection in writes]).tolist()
        for (target, key, bank, value, selection), equal in zip(writes, same):
            if equal:
                target[key] = bank
            else:
                target[key] = bank.clone()
                target[key][selection] = value

    def _attach_planar(self, scene, model, unchanged):
        """Recompose PlanarPEC ownership through ``attach_planar_pec``.

        The model is rebuilt in the full compile's key order: the volume PEC
        occupancy is restored, the four sheet keys are dropped, and
        ``attach_planar_pec`` runs at the point of the full compile, reusing the
        per-sheet ownership of unchanged sheets.
        """
        sheets = {new: self._planar_sheets[old] for new, old in unchanged.items() if old in self._planar_sheets}
        volume = model["pec_volume_occupancy"] if "pec_sheet_edges" in model else model["pec_occupancy"]
        head, tail = {}, {}
        target = head
        for key, value in model.items():
            if key in _PLANAR_PEC_MODEL_KEYS:
                continue
            target[key] = value
            if key == "pec_edge_fill":
                target = tail
        head["pec_occupancy"] = volume
        attach_planar_pec(scene, head, sheets=sheets)
        for new, old in unchanged.items():
            if new in sheets:
                self._planar_sheets.setdefault(old, sheets[new])
        head.update(tail)
        return head


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
    complex_dtype = torch.complex128 if model["eps_components"]["x"].dtype == torch.float64 else torch.complex64
    epsilon = {
        axis: model["eps_components"][axis].to(dtype=complex_dtype)
        for axis in _AXES
    }
    if _material_model_has_conductivity(model):
        for axis in _AXES:
            epsilon[axis] = epsilon[axis] - 1j * model["sigma_e_components"][axis].to(dtype=complex_dtype) / (
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
                epsilon[axis] = epsilon[axis] + entry["weight"].to(dtype=complex_dtype) * susceptibility

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
            for axis in entry.get("axes") or _AXES:
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
