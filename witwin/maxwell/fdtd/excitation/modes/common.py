"""Shared constants and mode-plane helpers for the mode-source solvers."""

from __future__ import annotations

import numpy as np
import torch
from witwin.core.material import VACUUM_PERMITTIVITY

from ....compiler.materials import evaluate_material_components
from ..spatial import physical_interior_indices
from ..tfsf_common import nearest_index


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


_FIELD_COMPONENTS = ("Ex", "Ey", "Ez")


_DENSE_EIGEN_LIMIT = 4096


_FULL_VECTOR_DENSE_LIMIT = 384


_LOBPCG_REQUEST_PADDING = 4


_LOBPCG_MAX_ITER = 200


_LOBPCG_TOL = 1.0e-8


_IMPLICIT_EIGEN_CG_MAX_ITER = 256


_IMPLICIT_EIGEN_CG_TOL = 1.0e-8


_VECTOR_EIGEN_REQUEST_PADDING = 4


_VECTOR_EIGS_MAX_ITER = 600


_VECTOR_EIGS_TOL = 1.0e-8


_PEC_OCCUPANCY_THRESHOLD = 0.5


_PEC_VECTOR_MATRIX_LIMIT = 4096


_VECTOR_DEGENERATE_RTOL = 5.0e-5


_VECTOR_DUPLICATE_BETA_RTOL = 1.0e-5


_VECTOR_DUPLICATE_OVERLAP_LIMIT = 0.99


_VECTOR_CHECKERBOARD_FRACTION_LIMIT = 0.35


# A transverse-uniform (kc^2 -> 0, beta -> k0) candidate is the discrete
# null-space branch of the transverse operator. It is the physically correct
# TEM answer for a doubly-connected metallic line (coax, solved on the separate
# electrostatic path), but for a *guided* (TE/TM/hybrid) mode request on a closed
# metallic aperture it is spurious: no hollow-guide mode has beta = k0. It is
# rejected only when BOTH its eigenvalue sits at beta^2 ~ k0^2 AND its transverse
# profile is UNIFORM in an absolute sense (min/max envelope ratio near 1). A
# genuine guided mode -- even a high-frequency one whose beta/k0 approaches 1
# (TE10 at 6 fc has beta/k0 = 0.986, EXECUTED) -- has a half-wave envelope that
# decays to the walls, so its min/max ratio is far below the uniformity limit and
# it is never rejected. This replaces the earlier squared-difference envelope
# threshold, which scaled as (dx)^2 and silently rejected legitimate fine-grid
# modes above ~5 fc (audit S1, executed).
_SPURIOUS_NEAR_K0_BETA_LIMIT = 0.995        # beta/k0 above this = candidate transverse null-space branch


_SPURIOUS_UNIFORMITY_LIMIT = 0.6            # min/max block envelope above this = transverse-uniform (plane-wave-like)


_RIGHT_HANDED_TANGENTIAL_AXES = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


def _cross(a, b):
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _mode_source_axes(source) -> tuple[str, tuple[str, str]]:
    normal_axis = str(source["normal_axis"])
    tangential_axes = _RIGHT_HANDED_TANGENTIAL_AXES[normal_axis]
    return normal_axis, tangential_axes


def _mode_source_field_name(source) -> str:
    return f"E{str(source['polarization_axis']).lower()}"


def _validate_mode_source_position(scene, source, *, plane_index: int) -> None:
    normal_axis = str(source["normal_axis"])
    direction_sign = int(source["direction_sign"])
    physical_lo, physical_hi = physical_interior_indices(scene, normal_axis)
    if direction_sign > 0:
        if plane_index <= physical_lo or plane_index >= physical_hi:
            raise ValueError(
                "ModeSource position must lie at least one cell inside the non-PML region "
                "along the propagation axis."
            )
        return
    if plane_index <= physical_lo or plane_index >= physical_hi:
        raise ValueError(
            "ModeSource position must lie at least one cell inside the non-PML region "
            "along the propagation axis."
        )


def _resolve_tangential_bounds(scene, source, axis_coords_by_axis=None) -> tuple[tuple[int, int], tuple[str, str]]:
    _, tangential_axes = _mode_source_axes(source)
    bounds = []
    for axis in tangential_axes:
        axis_index = _AXIS_TO_INDEX[axis]
        axis_coords = getattr(scene, axis) if axis_coords_by_axis is None else axis_coords_by_axis[axis]
        half_span = 0.5 * float(source["size"][axis_index])
        lower_coord = float(source["position"][axis_index]) - half_span
        upper_coord = float(source["position"][axis_index]) + half_span
        lower_index = nearest_index(axis_coords, lower_coord)
        upper_index = nearest_index(axis_coords, upper_coord)
        if upper_index < lower_index:
            lower_index, upper_index = upper_index, lower_index
        physical_lo, physical_hi = physical_interior_indices(scene, axis)
        lower_limit = float(getattr(scene, axis)[physical_lo].item())
        upper_limit = float(getattr(scene, axis)[physical_hi].item())
        if lower_coord < lower_limit - 1e-12 or upper_coord > upper_limit + 1e-12:
            raise ValueError("ModeSource aperture must remain inside the non-PML region.")
        if upper_index - lower_index < 2:
            raise ValueError(
                "ModeSource aperture must span at least three grid nodes along each tangential axis."
            )
        bounds.append((int(lower_index), int(upper_index)))
    return (bounds[0], bounds[1]), tangential_axes


# Relative transverse-spacing spread at or below which the mode plane is treated
# as exactly uniform and the legacy scalar spacing is returned bit-for-bit.
_UNIFORM_SPACING_RTOL = 1e-6


# Maximum fractional transverse-spacing variation tolerated across a mode-plane
# aperture. The 2D finite-difference mode operator is assembled from a single
# du/dv; on a graded transverse grid a centered stencil evaluated with the mean
# spacing loses one order of accuracy and its leading relative error scales with
# the fractional spacing spread (d_max - d_min)/d_mean. 1e-2 keeps the induced
# effective-index / operator error at roughly the one-percent level.
_MODE_PLANE_SPACING_SPREAD_BOUND = 1e-2


def _local_uniform_plane_spacing(scene, axis: str, lower_index: int, upper_index: int) -> float:
    """Effective transverse spacing across a mode-plane aperture window.

    The 2D mode solver assembles its first/second-difference operators from a
    single transverse spacing. A perfectly uniform window (spread <=
    ``_UNIFORM_SPACING_RTOL`` relative) returns its exact spacing, bit-for-bit as
    before. A mildly graded window is accepted when its fractional spacing
    variation stays below ``_MODE_PLANE_SPACING_SPREAD_BOUND`` and the region-mean
    spacing is used; beyond that the finite-difference operator error is no longer
    controlled and the window is rejected with the predicted variation.
    """
    primal = {"x": scene.dx_primal64, "y": scene.dy_primal64, "z": scene.dz_primal64}[axis]
    window = primal[max(int(lower_index) - 1, 0) : min(int(upper_index) + 1, len(primal))]
    d_min = float(window.min())
    d_max = float(window.max())
    if (d_max - d_min) <= _UNIFORM_SPACING_RTOL * d_max:
        return d_min

    d_eff = float(window.mean())
    fractional_spread = (d_max - d_min) / d_eff
    if fractional_spread > _MODE_PLANE_SPACING_SPREAD_BOUND:
        raise ValueError(
            "ModeSource/ModeMonitor mode solving is too graded across the mode "
            f"plane along axis '{axis}': the fractional transverse-spacing "
            f"variation is {fractional_spread:.3e} (min={d_min:g}, max={d_max:g}), "
            f"above the bound {_MODE_PLANE_SPACING_SPREAD_BOUND:.0e}. Refine "
            f"GridSpec.custom for locally uniform grid spacing along axis '{axis}' "
            "across the mode plane, or move the plane."
        )
    return d_eff


def _field_component_axis_coords(scene, field_name: str, axis: str) -> torch.Tensor:
    if field_name == "Ex":
        mapping = {
            "x": scene.x_half,
            "y": scene.y,
            "z": scene.z,
        }
    elif field_name == "Ey":
        mapping = {
            "x": scene.x,
            "y": scene.y_half,
            "z": scene.z,
        }
    elif field_name == "Ez":
        mapping = {
            "x": scene.x,
            "y": scene.y,
            "z": scene.z_half,
        }
    else:
        raise ValueError(f"Unsupported ModeSource field component {field_name!r}.")
    return mapping[axis]


def _average_node_tensor_to_component(node_tensor: torch.Tensor, field_name: str) -> torch.Tensor:
    if field_name == "Ex":
        return (0.5 * (node_tensor[:-1, :, :] + node_tensor[1:, :, :])).contiguous()
    if field_name == "Ey":
        return (0.5 * (node_tensor[:, :-1, :] + node_tensor[:, 1:, :])).contiguous()
    if field_name == "Ez":
        return (0.5 * (node_tensor[:, :, :-1] + node_tensor[:, :, 1:])).contiguous()
    raise ValueError(f"Unsupported ModeSource field component {field_name!r}.")


def _mode_source_node_axis_coords(scene, axis: str) -> torch.Tensor:
    return getattr(scene, axis)


def _mode_source_relative_material_slices(
    solver,
    *,
    frequency: float,
    normal_axis: str,
    plane_index: int,
    tangential_bounds,
):
    """Per-axis diagonal eps/mu aperture slices for the full-vector mode solve.

    Returns ``(eps_by_axis, mu_by_axis)`` where each maps ``"x"/"y"/"z"`` to the
    diagonal permittivity/permeability component sliced on the source plane. The
    full-vector operator consumes the three components separately so a
    diagonal-anisotropic aperture is resolved with its true per-axis tensor
    instead of an isotropic average.
    """
    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is None:
        raise RuntimeError("Full-vector ModeSource currently requires solver._compiled_material_model.")

    eps_components, mu_components = evaluate_material_components(compiled_material_model, float(frequency))
    eps_by_axis = {
        axis: _mode_slice(eps_components[axis], axis=normal_axis, plane_index=plane_index, tangential_bounds=tangential_bounds)
        for axis in ("x", "y", "z")
    }
    mu_by_axis = {
        axis: _mode_slice(mu_components[axis], axis=normal_axis, plane_index=plane_index, tangential_bounds=tangential_bounds)
        for axis in ("x", "y", "z")
    }
    return eps_by_axis, mu_by_axis


def _mode_source_pec_slice(
    solver,
    *,
    normal_axis: str,
    plane_index: int,
    tangential_bounds,
) -> torch.Tensor | None:
    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is None:
        return None
    occupancy = compiled_material_model.get("pec_occupancy")
    if occupancy is None:
        return None
    return _mode_slice(
        occupancy,
        axis=normal_axis,
        plane_index=plane_index,
        tangential_bounds=tangential_bounds,
    )


def _max_component_imag(components: dict) -> float:
    """Largest imaginary magnitude across the per-axis diagonal aperture slices."""
    worst = 0.0
    for component in components.values():
        if torch.is_complex(component):
            worst = max(worst, float(torch.max(torch.abs(torch.imag(component))).item()))
    return worst


def _min_component_real(components: dict) -> float:
    """Smallest real part across the per-axis diagonal aperture slices."""
    return min(
        float(torch.min(torch.real(component) if torch.is_complex(component) else component).item())
        for component in components.values()
    )


def _label_connected_components(active: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected components of a boolean 2D mask (True = member).

    Returns ``(labels, count)`` with ``labels`` a ``int32`` array (0 for non-members,
    1..count for members) and ``count`` the number of connected components. A small
    flood-fill keeps this dependency-free (SciPy's ``label`` is avoided so the mode
    solver has no new import).
    """
    active = np.asarray(active, dtype=bool)
    labels = np.zeros(active.shape, dtype=np.int32)
    count = 0
    nu, nv = active.shape
    for start_u in range(nu):
        for start_v in range(nv):
            if not active[start_u, start_v] or labels[start_u, start_v] != 0:
                continue
            count += 1
            stack = [(start_u, start_v)]
            labels[start_u, start_v] = count
            while stack:
                cu, cv = stack.pop()
                for du_step, dv_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nu_i, nv_i = cu + du_step, cv + dv_step
                    if 0 <= nu_i < nu and 0 <= nv_i < nv and active[nu_i, nv_i] and labels[nu_i, nv_i] == 0:
                        labels[nu_i, nv_i] = count
                        stack.append((nu_i, nv_i))
    return labels, count


def _normalize_mode_profiles_to_unit_power(
    component_profiles: dict[str, torch.Tensor],
    *,
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
    normal_axis: str,
) -> dict[str, torch.Tensor]:
    """Scale a common-node modal profile to one watt on the transverse Yee grids."""
    power = _discrete_mode_profile_power(
        component_profiles,
        coords_u=coords_u,
        coords_v=coords_v,
        normal_axis=normal_axis,
    )
    abs_power = torch.abs(power)
    if float(abs_power.detach().item()) <= torch.finfo(abs_power.dtype).eps:
        raise RuntimeError("ModeSource eigenmode profile has zero integrated Poynting power.")
    scale = torch.rsqrt(abs_power)
    return {name: profile * scale for name, profile in component_profiles.items()}


def _discrete_mode_profile_power(
    component_profiles: dict[str, torch.Tensor],
    *,
    coords_u: torch.Tensor,
    coords_v: torch.Tensor,
    normal_axis: str,
) -> torch.Tensor:
    """Integrate signed modal power after interpolation to the transverse Yee grids."""
    if coords_u.ndim != 1 or coords_v.ndim != 1 or coords_u.numel() < 2 or coords_v.numel() < 2:
        raise ValueError("ModeSource profile coordinates must be one-dimensional with at least two samples.")
    expected_shape = (int(coords_u.numel()), int(coords_v.numel()))
    for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        profile = component_profiles[field_name]
        if tuple(int(dim) for dim in profile.shape) != expected_shape:
            raise ValueError(
                f"ModeSource {field_name} profile shape {tuple(profile.shape)} does not match "
                f"coordinate shape {expected_shape}."
            )

    axis_u, axis_v = _RIGHT_HANDED_TANGENTIAL_AXES[normal_axis]
    electric_u = component_profiles[f"E{axis_u}"]
    electric_v = component_profiles[f"E{axis_v}"]
    magnetic_u = component_profiles[f"H{axis_u}"]
    magnetic_v = component_profiles[f"H{axis_v}"]

    coords_u_half = 0.5 * (coords_u[:-1] + coords_u[1:])
    coords_v_half = 0.5 * (coords_v[:-1] + coords_v[1:])
    electric_u_half = 0.5 * (electric_u[:-1, :] + electric_u[1:, :])
    magnetic_v_half = 0.5 * (magnetic_v[:-1, :] + magnetic_v[1:, :])
    electric_v_half = 0.5 * (electric_v[:, :-1] + electric_v[:, 1:])
    magnetic_u_half = 0.5 * (magnetic_u[:, :-1] + magnetic_u[:, 1:])

    positive_density = 0.5 * torch.real(electric_u_half * torch.conj(magnetic_v_half))
    negative_density = 0.5 * torch.real(electric_v_half * torch.conj(magnetic_u_half))
    positive_power = torch.trapezoid(
        torch.trapezoid(positive_density, x=coords_v, dim=1),
        x=coords_u_half,
        dim=0,
    )
    negative_power = torch.trapezoid(
        torch.trapezoid(negative_density, x=coords_v_half, dim=1),
        x=coords_u,
        dim=0,
    )
    return positive_power - negative_power


def _mode_slice(tensor: torch.Tensor, *, axis: str, plane_index: int, tangential_bounds):
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    if axis == "x":
        return tensor[plane_index, u_lo : u_hi + 1, v_lo : v_hi + 1]
    if axis == "y":
        return tensor[u_lo : u_hi + 1, plane_index, v_lo : v_hi + 1]
    return tensor[u_lo : u_hi + 1, v_lo : v_hi + 1, plane_index]


def _bend_conformal_factor(source, *, tangential_axes, tangential_bounds, coord_map):
    """Heiblum-Harris conformal index factor over a bent-port mode plane.

    Returns ``None`` for a straight port. A curved (bent) waveguide keeps its
    cross-section plane axis-aligned but its guided mode is solved on the
    equivalent straight guide produced by the conformal map: for a bend of signed
    radius ``R`` about the cylinder axis ``bend_axis``, the transformed
    permittivity is ``eps_eq(u, v) = eps(u, v) * (1 + r/R)**2`` where ``r`` is the
    signed offset of each node from the port centre along the in-plane radial axis
    (the tangential axis that is not the cylinder axis). ``R > 0`` grades the index
    up on the ``+r`` side, shifting the mode outward and raising the effective
    index, exactly as a physical bend does. The returned float64 tensor has shape
    ``(nu, 1)`` or ``(1, nv)`` so it broadcasts against a mode-plane aperture slice.
    """
    bend_radius = source.get("bend_radius")
    if bend_radius is None:
        return None
    bend_axis = str(source["bend_axis"])
    axis_u, axis_v = tangential_axes
    radial_is_u = axis_v == bend_axis
    radial_axis = axis_u if radial_is_u else axis_v
    (u_lo, u_hi), (v_lo, v_hi) = tangential_bounds
    lo, hi = (u_lo, u_hi) if radial_is_u else (v_lo, v_hi)
    coords_radial = coord_map[radial_axis][lo : hi + 1].to(dtype=torch.float64)
    radial_center = float(source["position"][_AXIS_TO_INDEX[radial_axis]])
    ratio = 1.0 + (coords_radial - radial_center) / float(bend_radius)
    if float(torch.min(ratio).item()) <= 0.0:
        raise ValueError(
            "Bent mode-plane aperture spans the bend centre of curvature "
            f"(1 + r/R <= 0 for bend_radius={float(bend_radius):g} about axis {bend_axis!r}); "
            "the conformal transform is singular there. Increase |bend_radius| or shrink the aperture."
        )
    factor_1d = ratio * ratio
    return factor_1d[:, None] if radial_is_u else factor_1d[None, :]


def _apply_bend_factor(tensor: torch.Tensor, factor) -> torch.Tensor:
    """Multiply a mode-plane eps slice by the conformal bend factor (identity if None)."""
    if factor is None:
        return tensor
    real_dtype = torch.real(tensor).dtype if torch.is_complex(tensor) else tensor.dtype
    return tensor * factor.to(device=tensor.device, dtype=real_dtype)


def _regular_grid_sample(coords: torch.Tensor, values: torch.Tensor):
    if values.numel() < 2:
        raise ValueError("ModeSource profile sampling requires at least two grid points per axis.")
    spacing = float(coords[1].item() - coords[0].item())
    if spacing <= 0.0:
        raise ValueError("ModeSource tangential coordinates must be strictly increasing.")
    return float(coords[0].item()), spacing


def _bilinear_sample(profile: torch.Tensor, coords_u: torch.Tensor, coords_v: torch.Tensor, positions: torch.Tensor):
    u_axis = int(profile.shape[0])
    v_axis = int(profile.shape[1])
    u0, du = _regular_grid_sample(coords_u, profile[:, 0])
    v0, dv = _regular_grid_sample(coords_v, profile[0, :])

    pos_u = positions[..., 0]
    pos_v = positions[..., 1]
    mask = (
        (pos_u >= float(coords_u[0].item()))
        & (pos_u <= float(coords_u[-1].item()))
        & (pos_v >= float(coords_v[0].item()))
        & (pos_v <= float(coords_v[-1].item()))
    )

    u_coord = torch.clamp((pos_u - u0) / du, min=0.0, max=max(u_axis - 1, 0))
    v_coord = torch.clamp((pos_v - v0) / dv, min=0.0, max=max(v_axis - 1, 0))
    u_lower = torch.floor(u_coord).to(torch.int64)
    v_lower = torch.floor(v_coord).to(torch.int64)
    u_upper = torch.clamp(u_lower + 1, max=u_axis - 1)
    v_upper = torch.clamp(v_lower + 1, max=v_axis - 1)
    u_frac = u_coord - u_lower.to(dtype=profile.dtype)
    v_frac = v_coord - v_lower.to(dtype=profile.dtype)

    f00 = profile[u_lower, v_lower]
    f10 = profile[u_upper, v_lower]
    f01 = profile[u_lower, v_upper]
    f11 = profile[u_upper, v_upper]
    sampled = (
        (1.0 - u_frac) * (1.0 - v_frac) * f00
        + u_frac * (1.0 - v_frac) * f10
        + (1.0 - u_frac) * v_frac * f01
        + u_frac * v_frac * f11
    )
    return torch.where(mask, sampled, torch.zeros_like(sampled))


def _mode_source_component_permittivity(solver, source, *, frequency: float) -> torch.Tensor:
    field_name = _mode_source_field_name(source)
    component_name = f"eps_{field_name}"
    field_tensor = getattr(solver, component_name, None)
    use_component_fields = bool(getattr(solver, "_mode_source_rebuild_from_fields", False))
    use_component_fields = use_component_fields or (
        isinstance(field_tensor, torch.Tensor) and bool(field_tensor.requires_grad)
    )

    if use_component_fields:
        if field_tensor is None:
            raise RuntimeError(
                f"ModeSource differentiable rebuild requires {component_name} on the temporary solver context."
            )
        eps0 = float(getattr(solver, "eps0", VACUUM_PERMITTIVITY))
        return field_tensor / eps0

    compiled_material_model = getattr(solver, "_compiled_material_model", None)
    if compiled_material_model is not None:
        # A diagonal-anisotropic aperture must be solved with the permittivity
        # component that matches the injected polarization (E_p sees eps_pp), the
        # same per-axis component the forward Yee update uses for that field
        # (solver.eps_E{p}). Averaging the three diagonal components would inject a
        # mode computed for an isotropic medium the forward solve never sees. For an
        # isotropic plane the three components coincide and this is unchanged.
        polarization_axis = field_name[1].lower()
        eps_components, _ = evaluate_material_components(compiled_material_model, float(frequency))
        return _average_node_tensor_to_component(eps_components[polarization_axis], field_name)

    if field_tensor is None:
        raise RuntimeError(
            "ModeSource requires either solver._compiled_material_model or component permittivity tensors."
        )
    eps0 = float(getattr(solver, "eps0", VACUUM_PERMITTIVITY))
    return field_tensor / eps0


def _normalize_profile_torch(profile: torch.Tensor) -> torch.Tensor:
    peak = torch.max(torch.abs(profile))
    if float(peak.item()) <= 0.0:
        raise RuntimeError("ModeSource eigenmode solve returned a zero profile.")
    normalized = profile / peak
    peak_index = int(torch.argmax(torch.abs(normalized).reshape(-1)).item())
    if float(normalized.reshape(-1)[peak_index].item()) < 0.0:
        normalized = -normalized
    return normalized


def _vector_mode_supported(
    solver,
    *,
    eps_by_axis: dict,
    mu_by_axis: dict,
    unknowns: int,
) -> bool:
    if getattr(solver, "_mode_source_rebuild_from_fields", False):
        return False
    if getattr(solver, "_compiled_material_model", None) is None:
        return False
    if any(component.requires_grad for component in eps_by_axis.values()):
        return False
    if any(component.requires_grad for component in mu_by_axis.values()):
        return False
    if unknowns <= 0:
        return False
    return True


def _real_plane_numpy(component: torch.Tensor) -> np.ndarray:
    real = torch.real(component) if torch.is_complex(component) else component
    return real.detach().cpu().numpy().astype(np.float64, copy=False)


def _boundary_connected_conductor(conductor: torch.Tensor) -> torch.Tensor:
    connected = torch.zeros_like(conductor)
    connected[0, :] = conductor[0, :]
    connected[-1, :] = conductor[-1, :]
    connected[:, 0] |= conductor[:, 0]
    connected[:, -1] |= conductor[:, -1]
    for _ in range(int(conductor.shape[0] + conductor.shape[1])):
        adjacent = torch.zeros_like(conductor)
        adjacent[1:, :] |= connected[:-1, :]
        adjacent[:-1, :] |= connected[1:, :]
        adjacent[:, 1:] |= connected[:, :-1]
        adjacent[:, :-1] |= connected[:, 1:]
        connected |= adjacent & conductor
    return connected
