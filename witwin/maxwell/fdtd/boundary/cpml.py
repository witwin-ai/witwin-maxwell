import math

import torch

from .common import BOUNDARY_PML


DEFAULT_CPML_CONFIG = {
    "grading_order": 4.0,
    "kappa_max": 7.0,
    "alpha_max": 0.1,
    "reflection": 1e-8,
    "memory_mode": "auto",
    "dense_memory_limit_mib": 64.0,
}

_AUTO_CPML_DENSE_FREE_FRACTION = 0.125

_CPML_VECTOR_ATTRS = (
    "cpml_kappa_e_x",
    "cpml_kappa_e_y",
    "cpml_kappa_e_z",
    "cpml_inv_kappa_e_x",
    "cpml_inv_kappa_e_y",
    "cpml_inv_kappa_e_z",
    "cpml_b_e_x",
    "cpml_b_e_y",
    "cpml_b_e_z",
    "cpml_c_e_x",
    "cpml_c_e_y",
    "cpml_c_e_z",
    "cpml_kappa_h_x",
    "cpml_kappa_h_y",
    "cpml_kappa_h_z",
    "cpml_inv_kappa_h_x",
    "cpml_inv_kappa_h_y",
    "cpml_inv_kappa_h_z",
    "cpml_b_h_x",
    "cpml_b_h_y",
    "cpml_b_h_z",
    "cpml_c_h_x",
    "cpml_c_h_y",
    "cpml_c_h_z",
)

_CPML_MEMORY_ATTRS = (
    "psi_ex_y",
    "psi_ex_z",
    "psi_ey_x",
    "psi_ey_z",
    "psi_ez_x",
    "psi_ez_y",
    "psi_hx_y",
    "psi_hx_z",
    "psi_hy_x",
    "psi_hy_z",
    "psi_hz_x",
    "psi_hz_y",
)
_CPML_IMAG_MEMORY_ATTRS = tuple(f"{attr_name}_imag" for attr_name in _CPML_MEMORY_ATTRS)

_CPML_MEMORY_SPECS = {
    "psi_ex_y": ("Ex", 1),
    "psi_ex_z": ("Ex", 2),
    "psi_ey_x": ("Ey", 0),
    "psi_ey_z": ("Ey", 2),
    "psi_ez_x": ("Ez", 0),
    "psi_ez_y": ("Ez", 1),
    "psi_hx_y": ("Hx", 1),
    "psi_hx_z": ("Hx", 2),
    "psi_hy_x": ("Hy", 0),
    "psi_hy_z": ("Hy", 2),
    "psi_hz_x": ("Hz", 0),
    "psi_hz_y": ("Hz", 1),
}

_SIGMA_ATTRS = (
    "sigma_x",
    "sigma_y",
    "sigma_z",
)


def _legacy_strength_scale(strength):
    value = float(strength)
    if value <= 0:
        return 1.0
    if value > 100.0:
        return 1.0
    return value


def _cpml_profile_1d(
    length,
    thickness,
    delta,
    dt,
    vacuum_constant,
    sigma_scale,
    impedance,
    config,
    *,
    half_step=False,
    apply_low=True,
    apply_high=True,
):
    sigma = torch.zeros(length, device=sigma_scale.device, dtype=torch.float32)
    kappa = torch.ones(length, device=sigma_scale.device, dtype=torch.float32)
    alpha = torch.zeros(length, device=sigma_scale.device, dtype=torch.float32)

    if thickness <= 0:
        b = torch.zeros_like(sigma)
        c = torch.zeros_like(sigma)
        return kappa, b, c

    grading_order = float(config["grading_order"])
    kappa_max = float(config["kappa_max"])
    alpha_max = float(config["alpha_max"])
    reflection = float(config["reflection"])
    sigma_max = -(grading_order + 1.0) * math.log(reflection) / (2.0 * impedance * thickness * delta)
    sigma_max *= float(sigma_scale.item())

    positions = torch.arange(length, device=sigma_scale.device, dtype=torch.float32)
    positions = positions + (0.5 if half_step else 0.0)
    distance = torch.zeros_like(positions)
    if apply_low:
        left_distance = torch.clamp(thickness - positions, min=0.0) / thickness
        distance = torch.maximum(distance, left_distance)
    if apply_high:
        right_origin = length - thickness - (0.5 if half_step else 1.0)
        right_distance = torch.clamp(positions - right_origin, min=0.0) / thickness
        distance = torch.maximum(distance, right_distance)
    active = distance > 0.0
    graded = torch.zeros_like(distance)
    graded[active] = distance[active] ** grading_order
    sigma[active] = sigma_max * graded[active]
    kappa[active] = 1.0 + (kappa_max - 1.0) * graded[active]
    alpha[active] = alpha_max * (1.0 - distance[active])

    decay = torch.exp(-(sigma / kappa + alpha) * dt / vacuum_constant)
    denom = sigma + alpha * kappa
    c = torch.zeros_like(sigma)
    mask = denom > 1e-12
    c[mask] = sigma[mask] * (decay[mask] - 1.0) / (denom[mask] * kappa[mask])
    b = decay
    return kappa.contiguous(), b.contiguous(), c.contiguous()


def _axis_face_uses_pml(solver, axis: str, *, low: bool) -> bool:
    attr_name = f"boundary_{axis}_{'low' if low else 'high'}_code"
    return int(getattr(solver, attr_name, solver.boundary_code)) == BOUNDARY_PML


def _build_cpml_memory_layout(solver, attr_name):
    field_name, axis = _CPML_MEMORY_SPECS[attr_name]
    field = getattr(solver, field_name)
    axis_name = "xyz"[axis]
    field_shape = tuple(int(size) for size in field.shape)
    axis_size = int(field_shape[axis])
    thickness = min(int(solver.scene.pml_thickness), axis_size)
    low_length = thickness if _axis_face_uses_pml(solver, axis_name, low=True) else 0
    high_capacity = max(axis_size - low_length, 0)
    high_length = min(thickness if _axis_face_uses_pml(solver, axis_name, low=False) else 0, high_capacity)
    compressed_shape = list(field_shape)
    compressed_shape[axis] = low_length + high_length
    layout = {
        "field_name": field_name,
        "field_shape": field_shape,
        "compressed_shape": tuple(compressed_shape),
        "axis": axis,
        "regions": [],
    }
    local_start = 0
    if low_length > 0:
        layout["regions"].append(
            {
                "side": "low",
                "global_start": 0,
                "length": low_length,
                "local_start": local_start,
            }
        )
        local_start += low_length
    if high_length > 0:
        layout["regions"].append(
            {
                "side": "high",
                "global_start": axis_size - high_length,
                "length": high_length,
                "local_start": local_start,
            }
        )
    return layout


def _estimate_cpml_memory_bytes(layouts, solver):
    dense_bytes = 0
    slab_bytes = 0
    for layout in layouts.values():
        field = getattr(solver, layout["field_name"])
        element_bytes = int(field.element_size())
        dense_bytes += math.prod(layout["field_shape"]) * element_bytes
        slab_bytes += math.prod(layout["compressed_shape"]) * element_bytes
    return dense_bytes, slab_bytes


def _requested_cpml_memory_mode(solver):
    requested = str(solver.cpml_config.get("memory_mode", "auto")).strip().lower()
    if requested not in {"auto", "dense", "slab"}:
        raise ValueError(
            "cpml_config['memory_mode'] must be one of 'auto', 'dense', or 'slab'."
        )
    return requested


def _resolve_cpml_memory_mode(solver, dense_bytes):
    requested = _requested_cpml_memory_mode(solver)
    solver._cpml_memory_mode_requested = requested
    if requested != "auto":
        solver._cpml_dense_memory_limit_bytes = None
        solver._cpml_auto_free_bytes = None
        return requested

    limit_mib = float(
        solver.cpml_config.get("dense_memory_limit_mib", DEFAULT_CPML_CONFIG["dense_memory_limit_mib"])
    )
    limit_bytes = max(int(limit_mib * 1024 * 1024), 0)
    free_bytes = None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(solver.device)
    except (RuntimeError, ValueError):
        free_bytes = None
    if free_bytes is not None:
        limit_bytes = min(limit_bytes, int(float(free_bytes) * _AUTO_CPML_DENSE_FREE_FRACTION))
    solver._cpml_dense_memory_limit_bytes = limit_bytes
    solver._cpml_auto_free_bytes = None if free_bytes is None else int(free_bytes)
    return "dense" if dense_bytes <= limit_bytes else "slab"


def _allocate_cpml_memory_variables(solver):
    layouts = {attr_name: _build_cpml_memory_layout(solver, attr_name) for attr_name in _CPML_MEMORY_ATTRS}
    dense_bytes, slab_bytes = _estimate_cpml_memory_bytes(layouts, solver)
    memory_multiplier = 2 if bool(getattr(solver, "complex_fields_enabled", False)) else 1
    dense_bytes *= memory_multiplier
    slab_bytes *= memory_multiplier
    memory_mode = _resolve_cpml_memory_mode(solver, dense_bytes)

    solver._cpml_memory_layouts = {}
    solver._cpml_memory_mode = memory_mode
    solver._cpml_dense_memory_bytes = dense_bytes
    solver._cpml_slab_memory_bytes = slab_bytes
    solver._cpml_allocated_memory_bytes = 0

    for attr_name, layout in layouts.items():
        field = getattr(solver, layout["field_name"])
        target_shape = layout["field_shape"] if memory_mode == "dense" else layout["compressed_shape"]
        tensor = torch.zeros(target_shape, device=solver.device, dtype=field.dtype)
        if memory_mode == "slab":
            solver._cpml_memory_layouts[attr_name] = layout
        solver._cpml_allocated_memory_bytes += tensor.numel() * tensor.element_size()
        setattr(solver, attr_name, tensor)
        if bool(getattr(solver, "complex_fields_enabled", False)):
            imag_attr_name = f"{attr_name}_imag"
            imag_tensor = torch.zeros(target_shape, device=solver.device, dtype=field.dtype)
            if memory_mode == "slab":
                solver._cpml_memory_layouts[imag_attr_name] = layout
            solver._cpml_allocated_memory_bytes += imag_tensor.numel() * imag_tensor.element_size()
            setattr(solver, imag_attr_name, imag_tensor)


def expand_cpml_memory_tensor(solver, attr_name):
    tensor = getattr(solver, attr_name)
    layout = getattr(solver, "_cpml_memory_layouts", {}).get(attr_name)
    if layout is None:
        return tensor

    dense = torch.zeros(layout["field_shape"], device=tensor.device, dtype=tensor.dtype)
    axis = int(layout["axis"])
    for region in layout["regions"]:
        length = int(region["length"])
        if length <= 0:
            continue
        dense.narrow(axis, int(region["global_start"]), length).copy_(
            tensor.narrow(axis, int(region["local_start"]), length)
        )
    return dense


def _initialize_neutral_cpml_buffers(solver):
    solver.cpml_kappa_e_x = torch.ones(solver.Nx, device=solver.device, dtype=torch.float32)
    solver.cpml_kappa_e_y = torch.ones(solver.Ny, device=solver.device, dtype=torch.float32)
    solver.cpml_kappa_e_z = torch.ones(solver.Nz, device=solver.device, dtype=torch.float32)
    solver.cpml_b_e_x = torch.zeros(solver.Nx, device=solver.device, dtype=torch.float32)
    solver.cpml_b_e_y = torch.zeros(solver.Ny, device=solver.device, dtype=torch.float32)
    solver.cpml_b_e_z = torch.zeros(solver.Nz, device=solver.device, dtype=torch.float32)
    solver.cpml_c_e_x = torch.zeros(solver.Nx, device=solver.device, dtype=torch.float32)
    solver.cpml_c_e_y = torch.zeros(solver.Ny, device=solver.device, dtype=torch.float32)
    solver.cpml_c_e_z = torch.zeros(solver.Nz, device=solver.device, dtype=torch.float32)

    solver.cpml_kappa_h_x = torch.ones(solver.Nx - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_kappa_h_y = torch.ones(solver.Ny - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_kappa_h_z = torch.ones(solver.Nz - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_b_h_x = torch.zeros(solver.Nx - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_b_h_y = torch.zeros(solver.Ny - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_b_h_z = torch.zeros(solver.Nz - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_c_h_x = torch.zeros(solver.Nx - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_c_h_y = torch.zeros(solver.Ny - 1, device=solver.device, dtype=torch.float32)
    solver.cpml_c_h_z = torch.zeros(solver.Nz - 1, device=solver.device, dtype=torch.float32)


def _allocate_sigma_tensors(solver):
    solver.sigma_x = torch.zeros((solver.Nx, solver.Ny, solver.Nz), device=solver.device, dtype=torch.float32)
    solver.sigma_y = torch.zeros((solver.Nx, solver.Ny, solver.Nz), device=solver.device, dtype=torch.float32)
    solver.sigma_z = torch.zeros((solver.Nx, solver.Ny, solver.Nz), device=solver.device, dtype=torch.float32)


def _set_inv_kappa_buffers(solver):
    solver.cpml_inv_kappa_e_x = torch.reciprocal(solver.cpml_kappa_e_x).contiguous()
    solver.cpml_inv_kappa_e_y = torch.reciprocal(solver.cpml_kappa_e_y).contiguous()
    solver.cpml_inv_kappa_e_z = torch.reciprocal(solver.cpml_kappa_e_z).contiguous()
    solver.cpml_inv_kappa_h_x = torch.reciprocal(solver.cpml_kappa_h_x).contiguous()
    solver.cpml_inv_kappa_h_y = torch.reciprocal(solver.cpml_kappa_h_y).contiguous()
    solver.cpml_inv_kappa_h_z = torch.reciprocal(solver.cpml_kappa_h_z).contiguous()


def _clear_auxiliary_state(solver):
    for attr in _CPML_VECTOR_ATTRS + _CPML_MEMORY_ATTRS + _CPML_IMAG_MEMORY_ATTRS + _SIGMA_ATTRS:
        setattr(solver, attr, None)
    solver._cpml_memory_layouts = {}
    solver._cpml_memory_mode = "none"
    solver._cpml_dense_memory_bytes = 0
    solver._cpml_slab_memory_bytes = 0
    solver._cpml_allocated_memory_bytes = 0
    solver._cpml_dense_memory_limit_bytes = None
    solver._cpml_auto_free_bytes = None


def build_cpml_profiles(solver):
    strength_scale = _legacy_strength_scale(solver.scene.pml_strength)
    scale_tensor = torch.tensor(strength_scale, device=solver.device, dtype=torch.float32)
    thickness = solver.scene.pml_thickness
    eta0 = math.sqrt(solver.mu0 / solver.eps0)

    solver.cpml_kappa_e_x, solver.cpml_b_e_x, solver.cpml_c_e_x = _cpml_profile_1d(
        solver.Nx,
        thickness,
        solver.dx,
        solver.dt,
        solver.eps0,
        scale_tensor,
        eta0,
        solver.cpml_config,
        half_step=False,
        apply_low=_axis_face_uses_pml(solver, "x", low=True),
        apply_high=_axis_face_uses_pml(solver, "x", low=False),
    )
    solver.cpml_kappa_e_y, solver.cpml_b_e_y, solver.cpml_c_e_y = _cpml_profile_1d(
        solver.Ny,
        thickness,
        solver.dy,
        solver.dt,
        solver.eps0,
        scale_tensor,
        eta0,
        solver.cpml_config,
        half_step=False,
        apply_low=_axis_face_uses_pml(solver, "y", low=True),
        apply_high=_axis_face_uses_pml(solver, "y", low=False),
    )
    solver.cpml_kappa_e_z, solver.cpml_b_e_z, solver.cpml_c_e_z = _cpml_profile_1d(
        solver.Nz,
        thickness,
        solver.dz,
        solver.dt,
        solver.eps0,
        scale_tensor,
        eta0,
        solver.cpml_config,
        half_step=False,
        apply_low=_axis_face_uses_pml(solver, "z", low=True),
        apply_high=_axis_face_uses_pml(solver, "z", low=False),
    )

    magnetic_scale = scale_tensor * (solver.mu0 / solver.eps0)
    solver.cpml_kappa_h_x, solver.cpml_b_h_x, solver.cpml_c_h_x = _cpml_profile_1d(
        solver.Nx - 1,
        thickness,
        solver.dx,
        solver.dt,
        solver.mu0,
        magnetic_scale,
        eta0,
        solver.cpml_config,
        half_step=True,
        apply_low=_axis_face_uses_pml(solver, "x", low=True),
        apply_high=_axis_face_uses_pml(solver, "x", low=False),
    )
    solver.cpml_kappa_h_y, solver.cpml_b_h_y, solver.cpml_c_h_y = _cpml_profile_1d(
        solver.Ny - 1,
        thickness,
        solver.dy,
        solver.dt,
        solver.mu0,
        magnetic_scale,
        eta0,
        solver.cpml_config,
        half_step=True,
        apply_low=_axis_face_uses_pml(solver, "y", low=True),
        apply_high=_axis_face_uses_pml(solver, "y", low=False),
    )
    solver.cpml_kappa_h_z, solver.cpml_b_h_z, solver.cpml_c_h_z = _cpml_profile_1d(
        solver.Nz - 1,
        thickness,
        solver.dz,
        solver.dt,
        solver.mu0,
        magnetic_scale,
        eta0,
        solver.cpml_config,
        half_step=True,
        apply_low=_axis_face_uses_pml(solver, "z", low=True),
        apply_high=_axis_face_uses_pml(solver, "z", low=False),
    )


def initialize_cpml_state(solver):
    _clear_auxiliary_state(solver)
    build_cpml_profiles(solver)
    _set_inv_kappa_buffers(solver)
    _allocate_cpml_memory_variables(solver)


def initialize_neutral_boundary_state(solver):
    _clear_auxiliary_state(solver)


def initialize_simple_pml_state(solver):
    _clear_auxiliary_state(solver)

    thickness = solver.scene.pml_thickness
    _allocate_sigma_tensors(solver)
    sigma_max = solver.scene.pml_strength * 2.0 / (solver.dt * min(solver.dx, solver.dy, solver.dz))
    for i in range(thickness):
        x = (thickness - i) / thickness
        sigma = sigma_max * x**4
        if _axis_face_uses_pml(solver, "x", low=True):
            solver.sigma_x[i, :, :] = sigma
        if _axis_face_uses_pml(solver, "x", low=False):
            solver.sigma_x[-(i + 1), :, :] = sigma
        if _axis_face_uses_pml(solver, "y", low=True):
            solver.sigma_y[:, i, :] = sigma
        if _axis_face_uses_pml(solver, "y", low=False):
            solver.sigma_y[:, -(i + 1), :] = sigma
        if _axis_face_uses_pml(solver, "z", low=True):
            solver.sigma_z[:, :, i] = sigma
        if _axis_face_uses_pml(solver, "z", low=False):
            solver.sigma_z[:, :, -(i + 1)] = sigma
