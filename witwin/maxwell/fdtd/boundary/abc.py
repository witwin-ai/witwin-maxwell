"""Second-order Mur faces, mode-matched one-way faces and internal absorber planes.

All are post-E-update one-way boundary passes:

* Mode-matched faces and internal absorber planes use the one-step update
  ``E_b(n+1) = E_a(n) + p (E_a(n+1) - E_b(n))`` and are appended to
  ``solver._mur_state``, so they share the native Mur kernel, checkpoint/resume
  histories and the discrete adjoint transpose with first-order Mur faces.
* Broadband mode-matched planes are Higdon products of several such one-step
  operators (``solver._abc_higdon_state``).
* Second-order Mur faces are the Higdon product of two normal-incidence Mur
  operators (same state list).

The multi-level entries are applied in PyTorch after the native one-step kernels.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_AXES = ("x", "y", "z")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Tangential electric components on a plane normal to each axis.
_TANGENTIAL_COMPONENTS = {
    "x": ("Ey", "Ez"),
    "y": ("Ex", "Ez"),
    "z": ("Ex", "Ey"),
}


def _component_names(solver, base_name):
    return (base_name, base_name + "_imag") if solver.complex_fields_enabled else (base_name,)


def _axis_primal(scene, axis):
    return {"x": scene.dx_primal64, "y": scene.dy_primal64, "z": scene.dz_primal64}[axis]


def initialize_second_order_mur_faces(solver, faces, speed_dt):
    """Second-order one-way faces as the Higdon product of two Mur operators.

    Both factors use the normal-incidence Mur coefficient
    ``(v dt - d)/(v dt + d)``, so the reflection is the square of the
    first-order Mur reflection: second-order accurate in the incidence angle,
    like the Engquist-Majda condition, but without a transverse stencil, so it
    stays passive next to transverse PML, at edges and in media slower than
    ``v``.
    """
    scene = solver.scene
    for axis, side in faces:
        primal = _axis_primal(scene, axis)
        size = int(len(primal)) + 1
        delta = float(primal[0] if side == "low" else primal[-1])
        coef = (speed_dt - delta) / (speed_dt + delta)
        _append_one_way_plane(
            solver,
            axis=axis,
            boundary_index=0 if side == "low" else size - 1,
            adjacent_index=1 if side == "low" else size - 2,
            coefficients=(coef, coef),
            source=("mur", axis, side),
        )


# --------------------------------------------------------------------------- #
# Mode-matched one-way coefficients                                           #
# --------------------------------------------------------------------------- #


def _numerical_frequency(frequency, dt):
    """Frequency whose continuous dispersion matches the leapfrog time stencil."""
    angle = math.pi * float(frequency) * float(dt)
    return math.sin(angle) / (math.pi * float(dt))


def _solve_plane_mode_beta(solver, spec, *, axis, plane_index, direction, frequency):
    """Solve the guided mode on a full-domain cross-section; return its beta.

    The plane spans the physical (non-PML) transverse extent of the domain and
    reuses the ModeSource eigen-solver, so material sampling, polarization
    families and mode ordering match ModeSource/ModeMonitor.
    """
    from ...compiler.sources import _compile_mode_source
    from ...sources import CW, ModeSource
    from ..excitation.modes import solve_mode_source_profile
    from ..excitation.spatial import physical_interior_indices

    scene = solver.scene
    axis_index = _AXIS_INDEX[axis]
    coords = np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64)
    position = [0.0, 0.0, 0.0]
    size = [0.0, 0.0, 0.0]
    for other_index, other_axis in enumerate(_AXES):
        if other_index == axis_index:
            continue
        # Use the solver's own physical edge nodes so the aperture check, which
        # compares against those sampled coordinates, accepts the full span.
        lo_index, hi_index = physical_interior_indices(scene, other_axis)
        axis_coords = getattr(scene, other_axis)
        low = float(axis_coords[lo_index].item())
        high = float(axis_coords[hi_index].item())
        position[other_index] = 0.5 * (low + high)
        size[other_index] = high - low
    position[axis_index] = float(coords[plane_index])
    source = ModeSource(
        position=tuple(position),
        size=tuple(size),
        mode_index=spec.mode_index,
        direction=direction,
        polarization=spec.polarization,
        source_time=CW(frequency=float(frequency), amplitude=1.0),
    )
    compiled = _compile_mode_source(source, default_frequency=float(frequency))
    mode_data = solve_mode_source_profile(solver, compiled)
    beta_tensor = mode_data.get("beta_tensor")
    if isinstance(beta_tensor, torch.Tensor) and beta_tensor.requires_grad:
        raise NotImplementedError(
            "ModeABC does not differentiate its one-way coefficient through trainable materials on "
            "the boundary cross-section; keep design regions off the ModeABC plane."
        )
    beta_complex = mode_data.get("beta_complex")
    beta = float(mode_data["beta"])
    if beta_complex is not None and abs(complex(beta_complex).imag) > 1e-3 * max(abs(beta), 1e-30):
        raise NotImplementedError(
            "ModeABC requires a lossless guided mode; the one-step real update cannot absorb a "
            "mode with a complex propagation constant."
        )
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("ModeABC mode solve returned a non-propagating mode.")
    return beta


def _discrete_beta(beta_continuous, delta):
    """Map a continuous longitudinal wavenumber onto the Yee second difference."""
    argument = 0.5 * float(beta_continuous) * float(delta)
    if argument >= 1.0:
        raise ValueError(
            "ModeABC mode is under-resolved along the propagation axis "
            "(beta * dx / 2 >= 1); refine the grid normal to the boundary."
        )
    return 2.0 / float(delta) * math.asin(argument)


def one_way_mode_coefficient(omega_dt, beta_delta):
    """Real one-step coefficient annihilating exp(j(w t -/+ beta x)) exactly.

    Substituting the outgoing discrete wave into
    ``E_b(n+1) = E_a(n) + p (E_a(n+1) - E_b(n))`` gives
    ``p = sin((w dt - beta d)/2) / sin((w dt + beta d)/2)``; its small-argument
    limit is the Mur coefficient ``(v dt - d)/(v dt + d)`` with ``v = w/beta``.
    """
    return math.sin(0.5 * (omega_dt - beta_delta)) / math.sin(0.5 * (omega_dt + beta_delta))


def chebyshev_band_frequencies(low, high, count):
    """Chebyshev nodes of ``[low, high]``: they minimize the band maximum of the
    Higdon product of ``count`` factors that each vanish at one node."""
    middle = 0.5 * (float(low) + float(high))
    half = 0.5 * (float(high) - float(low))
    return np.asarray(
        [middle + half * math.cos((2 * k - 1) * math.pi / (2 * count)) for k in range(count, 0, -1)],
        dtype=np.float64,
    )


# Spatial damping carried by the last Higdon factor. Every undamped factor
# vanishes for a static uniform field (Z = S = 1), so an undamped product has a
# repeated static root that lets closed one-way boxes drift; 0.5% damping makes
# that root simple without measurably changing the propagating reflection.
HIGDON_DAMPING = 0.995


def higdon_coefficients(factors):
    """Coefficients ``c[m, l]`` of ``prod_k [(Z + p_k) - a_k (1 + p_k Z) S]``.

    ``Z`` advances one time step and ``S`` moves one plane into the interior;
    ``a_k = 1`` except ``HIGDON_DAMPING`` on the last factor of a product. A
    single factor is the one-step Mur update; the product's reflection is the
    product of the factor reflections. ``c[K, 0] = 1`` is the new boundary value.
    """
    coefficients = np.ones((1, 1), dtype=np.float64)
    for index, p in enumerate(factors):
        alpha = HIGDON_DAMPING if len(factors) > 1 and index == len(factors) - 1 else 1.0
        factor = np.array([[float(p), -alpha], [1.0, -alpha * float(p)]], dtype=np.float64)
        grown = np.zeros((coefficients.shape[0] + 1, coefficients.shape[1] + 1), dtype=np.float64)
        for m in range(coefficients.shape[0]):
            for l in range(coefficients.shape[1]):
                grown[m:m + 2, l:l + 2] += coefficients[m, l] * factor
        coefficients = grown
    return coefficients


def _mode_one_way_coefficient(solver, spec, *, axis, plane_index, direction, delta):
    """Return ``(factors, info)``: one exact mode-matched one-step coefficient per
    matched frequency (a single factor for a single-frequency spec)."""
    dt = float(solver.dt)
    if spec.is_broadband:
        frequencies = chebyshev_band_frequencies(*spec.frequency_range, int(spec.order))
    else:
        frequencies = np.asarray(
            [float(spec.frequency) if spec.frequency is not None else float(solver.source_frequency)]
        )
    factors = []
    beta = []
    for frequency in frequencies:
        beta_numeric = _solve_plane_mode_beta(
            solver,
            spec,
            axis=axis,
            plane_index=plane_index,
            direction=direction,
            frequency=_numerical_frequency(float(frequency), dt),
        )
        beta_discrete = _discrete_beta(beta_numeric, delta)
        beta.append(beta_discrete)
        factors.append(one_way_mode_coefficient(2.0 * math.pi * float(frequency) * dt, beta_discrete * float(delta)))
    beta = np.asarray(beta, dtype=np.float64)
    effective_index = beta * float(solver.c) / (2.0 * np.pi * frequencies)
    info = {
        "frequencies": tuple(float(value) for value in frequencies),
        "betas": tuple(float(value) for value in beta),
        "effective_indices": tuple(float(value) for value in effective_index),
    }
    if not spec.is_broadband:
        info.update(frequency=float(frequencies[0]), beta=float(beta[0]), effective_index=float(effective_index[0]))
    return tuple(float(value) for value in factors), info


def _require_uniform_normal_cells(scene, axis, boundary_index, step, count):
    primal = np.asarray(_axis_primal(scene, axis), dtype=np.float64)
    cells = [boundary_index + (k if step > 0 else -k - 1) for k in range(count)]
    if min(cells) < 0 or max(cells) >= primal.size:
        raise ValueError("Higdon ModeABC needs as many interior planes as its order next to the boundary.")
    spacing = primal[cells]
    if float(np.max(spacing) - np.min(spacing)) > 1e-6 * float(np.max(spacing)):
        raise ValueError(
            "Higdon ModeABC requires uniform normal spacing over its order cells next to the boundary."
        )


def _append_one_way_plane(solver, *, axis, boundary_index, adjacent_index, coefficients, source):
    """Register a one-way plane: a single factor goes to ``_mur_state`` (native
    Mur kernel and adjoint transpose), a Higdon product of order > 1 to
    ``_abc_higdon_state``."""
    normal_index = _AXIS_INDEX[axis]
    order = len(coefficients)
    step = int(adjacent_index) - int(boundary_index)
    if order > 1:
        _require_uniform_normal_cells(solver.scene, axis, int(boundary_index), step, order)
        table = higdon_coefficients(coefficients)
    for base_name in _TANGENTIAL_COMPONENTS[axis]:
        for field_name in _component_names(solver, base_name):
            field = getattr(solver, field_name)
            plane_shape = tuple(field.shape[:normal_index] + field.shape[normal_index + 1:])
            entry = {
                "field": field_name,
                "axis": int(normal_index),
                "boundary_index": int(boundary_index),
                "adjacent_index": int(adjacent_index),
                "source": source,
            }
            if order == 1:
                entry["coef"] = float(coefficients[0])
                entry["prev_boundary"] = torch.zeros(plane_shape, device=solver.device, dtype=field.dtype)
                entry["prev_adjacent"] = torch.zeros(plane_shape, device=solver.device, dtype=field.dtype)
                solver._mur_state.append(entry)
                continue
            # E_0(n+1) = -sum_{(m, l) != (K, 0)} c[m, l] E_l(n + 1 - K + m), planes l = 0..K
            # counted from the boundary into the interior.
            entry["plane_indices"] = tuple(int(boundary_index) + step * l for l in range(order + 1))
            entry["current_weights"] = torch.tensor(
                np.ascontiguousarray(table[order, 1:]), device=solver.device, dtype=field.dtype
            ).view(-1, 1, 1)
            # history[h, l] holds plane l at time n - h, weighted by c[K - 1 - h, l].
            entry["history_weights"] = torch.tensor(
                np.ascontiguousarray(table[order - 1::-1, :]), device=solver.device, dtype=field.dtype
            ).view(order, order + 1, 1, 1)
            entry["history"] = torch.zeros((order, order + 1) + plane_shape, device=solver.device, dtype=field.dtype)
            # Plane rims meet the neighbouring faces; there the one-step update with
            # the first factor keeps edge and corner coupling passive.
            rim = torch.ones(plane_shape, device=solver.device, dtype=torch.bool)
            rim[1:-1, 1:-1] = False
            entry["rim_mask"] = rim
            entry["rim_coef"] = float(coefficients[0])
            solver._abc_higdon_state.append(entry)


def initialize_mode_abc_state(solver):
    """Append mode-matched one-way entries for every ``mode_abc`` outer face."""
    faces = tuple(getattr(solver, "mode_abc_faces", ()))
    solver._mode_abc_info = []
    if not faces:
        return
    scene = solver.scene
    spec = scene.boundary.mode_abc
    for axis, side in faces:
        primal = _axis_primal(scene, axis)
        size = int(len(primal)) + 1
        boundary_index = 0 if side == "low" else size - 1
        adjacent_index = 1 if side == "low" else size - 2
        delta = float(primal[0] if side == "low" else primal[-1])
        coefficients, info = _mode_one_way_coefficient(
            solver,
            spec,
            axis=axis,
            plane_index=adjacent_index,
            direction="-" if side == "low" else "+",
            delta=delta,
        )
        solver._mode_abc_info.append({"face": (axis, side), "coefficients": coefficients, "order": len(coefficients), **info})
        _append_one_way_plane(
            solver,
            axis=axis,
            boundary_index=boundary_index,
            adjacent_index=adjacent_index,
            coefficients=coefficients,
            source=("mode_abc", axis, side),
        )


# --------------------------------------------------------------------------- #
# Internal absorbers                                                          #
# --------------------------------------------------------------------------- #


def _internal_absorber_plane_index(scene, absorber):
    from ..excitation.spatial import physical_interior_indices

    axis = absorber.normal_axis
    axis_index = _AXIS_INDEX[axis]
    for other_index, other_axis in enumerate(_AXES):
        if other_index == axis_index:
            continue
        low, high = scene.domain.bounds[other_index]
        extent = float(high) - float(low)
        if absorber.size[other_index] < extent * (1.0 - 1e-9):
            raise NotImplementedError(
                f"InternalAbsorber {absorber.name or ''!s} must span the full domain cross-section along "
                f"{other_axis}; finite absorber apertures (with their PEC frame and backing plate) are "
                "not implemented. Use size=inf on the transverse axes."
            )
    nodes = np.asarray(getattr(scene, f"{axis}_nodes64"), dtype=np.float64)
    index = int(np.argmin(np.abs(nodes - float(absorber.position[axis_index])))) + int(absorber.grid_shift)
    physical_lo, physical_hi = physical_interior_indices(scene, axis)
    if index <= physical_lo or index >= physical_hi:
        raise ValueError(
            "InternalAbsorber plane must lie strictly inside the non-PML region along its normal axis."
        )
    return index


def initialize_internal_absorber_state(solver):
    """Append one-way entries for every ``InternalAbsorber`` plane."""
    from ...absorbers import ModeABC

    absorbers = tuple(getattr(solver, "internal_absorbers", ()))
    solver._internal_absorber_info = []
    if not absorbers:
        return
    scene = solver.scene
    for index, absorber in enumerate(absorbers):
        axis = absorber.normal_axis
        plane_index = _internal_absorber_plane_index(scene, absorber)
        primal = _axis_primal(scene, axis)
        if absorber.direction_sign > 0:
            adjacent_index = plane_index - 1
            delta = float(primal[plane_index - 1])
        else:
            adjacent_index = plane_index + 1
            delta = float(primal[plane_index])
        spec = absorber.boundary
        if isinstance(spec, ModeABC):
            coefficients, info = _mode_one_way_coefficient(
                solver,
                spec,
                axis=axis,
                plane_index=plane_index,
                direction=absorber.direction,
                delta=delta,
            )
        else:
            speed_dt = float(solver.c) * float(solver.dt)
            if spec.permittivity is not None:
                speed_dt = speed_dt / math.sqrt(spec.permittivity)
            coefficients = ((speed_dt - delta) / (speed_dt + delta),) * int(spec.order)
            info = {}
        solver._internal_absorber_info.append(
            {
                "index": index,
                "name": absorber.name,
                "axis": axis,
                "plane_index": plane_index,
                "coefficients": coefficients,
                **info,
            }
        )
        _append_one_way_plane(
            solver,
            axis=axis,
            boundary_index=plane_index,
            adjacent_index=adjacent_index,
            coefficients=coefficients,
            source=("internal_absorber", index),
        )
