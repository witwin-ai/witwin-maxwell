from __future__ import annotations

import math
import os

import numpy as np
import torch

from ...scene import prepare_scene
from ...sources import evaluate_source_time


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

# Relative spacing spread at or below which a tangential aperture window is
# treated as exactly uniform: the exact minimum cell spacing is returned
# bit-for-bit (equal to the global minimum used before), so uniform grids keep
# their numerical-dispersion phase correction unchanged.
_SOFT_PLANE_UNIFORM_RTOL = 1e-6


def _pml_margin(scene, axis: str, side: str, *, extra: int = 0) -> int:
    base = scene.pml_thickness_for_face(axis, side)
    return int(base) + int(extra)


def resolve_injection_axis(direction, injection_axis=None) -> str:
    if injection_axis is not None:
        axis = str(injection_axis).lower()
        if axis not in _AXIS_TO_INDEX:
            raise ValueError("injection_axis must be 'x', 'y', 'z', or None.")
        axis_index = _AXIS_TO_INDEX[axis]
        if abs(float(direction[axis_index])) <= 1e-6:
            raise ValueError("direction must have a non-zero component along injection_axis.")
        return axis

    dominant_index = max(range(3), key=lambda index: abs(float(direction[index])))
    if abs(float(direction[dominant_index])) <= 1e-6:
        raise ValueError("direction must be non-zero.")
    return "xyz"[dominant_index]


def source_plane_index(scene, axis: str, direction_component: float) -> int:
    scene = prepare_scene(scene)
    axis_index = _AXIS_TO_INDEX[axis]
    domain_counts = (scene.Nx, scene.Ny, scene.Nz)
    low_margin = min(_pml_margin(scene, axis, "low", extra=1), max(domain_counts[axis_index] - 1, 0))
    high_margin = min(_pml_margin(scene, axis, "high", extra=1), max(domain_counts[axis_index] - 1, 0))
    if direction_component >= 0.0:
        return low_margin
    return max(domain_counts[axis_index] - high_margin - 1, 0)


def physical_interior_indices(scene, axis: str) -> tuple[int, int]:
    scene = prepare_scene(scene)
    size = {"x": scene.Nx, "y": scene.Ny, "z": scene.Nz}[axis]
    lo = _pml_margin(scene, axis, "low")
    hi = size - _pml_margin(scene, axis, "high") - 1
    if hi <= lo:
        raise ValueError(f"Scene has no physical interior along {axis}.")
    return lo, hi


def soft_plane_wave_index(scene, axis: str, direction_component: float, *, fraction: float = 0.05) -> int:
    scene = prepare_scene(scene)
    axis_coords = {"x": scene.x, "y": scene.y, "z": scene.z}[axis]
    lo, hi = physical_interior_indices(scene, axis)
    lo_coord = float(axis_coords[lo].item())
    hi_coord = float(axis_coords[hi].item())
    span = hi_coord - lo_coord
    placement = float(fraction)
    if direction_component >= 0.0:
        target = lo_coord + placement * span
    else:
        target = hi_coord - placement * span
    return int(torch.argmin(torch.abs(axis_coords - target)).item())


def soft_plane_wave_coordinate(scene, axis: str, direction_component: float, *, fraction: float = 0.05) -> float:
    scene = prepare_scene(scene)
    index = soft_plane_wave_index(scene, axis, direction_component, fraction=fraction)
    axis_coords = {"x": scene.x, "y": scene.y, "z": scene.z}[axis]
    return float(axis_coords[index].item())


def _window_is_uniform(cells: np.ndarray) -> bool:
    cells = np.asarray(cells, dtype=np.float64)
    d_min = float(cells.min())
    d_max = float(cells.max())
    return (d_max - d_min) <= _SOFT_PLANE_UNIFORM_RTOL * max(d_max, 1e-300)


def soft_plane_wave_region_spacing(scene, *, injection_axis: str, plane_index: int, direction_sign: int) -> dict:
    """Per-axis primal spacing local to the soft plane-wave launch footprint.

    The soft plane-wave numerical-dispersion phase correction solves the discrete
    dispersion relation with one effective spacing per axis. On a grid whose
    interior is uniform along an axis (to ``_SOFT_PLANE_UNIFORM_RTOL``) that axis
    returns its exact minimum cell spacing, bit-for-bit the global-minimum value
    the correction used before, so uniform grids are unchanged.

    On a graded axis the global minimum spacing can belong to a cell far from the
    source, which mis-tunes the injected wavefront's phase velocity. Instead, the
    injection axis returns the spacing of the cell the wavefront is launched into
    (the Yee E/H half-cell offset that governs the one-way phase match lives
    there), and each tangential axis returns the mean spacing over the physical
    aperture (the best single scalar for the oblique-incidence phase across the
    plane).
    """
    scene = prepare_scene(scene)
    primal = {
        "x": np.asarray(scene.dx_primal64, dtype=np.float64),
        "y": np.asarray(scene.dy_primal64, dtype=np.float64),
        "z": np.asarray(scene.dz_primal64, dtype=np.float64),
    }
    deltas = {}
    for axis in "xyz":
        cells = primal[axis]
        lo, hi = physical_interior_indices(scene, axis)
        interior = cells[lo:hi] if hi > lo else cells
        if _window_is_uniform(interior):
            # Uniform interior: exact spacing, bit-for-bit the previous value.
            deltas[axis] = float(interior.min())
        elif axis == injection_axis:
            # Cell straddled by the launch plane on the propagation side.
            cell_index = plane_index if direction_sign > 0 else plane_index - 1
            cell_index = min(max(int(cell_index), 0), len(cells) - 1)
            deltas[axis] = float(cells[cell_index])
        else:
            deltas[axis] = float(interior.mean())
    return deltas


def plane_center(scene, axis: str, plane_coordinate: float, *, device, dtype):
    scene = prepare_scene(scene)
    bounds = scene.domain.domain_range
    center = torch.tensor(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        device=device,
        dtype=dtype,
    )
    center[_AXIS_TO_INDEX[axis]] = float(plane_coordinate)
    return center


def plane_wave_profile(positions, *, direction, reference_point, propagation_speed):
    direction_tensor = positions.new_tensor(direction)
    reference_tensor = (
        reference_point.to(device=positions.device, dtype=positions.dtype)
        if isinstance(reference_point, torch.Tensor)
        else positions.new_tensor(reference_point)
    )
    delay = torch.sum((positions - reference_tensor) * direction_tensor, dim=-1) / float(propagation_speed)
    amplitude = torch.ones_like(delay)
    return amplitude, delay


def _astigmatic_curvature_delay(transverse, z, z_rayleigh, propagation_speed):
    radius_of_curvature = torch.where(
        torch.abs(z) > 1e-9,
        z * (1.0 + (z_rayleigh / torch.clamp(torch.abs(z), min=1e-30)).square()),
        torch.full_like(z, float("inf")),
    )
    return torch.where(
        torch.isfinite(radius_of_curvature),
        transverse.square() / (2.0 * radius_of_curvature * float(propagation_speed)),
        torch.zeros_like(z),
    )


def _beam_profile_core(
    positions,
    *,
    direction,
    polarization,
    beam_waist_u,
    beam_waist_v,
    focus,
    focus_offset_u,
    focus_offset_v,
    frequency,
    propagation_speed,
):
    direction_tensor = positions.new_tensor(direction)
    polarization_tensor = positions.new_tensor(polarization)
    binormal_tensor = torch.cross(direction_tensor, polarization_tensor, dim=0)
    focus_tensor = positions.new_tensor(focus)

    rel = positions - focus_tensor
    longitudinal = torch.sum(rel * direction_tensor, dim=-1)
    transverse_u = torch.sum(rel * polarization_tensor, dim=-1)
    transverse_v = torch.sum(rel * binormal_tensor, dim=-1)

    # Per-axis longitudinal coordinate measured from each transverse-axis waist plane.
    z_u = longitudinal - float(focus_offset_u)
    z_v = longitudinal - float(focus_offset_v)

    wavelength = float(propagation_speed) / float(frequency)
    z_rayleigh_u = math.pi * float(beam_waist_u) ** 2 / max(wavelength, 1e-30)
    z_rayleigh_v = math.pi * float(beam_waist_v) ** 2 / max(wavelength, 1e-30)

    beam_radius_u = float(beam_waist_u) * torch.sqrt(1.0 + (z_u / max(z_rayleigh_u, 1e-30)).square())
    beam_radius_v = float(beam_waist_v) * torch.sqrt(1.0 + (z_v / max(z_rayleigh_v, 1e-30)).square())

    amplitude = (
        torch.sqrt(float(beam_waist_u) / beam_radius_u)
        * torch.sqrt(float(beam_waist_v) / beam_radius_v)
        * torch.exp(
            -transverse_u.square() / beam_radius_u.square()
            - transverse_v.square() / beam_radius_v.square()
        )
    )

    curvature_delay = _astigmatic_curvature_delay(
        transverse_u, z_u, z_rayleigh_u, propagation_speed
    ) + _astigmatic_curvature_delay(transverse_v, z_v, z_rayleigh_v, propagation_speed)
    gouy_phase = 0.5 * (
        torch.atan(z_u / max(z_rayleigh_u, 1e-30)) + torch.atan(z_v / max(z_rayleigh_v, 1e-30))
    )
    gouy_delay = gouy_phase / (2.0 * math.pi * float(frequency))
    delay = longitudinal / float(propagation_speed) + curvature_delay - gouy_delay
    return amplitude, delay


def gaussian_beam_profile(
    positions,
    *,
    direction,
    polarization,
    beam_waist,
    focus,
    frequency,
    propagation_speed,
):
    return _beam_profile_core(
        positions,
        direction=direction,
        polarization=polarization,
        beam_waist_u=beam_waist,
        beam_waist_v=beam_waist,
        focus=focus,
        focus_offset_u=0.0,
        focus_offset_v=0.0,
        frequency=frequency,
        propagation_speed=propagation_speed,
    )


def astigmatic_gaussian_beam_profile(
    positions,
    *,
    direction,
    polarization,
    beam_waist,
    focus,
    focus_offsets,
    frequency,
    propagation_speed,
):
    beam_waist_u, beam_waist_v = beam_waist
    focus_offset_u, focus_offset_v = focus_offsets
    return _beam_profile_core(
        positions,
        direction=direction,
        polarization=polarization,
        beam_waist_u=beam_waist_u,
        beam_waist_v=beam_waist_v,
        focus=focus,
        focus_offset_u=focus_offset_u,
        focus_offset_v=focus_offset_v,
        frequency=frequency,
        propagation_speed=propagation_speed,
    )


def beam_profile_from_source(
    positions,
    source,
    *,
    frequency,
    propagation_speed,
    polarization=None,
):
    polarization = source["polarization"] if polarization is None else polarization
    if source["kind"] == "astigmatic_gaussian_beam":
        return astigmatic_gaussian_beam_profile(
            positions,
            direction=source["direction"],
            polarization=polarization,
            beam_waist=(source["beam_waist_u"], source["beam_waist_v"]),
            focus=source["focus"],
            focus_offsets=(source["focus_u"], source["focus_v"]),
            frequency=frequency,
            propagation_speed=propagation_speed,
        )
    return gaussian_beam_profile(
        positions,
        direction=source["direction"],
        polarization=polarization,
        beam_waist=source["beam_waist"],
        focus=source["focus"],
        frequency=frequency,
        propagation_speed=propagation_speed,
    )


class AuxiliaryGrid1D:
    def __init__(
        self,
        *,
        s_min,
        s_max,
        ds,
        dt,
        wave_speed,
        impedance,
        source_time,
        device,
        dtype,
        absorber_cells=20,
        source_buffer_cells=6,
        fdtd_module=None,
    ):
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.ds = float(ds)
        self.dt = float(dt)
        self.wave_speed = float(wave_speed)
        self.impedance = float(impedance)
        self.source_time = source_time
        self.device = torch.device(device)
        self.dtype = dtype
        self.absorber_cells = max(int(absorber_cells), 0)
        self.source_buffer_cells = max(int(source_buffer_cells), 0)
        self.fdtd_module = fdtd_module

        if self.ds <= 0.0:
            raise ValueError("AuxiliaryGrid1D ds must be > 0.")
        if self.dt <= 0.0:
            raise ValueError("AuxiliaryGrid1D dt must be > 0.")
        if self.wave_speed <= 0.0:
            raise ValueError("AuxiliaryGrid1D wave_speed must be > 0.")
        if self.s_max <= self.s_min:
            raise ValueError("AuxiliaryGrid1D requires s_max > s_min.")

        electric_cells = max(int(math.ceil((self.s_max - self.s_min) / self.ds)) + 1, 4)
        self.electric = torch.zeros(electric_cells, device=self.device, dtype=dtype)
        self.magnetic = torch.zeros(electric_cells - 1, device=self.device, dtype=dtype)
        self.source_index = 0

        self.eps = 1.0 / (self.impedance * self.wave_speed)
        self.mu = self.impedance / self.wave_speed
        self._build_loss_profiles()
        self.time_step = 0

    @property
    def electric_time(self):
        return self.time_step * self.dt

    @property
    def magnetic_time(self):
        return (self.time_step - 0.5) * self.dt

    def _can_use_compiled_auxiliary(self) -> bool:
        return self.device.type == "cuda" and self.dtype == torch.float32

    def _resolve_fdtd_module(self):
        if self.fdtd_module is not None:
            return self.fdtd_module
        if not self._can_use_compiled_auxiliary():
            return None

        from ..cuda.backend import get_native_fdtd_module

        self.fdtd_module = get_native_fdtd_module()
        return self.fdtd_module

    def _build_loss_profiles(self):
        sigma_e = torch.zeros_like(self.electric)
        sigma_h = torch.zeros_like(self.magnetic)
        edge_cells_e = min(self.absorber_cells, max(self.electric.numel() - 1, 0))
        edge_cells_h = min(self.absorber_cells, max(self.magnetic.numel(), 0))
        if edge_cells_e > 0:
            max_sigma = 0.35
            ramp_e = ((torch.arange(edge_cells_e, device=self.device, dtype=self.dtype) + 1.0) / edge_cells_e) ** 2
            sigma_e[-edge_cells_e:] = max_sigma * ramp_e
        if edge_cells_h > 0:
            max_sigma = 0.35
            ramp_h = ((torch.arange(edge_cells_h, device=self.device, dtype=self.dtype) + 0.5) / edge_cells_h) ** 2
            sigma_h[-edge_cells_h:] = max_sigma * ramp_h

        sigma_factor_e = 0.5 * self.dt * sigma_e / max(self.eps, 1e-30)
        sigma_factor_h = 0.5 * self.dt * sigma_h / max(self.mu, 1e-30)
        self.electric_decay = (1.0 - sigma_factor_e) / (1.0 + sigma_factor_e)
        self.electric_curl = (self.dt / (self.eps * self.ds)) / (1.0 + sigma_factor_e)
        self.magnetic_decay = (1.0 - sigma_factor_h) / (1.0 + sigma_factor_h)
        self.magnetic_curl = (self.dt / (self.mu * self.ds)) / (1.0 + sigma_factor_h)

    def _interpolate(self, field, *, origin, positions, out=None):
        positions_tensor = torch.as_tensor(positions, device=self.device, dtype=self.dtype)
        coord = torch.clamp((positions_tensor - float(origin)) / self.ds, min=0.0, max=field.numel() - 1.0)
        lower = torch.floor(coord).to(torch.int64)
        upper = torch.clamp(lower + 1, max=field.numel() - 1)
        frac = coord - lower.to(self.dtype)
        if out is None:
            out = torch.empty_like(positions_tensor)
        out.copy_(field[lower])
        out.lerp_(field[upper], frac)
        return out

    def sample_e(self, positions, out=None):
        return self._interpolate(self.electric, origin=self.s_min, positions=positions, out=out)

    def sample_h(self, positions, out=None):
        return self._interpolate(self.magnetic, origin=self.s_min + 0.5 * self.ds, positions=positions, out=out)

    def _apply_source(self):
        self.electric[self.source_index] = evaluate_source_time(self.source_time, self.electric_time)

    def _advance_magnetic_compiled(self):
        self._resolve_fdtd_module().updateAuxiliaryMagnetic1D(
            Magnetic=self.magnetic,
            Electric=self.electric,
            MagneticDecay=self.magnetic_decay,
            MagneticCurl=self.magnetic_curl,
        ).launchRaw()

    def _advance_electric_compiled(self):
        source_value = float(evaluate_source_time(self.source_time, self.electric_time))
        self._resolve_fdtd_module().updateAuxiliaryElectric1D(
            Electric=self.electric,
            Magnetic=self.magnetic,
            ElectricDecay=self.electric_decay,
            ElectricCurl=self.electric_curl,
            sourceIndex=int(self.source_index),
            sourceValue=source_value,
        ).launchRaw()

    def advance_magnetic(self):
        if self._can_use_compiled_auxiliary():
            self._advance_magnetic_compiled()
            return
        if self.magnetic.numel() > 0:
            curl_e = self.electric[1:] - self.electric[:-1]
            self.magnetic = self.magnetic_decay * self.magnetic - self.magnetic_curl * curl_e

    def advance_electric(self):
        if self._can_use_compiled_auxiliary():
            self._advance_electric_compiled()
            self.time_step += 1
            return
        if self.electric.numel() > 2:
            curl_h = self.magnetic[1:] - self.magnetic[:-1]
            self.electric[1:-1] = self.electric_decay[1:-1] * self.electric[1:-1] - self.electric_curl[1:-1] * curl_h

        self._apply_source()
        self.electric[-1] = 0.0
        self.time_step += 1

    def advance(self):
        self.advance_magnetic()
        self.advance_electric()

    def step(self):
        self.advance()
