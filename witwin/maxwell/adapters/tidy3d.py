"""Convert a maxwell Scene to a Tidy3D Simulation.

Usage::

    td_sim = scene.to_tidy3d(frequencies=frequencies)
    td_result = tidy3d.web.run(td_sim)

Requires ``tidy3d`` as an optional dependency.

**Unit convention**: maxwell uses metres; Tidy3D uses micrometres (um).
All spatial quantities are multiplied by ``length_scale`` (default 1e6)
during conversion.  Frequencies stay in Hz.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
from ..fdtd.excitation.spatial import resolve_injection_axis, soft_plane_wave_coordinate

if TYPE_CHECKING:
    import tidy3d

# Default length conversion factor: metres to micrometres.
_M_TO_UM = 1e6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_tidy3d():
    """Import and return tidy3d, raising a clear error if missing."""
    try:
        import tidy3d as td
        return td
    except ImportError:
        raise ImportError(
            "tidy3d is required for Scene.to_tidy3d(). "
            "Install it with: pip install tidy3d"
        ) from None


def _scale3(vec, s):
    """Scale a 3-element vector by *s*, returning a Python-float tuple."""
    return (float(vec[0]) * s, float(vec[1]) * s, float(vec[2]) * s)


def _domain_to_center_size(domain, s):
    """Convert Domain bounds to Tidy3D (center, size) in Tidy3D units."""
    bounds = domain.bounds
    center = tuple((lo + hi) / 2.0 * s for lo, hi in bounds)
    size = tuple((hi - lo) * s for lo, hi in bounds)
    return center, size


def _polarization_to_component(polarization: tuple[float, float, float]) -> str:
    """Map a unit polarization vector to 'Ex', 'Ey', or 'Ez'."""
    px, py, pz = polarization
    mapping = {(1, 0, 0): "Ex", (0, 1, 0): "Ey", (0, 0, 1): "Ez"}
    key = (int(round(px)), int(round(py)), int(round(pz)))
    if key in mapping:
        return mapping[key]
    raise ValueError(
        f"Cannot map polarization {polarization} to a single Tidy3D component. "
        "Only axis-aligned unit vectors are supported."
    )


def _axis_name_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


# ---------------------------------------------------------------------------
# Source-time conversion (no length scaling - purely temporal / frequency)
# ---------------------------------------------------------------------------

def _convert_source_time(source_time, td):
    """Convert maxwell source_time to a Tidy3D SourceTime."""
    from ..sources import CW, GaussianPulse, RickerWavelet

    if source_time is None:
        raise ValueError("source_time must be set for Tidy3D export.")

    if isinstance(source_time, CW):
        return td.ContinuousWave(
            freq0=source_time.frequency,
            amplitude=source_time.amplitude,
            phase=source_time.phase,
        )

    if isinstance(source_time, GaussianPulse):
        # Maxwell's pulsed source keeps the carrier delay inside the sampled
        # waveform, while Tidy3D's GaussianPulse encodes the carrier spectrum
        # phase separately from the envelope offset. Fold the carrier delay
        # into the exported phase so single-frequency benchmark fields compare
        # against the same source spectrum at the target frequency.
        offset = source_time.delay / source_time.sigma_t
        phase = source_time.phase + 2.0 * math.pi * source_time.frequency * source_time.delay
        return td.GaussianPulse(
            freq0=source_time.frequency,
            fwidth=source_time.fwidth,
            amplitude=source_time.amplitude,
            phase=phase,
            offset=offset,
        )

    if isinstance(source_time, RickerWavelet):
        return td.GaussianPulse(
            freq0=source_time.frequency,
            fwidth=source_time.frequency,
            amplitude=source_time.amplitude,
        )

    raise TypeError(f"Unsupported source_time type: {type(source_time).__name__}")


# ---------------------------------------------------------------------------
# Material conversion (no length scaling - dimensionless or frequency-based)
# ---------------------------------------------------------------------------

def _convert_material(material, td):
    """Convert a Maxwell Material to a Tidy3D medium."""
    has_debye = bool(material.debye_poles)
    has_drude = bool(material.drude_poles)
    has_lorentz = bool(material.lorentz_poles)
    if material.is_anisotropic:
        raise NotImplementedError("Tidy3D export for anisotropic Material is not implemented yet.")
    if material.is_nonlinear:
        raise NotImplementedError("Tidy3D export for Kerr nonlinear Material is not implemented yet.")
    if material.is_magnetic_dispersive:
        raise NotImplementedError("Tidy3D export for magnetic dispersive Material is not implemented yet.")
    if not math.isclose(float(material.mu_r), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise NotImplementedError(
            "Tidy3D export currently assumes mu_r = 1. Static magnetic materials are not implemented yet."
        )

    if not material.is_dispersive:
        kwargs = {"permittivity": material.eps_r}
        if getattr(material, "sigma_e", 0.0) != 0.0:
            kwargs["conductivity"] = material.sigma_e
        return td.Medium(**kwargs)

    if getattr(material, "sigma_e", 0.0) != 0.0:
        raise NotImplementedError("Tidy3D export does not yet support combining sigma_e with dispersive pole materials.")

    if has_drude and not has_debye and not has_lorentz:
        coeffs = [
            (pole.plasma_frequency, pole.gamma)
            for pole in material.drude_poles
        ]
        return td.Drude(eps_inf=material.eps_r, coeffs=coeffs)

    if has_lorentz and not has_debye and not has_drude:
        coeffs = [
            (pole.delta_eps, pole.resonance_frequency, pole.gamma)
            for pole in material.lorentz_poles
        ]
        return td.Lorentz(eps_inf=material.eps_r, coeffs=coeffs)

    if has_debye and not has_drude and not has_lorentz:
        coeffs = [
            (pole.delta_eps, pole.tau)
            for pole in material.debye_poles
        ]
        return td.Debye(eps_inf=material.eps_r, coeffs=coeffs)

    # Mixed pole types to PoleResidue
    poles = []
    omega_factor = 2.0 * math.pi
    for p in material.drude_poles:
        wp = p.plasma_frequency * omega_factor
        g = p.gamma * omega_factor
        a = complex(0, -g / 2)
        c = complex(0, -wp * wp / (2.0 * g)) if g > 0 else complex(-wp * wp / 2, 0)
        poles.append((a, c))
    for p in material.lorentz_poles:
        w0 = p.resonance_frequency * omega_factor
        g = p.gamma * omega_factor
        disc = g * g / 4.0 - w0 * w0
        if disc >= 0:
            sq = math.sqrt(disc)
            a1 = complex(0, -g / 2 + sq)
            a2 = complex(0, -g / 2 - sq)
        else:
            sq = math.sqrt(-disc)
            a1 = complex(sq, -g / 2)
            a2 = complex(-sq, -g / 2)
        c_val = p.delta_eps * w0 * w0 / (2.0 * (a1 - a2)) if abs(a1 - a2) > 0 else 0
        poles.append((a1, complex(c_val)))
        poles.append((a2, complex(-c_val)))
    for p in material.debye_poles:
        tau = p.tau
        a = complex(0, -1.0 / tau)
        c = complex(0, p.delta_eps / tau)
        poles.append((a, c))

    return td.PoleResidue(eps_inf=material.eps_r, poles=poles)


# ---------------------------------------------------------------------------
# Geometry conversion (all lengths scaled)
# ---------------------------------------------------------------------------

def _convert_geometry(geometry, td, s):
    """Convert maxwell geometry to a Tidy3D geometry (lengths x *s*)."""
    kind = geometry.kind

    if kind == "box":
        return td.Box(center=_scale3(geometry.position, s), size=_scale3(geometry.size, s))

    if kind == "sphere":
        return td.Sphere(center=_scale3(geometry.position, s), radius=float(geometry.radius) * s)

    if kind == "cylinder":
        axis_idx = _axis_name_to_index(geometry.axis)
        return td.Cylinder(
            center=_scale3(geometry.position, s),
            radius=float(geometry.radius) * s,
            length=float(geometry.height) * s,
            axis=axis_idx,
        )

    if kind == "cone":
        axis_idx = _axis_name_to_index(geometry.axis)
        sidewall_angle = math.atan2(float(geometry.radius), float(geometry.height))
        return td.Cylinder(
            center=_scale3(geometry.position, s),
            radius=float(geometry.radius) * s,
            length=float(geometry.height) * s,
            axis=axis_idx,
            sidewall_angle=sidewall_angle,
        )

    if kind == "ellipsoid":
        rx, ry, rz = float(geometry.radii[0]), float(geometry.radii[1]), float(geometry.radii[2])
        if abs(rx - ry) < 1e-12 and abs(ry - rz) < 1e-12:
            return td.Sphere(center=_scale3(geometry.position, s), radius=rx * s)
        raise NotImplementedError(
            "Tidy3D has no native Ellipsoid geometry. "
            "Consider using a Box approximation or a GDS-based PolySlab."
        )

    raise NotImplementedError(
        f"Geometry type '{kind}' has no Tidy3D mapping yet. "
        f"Supported: box, sphere, cylinder, cone."
    )


# ---------------------------------------------------------------------------
# Structure conversion
# ---------------------------------------------------------------------------

def _convert_structure(structure, td, s):
    """Convert maxwell Structure to a Tidy3D Structure."""
    td_geometry = _convert_geometry(structure.geometry, td, s)
    td_material = _convert_material(structure.material, td)
    return td.Structure(geometry=td_geometry, medium=td_material)


# ---------------------------------------------------------------------------
# Source conversion
# ---------------------------------------------------------------------------

def _convert_source(source, scene, td, s):
    """Convert a maxwell source to a Tidy3D source (lengths x *s*)."""
    from ..sources import PointDipole, PlaneWave, GaussianBeam
    domain_bounds = scene.domain.bounds

    if isinstance(source, PointDipole):
        component = _polarization_to_component(source.polarization)
        td_source_time = _convert_source_time(source.source_time, td)
        return td.PointDipole(
            center=_scale3(source.position, s),
            source_time=td_source_time,
            polarization=component,
            name=source.name or "point_dipole",
        )

    if isinstance(source, PlaneWave):
        td_source_time = _convert_source_time(source.source_time, td)
        direction = source.direction

        injection_axis = resolve_injection_axis(direction, source.injection_axis)
        axis_idx = _axis_name_to_index(injection_axis)
        inject_dir = "+" if direction[axis_idx] > 0 else "-"

        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [td.inf, td.inf, td.inf]
        center[axis_idx] = soft_plane_wave_coordinate(scene, injection_axis, float(direction[axis_idx])) * s
        size[axis_idx] = 0.0

        pol_angle, angle_theta, angle_phi = _direction_to_angles(direction, axis_idx)

        return td.PlaneWave(
            center=tuple(center),
            size=tuple(size),
            source_time=td_source_time,
            direction=inject_dir,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            name=source.name or "plane_wave",
        )

    if isinstance(source, GaussianBeam):
        td_source_time = _convert_source_time(source.source_time, td)
        direction = source.direction

        abs_dir = [abs(d) for d in direction]
        dominant_axis = int(np.argmax(abs_dir))
        inject_dir = "+" if direction[dominant_axis] > 0 else "-"

        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[dominant_axis] = source.focus[dominant_axis] * s
        size[dominant_axis] = 0.0

        pol_angle, angle_theta, angle_phi = _direction_to_angles(direction, dominant_axis)

        return td.GaussianBeam(
            center=tuple(center),
            size=tuple(size),
            source_time=td_source_time,
            direction=inject_dir,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            waist_radius=source.beam_waist * s,
            name=source.name or "gaussian_beam",
        )

    raise NotImplementedError(
        f"Source type '{type(source).__name__}' has no Tidy3D mapping yet."
    )


def _direction_to_angles(direction, dominant_axis):
    """Compute Tidy3D (pol_angle, angle_theta, angle_phi) from direction vector."""
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-15:
        return 0.0, 0.0, 0.0
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    tol = 1e-6
    if dominant_axis == 0 and abs(dy) < tol and abs(dz) < tol:
        return 0.0, 0.0, 0.0
    if dominant_axis == 1 and abs(dx) < tol and abs(dz) < tol:
        return 0.0, 0.0, 0.0
    if dominant_axis == 2 and abs(dx) < tol and abs(dy) < tol:
        return 0.0, 0.0, 0.0

    if dominant_axis == 2:
        angle_theta = math.acos(abs(dz))
        angle_phi = math.atan2(dy, dx)
    elif dominant_axis == 0:
        angle_theta = math.acos(abs(dx))
        angle_phi = math.atan2(dz, dy)
    else:
        angle_theta = math.acos(abs(dy))
        angle_phi = math.atan2(dx, dz)

    return 0.0, angle_theta, angle_phi


# ---------------------------------------------------------------------------
# Monitor conversion
# ---------------------------------------------------------------------------

def _convert_monitor(monitor, domain_bounds, frequencies, td, s):
    """Convert a maxwell monitor to a Tidy3D monitor (lengths x *s*)."""
    from ..monitors import FinitePlaneMonitor, ModeMonitor, PointMonitor, PlaneMonitor

    monitor_frequencies = (
        monitor.frequencies
        if hasattr(monitor, "frequencies") and monitor.frequencies
        else frequencies
    )
    if not monitor_frequencies:
        raise ValueError(
            f"Monitor '{monitor.name}' has no frequencies and none were passed to to_tidy3d()."
        )

    if isinstance(monitor, PointMonitor):
        return td.FieldMonitor(
            center=_scale3(monitor.position, s),
            size=(0.0, 0.0, 0.0),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, ModeMonitor):
        axis_idx = _axis_name_to_index(monitor.axis)
        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[axis_idx] = monitor.plane_position * s
        size[axis_idx] = 0.0
        return td.FieldMonitor(
            center=tuple(center),
            size=tuple(size),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, FinitePlaneMonitor):
        center = _scale3(monitor.position, s)
        size = _scale3(monitor.size, s)

        if monitor.compute_flux:
            return td.FluxMonitor(
                center=center,
                size=size,
                freqs=list(monitor_frequencies),
                name=monitor.name,
                normal_dir=monitor.normal_direction,
            )

        return td.FieldMonitor(
            center=center,
            size=size,
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, PlaneMonitor):
        axis_idx = _axis_name_to_index(monitor.axis)
        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[axis_idx] = monitor.position * s
        size[axis_idx] = 0.0

        if monitor.compute_flux:
            return td.FluxMonitor(
                center=tuple(center),
                size=tuple(size),
                freqs=list(monitor_frequencies),
                name=monitor.name,
                normal_dir=monitor.normal_direction,
            )

        return td.FieldMonitor(
            center=tuple(center),
            size=tuple(size),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    raise NotImplementedError(
        f"Monitor type '{type(monitor).__name__}' has no Tidy3D mapping yet."
    )


# ---------------------------------------------------------------------------
# Boundary conversion (no length scaling)
# ---------------------------------------------------------------------------

def _convert_boundary(boundary, td):
    """Convert maxwell BoundarySpec to Tidy3D BoundarySpec."""
    kind = boundary.kind
    if boundary.bloch_wavevector == "auto":
        raise ValueError(
            "Automatic Bloch wavevectors require Simulation.prepare() and cannot be exported to Tidy3D unresolved."
        )

    if kind == "pml":
        pml = td.PML(num_layers=boundary.num_layers)
        return td.BoundarySpec.all_sides(boundary=pml)

    if kind == "periodic":
        periodic = td.Periodic()
        return td.BoundarySpec.all_sides(boundary=periodic)

    if kind == "pec":
        pec = td.PECBoundary()
        return td.BoundarySpec.all_sides(boundary=pec)

    if kind == "pmc":
        pmc = td.PMCBoundary()
        return td.BoundarySpec.all_sides(boundary=pmc)

    if kind == "bloch":
        bloch_x = td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[0])
        bloch_y = td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[1])
        bloch_z = td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[2])
        return td.BoundarySpec(
            x=td.Boundary(plus=bloch_x, minus=bloch_x),
            y=td.Boundary(plus=bloch_y, minus=bloch_y),
            z=td.Boundary(plus=bloch_z, minus=bloch_z),
        )

    if kind == "none":
        pec = td.PECBoundary()
        return td.BoundarySpec.all_sides(boundary=pec)

    if kind == "mixed":
        axis_index = {"x": 0, "y": 1, "z": 2}

        def convert_face(face_kind, axis):
            if face_kind == "pml":
                return td.PML(num_layers=boundary.num_layers)
            if face_kind == "periodic":
                return td.Periodic()
            if face_kind == "pec":
                return td.PECBoundary()
            if face_kind == "pmc":
                return td.PMCBoundary()
            if face_kind == "bloch":
                return td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[axis_index[axis]])
            if face_kind == "none":
                return td.PECBoundary()
            raise ValueError(f"Unsupported boundary kind: {face_kind}")

        return td.BoundarySpec(
            x=td.Boundary(
                minus=convert_face(boundary.face_kind("x", "low"), "x"),
                plus=convert_face(boundary.face_kind("x", "high"), "x"),
            ),
            y=td.Boundary(
                minus=convert_face(boundary.face_kind("y", "low"), "y"),
                plus=convert_face(boundary.face_kind("y", "high"), "y"),
            ),
            z=td.Boundary(
                minus=convert_face(boundary.face_kind("z", "low"), "z"),
                plus=convert_face(boundary.face_kind("z", "high"), "z"),
            ),
        )

    raise ValueError(f"Unsupported boundary kind: {kind}")


# ---------------------------------------------------------------------------
# Grid conversion
# ---------------------------------------------------------------------------

def _convert_grid(grid, td, s):
    """Convert maxwell GridSpec to Tidy3D GridSpec (dl x *s*)."""
    if grid.is_custom or grid.is_auto:
        raise NotImplementedError(
            "Tidy3D export does not support nonuniform (GridSpec.custom / "
            "GridSpec.auto) grids; use a uniform GridSpec."
        )
    if grid.dx == grid.dy == grid.dz:
        return td.GridSpec.uniform(dl=grid.dx * s)
    return td.GridSpec(
        grid_x=td.UniformGrid(dl=grid.dx * s),
        grid_y=td.UniformGrid(dl=grid.dy * s),
        grid_z=td.UniformGrid(dl=grid.dz * s),
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def scene_to_tidy3d(
    scene,
    *,
    frequencies: float | Sequence[float] | None = None,
    run_time: float | None = None,
    length_scale: float = _M_TO_UM,
    **kwargs,
) -> tidy3d.Simulation:
    """Convert a maxwell ``Scene`` to a ``tidy3d.Simulation``.

    Parameters
    ----------
    scene : Scene
        The maxwell scene to convert.
    frequencies : float or sequence of float, optional
        Monitoring / source frequencies in Hz.  Required for monitors.
    run_time : float, optional
        Simulation run time in seconds.  If not given, an estimate is made
        from the domain size (10 light-crossing times).
    length_scale : float
        Factor to convert maxwell lengths (metres) to Tidy3D lengths.
        Default ``1e6`` (metres tomicrometres, the Tidy3D convention).
    **kwargs
        Extra keyword arguments forwarded to ``tidy3d.Simulation``.

    Returns
    -------
    tidy3d.Simulation
    """
    td = _ensure_tidy3d()
    s = length_scale  # shorthand

    # -- frequencies -----------------------------------------------------------
    if isinstance(frequencies, (int, float)):
        frequencies = (float(frequencies),)
    elif frequencies is not None:
        frequencies = tuple(float(frequency) for frequency in frequencies)

    # -- domain ----------------------------------------------------------------
    if scene.domain is None:
        raise ValueError("Scene must have a Domain for Tidy3D export.")
    center, size = _domain_to_center_size(scene.domain, s)

    # -- run_time estimate -----------------------------------------------------
    if run_time is None:
        c0 = 299_792_458.0
        max_extent_m = max(
            b[1] - b[0] for b in scene.domain.bounds
        )
        run_time = 10.0 * max_extent_m / c0

    # -- grid ------------------------------------------------------------------
    td_grid = None
    if scene.grid is not None:
        td_grid = _convert_grid(scene.grid, td, s)

    # -- boundary --------------------------------------------------------------
    td_boundary = td.BoundarySpec.all_sides(boundary=td.PML())
    if scene.boundary is not None:
        td_boundary = _convert_boundary(scene.boundary, td)

    # -- structures ------------------------------------------------------------
    td_structures = []
    for structure in (scene.structures or []):
        td_structures.append(_convert_structure(structure, td, s))

    # -- sources ---------------------------------------------------------------
    domain_bounds = scene.domain.bounds
    td_sources = []
    sources = scene.resolved_sources() if hasattr(scene, "resolved_sources") else (scene.sources or [])
    for source in sources:
        td_sources.append(_convert_source(source, scene, td, s))

    # -- monitors --------------------------------------------------------------
    td_monitors = []
    monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else (scene.monitors or [])
    for monitor in monitors:
        td_monitors.append(_convert_monitor(monitor, domain_bounds, frequencies, td, s))

    # -- symmetry --------------------------------------------------------------
    td_symmetry = (0, 0, 0)
    if scene.symmetry is not None:
        # Tidy3D encodes symmetry about the domain center only; the folded face
        # (low/high) has no Tidy3D counterpart and is dropped in the export.
        sym_map = {"PEC": -1, "PMC": 1}
        td_symmetry = tuple(
            0 if entry is None else sym_map.get(entry[0], 0)
            for entry in scene.symmetry
        )

    # -- build simulation ------------------------------------------------------
    sim_kwargs = dict(
        center=center,
        size=size,
        run_time=run_time,
        structures=td_structures,
        sources=td_sources,
        monitors=td_monitors,
        boundary_spec=td_boundary,
        symmetry=td_symmetry,
    )
    if td_grid is not None:
        sim_kwargs["grid_spec"] = td_grid
    sim_kwargs.update(kwargs)

    return td.Simulation(**sim_kwargs)
