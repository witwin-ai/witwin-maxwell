"""Shared scenes and measurements for the conformal thin-sheet CPU tests."""
from __future__ import annotations

import math

import numpy as np
import torch

import witwin.maxwell as mw

C0 = 299792458.0
ETA0 = 376.730313668


def rectangle_mesh(center, normal, half_u, half_v, *, u_direction=None, **kwargs):
    """Planar rectangle SurfaceMesh with outward ``normal`` (triangle orientation)."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if u_direction is None:
        seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
        u = np.cross(normal, seed)
    else:
        u = np.asarray(u_direction, dtype=np.float64) - normal * float(np.dot(u_direction, normal))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p = np.asarray(center, dtype=np.float64)
    vertices = np.stack([p - half_u * u - half_v * v, p + half_u * u - half_v * v,
                         p + half_u * u + half_v * v, p - half_u * u + half_v * v])
    return mw.SurfaceMesh(vertices, [(0, 1, 2), (0, 2, 3)], **kwargs)


def extruded_polyline_mesh(points, z_range, *, closed=False, **kwargs):
    """Faceted sheet extruded along z through an xy polyline (one planar strip per segment).

    The normal of each strip is (dy, -dx) / |d| for its segment direction d, so a
    counter-clockwise closed polygon has outward normals.
    """
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    z0, z1 = z_range
    vertices = np.concatenate((np.c_[points, np.full(count, z0)], np.c_[points, np.full(count, z1)]))
    triangles = []
    for i in range(count if closed else count - 1):
        j = (i + 1) % count
        # Strip i: (i, j) at z0 and (i, j) at z1, normal (dy, -dx).
        triangles += [(i, j, count + i), (j, count + j, count + i)]
    return mw.SurfaceMesh(vertices, triangles, **kwargs)


def polygon_points(radius, sides, *, center=(0.0, 0.0), phase=0.0):
    angles = phase + 2 * np.pi * np.arange(sides) / sides
    return np.c_[center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]


def pec_material():
    return mw.Material.pec() if hasattr(mw.Material, "pec") else mw.PEC


def resistive_foil(sheet_conductance, *, conductivity=1.0e5, band=(0.3e9, 2.0e9)):
    """Electrically thin CopperFoil with sigma*t = sheet_conductance (siemens)."""
    return mw.CopperFoil(thickness=sheet_conductance / conductivity, conductivity=conductivity, frequency_range=band)


def oblique_transmission_scene(*, material, angle_deg, polarization, dl, frequency=1.0e9, half_length=None,
                               observer_offset=None, with_sheet=True, pml_layers=10, sheet_shift=0.0, nz=2,
                               plus_side_eps=None, device="cpu", aligned_foil=False):
    """Quasi-2D plane-wave transmission through a sheet tilted about z.

    The sheet normal is (cos a, sin a, 0); the plane wave travels along +x so
    the incidence angle is ``a``. Two z cells with PEC (TE: E along z) or PMC
    (TM: H along z) walls make the problem exactly z-invariant. The sheet is
    finite along its in-plane direction and ends outside the gated window.
    ``aligned_foil`` rotates the incidence onto a native foil at x = 0,
    retaining the incidence angle and one wavelength of tangential padding.
    """
    wavelength = C0 / frequency
    angle = math.radians(angle_deg)
    half_length = 6 * wavelength if half_length is None else half_length
    observer_offset = 0.5 * wavelength if observer_offset is None else observer_offset
    sheet_angle = 0.0 if aligned_foil else angle
    normal = np.array((math.cos(sheet_angle), math.sin(sheet_angle), 0.0))
    tangent = np.array((-math.sin(sheet_angle), math.cos(sheet_angle), 0.0))
    direction = (math.cos(angle), -math.sin(angle), 0.0) if aligned_foil else (1.0, 0.0, 0.0)
    # Extents are whole multiples of dl so the uniform grid spacing is exactly dl
    # and the observer sample positions below coincide with the solver's nodes.
    x_extent = math.ceil((half_length * abs(math.sin(angle)) + 1.6 * wavelength) / dl - 1e-9) * dl
    y_extent = math.ceil((half_length * abs(math.cos(sheet_angle)) + 1.0 * wavelength) / dl - 1e-9) * dl
    bounds = ((-x_extent, x_extent), (-y_extent, y_extent), (0.0, nz * dl))
    wall = "pec" if polarization == "te" else "pmc"
    e_pol = (0.0, 0.0, 1.0) if polarization == "te" else (0.0, 1.0, 0.0)
    if aligned_foil and polarization == "tm":
        e_pol = (math.sin(angle), math.cos(angle), 0.0)
    structures = []
    if with_sheet:
        if aligned_foil:
            from witwin.core import Box
            geometry = Box(position=(sheet_shift, 0.0, 0.5 * nz * dl), size=(0.0, 2 * half_length, 8 * nz * dl))
        else:
            geometry = rectangle_mesh((sheet_shift * normal[0], sheet_shift * normal[1], 0.5 * nz * dl),
                                      normal, half_length, 4 * nz * dl, u_direction=tangent,
                                      sheet_model="conformal_two_sided")
        structures.append(mw.Structure(geometry=geometry, material=material))
    if plus_side_eps is not None:
        # Dielectric half-space on the sheet's + side; its face coincides with
        # the (possibly tilted) sheet plane.
        from witwin.core import Box
        far = 2 * (x_extent + y_extent) + 2.0 * wavelength
        center = (sheet_shift + 0.5 * far) * normal + np.array((0.0, 0.0, 0.5 * nz * dl))
        rotation = (math.cos(0.5 * sheet_angle), 0.0, 0.0, math.sin(0.5 * sheet_angle))
        structures.append(mw.Structure(geometry=Box(position=tuple(center), size=(far, far, 8 * nz * dl),
                                                    rotation=rotation),
                                       material=mw.Material(eps_r=plus_side_eps)))
    z_obs = 0.5 * nz * dl if polarization == "tm" else 0.5 * dl
    # Point monitors sample the nearest Yee node, so observers are placed on
    # exact sample positions. The reflected field at ``front`` equals R times
    # the incident plane wave at the mirror point, whose x coordinate is in
    # general not a node: the incident run records the two nodes bracketing
    # it ("ref0", "ref1") and the analysis interpolates with the measured
    # numerical phase step, exact for a discrete plane wave along x.
    x_nodes = np.linspace(-x_extent, x_extent, int(round(2 * x_extent / dl)) + 1)
    y_count = int(round(2 * y_extent / dl))
    y_nodes = np.linspace(-y_extent, y_extent, y_count + 1)
    y_samples = y_nodes if polarization == "te" else 0.5 * (y_nodes[1:] + y_nodes[:-1])

    def snap(point):
        return (float(x_nodes[np.argmin(np.abs(x_nodes - point[0]))]),
                float(y_samples[np.argmin(np.abs(y_samples - point[1]))]), z_obs)

    front = snap(-observer_offset * normal + sheet_shift * normal)
    back = snap(observer_offset * normal + sheet_shift * normal)
    distance = float(np.dot(front[:2], normal[:2])) - sheet_shift
    mirror_x = front[0] - 2 * distance * normal[0]
    ref_index = int(np.searchsorted(x_nodes, mirror_x)) - 1
    observers = {"back": back, "front": front,
                 "ref0": (float(x_nodes[ref_index]), front[1], z_obs),
                 "ref1": (float(x_nodes[ref_index + 1]), front[1], z_obs)}
    mirror_fraction = (mirror_x - x_nodes[ref_index]) / dl
    monitors = [mw.FieldTimeMonitor(name, components=("Ez",) if polarization == "te" else ("Ey", "Ex"),
                                    position=position) for name, position in observers.items()]
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.uniform(dl),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=pml_layers, z=wall), device=device,
        sources=[mw.PlaneWave(direction=direction, polarization=e_pol,
                              source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.5 * frequency))],
        structures=structures, monitors=monitors)
    return scene, dict(normal=normal, tangent=tangent, half_length=half_length, observers=observers,
                       mirror_fraction=float(mirror_fraction))


def backend_for(scene):
    """CPU scenes use the explicit CPU reference profile, CUDA scenes the native kernels."""
    return "cpu_reference" if torch.device(scene.device).type == "cpu" else "native"


def run(scene, *, time_steps=None, run_time=None, frequency=1.0e9):
    config = mw.TimeConfig(time_steps=time_steps) if time_steps is not None else run_time
    return mw.Simulation.fdtd(scene, absorber="cpml", frequencies=[frequency], run_time=config, backend=backend_for(scene)).run()


def time_series(result, name, component):
    data = result.monitors[name]
    values = data["values"][component] if "values" in data else data[component]
    return np.asarray(torch.as_tensor(values).detach().cpu().double()).reshape(-1)


def field_series(result, name, component):
    data = result.monitors[name]
    return (np.asarray(torch.as_tensor(data["t"]).detach().cpu().double()).reshape(-1),
            np.asarray(torch.as_tensor(data["components"][component]).detach().cpu().double()).reshape(-1))


def gated_ratio(t, total, incident, frequencies, *, center, half_width):
    """Complex spectral ratio of two signals inside one smooth time gate."""
    x = (t - center) / half_width
    gate = np.where(np.abs(x) < 1, np.cos(0.5 * np.pi * x) ** 2, 0.0)
    phase = np.exp(2j * np.pi * np.outer(frequencies, t))
    return (phase @ (gate * total)) / (phase @ (gate * incident))


def gaussian_sigma(frequency):
    return 1.0 / (2 * math.pi * 0.5 * frequency)


def thin_sheet_reflection_te(sheet_conductance, angle_deg):
    """Ez reflection coefficient of a resistive sheet (E tangential), PEC limit -1."""
    g = ETA0 * sheet_conductance
    c = math.cos(math.radians(angle_deg))
    return -g / (2 * c + g)


def thin_sheet_reflection_tm_ey(sheet_conductance, angle_deg):
    """Ey reflection ratio for the TM scene: (T_TM - 1) cos(2a); PEC limit -cos(2a)."""
    return (thin_sheet_transmission(sheet_conductance, angle_deg, "tm") - 1) * math.cos(math.radians(2 * angle_deg))


def thin_sheet_transmission(sheet_conductance, angle_deg, polarization):
    g = ETA0 * sheet_conductance
    c = math.cos(math.radians(angle_deg))
    if polarization == "te":
        return 2 * c / (2 * c + g)
    return 2 / (2 + g * c)


def prepared_dt(scene, *, frequency=1.0e9, absorber="cpml"):
    probe = mw.Simulation.fdtd(scene, absorber=absorber, frequencies=[frequency], run_time=mw.TimeConfig(time_steps=1),
                               backend=backend_for(scene)).prepare()
    # Compare grids at the accepted PEC bound, keeping dt/h consistent.
    timing = probe.solver._time_step_metadata
    dt = timing["safety_factor"] * timing["stability_limit"]
    if probe.solver._thin_sheets:
        dt = min(dt, .985 * float(probe.solver._thin_sheet_cfl_limit))
    return dt


def run_with_dt(scene, dt, duration, *, frequency=1.0e9, absorber="cpml"):
    steps = int(math.ceil(duration / dt))
    return mw.Simulation.fdtd(scene, absorber=absorber, frequencies=[frequency], run_time=mw.TimeConfig(time_steps=steps, dt=dt),
                              backend=backend_for(scene)).run()


def measure_transmission(material, angle_deg, polarization, dl, *, half_length_wavelengths=6.0, sheet_shift=0.0,
                         observer_offset=None, plus_side_eps=None, reference_material=None,
                         frequency=1.0e9, frequencies=(0.8e9, 1.0e9, 1.2e9), device="cpu",
                         aligned_foil=False, dt=None, absorber="cpml"):
    wavelength = C0 / frequency
    series = {}
    sigma = gaussian_sigma(frequency)
    # ``reference_material`` replaces the sheet-free reference run by the same
    # sheet geometry with another material, so both runs share one discrete
    # interface (needed when a dielectric face coincides with the sheet).
    scenes = {with_sheet: oblique_transmission_scene(
                  material=material if with_sheet else reference_material, angle_deg=angle_deg,
                  polarization=polarization, dl=dl, half_length=half_length_wavelengths * wavelength,
                  with_sheet=with_sheet or reference_material is not None, frequency=frequency,
                  sheet_shift=sheet_shift, observer_offset=observer_offset, plus_side_eps=plus_side_eps,
                  device=device, aligned_foil=aligned_foil)
              for with_sheet in (False, True)}
    # Both runs share one dt (the sheet-tightened step, capped below the
    # explicit-dt safety limit) so source and sampling clocks are identical.
    if dt is None:
        dt = min(prepared_dt(scenes[True][0], frequency=frequency, absorber=absorber), 0.9 * dl / (C0 * math.sqrt(3.0)))
    for with_sheet in (False, True):
        scene, info = scenes[with_sheet]
        x_min = scene.domain.bounds[0][0]
        duration = (info["observers"]["back"][0] - x_min) / C0 + 14 * sigma
        result = run_with_dt(scene, dt, duration, frequency=frequency, absorber=absorber)
        if with_sheet:
            chord_unknowns = sum(runtime.chord_e.numel() for runtime in result.solver._thin_sheets
                                 if runtime.copper is not None)
            foil_entries = len(result.solver._copper_foils)
        component = "Ez" if polarization == "te" else "Ey"
        for name in ("back", "front", "ref0", "ref1"):
            series[(with_sheet, name)] = field_series(result, name, component)
        series[(with_sheet, "dt")] = float(result.solver.dt)
        del result
    t_inc, back_inc = series[(False, "back")]
    _, front_inc = series[(False, "front")]
    t_tot, back_tot = series[(True, "back")]
    _, front_tot = series[(True, "front")]
    back_tot = np.interp(t_inc, t_tot, back_tot)
    front_tot = np.interp(t_inc, t_tot, front_tot)
    freqs = np.asarray(frequencies)
    center = t_inc[int(np.argmax(np.abs(back_inc)))]
    transmission = gated_ratio(t_inc, back_tot, back_inc, freqs, center=center, half_width=4.0 * sigma)
    ref0, ref1 = series[(False, "ref0")][1], series[(False, "ref1")][1]
    center = t_inc[int(np.argmax(np.abs(ref0)))]
    scattered = gated_ratio(t_inc, front_tot - front_inc, ref0, freqs, center=center, half_width=4.0 * sigma)
    step = gated_ratio(t_inc, ref1, ref0, freqs, center=center, half_width=4.0 * sigma)
    u = scenes[False][1]["mirror_fraction"]
    mirror = np.abs(step) ** u * np.exp(1j * u * np.angle(step))
    reflection = scattered / mirror
    return transmission, reflection, dict(dt_sheet=series[(True, "dt")], dt_free=series[(False, "dt")],
                                          chord_unknowns=chord_unknowns, foil_entries=foil_entries)


def _bessel_series(kind, ka, kr, phi, terms=None):
    """Scattered field / incident amplitude of a PEC circular cylinder (exp(-i w t)).

    ``kind`` "te": field along the axis is E (E_z), "tm": it is H (H_z).
    """
    from scipy.special import hankel1, h1vp, jv, jvp
    terms = int(ka + 4 * ka ** (1 / 3) + 12) if terms is None else terms
    n = np.arange(-terms, terms + 1)
    if kind == "te":
        coefficient = jv(n, ka) / hankel1(n, ka)
    else:
        coefficient = jvp(n, ka) / h1vp(n, ka)
    return -np.sum((1j ** n)[:, None] * coefficient[:, None] * hankel1(n[:, None], kr[None])
                   * np.exp(1j * n[:, None] * phi[None]), 0)


def cylinder_scattering(material, polarization, dl, *, sides=32, radius_wl=0.5, observer_wl=1.0, frequency=1.0e9,
                        device="cuda", angles=8, center_cells=(0.37, 0.21), phase=0.1, pml_layers=12, absorber="cpml"):
    """Quasi-2D scattering by a faceted closed sheet vs the circular-cylinder series.

    The prism has ``sides`` planar facets extruded across the two z cells (PEC
    walls for TE, E along the axis; PMC walls for TM, H along the axis) and the
    area-equivalent radius ``radius_wl`` wavelengths. The scattered phasor
    (total minus incident run, same grid and dt) is sampled at ``angles``
    observers on a circle of ``observer_wl`` wavelengths, snapped onto exact
    sample nodes, and normalized by the incident phasor at a reference node.
    Returns (relative RMS error, info).
    """
    wavelength = C0 / frequency
    k = 2 * np.pi / wavelength
    radius = radius_wl * wavelength
    circumradius = radius / math.sqrt(sides / (2 * np.pi) * math.sin(2 * np.pi / sides))
    center = np.array(center_cells) * dl
    extent = math.ceil((observer_wl + 0.3) * wavelength / dl - 1e-9) * dl
    nz = 2
    bounds = ((-extent, extent), (-extent, extent), (0.0, nz * dl))
    wall = "pec" if polarization == "te" else "pmc"
    e_pol = (0.0, 0.0, 1.0) if polarization == "te" else (0.0, 1.0, 0.0)
    component = "Ez" if polarization == "te" else "Hz"
    # Sample lattice of the axial component in the xy plane.
    nodes = np.linspace(-extent, extent, int(round(2 * extent / dl)) + 1)
    lattice = nodes if polarization == "te" else 0.5 * (nodes[1:] + nodes[:-1])
    z_obs = 0.5 * dl if polarization == "te" else 0.5 * nz * dl

    def snap(point):
        return np.array((lattice[np.argmin(np.abs(lattice - point[0]))], lattice[np.argmin(np.abs(lattice - point[1]))]))

    theta = 0.2 + 2 * np.pi * np.arange(angles) / angles
    observers = [snap(center + observer_wl * wavelength * np.array((np.cos(t), np.sin(t)))) for t in theta]
    reference = snap(np.zeros(2))
    monitors = [mw.FieldTimeMonitor(f"o{i}", components=(component,), position=(p[0], p[1], z_obs))
                for i, p in enumerate(observers)]
    monitors.append(mw.FieldTimeMonitor("ref", components=(component,), position=(reference[0], reference[1], z_obs)))
    mesh_kwargs = {} if getattr(material, "is_copper_foil", False) or isinstance(material, mw.CopperFoil) else dict(
        sheet_model="conformal_two_sided")
    mesh = extruded_polyline_mesh(polygon_points(circumradius, sides, center=center, phase=phase),
                                  (-2 * dl, (nz + 2) * dl), closed=True, **mesh_kwargs)
    scenes = {}
    for with_sheet in (False, True):
        scenes[with_sheet] = mw.Scene(
            domain=mw.Domain(bounds=bounds), grid=mw.GridSpec.uniform(dl),
            boundary=mw.BoundarySpec.faces(default="pml", num_layers=pml_layers, z=wall), device=device,
            sources=[mw.PlaneWave(direction=(1.0, 0.0, 0.0), polarization=e_pol,
                                  source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.5 * frequency))],
            structures=[mw.Structure(geometry=mesh, material=material)] if with_sheet else [], monitors=monitors)
    dt = min(prepared_dt(scenes[True], frequency=frequency, absorber=absorber), 0.9 * dl / (C0 * math.sqrt(3.0)))
    duration = (4 * extent) / C0 + 14 * gaussian_sigma(frequency)
    series = {}
    for with_sheet, scene in scenes.items():
        result = run_with_dt(scene, dt, duration, frequency=frequency, absorber=absorber)
        for name in [f"o{i}" for i in range(angles)] + ["ref"]:
            series[(with_sheet, name)] = field_series(result, name, component)
        if with_sheet:
            info = dict(result.solver._thin_sheets[0].complex.statistics)
        del result
    t = series[(False, "ref")][0]
    phase_factor = np.exp(2j * np.pi * frequency * t)
    incident = series[(False, "ref")][1] @ phase_factor
    amplitude = incident / np.exp(1j * k * reference[0])          # incident phasor at x = 0
    measured, expected = [], []
    for i, point in enumerate(observers):
        total = series[(True, f"o{i}")][1] @ phase_factor
        free = series[(False, f"o{i}")][1] @ phase_factor
        measured.append((total - free) / amplitude)
        local = point - center
        series_value = _bessel_series(polarization, k * radius, np.array([k * np.linalg.norm(local)]),
                                      np.array([math.atan2(local[1], local[0])]))[0]
        expected.append(series_value * np.exp(1j * k * center[0]))
    measured, expected = np.array(measured), np.array(expected)
    error = float(np.sqrt(np.mean(np.abs(measured - expected) ** 2) / np.mean(np.abs(expected) ** 2)))
    return error, dict(info=info, dt=dt, measured=measured, expected=expected,
                       facet_width=2 * circumradius * math.sin(math.pi / sides))
