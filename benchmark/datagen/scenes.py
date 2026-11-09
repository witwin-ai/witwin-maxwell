"""Shared, deterministic FDTD sample workloads, expressed through the public API.

Cell counts include eight external PML layers on each absorbing face. Sparse
point readouts use PointMonitor's native Yee sampling, not interpolation.
The 2.4 GHz board uses local feed refinement; copper thickness is a sheet-model
parameter and does not introduce a volumetric 35 micrometre grid interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial
import math
from typing import Callable

import numpy as np
import torch

import witwin.maxwell as mw

C0 = 299792458.0
PML = 8
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@dataclass
class Sample:
    name: str
    family: str
    scene: mw.Scene
    fdtd_kwargs: dict
    readouts: list[tuple[str, Callable[[mw.Result], torch.Tensor | float]]]
    cache_unavailable: str | None = None


@dataclass(frozen=True)
class SceneBuilder:
    name: str
    family: str
    expected_cells: int
    build: Callable[..., Sample]


def _grid_scene(axes, *, device, boundary=None, **kwargs):
    return mw.Scene(
        domain=mw.Domain(tuple((float(axis[0]), float(axis[-1])) for axis in axes)),
        grid=mw.GridSpec.custom(*axes),
        boundary=mw.BoundarySpec.pml(num_layers=PML) if boundary is None else boundary,
        device=device,
        **kwargs,
    )


def _cube(n, span, *, device):
    nodes = np.linspace(-span / 2, span / 2, n + 1)
    return _grid_scene((nodes,) * 3, device=device)


def _kwargs(frequencies, *, antenna=False, decay=False):
    # decay=True follows the data-plan label rule: stop on field-energy decay, with a
    # longer horizon as the cap, instead of a fixed truncated horizon.
    kwargs = dict(
        frequencies=frequencies,
        run_time=mw.TimeConfig.auto(steady_cycles=40 if decay else 12, transient_cycles=8),
        shutoff=1e-5 if decay else 0.0,
        full_field_dft=False,
    )
    if antenna:
        center = (frequencies[0] + frequencies[-1]) / 2
        kwargs["excitations"] = (mw.PortExcitation(
            "feed", source_time=mw.GaussianPulse(frequency=center, fwidth=0.8 * center),
        ),)
    return kwargs


def _box(position, size):
    return mw.Box(position=torch.tensor(position, dtype=torch.float64),
                  size=torch.tensor(size, dtype=torch.float64))


def _flux(name, result):
    return result.monitor(name)["flux"]


def _point_fields(names, result):
    monitors = result.monitors
    return torch.stack([
        torch.stack([monitors[name][component] for component in COMPONENTS])
        for name in names
    ])


def _surface(scene, frequencies, size):
    scene.add_monitor(mw.ClosedSurfaceMonitor.box(
        "radiation", size=size, frequencies=frequencies,
    ))


def _antenna(result):
    return result.antenna(surface="radiation", driven_port="feed",
                          theta_points=19, phi_points=37)


def _port(result):
    return result.port("feed")


def _tensor_readouts(prefix, factory, attributes):
    # Share one expensive transform among named tensor readouts for this sample.
    @cache
    def data(result):
        return factory(result)

    return [(f"{prefix}_{name}", lambda result, name=name: getattr(data(result), name))
            for name in attributes]


def _antenna_sample(name, family, scene, frequencies, cache_unavailable=None, decay=False):
    readouts = _tensor_readouts("port", _port, (
        "frequencies", "reflection_coefficient", "z_in", "incident_power",
        "reflected_power", "accepted_power", "delivered_power",
    ))
    readouts += _tensor_readouts("antenna", _antenna, (
        "e_theta", "e_phi", "directivity", "gain", "realized_gain",
        "radiation_efficiency", "mismatch_efficiency", "system_efficiency",
        "p_rad", "p_accepted", "p_incident",
    ))
    readouts.append(("closed_surface_flux", partial(_flux, "radiation")))
    return Sample(name, family, scene, _kwargs(frequencies, antenna=True, decay=decay), readouts,
                  cache_unavailable)


DIELECTRIC_SOURCE_PHASE = CONDUCTOR_SOURCE_PHASE = math.pi / 2


def dielectric(name, *, n, wavelengths, bodies, kind, eps, lossy, device):
    """One to four separated bodies and 16,384 seeded sparse Yee readouts."""
    frequency = 2.4e9
    wavelength = C0 / frequency
    span = wavelengths * wavelength
    scene = _cube(n, span, device=device)
    for index in range(bodies):
        position = ((index - (bodies - 1) / 2) * 0.48 * wavelength, 0.0, 0.0)
        shape = ("box", "sphere", "cylinder", "mesh")[index] if kind == "mixed" else kind
        if shape == "box":
            geometry = _box(position, (0.3 * wavelength,) * 3)
        elif shape == "sphere":
            geometry = mw.Sphere(position=position, radius=0.18 * wavelength)
        elif shape == "cylinder":
            geometry = mw.Cylinder(position=position, radius=0.16 * wavelength,
                                   height=0.36 * wavelength, axis="z")
        elif shape == "mesh":
            vertices, faces = mw.Pyramid(position=position, base_size=0.35 * wavelength,
                                         height=0.4 * wavelength, axis="z").to_mesh()
            geometry = mw.Mesh(vertices, faces, recenter=False, fill_mode="solid")
        scene.add_structure(mw.Structure(
            geometry, mw.Material(eps_r=eps, sigma_e=0.02 if lossy else 0.0),
            name=f"body_{index}",
        ))
    # Sine carrier (phase pi/2): odd about the delay, so the dipole leaves no static
    # charge and the E energy decays; the phase-0 pulse has a DC fraction of 0.87.
    scene.add_source(mw.PointDipole(
        position=(0.0, 0.0, -0.65 * wavelength), polarization="Ez", profile="ideal",
        source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.8 * frequency,
                                     phase=DIELECTRIC_SOURCE_PHASE),
    ))
    rng = np.random.default_rng(1701)
    # Unique physical-grid nodes, excluding the outermost cells.
    indices = rng.choice((n - 4) ** 3, 16384, replace=False)
    indices = np.column_stack(np.unravel_index(indices, (n - 4,) * 3)) + 2
    points = -span / 2 + indices * (span / n)
    names = tuple(f"point_{index:05d}" for index in range(len(points)))
    for point_name, point in zip(names, points):
        scene.add_monitor(mw.PointMonitor(point_name, point, fields=COMPONENTS))
    unsupported = ("FDTDMaterialCache accepts Box/Mesh/PlanarPEC, not analytic curved bodies"
                   if kind in ("sphere", "cylinder", "mixed") else None)
    return Sample(name, "F1", scene, _kwargs((frequency,)),
                  [("fields", partial(_point_fields, names))], unsupported)


def conductor(name, *, n, kind, device):
    frequency = 2.4e9
    wavelength = C0 / frequency
    span = n * wavelength / 20
    scene = _cube(n, span, device=device)
    if kind == "box":
        geometry = _box((0.0, 0.0, 0.0), (0.5 * wavelength,) * 3)
    else:
        holes = ()
        if kind == "holes":
            holes = tuple(tuple((x * wavelength, y * wavelength) for x, y in (
                (center - 0.06, -0.08), (center + 0.06, -0.08),
                (center + 0.06, 0.08), (center - 0.06, 0.08),
            )) for center in (-0.18, 0.18))
        geometry = mw.PlanarPEC.rectangle(
            ((-0.4 * wavelength, 0.4 * wavelength),) * 2, holes=holes,
        )
    scene.add_structure(mw.Structure(geometry, mw.Material.pec(), name="scatterer"))
    # The Huygens surface lies outside the TFSF box and records scattered fields.
    scene.add_source(mw.PlaneWave(
        direction=(0, 0, 1), polarization="Ex",
        source_time=mw.GaussianPulse(frequency=frequency, fwidth=0.8 * frequency,
                                     phase=CONDUCTOR_SOURCE_PHASE),
        injection=mw.TFSF(((-0.55 * wavelength, 0.55 * wavelength),) * 3),
    ))
    _surface(scene, (frequency,), (1.4 * wavelength,) * 3)
    readouts = _tensor_readouts("far_field", _projected_fields, ("E", "H", "directions"))
    readouts.append(("closed_surface_flux", partial(_flux, "radiation")))
    return Sample(name, "F2", scene, _kwargs((frequency,)), readouts)


def _projected_fields(result):
    projector = mw.postprocess.FieldProjector.from_monitor(
        result, "radiation", device=result.scene.device,
    )
    return projector.angles(np.linspace(0, math.pi, 19), np.linspace(0, 2 * math.pi, 37))


def _partition(anchors, step, *, fine=(), half_planes=()):
    """Partition fixed physical anchors, preserving native H-contour half nodes."""
    contour_step = min(step, .0004)
    brackets = tuple((plane - contour_step / 2, plane + contour_step / 2)
                     for plane in half_planes)
    required = sorted(set((*anchors, *(x for lo, hi, _ in fine for x in (lo, hi)),
                           *(x for pair in brackets for x in pair))))
    nodes = [required[0]]
    for lo, hi in zip(required, required[1:]):
        target = step
        for first, last, local_step in fine:
            if first <= (lo + hi) / 2 <= last:
                target = min(target, local_step)
        count = (1 if (lo, hi) in brackets else
                 max(1, math.ceil((hi - lo) / target - 1e-12)))
        nodes.extend(np.linspace(lo, hi, count + 1)[1:])
    return np.asarray(nodes)


def _printed_axes(lateral, air, finest, height):
    x = _partition((-.065, -.045, -.040, -.001, 0., .001, .040, .045, .065), air,
                   fine=((-.045, .045, lateral), (-.002, .002, finest)))
    y = _partition((-.025, -.0055, -.001, 0., .001, .0055, .025), air,
                   fine=((-.0055, .0055, lateral),), half_planes=(-.0035, .0035))
    z = _partition((-.025, -height, 0., .025), air,
                   fine=((-height, 0., min(height / 7, finest)),),
                   half_planes=(-.0025, .0025))
    return x, y, z


def printed(name, *, lateral, air, finest, height, count, lossy, device):
    """90 x 11 mm copper printed dipole with an explicit finite-width LumpedPort."""
    frequencies = tuple(np.linspace(1.8e9, 3.0e9, count))
    axes = _printed_axes(lateral, air, finest, height)
    scene = _grid_scene(axes, device=device)
    scene.add_structure(mw.Structure(
        _box((0., 0., -height / 2), (.090, .011, height)),
        mw.Material(eps_r=3.5, sigma_e=.01 if lossy else 0.), name="substrate",
    ))
    for arm, lo, hi in (("negative", -.040, -.001), ("positive", .001, .040)):
        scene.add_structure(mw.Structure(
            _box(((lo + hi) / 2, 0., 0.), (hi - lo, .002, 0.)),
            mw.CopperFoil(thickness=35e-6, frequency_range=(.1e9, 10e9)), name=arm,
        ))
    scene.add_port(mw.LumpedPort(
        "feed", positive=(.001, 0., 0.), negative=(-.001, 0., 0.),
        voltage_path=mw.AxisPath("x"), reference_impedance=50.,
        current_surface=_box((axes[0][np.searchsorted(axes[0], 0.) + 1] / 2, 0., 0.),
                             (0., .007, .005)),
        voltage_aperture=_box((0., 0., 0.), (.002, .002, 0.)),
    ))
    _surface(scene, frequencies, (.110, .035, .035))
    return _antenna_sample(name, "F3", scene, frequencies)


def wire(name, *, n, kind, count, finite, device):
    frequency = 2.4e9
    wavelength = C0 / frequency
    span = 1.6 * wavelength
    dx = span / n
    scene = _cube(n, span, device=device)
    conductor = mw.WireConductor.finite(conductivity=5.8e7) if finite else mw.WireConductor.pec()
    if kind == "loop":
        radius = .18 * wavelength
        angle = math.asin(dx / (2 * radius))
        angles = np.linspace(angle, 2 * math.pi - angle,
                             math.ceil(2 * math.pi * radius / dx) + 1)
        points = [(radius * math.sin(t), -radius * math.cos(t), 0.) for t in angles]
        points[0] = (dx / 2, -radius * math.cos(angle), 0.)
        points[-1] = (-dx / 2, points[0][1], 0.)
        # Close the loop through the feed chord; the gap binding removes that segment.
        points.append(points[0])
        scene.add_thin_wire(mw.ThinWire("loop", points, .12 * dx, conductor, snap="continuous"))
        negative = mw.WireNodeRef("loop", len(points) - 2)
        positive = mw.WireNodeRef("loop", len(points) - 1)
    else:
        arms = []
        for sign in (-1, 1):
            segments = max(20, math.ceil(.6 * wavelength / dx))
            if kind == "dipole":
                # About half a cell per segment; much shorter sub-cell segments
                # trip the thin-wire proximity check on coarse grids.
                segments = max(8, math.ceil((.24 * wavelength - dx / 2) / (.5 * dx)))
                points = [(0., 0., sign * value) for value in
                          np.linspace(dx / 2, .24 * wavelength, segments + 1)]
            else:
                points = [(.07 * wavelength * (math.cos(t) - 1),
                           .07 * wavelength * math.sin(t),
                           sign * (dx / 2 + .12 * wavelength * t / (4 * math.pi)))
                          for t in np.linspace(0., 4 * math.pi, segments + 1)]
            arms.append(points)
        # One wire through the feed: the gap binding removes the segment between the arm
        # roots at z = -dx/2 and +dx/2. The node binding's voltage diverges with refinement.
        scene.add_thin_wire(mw.ThinWire("wire", arms[0][::-1] + arms[1], .12 * dx, conductor,
                                        snap="continuous"))
        negative = mw.WireNodeRef("wire", len(arms[0]) - 1)
        positive = mw.WireNodeRef("wire", len(arms[0]))
    scene.add_port(mw.LumpedPort(
        "feed", wire_binding=mw.WirePortBinding.gap(negative=negative, positive=positive),
        reference_impedance=50.,
    ))
    frequencies = tuple(np.linspace(.8 * frequency, 1.2 * frequency, count))
    _surface(scene, frequencies, (1.2 * wavelength,) * 3)
    return _antenna_sample(name, "F4", scene, frequencies,
                          "FDTDMaterialCache does not support scene.thin_wires", decay=True)


def periodic(name, *, n, nz, loaded, frequency, device):
    wavelength = C0 / 2.4e9
    transverse = np.linspace(-.4 * wavelength, .4 * wavelength, n + 1)
    z = np.linspace(-wavelength, wavelength, nz + 1)
    boundary = mw.BoundarySpec.faces(
        default="pml", num_layers=PML, x="periodic" if loaded else "bloch",
        y="periodic" if loaded else "bloch", z="pml",
        bloch_wavevector=(0., 0., 0.) if loaded else "auto",
    )
    scene = _grid_scene((transverse, transverse, z), device=device, boundary=boundary)
    gap = 1.6 * wavelength / n
    for side, lo, hi in (("negative", -.28 * wavelength, -gap / 2),
                         ("positive", gap / 2, .28 * wavelength)):
        scene.add_structure(mw.Structure(
            mw.PlanarPEC.rectangle(((lo, hi), (-.28 * wavelength, .28 * wavelength))),
            mw.Material.pec(), name=side,
        ))
    if loaded:
        dz = 2 * wavelength / nz
        scene.add_port(mw.LumpedPort(
            "load", positive=(gap / 2, 0., 0.), negative=(-gap / 2, 0., 0.),
            voltage_path=mw.AxisPath("x"),
            current_surface=_box((gap / 4, 0., 0.), (0., 1.5 * gap, 3 * dz)),
            termination=mw.ParallelRLC(r=100., l=2e-9, c=.2e-12),
        ))
    angle = 0. if loaded else math.radians(35)
    scene.add_source(mw.PlaneWave(
        direction=(math.sin(angle), 0., math.cos(angle)),
        polarization=(math.cos(angle), 0., -math.sin(angle)),
        source_time=mw.GaussianPulse(frequency=frequency, fwidth=.8 * frequency) if loaded
        else mw.CW(frequency=frequency),
        injection=mw.TFSF.slab(axis="z", bounds=(-.4 * wavelength, .8 * wavelength)),
    ))
    for label, position, normal in (("reflected", -.6 * wavelength, "-"),
                                    ("transmitted", .6 * wavelength, "+")):
        scene.add_monitor(mw.FluxMonitor(label, axis="z", position=position,
                                         normal_direction=normal, frequencies=(frequency,)))
    kwargs = _kwargs((frequency,))
    if loaded:
        # Real periodic + PML z: absorber="auto" would select the graded absorber, which
        # needs >= 40 layers. Pin CPML so the frozen datagen rows stay bit-unchanged;
        # F5 production chooses its absorber when F5 is defined in the label runner.
        kwargs["absorber"] = "cpml"
    return Sample(name, "F5", scene, kwargs, [
        ("reflected_flux", partial(_flux, "reflected")),
        ("transmitted_flux", partial(_flux, "transmitted")),
    ])


def scene_builders() -> tuple[SceneBuilder, ...]:
    """Return the catalog without constructing scenes or touching CUDA."""
    builders = []
    dielectric_cases = (
        ("box_small", 32, 2, 1, "box", 1.5, False),
        ("sphere_small", 36, 2, 1, "sphere", 4., False),
        ("cylinder_small", 40, 2, 1, "cylinder", 8., True),
        ("boxes_medium", 64, 4, 2, "box", 2.5, False),
        ("mesh_medium", 72, 4, 1, "mesh", 6., False),
        ("mixed_medium", 80, 4, 4, "mixed", 4., True),
        ("boxes_large", 128, 8, 4, "box", 8., True),
        ("spheres_large", 144, 8, 3, "sphere", 1.5, False),
        ("cylinders_large", 160, 8, 2, "cylinder", 6., True),
        ("meshes_large", 160, 8, 4, "mesh", 8., False),
    )
    for label, n, wavelengths, bodies, kind, eps, lossy in dielectric_cases:
        name = f"f1_{label}"
        builders.append(SceneBuilder(name, "F1", (n + 2 * PML) ** 3, partial(
            dielectric, name, n=n, wavelengths=wavelengths, bodies=bodies,
            kind=kind, eps=eps, lossy=lossy)))
    for index, n in enumerate((32, 40, 48, 64, 80, 96, 112, 128, 160, 180)):
        kind = ("box", "plate", "holes")[index % 3]
        name = f"f2_{kind}_{n}"
        builders.append(SceneBuilder(name, "F2", (n + 2 * PML) ** 3,
                                     partial(conductor, name, n=n, kind=kind)))
    printed_cases = (
        ("coarse_11f", .002, .004, .0002, .0014, 11, False),
        ("medium_21f", .001, .002, .0002, .0014, 21, False),
        ("lossy_21f", .001, .002, .0002, .0014, 21, True),
        ("board_020mm", .0005, .002, .0002, .0014, 21, False),
        ("board_010mm", .0005, .002, .0001, .0014, 21, False),
        ("board_005mm", .0005, .002, .00005, .0014, 41, True),
        ("thin_substrate", .0005, .001, .0001, .0007, 21, False),
        ("fine_11f", .00025, .001, .0001, .0014, 11, False),
        ("fine_41f", .00025, .001, .0001, .0014, 41, True),
        ("large_41f", .0002, .001, .00005, .0014, 41, True),
    )
    for label, lateral, air, finest, height, count, lossy in printed_cases:
        name = f"f3_{label}"
        cells = math.prod(len(axis) - 1 + 2 * PML
                          for axis in _printed_axes(lateral, air, finest, height))
        builders.append(SceneBuilder(name, "F3", cells, partial(
            printed, name, lateral=lateral, air=air, finest=finest,
            height=height, count=count, lossy=lossy)))
    for index, n in enumerate((32, 40, 48, 64, 80, 96, 112, 128, 160, 180)):
        kind = ("dipole", "loop", "helix")[index % 3]
        name = f"f4_{kind}_{n}"
        builders.append(SceneBuilder(name, "F4", (n + 2 * PML) ** 3, partial(
            wire, name, n=n, kind=kind, count=(11, 21, 41)[index % 3], finite=False)))
    for n, nz in ((36, 64), (48, 80), (80, 112), (192, 192)):
        name = f"f5_loaded_{n}"
        builders.append(SceneBuilder(name, "F5", n * n * (nz + 2 * PML), partial(
            periodic, name, n=n, nz=nz, loaded=True, frequency=2.4e9)))
    for n, nz in ((48, 80), (192, 192)):
        for frequency in (2.1e9, 2.4e9, 2.7e9):
            name = f"f5_bloch_{n}_{frequency / 1e9:.1f}ghz"
            builders.append(SceneBuilder(name, "F5", n * n * (nz + 2 * PML), partial(
                periodic, name, n=n, nz=nz, loaded=False, frequency=frequency)))
    return tuple(builders)
