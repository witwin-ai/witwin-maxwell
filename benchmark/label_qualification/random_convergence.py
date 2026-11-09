"""Seeded small-scene label convergence; no solver is prepared by --dry-run.

Each scene has 3 resolutions x 2 PML depths x 3 stopping rules. F5 is
deliberately deferred. The shared builders supply the physical workloads;
wrappers freeze geometry, feeds and queries across all numerical controls.
The scheduler owns spawning, environment setup, two slots/device and failures.
Cost-balanced shards write separate directories; --report --shard-count N
combines their measurements before convergence analysis.

JSONL holds per-run measurements/errors; adjacent scheduler .pt files retain
complex labels. Recommendations require usable data/QC for every scene, tier
error at the 90th percentile, and stopping agreement within 0.5% for every
scene. A finest-grid self-comparison never qualifies a resolution.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from benchmark.datagen import scheduler

FAMILIES = ("F1", "F2", "F3", "F4")
DEFERRED_FAMILIES = {"F5": "Awaiting periodic/Bloch qualification"}
LEVELS = (1.0, 1.5, 2.0)
STOPS = {"energy_1e-5": 1e-5, "energy_1e-7": 1e-7, "fixed": 0.0}
C0 = 299792458.0
F0 = 2.4e9
BAND = (0.8 * F0, 1.2 * F0)
CELL_UPDATES_PER_SECOND = 2.7e9
STEP_OVERHEAD = {"F3": .96e-3, "F4": .15e-3}
CLASSES = {"points_E": "field", "points_H": "field", "plane_E": "field",
           "plane_H": "field", "S11": "port", "Z_in": "port",
           "P_acc_over_P_inc": "power_ratio", "closed_surface_flux": "integral",
           "C_sca": "integral", "flux_over_P_inc": "power_ratio"}
# F4 wires are built once on the n=40 base grid with radius 0.12 * dx (benchmark/datagen
# wire builder) and keep that physical radius on every refinement level, so a/delta_perp
# grows as 0.12 * wire_radius_scale * level. Keep the whole ladder inside the thin-wire
# validity band a/delta_perp <= 0.2 (compiler/thin_wire.py).
F4_BASE_RADIUS_TO_SPACING = 0.12
THIN_WIRE_MAX_RADIUS_TO_SPACING = 0.2
F4_RADIUS_SCALE_RANGE = (0.5, min(1.5, THIN_WIRE_MAX_RADIUS_TO_SPACING
                                   / (F4_BASE_RADIUS_TO_SPACING * max(LEVELS)) * (1 - 1e-9)))


def f4_radius_scale(draw):
    """Map the unchanged U(0.5, 1.5) draw affinely onto the in-band range."""
    lo, hi = F4_RADIUS_SCALE_RANGE
    return lo + (draw - .5) * (hi - lo)


TIERS = {"integral_1pct": ("port", "integral", "power_ratio", .01),
         "integral_2pct": ("port", "integral", "power_ratio", .02),
         "pretraining": ("field", .05), "posttraining": ("field", .02)}


def draw_scenes(families, count, seed):
    """Stratify discrete builder cases; draw continuous parameters independently.

    Small-scene domain spans stop at 3.2 wavelengths. F1 uses at least 20 cells
    per vacuum wavelength and 10 per shortest material wavelength.
    F3 retains the coarse printed grid and its physical contour anchors.
    """
    scenes = []
    for family in families:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(family[1:])]))
        kinds = {"F1": ("box", "sphere", "cylinder", "mesh", "mixed"),
                 "F2": ("box", "plate", "holes"), "F3": ("printed",),
                 "F4": ("dipole", "loop", "helix")}[family]
        cases = list(itertools.product(kinds, (False, True)))
        order = rng.permutation(len(cases))
        for index in range(count):
            kind, loss = cases[int(order[index % len(cases)])]
            span = float(rng.uniform(2., 3.2)) if family == "F1" else float(rng.uniform(1.6, 3.2))
            n = 4 * math.ceil(20 * BAND[1] / F0 * span / 4)
            scenes.append(dict(
                family=family, scene_id=f"{family.lower()}_{index:03d}",
                seed=int(rng.integers(0, 2**31)), kind=kind, lossy=loss,
                wavelengths=span, n=n, bodies=4 if kind == "mixed" else int(rng.integers(1, 5)),
                eps=float(rng.uniform(1.5, 8.)), scale=float(rng.uniform(.7, 1.)),
                height=float(rng.uniform(.0007, .0014)), arm_length=float(rng.uniform(.025, .040)),
                wire_radius_scale=float(rng.uniform(.5, 1.5)),
            ))
            if family == "F4":
                # Same RNG draw (the stream is unchanged), mapped into the thin-wire band.
                scenes[-1]["wire_radius_scale"] = f4_radius_scale(scenes[-1]["wire_radius_scale"])
            if family == "F1":
                spec = scenes[-1]
                n_max = math.sqrt(spec["eps"])
                ppw = max(20., 10 * n_max)
                spec.update(n_max=n_max, ppw=ppw, n=math.ceil(ppw * span))
    return scenes


def f2_box_size(spec):
    return .5 * C0 / F0 * spec["scale"]


def axes_for(spec, level):
    from benchmark.datagen.scenes import _partition, _printed_axes

    if spec["family"] == "F3":
        axes = list(_printed_axes(.002 / level, .004 / level, .0002 / level, spec["height"]))
        end = spec["arm_length"]
        axes[0] = _partition((-.065, -.045, -end, -.001, 0., .001, end, .045, .065),
                             .004 / level,
                             fine=((-.045, .045, .002 / level), (-.002, .002, .0002 / level)))
        axes[1] = _partition((-.025, -.0055, -.001, 0., .001, .0055, .025), .004 / level,
                             fine=((-.0055, .0055, .002 / level), (-.001, .001, .0005 / level)),
                             half_planes=(-.0035, .0035))
        height = spec["height"]
        axes[2] = _partition((-.025, -height, 0., .025), .004 / level,
                             fine=((-height, 0., min(height / 7, .0002) / level),),
                             half_planes=(-.0025, .0025))
        # Keep the base builder's current contour at the SAME native half node
        # in physical space, instead of moving the port when refining the grid.
        base_x = _printed_axes(.002, .004, .0002, spec["height"])[0]
        contour = base_x[np.searchsorted(base_x, 0.) + 1] / 2
        if level != 1:
            lo, hi = contour - contour / level, contour + contour / level
            axes[0] = np.sort(np.concatenate((axes[0][(axes[0] < lo) | (axes[0] > hi)], (lo, hi))))
        return tuple(axes)
    if spec["family"] == "F4":
        n, span = 40, 1.6 * C0 / F0
    elif spec["family"] == "F2":
        # Match the builder's physical domain, independent of the final grid.
        span = math.ceil(20 * spec["wavelengths"]) * C0 / F0 / 20
        n = 4 * math.ceil(20 * BAND[1] * span / C0 / 4)
        if spec["kind"] == "box":
            # Pin faces to nodes: the snapped staircase size changed sign along the ladder.
            # An even base count keeps the exact 1 : 1.5 : 2 ratios Richardson assumes.
            h = f2_box_size(spec) / 2
            base = 2 * max(1, round(h / (span / n) / 2))
            m = round(base * level)
            dx = h / m
            N = round((span / 2) / dx)
            axis = np.arange(-N, N + 1) * dx
            axis[N - m], axis[N + m], axis[N] = -h, h, 0.0
            return (axis,) * 3
    else:
        n, span = spec["n"], spec["wavelengths"] * C0 / F0
    return (np.linspace(-span / 2, span / 2, int(n * level) + 1),) * 3


def estimate(spec, level, pml, cycles):
    axes = axes_for(spec, level)
    cells = math.prod(len(axis) - 1 + 2 * pml for axis in axes)
    # Vacuum CFL estimate; preparation records the actual material/wire dt.
    dt = .9 / (C0 * math.sqrt(sum(float(np.diff(a).min()) ** -2 for a in axes)))
    dt = min(dt, 1 / (30 * (F0 + 3 * .8 * F0)))
    fmin = 1.8e9 if spec["family"] == "F3" else BAND[0]
    center = 2.4e9
    settling = 12 / (2 * math.pi * .8 * center)
    propagation = 10 * max(a[-1] - a[0] for a in axes) / C0
    horizon = max(8 / fmin, settling + propagation) + cycles / fmin
    steps = math.ceil(horizon / dt)
    seconds = steps * (cells / CELL_UPDATES_PER_SECOND + STEP_OVERHEAD.get(spec["family"], 0.))
    return dict(estimated_cells=cells, estimated_steps=steps, estimated_seconds=seconds,
                estimated_dt=dt, minimum_spacing=[float(np.diff(a).min()) for a in axes])


def task_list(args):
    families = FAMILIES if args.family == "all" else (args.family,)
    if args.family in DEFERRED_FAMILIES:
        raise NotImplementedError(DEFERRED_FAMILIES[args.family])
    tasks, plans = [], {}
    for spec in draw_scenes(families, args.scenes, args.seed):
        if spec["family"] == "F1":
            spec["f1_averaging"] = args.f1_averaging
        for level, pml, stop in itertools.product(LEVELS, (8, 16), STOPS):
            task_id = f"{spec['scene_id']}_r{level:g}_p{pml}_{stop}"
            plan = dict(spec=spec, level=level, pml=pml, stopping=stop,
                        long_cycles=args.long_cycles, points=args.points,
                        plane_side=args.plane_side, mesh_fill_tolerance=args.mesh_fill_tolerance,
                        **estimate(spec, level, pml, args.long_cycles))
            if spec["family"] == "F1":
                from benchmark.datagen.scenes import DIELECTRIC_SOURCE_PHASE

                plan["f1_source"] = dict(kind="gaussian_pulse", frequency=F0,
                                         fwidth=0.8 * F0, phase=DIELECTRIC_SOURCE_PHASE)
            tasks.append(scheduler.Task(task_id, json.dumps(plan), spec["seed"]))
            plans[task_id] = plan
    return tasks, plans


def shard_tasks(tasks, plans, count):
    """Longest estimated tasks first; shard index breaks equal-load ties."""
    shards, loads = [[] for _ in range(count)], [0.] * count
    for task in sorted(tasks, key=lambda t: (-plans[t.task_id]["estimated_seconds"], t.task_id)):
        index = min(range(count), key=loads.__getitem__)
        shards[index].append(task)
        loads[index] += plans[task.task_id]["estimated_seconds"]
    return shards


def query_positions(plan):
    spec = plan["spec"]
    rng = np.random.default_rng(spec["seed"])
    if spec["family"] == "F3":
        extents, z, depth = (.030, .004), .012, .004
    else:
        wavelength = C0 / F0
        extents = ((.8, .32) if spec["family"] == "F1" else (.32, .32))
        extents = tuple(x * wavelength for x in extents)
        z, depth = .4 * wavelength, .02 * wavelength
    points = rng.uniform(-1., 1., (plan["points"], 3))
    points *= (*extents, depth)
    points[:, 2] += z
    x, y = np.meshgrid(*(np.linspace(-v, v, plan["plane_side"]) for v in extents), indexing="ij")
    plane = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
    return points, plane


def interpolation_stencil(axes, points):
    """Indices and multilinear weights on exact component-specific Yee axes."""
    lower = np.column_stack([np.searchsorted(a, points[:, d], side="right") - 1
                             for d, a in enumerate(axes)])
    fraction = np.column_stack([(points[:, d] - a[lower[:, d]]) /
                                (a[lower[:, d] + 1] - a[lower[:, d]]) for d, a in enumerate(axes)])
    offsets = np.asarray(tuple(itertools.product((0, 1), repeat=len(axes))))
    weights = np.where(offsets[None, :, :], fraction[:, None, :], 1 - fraction[:, None, :]).prod(axis=-1)
    return lower[:, None, :] + offsets[None, :, :], weights


def add_field_monitors(scene, plan, frequencies):
    """Retain exact Yee stencils and native planes for common physical queries."""
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.coords import component_coords
    from witwin.maxwell.scene import prepare_scene

    points, plane = query_positions(plan)
    grid = prepare_scene(scene)
    sampling = {}
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        axes = component_coords(grid, component)
        indices, weights = interpolation_stencil(axes, points)
        unique, gather = np.unique(indices.reshape(-1, 3), axis=0, return_inverse=True)
        names = []
        for index, loc in enumerate(unique):
            name = f"sparse_{component}_{index}"
            names.append(name)
            scene.add_monitor(mw.PointMonitor(
                name, tuple(axes[d][loc[d]] for d in range(3)),
                fields=(component,),
            ))
        high = np.searchsorted(axes[2], plane[0, 2])
        plane_names = []
        for index in (high - 1, high):
            name = f"plane_{component}_{index}"
            plane_names.append(name)
            scene.add_monitor(mw.PlaneMonitor(
                name, axis="z", position=axes[2][index],
                fields=(component,), frequencies=frequencies,
            ))
        plane_indices, plane_weights = interpolation_stencil(axes[:2], plane[:, :2])
        normal_weight = (plane[0, 2] - axes[2][high - 1]) / (axes[2][high] - axes[2][high - 1])
        sampling[component] = dict(point_names=names, point_gather=gather.reshape(len(points), 8),
                                   point_weights=weights, plane_names=plane_names,
                                   plane_gather=plane_indices[:, :, 0] * len(axes[1]) + plane_indices[:, :, 1],
                                   plane_weights=plane_weights, normal_weight=normal_weight)
    return sampling


def recorded_fields(result, sampling):
    """Interpolate actual native DFT samples, including early-shutoff runs.

    The physical-query API requires a complete horizon. Here all controls use
    the same explicit measurement interpolation and the runtime's restored
    planned-horizon DFT normalizers; actual sample counts remain untouched.
    """
    import torch

    points, planes = {}, {}
    for component, s in sampling.items():
        values = torch.stack([result.monitor(name)["components"][component].reshape(-1)
                              for name in s["point_names"]], dim=-1)
        gather = torch.as_tensor(s["point_gather"], device=values.device)
        weights = torch.as_tensor(s["point_weights"], device=values.device)
        points[component] = (values[:, gather] * weights).sum(dim=-1)
        gather = torch.as_tensor(s["plane_gather"], device=values.device)
        weights = torch.as_tensor(s["plane_weights"], device=values.device)
        slices = [result.monitor(name)["components"][component]["data"].reshape(len(result.frequencies), -1)
                  for name in s["plane_names"]]
        lower, upper = [(v[:, gather] * weights).sum(dim=-1) for v in slices]
        planes[component] = lower * (1 - s["normal_weight"]) + upper * s["normal_weight"]
    return {prefix + "_" + field: torch.stack([values[field + axis] for axis in "xyz"], dim=-1)
            for prefix, values in (("points", points), ("plane", planes)) for field in ("E", "H")}


def build_sample(plan, *, device):
    import torch
    import witwin.maxwell as mw
    from benchmark.datagen import scenes as builders

    s = plan["spec"]
    family, name = s["family"], s["scene_id"]
    frequencies = tuple(np.linspace(1.8e9, 3.e9, 11)) if family == "F3" else tuple(np.linspace(*BAND, 11))
    if family == "F1":
        sample = builders.dielectric(
            name, n=s["n"], wavelengths=s["wavelengths"], bodies=s["bodies"],
            kind=s["kind"], eps=s["eps"], lossy=s["lossy"], device=device,
        )
        sample.scene = sample.scene.clone(subpixel_samples=mw.SubpixelSpec(
            samples=3 if s["f1_averaging"] == "polarized" else 4,
            averaging=s["f1_averaging"],
        ))
        # The builder drives the zero-DC pulse the plan records.
        (source,) = sample.scene.sources
        assert source.source_time == mw.GaussianPulse(frequency=F0, fwidth=.8 * F0,
                                                      phase=builders.DIELECTRIC_SOURCE_PHASE)
    elif family == "F2":
        sample = builders.conductor(name, n=math.ceil(20 * s["wavelengths"]),
                                    kind=s["kind"], device=device)
        structure = sample.scene.structures[0]
        if s["kind"] == "box":
            geometry = builders._box((0., 0., 0.), (f2_box_size(s),) * 3)
        else:
            g = structure.geometry
            geometry = mw.PlanarPEC(g.vertices * s["scale"],
                                    holes=tuple(h * s["scale"] for h in g.holes), device=device)
        sample.scene.structures[0] = replace(structure, geometry=geometry)
    elif family == "F3":
        sample = builders.printed(name, lateral=.002, air=.004, finest=.0002,
                                  height=s["height"], count=11, lossy=s["lossy"], device=device)
        for index in (1, 2):
            structure = sample.scene.structures[index]
            sign = -1 if index == 1 else 1
            end = s["arm_length"]
            geometry = builders._box((sign * (end + .001) / 2, 0., 0.), (end - .001, .002, 0.))
            sample.scene.structures[index] = replace(
                structure, geometry=geometry,
                material=mw.CopperFoil(thickness=35e-6, frequency_range=(frequencies[0], frequencies[-1])),
            )
    else:
        # Construct ONCE at the base wire grid: changing n in this builder would
        # change the physical feed gap, wire radius and centerline segmentation.
        sample = builders.wire(name, n=40, kind=s["kind"], count=11,
                               finite=s["lossy"], device=device)
        sample.scene = sample.scene.clone(thin_wires=[mw.ThinWire(
            w.name, torch.as_tensor(w.points, dtype=torch.float64, device=device) * s["scale"],
            w.radius * s["wire_radius_scale"], w.conductor, endpoints=w.endpoints or None,
            snap="continuous",
        ) for w in sample.scene.thin_wires])
    # Keep feeds, contours, geometry and source pulse fixed across all controls.
    axes = axes_for(s, plan["level"])
    extra = {}
    if family == "F2" and s["kind"] == "box":
        # The face-anchored box grid rounds the domain to whole cells (< dx/2 per side).
        extra["domain"] = mw.Domain(tuple((float(a[0]), float(a[-1])) for a in axes))
    scene = sample.scene.clone(grid=mw.GridSpec.custom(*axes),
                               boundary=mw.BoundarySpec.pml(num_layers=plan["pml"]), **extra)
    scene.monitors.clear()
    surface_size = {"F1": (1.9 * C0 / F0,) * 3, "F2": (1.4 * C0 / F0,) * 3,
                    "F3": (.110, .035, .035), "F4": (1.2 * C0 / F0,) * 3}[family]
    builders._surface(scene, frequencies, surface_size)
    sampling = add_field_monitors(scene, plan, frequencies)
    sample.scene = scene
    sample.fdtd_kwargs.update(
        frequencies=frequencies,
        run_time=mw.TimeConfig.auto(steady_cycles=plan["long_cycles"], transient_cycles=8),
        shutoff=STOPS[plan["stopping"]],
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        # Explicit, so raw pulse spectra do not depend on the default reference period.
        reference_frequency=F0,
    )
    sample.readouts.clear()
    return sample, sampling


def source_spectrum(solver, frequencies, stats):
    """Source DFT with the same pulse normalization as the raw observers: dt * f_ref * sum,
    using the solver's spectral integration step and the explicit reference frequency F0.
    Dividing observer spectra by it gives per-unit-source transfer functions that do not
    depend on the reference choice."""
    import torch
    from witwin.maxwell.waveforms import evaluate_time_shifted_waveform_torch

    t = torch.arange(stats["steps_run"], dtype=torch.float64, device=solver.device) * stats["dt"]
    source = solver._normalization_source_time
    waveform = evaluate_time_shifted_waveform_torch(source, t)
    omega = torch.tensor(frequencies, device=solver.device, dtype=torch.float64) * (2 * math.pi)
    scale = solver._spectral_integration_dt * solver._spectral_reference_frequency
    spectrum = (torch.exp(1j * omega[:, None] * t) * waveform).sum(dim=1) * scale
    return spectrum


def mesh_fill_qc(sample, solver, tolerance):
    """Compare compiled nodal epsilon occupancy with exact oriented mesh volume."""
    import torch
    import witwin.maxwell as mw

    checks = []
    grid = solver.scene
    for structure in sample.scene.structures:
        if not isinstance(structure.geometry, mw.Mesh):
            continue
        vertices, faces = structure.geometry.to_mesh(device=solver.device)
        triangles = vertices[faces.long()].double()
        exact = float((triangles[:, 0] * torch.linalg.cross(triangles[:, 1], triangles[:, 2])).sum().abs() / 6)
        lower, upper = vertices.amin(dim=0), vertices.amax(dim=0)
        selections, weights = [], []
        for dim, axis in enumerate("xyz"):
            nodes = torch.as_tensor(getattr(grid, axis + "_nodes64"), device=vertices.device)
            intervals = nodes[1:] - nodes[:-1]
            widths = torch.cat((intervals[:1] / 2, (intervals[1:] + intervals[:-1]) / 2, intervals[-1:] / 2))
            selected = (nodes >= lower[dim] - intervals.max()) & (nodes <= upper[dim] + intervals.max())
            selections.append(torch.where(selected)[0])
            weights.append(widths[selected])
        epsilon = solver.epsilon_r[tuple(torch.meshgrid(*selections, indexing="ij"))]
        occupancy = (epsilon - 1) / (float(structure.material.eps_r) - 1)
        volume = float((occupancy * weights[0][:, None, None] * weights[1][None, :, None]
                        * weights[2][None, None, :]).sum())
        error = abs(volume - exact) / exact
        checks.append(dict(structure=structure.name, exact_volume=exact, compiled_volume=volume,
                           relative_error=error, passed=error <= tolerance))
    return checks


def execute_task(task, output_dir, commit, builders, device_name):
    """Scheduler callback; its enclosing worker captures per-task exceptions."""
    import torch
    import witwin.maxwell as mw
    from witwin.maxwell.fdtd.runtime.stepping import _electric_field_energy

    plan = json.loads(task.builder)
    stages = {}

    def measure(name, operation):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        stages[name] = time.perf_counter() - start
        return value

    sample, sampling = measure("build", lambda: build_sample(plan, device="cuda:0"))
    prepared = measure("prepare", lambda: mw.Simulation.fdtd(sample.scene, **sample.fdtd_kwargs).prepare())
    result = measure("run", prepared.run)
    stats, solver = result.solver_stats, prepared.solver

    def readouts():
        spectrum = source_spectrum(solver, result.frequencies, stats)
        labels = {name: values / spectrum[:, None, None]
                  for name, values in recorded_fields(result, sampling).items()}
        flux = torch.as_tensor(result.monitor("radiation")["flux"], device=solver.device).reshape(-1)
        labels["closed_surface_flux"] = flux / spectrum.abs().square()
        if plan["spec"]["family"] == "F2":
            labels["C_sca"] = labels["closed_surface_flux"] * (2 * 376.730313668)
        valid = spectrum.abs() >= .1 * spectrum.abs().max()
        s11_max = None
        if sample.scene.ports:
            port = result.port("feed")
            ratio = port.accepted_power / port.incident_power
            labels.update(S11=port.reflection_coefficient, Z_in=port.z_in,
                          P_acc_over_P_inc=ratio, flux_over_P_inc=flux / port.incident_power)
            valid &= ratio >= 1e-3
            s11_max = float(port.reflection_coefficient.abs().max())
        return dict(labels={k: v.detach().cpu() for k, v in labels.items()},
                    valid=valid.cpu(), source_spectrum=spectrum.cpu(), s11_max=s11_max)

    data = measure("readouts", readouts)
    fill = measure("mesh_fill_qc", lambda: mesh_fill_qc(sample, solver, plan["mesh_fill_tolerance"]))
    energy = float(_electric_field_energy(solver))
    peak = float(solver._shutoff_peak)
    finite = (bool(data["valid"].any()) and math.isfinite(energy)
              and all(bool(torch.isfinite(v[data["valid"]]).all()) for v in data["labels"].values()))
    s11 = data.pop("s11_max")
    qc = dict(finite=finite, mesh_fill=fill,
              s11_max=s11 if s11 is not None and math.isfinite(s11) else None,
              s11_passed=s11 is None or (math.isfinite(s11) and s11 <= 1.001),
              final_energy=energy if math.isfinite(energy) else None,
              peak_energy=peak if math.isfinite(peak) else None,
              shutoff_triggered=stats["shutoff_triggered"])
    cells = math.prod(getattr(solver.scene, "N" + axis) - 1 for axis in "xyz")
    path = Path(output_dir) / f"{task.task_id}.pt"
    measure("save", lambda: torch.save(data, path))
    return dict(status="ok", path=str(path), stage_seconds=stages, qc=qc,
                cells=cells, steps=stats["steps_run"], planned_steps=stats["time_steps"],
                dt=stats["dt"], time_plan=stats["time_plan"], git_commit=commit,
                normalize_source=False)


def qualification_worker(*args):
    # run_tasks spawns this importable adapter. In the fresh child, _worker is
    # the original scheduler worker; its environment is set before witwin loads.
    scheduler.execute_task = execute_task
    scheduler._worker(*args)


def run(tasks, output_dir, devices):
    original_worker = scheduler._worker
    scheduler._worker = qualification_worker
    manifest = scheduler.run_tasks(tasks, output_dir, devices, slots_per_device=2)
    scheduler._worker = original_worker
    return manifest


def relative_error(value, reference, mask):
    """Worst retained-frequency relative complex L2, with no fitted phase/scale."""
    if not mask.any():
        return None
    value, reference = value[mask].reshape(mask.sum(), -1), reference[mask].reshape(mask.sum(), -1)
    denominator = np.linalg.norm(reference, axis=1)
    if np.any(denominator == 0) or not np.isfinite(value).all() or not np.isfinite(reference).all():
        return None
    return float(np.max(np.linalg.norm(value - reference, axis=1) / denominator))


def reference_for(values, mask):
    """Unequal-ratio Richardson; reject oscillatory/non-asymptotic differences.

    Sane means order in [0.5,4], <5% norm-ratio residual and >0.9 real
    alignment of successive complex differences. Without that evidence the
    finest result is only a comparison reference, with unresolved finest error.
    """
    coarse, middle, fine = values
    d01, d12 = coarse[mask] - middle[mask], middle[mask] - fine[mask]
    a, b = np.linalg.norm(d01), np.linalg.norm(d12)
    if not mask.any() or not np.isfinite(a + b) or min(a, b) == 0:
        return fine, None
    alignment = float(np.vdot(d01, d12).real / (a * b))
    orders = np.linspace(.5, 4., 701)
    ratios = (1 - (1 / 1.5)**orders) / ((1 / 1.5)**orders - .5**orders)
    index = int(np.argmin(abs(ratios - a / b)))
    order = float(orders[index])
    if alignment < .9 or abs(ratios[index] / (a / b) - 1) > .05:
        return fine, None
    return fine + (fine - middle) / ((2 / 1.5)**order - 1), order


def enrich(records):
    import torch

    groups = {}
    data = {}
    for record in records:
        groups.setdefault(record["plan"]["spec"]["scene_id"], []).append(record)
        if record["status"] == "ok":
            payload = torch.load(record["path"], map_location="cpu", weights_only=True)
            data[record["task_id"]] = dict(labels={k: v.numpy() for k, v in payload["labels"].items()},
                                           valid=payload["valid"].numpy())
    for members in groups.values():
        by_setting = {(r["plan"]["level"], r["plan"]["pml"], r["plan"]["stopping"]): r for r in members}
        complete = (set(by_setting) == set(itertools.product(LEVELS, (8, 16), STOPS))
                    and all(r["status"] == "ok" for r in members))
        if not complete:
            for r in members:
                r["scene_complete"] = False
            continue
        # One common mask per scene, including every control's accepted power.
        mask = np.logical_and.reduce([data[r["task_id"]]["valid"] for r in members])
        references, orders = {}, {}
        for label in data[members[0]["task_id"]]["labels"]:
            values = [data[by_setting[level, 16, "fixed"]["task_id"]]["labels"][label] for level in LEVELS]
            references[label], orders[label] = reference_for(values, mask)
        for r in members:
            p = r["plan"]
            strict = by_setting[p["level"], p["pml"], "energy_1e-7"]
            peak = strict["qc"]["peak_energy"] if p["stopping"] == "fixed" else r["qc"]["peak_energy"]
            threshold = 1e-7 if p["stopping"] == "fixed" else STOPS[p["stopping"]]
            q = r["qc"]
            q["energy_ratio"] = (q["final_energy"] / peak
                                  if peak is not None and peak > 0 and q["final_energy"] is not None else None)
            q["boundedness_passed"] = (q["energy_ratio"] is not None and q["energy_ratio"] <= threshold
                                        and (p["stopping"] == "fixed" or q["shutoff_triggered"]))
            q["passed"] = (q["finite"] and q["s11_passed"] and q["boundedness_passed"]
                            and all(check["passed"] for check in q["mesh_fill"]))
            r.update(scene_complete=True, valid_frequencies=int(mask.sum()), observed_order=orders,
                     errors={}, pml_sensitivity={}, stopping_vs_fixed={})
            values = data[r["task_id"]]["labels"]
            base_pml = data[by_setting[p["level"], 8, p["stopping"]]["task_id"]]["labels"]
            doubled_pml = data[by_setting[p["level"], 16, p["stopping"]]["task_id"]]["labels"]
            fixed = data[by_setting[p["level"], p["pml"], "fixed"]["task_id"]]["labels"]
            for label, value in values.items():
                # Without sane Richardson, finest vs itself has unknown error.
                r["errors"][label] = (None if p["level"] == 2 and orders[label] is None else
                                        relative_error(value, references[label], mask))
                r["pml_sensitivity"][label] = relative_error(base_pml[label], doubled_pml[label], mask)
                r["stopping_vs_fixed"][label] = relative_error(value, fixed[label], mask)
        passed = all(r["qc"]["passed"] for r in members
                     if r["plan"]["stopping"] == "fixed" and r["plan"]["pml"] == 16)
        for r in members:
            r["reference_qc_passed"] = passed
    return records


def summary(values):
    """Missing/nonfinite estimates remain explicit failures in cohort statistics."""
    usable = [v for v in values if v is not None and math.isfinite(v)]
    return dict(median=float(np.median(usable)) if usable else None,
                p90=float(np.percentile(usable, 90)) if len(usable) == len(values) and usable else None,
                maximum=max(usable) if len(usable) == len(values) and usable else None,
                available=len(usable), total=len(values))


def class_errors(records, key, classes):
    errors = []
    for r in records:
        if r["status"] != "ok" or not r["scene_complete"]:
            errors.append(None)
            continue
        values = [v for label, v in r[key].items() if CLASSES[label] in classes]
        errors.append(None if not values or any(v is None for v in values) else max(values))
    return summary(errors)


def recommend(cohort, classes, tier_classes, target):
    choices = []
    for level, pml, stop in itertools.product(LEVELS, (8, 16), STOPS):
        selected = [r for r in cohort if (r["plan"]["level"], r["plan"]["pml"], r["plan"]["stopping"]) == (level, pml, stop)]
        if not all(r["status"] == "ok" and r["scene_complete"] and r["reference_qc_passed"] and r["qc"]["passed"] for r in selected):
            continue
        error = class_errors(selected, "errors", tier_classes)["p90"]
        pml_error = class_errors(selected, "pml_sensitivity", tier_classes)["p90"]
        # Unlike the tier's cohort p90, the 0.5% stopping check must pass for
        # EVERY scene's normalized spectra and raw power ratios.
        stop_error = class_errors(selected, "stopping_vs_fixed", classes)["maximum"]
        if error is None or pml_error is None or stop_error is None:
            continue
        if error <= target and (pml == 16 or pml_error <= target) and stop_error <= .005:
            costs = [r["seconds"] for r in selected]
            choices.append(dict(resolution=level, pml=pml, stopping=stop,
                                error_p90=error, sample_seconds=summary(costs),
                                cells=summary([r["cells"] for r in selected]),
                                steps=summary([r["steps"] for r in selected])))
    return min(choices, key=lambda c: (c["resolution"], c["sample_seconds"]["median"])) if choices else None


def show_report(records):
    print("Errors: worst frequency relative complex/spatial L2, then worst label per class.")
    print("Median/p90 over scenes; unresolved finest estimates cannot qualify. F5 deferred.")
    print("PML=16 is the refined absorber reference; PML=8 must pass the doubling check.")
    for family in FAMILIES:
        cohort = [r for r in records if r["plan"]["spec"]["family"] == family]
        if not cohort:
            continue
        print(f"\n{family}: {len(cohort)} runs; {sum(r['status'] != 'ok' for r in cohort)} execution failures")
        classes = ("field", "integral") if family in ("F1", "F2") else ("field", "port", "integral", "power_ratio")
        for label_class in classes:
            for level in LEVELS:
                selected = [r for r in cohort if r["plan"]["level"] == level and r["plan"]["pml"] == 16 and r["plan"]["stopping"] == "fixed"]
                print(json.dumps(dict(label_class=label_class, resolution=level,
                                      error=class_errors(selected, "errors", (label_class,)))))
        # Print sensitivities at every level and PML/stopping combination.
        for level, pml, stop in itertools.product(LEVELS, (8, 16), STOPS):
            selected = [r for r in cohort if (r["plan"]["level"], r["plan"]["pml"], r["plan"]["stopping"]) == (level, pml, stop)]
            agreement = class_errors(selected, "stopping_vs_fixed", classes)
            print(json.dumps(dict(resolution=level, pml=pml, stopping=stop,
                                  pml_sensitivity=class_errors(selected, "pml_sensitivity", classes),
                                  shutoff_vs_fixed=agreement,
                                  agreement_0p5pct=agreement["maximum"] is not None and agreement["maximum"] <= .005,
                                  qc_failures=sum(r["status"] != "ok" or not r["scene_complete"] or not r["qc"]["passed"] for r in selected))))
        for tier, definition in TIERS.items():
            tier_classes, target = definition[:-1], definition[-1]
            present = tuple(c for c in tier_classes if c in classes)
            for label_class in present:
                print(json.dumps(dict(tier=tier, label_class=label_class, target=target,
                                      recommendation=recommend(cohort, classes, (label_class,), target))))
            print(json.dumps(dict(tier=tier, label_class="all_in_tier", target=target,
                                  recommendation=recommend(cohort, classes, present, target))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=(*FAMILIES, "F5", "all"), default="all")
    parser.add_argument("--scenes", type=int, default=20, help="Random scenes per family")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--long-cycles", type=int, default=120, help="Fixed horizon and common shutoff cap")
    parser.add_argument("--points", type=int, default=32, help="Common sparse physical query count")
    parser.add_argument("--plane-side", type=int, default=9, help="Common square-plane query side count")
    parser.add_argument("--mesh-fill-tolerance", type=float, default=.1)
    parser.add_argument("--f1-averaging", choices=("kottke_edge", "polarized"), default="kottke_edge")
    parser.add_argument("--shard-index", type=int, help="Zero-based shard to run or inspect")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/label_qualification/results/random-convergence"))
    parser.add_argument("--dry-run", action="store_true", help="List tasks, CFL-estimated cells/steps and GPU hours; CPU only")
    parser.add_argument("--report", action="store_true", help="Report the saved runs.jsonl without launching workers")
    args = parser.parse_args()
    if args.shard_count < 1 or (args.shard_index is not None and not 0 <= args.shard_index < args.shard_count):
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    if args.shard_count > 1 and args.shard_index is None and not (args.dry_run or args.report):
        parser.error("Select --shard-index to run a shard")
    output_dir = args.output_dir
    if args.shard_index is not None:
        output_dir /= f"shard-{args.shard_index}"
    if args.report:
        directories = ([args.output_dir / f"shard-{i}" for i in range(args.shard_count)]
                       if args.shard_count > 1 and args.shard_index is None else [output_dir])
        records = [json.loads(line) for directory in directories
                   for line in (directory / "runs.jsonl").read_text().splitlines()]
        show_report(enrich(records))
        return 0
    tasks, plans = task_list(args)
    shards = shard_tasks(tasks, plans, args.shard_count)
    shard_indices = range(args.shard_count) if args.shard_index is None else (args.shard_index,)
    if args.shard_index is not None:
        tasks = shards[args.shard_index]
        plans = {t.task_id: plans[t.task_id] for t in tasks}
    if args.dry_run:
        for task in tasks:
            plan = plans[task.task_id]
            print(json.dumps(dict(task_id=task.task_id, **plan)))
        for family in FAMILIES:
            members = [p for p in plans.values() if p["spec"]["family"] == family]
            print(json.dumps(dict(family=family, tasks=len(members),
                                  maximum_base_cells=max((p["estimated_cells"] for p in members if p["level"] == 1), default=None),
                                  estimated_gpu_hours=sum(p["estimated_seconds"] for p in members) / 3600)))
        for index in shard_indices:
            print(json.dumps(dict(shard_index=index, tasks=len(shards[index]),
                                  estimated_gpu_hours=sum(plans[t.task_id]["estimated_seconds"] for t in shards[index]) / 3600)))
        print(json.dumps(dict(tasks=len(tasks), estimated_gpu_hours=sum(p["estimated_seconds"] for p in plans.values()) / 3600,
                              caveat="CFL/full-horizon estimate; excludes setup/readouts; material and thin-wire limits may increase steps", deferred=DEFERRED_FAMILIES)))
        return 0
    manifest = run(tasks, output_dir, args.devices)
    records = [dict(task_id=t.task_id, plan=plans[t.task_id], **manifest["tasks"][t.task_id]) for t in tasks]
    records = enrich(records)
    (output_dir / "runs.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    return int(any(r["status"] != "ok" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
