"""FDTD subpixel-averaging convergence against analytic references.

Each invocation runs one case at one resolution for one or more averaging
methods and appends one JSON record per FDTD run to ``--output`` (JSON lines).
``report`` aggregates the records into tables, fitted orders and plots.

Cases (resolution = cells per wavelength in the densest material unless noted):

* ``planar``   two-layer parallel-plate cavity, TM0 between PEC plates (E normal
               to the interface) and TE0 between PMC plates (E tangential); the
               interface lies between grid planes at several sub-cell phases.
               Resolution = cells across the plate gap. Reference: exact
               dispersion relation with the Yee x/t substitution applied.
* ``pair``     printed coupled-pair cross-section (5 mm substrate, eps 2.25)
               as a PEC cavity in x on the line's own transverse grids;
               even-mode beta against an external cross-section reference.
               Resolution = transverse dx in mm.
* ``sphere``   dielectric sphere, TFSF + Huygens box: C_sca (box flux and
               far-field quadrature) and bistatic RCS cuts vs the Mie series.
* ``cylinder`` z-invariant dielectric cylinder, TE and TM, vs the 2D series.
* ``pec``      PEC sphere, staircase vs conformal, vs the PEC Mie series.
* ``mie-check`` CPU-only validation of the analytic series.

Examples::

    python -m scripts.dev.fdtd.subpixel_convergence sphere --resolution 20 \
        --methods staircase arithmetic polarized --output runs.jsonl
    python -m scripts.dev.fdtd.subpixel_convergence report --input runs.jsonl \
        --plots docs/assessments/subpixel-convergence-2026-10-29
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0, ETA_0
from witwin.maxwell.postprocess import find_resonances

from scripts.dev.fdtd import subpixel_analytic as A

METHODS = ("staircase", "arithmetic", "polarized")
PEC_METHODS = ("staircase", "conformal")
SPHERE_OFFSETS = ((0.0, 0.0, 0.0), (0.31, 0.17, 0.43), (0.62, 0.45, 0.11))


def subpixel_spec(method, samples):
    if method == "staircase":
        return mw.SubpixelSpec(samples=1, averaging="arithmetic")
    if method == "arithmetic":
        return mw.SubpixelSpec(samples=samples, averaging="arithmetic")
    if method == "polarized":
        return mw.SubpixelSpec(samples=samples, averaging="polarized")
    if method == "conformal":
        return mw.SubpixelSpec(samples=1, averaging="arithmetic", pec="conformal")
    raise ValueError(method)


def _cells(result):
    scene = result.solver.scene
    return int((len(scene.x_nodes64) - 1) * (len(scene.y_nodes64) - 1) * (len(scene.z_nodes64) - 1))


def _timed_run(simulation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = simulation.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = dict(
        wall_s=elapsed,
        peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
        time_steps=int((getattr(result, "_solver_stats", None) or {}).get("steps_run") or 0),
        step_loop_s=float(getattr(result.solver, "last_step_loop_elapsed_s", float("nan")) or float("nan")),
        dt_s=float(result.solver.dt),
        cells=_cells(result),
    )
    return result, stats


def _append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("pattern",)}), flush=True)


def _envelope(payload, component, start_time, segments=8):
    """Max |field| per time segment after ``start_time`` (growth/instability check)."""
    t = torch.as_tensor(payload["t"]).detach().cpu().numpy()
    trace = torch.as_tensor(payload["components"][component]).detach().abs().cpu().numpy().reshape(len(t), -1).max(axis=1)
    trace = trace[t >= start_time]
    return [float(chunk.max()) for chunk in np.array_split(trace, segments) if chunk.size]


def _pick_mode(modes, target):
    """Strongest harmonic-inversion mode; returns (frequency, decay, error estimate)."""
    freqs = modes.frequency.detach().cpu().numpy()
    if freqs.size == 0:
        raise RuntimeError("Harmonic inversion found no mode in the window.")
    index = int(np.argmax(modes.magnitude.detach().cpu().numpy()))
    return (float(freqs[index]), float(modes.decay_rate.detach().cpu().numpy()[index]),
            float(modes.error.detach().cpu().numpy()[index]))


# --------------------------------------------------------------------------
# planar: two-layer parallel-plate cavity
# --------------------------------------------------------------------------
PLANAR_H = 0.010
PLANAR_D_FRACTION = 0.4137
PLANAR_L = 0.060
PLANAR_NX = 12
# Golden-ratio sub-cell phases of the interface (no alignment with any sample lattice).
PLANAR_PHASES = tuple(float((0.1 + 0.6180339887498949 * j) % 1.0) for j in range(8))
PLANAR_MODES = {
    # label: (z boundary, source/probe component, analytic solver)
    "TM0": ("pec", "Ez", A.parallel_plate_tm0_omega),
    "TE0": ("pmc", "Ey", A.parallel_plate_te0_omega),
}


def _samples_for(args):
    if args.samples_z is not None:
        return (1, 1, int(args.samples_z))
    return int(args.samples)


def run_planar(args):
    """Quasi-TEM modes of a two-layer parallel-plate cavity.

    The interface sits at d = (round(0.4137 N) + phase) dz for several sub-cell
    phases so the reported error is not tied to one interface position. TM0
    (PEC plates) puts E normal to the interface, TE0 (PMC plates) tangential.
    Both are quasi-static, so the Yee z dispersion is far below the interface
    error; x and t discretization are removed exactly by the semi-discrete map.
    """
    eps = float(args.eps)
    n_gap = int(args.resolution)
    h = PLANAR_H
    dz = h / n_gap
    dl = PLANAR_L / PLANAR_NX
    x_nodes = np.linspace(0.0, PLANAR_L, PLANAR_NX + 1)
    y_nodes = np.array([0.0, dl, 2 * dl])
    z_nodes = np.linspace(0.0, h, n_gap + 1)
    kx = math.pi / PLANAR_L
    kx_tilde = 2.0 / dl * math.sin(0.5 * kx * dl)
    for phase_index, phase in enumerate(PLANAR_PHASES[: int(args.offsets)]):
        d = (round(PLANAR_D_FRACTION * n_gap) + phase) * dz
        for label in [m.upper() for m in args.modes]:
            z_kind, component, solver = PLANAR_MODES[label]
            f_cont = solver(kx, eps, d, h) / (2 * math.pi)
            pulse = mw.GaussianPulse(frequency=f_cont, fwidth=0.5 * f_cont)
            for method in args.methods:
                scene = mw.Scene(
                    domain=mw.Domain(bounds=((0.0, PLANAR_L), (0.0, 2 * dl), (0.0, h))),
                    grid=mw.GridSpec.custom(x_nodes, y_nodes, z_nodes),
                    boundary=mw.BoundarySpec.faces(default="pec", y="periodic", z=z_kind),
                    device="cuda",
                    subpixel_samples=subpixel_spec(method, _samples_for(args)),
                )
                scene.add_structure(mw.Structure(
                    geometry=mw.Box(position=(0.5 * PLANAR_L, dl, d - h), size=(4 * PLANAR_L, 8 * dl, 2 * h)),
                    material=mw.Material(eps_r=eps), name="layer"))
                # Off the x standing-wave node and inside the dielectric.
                scene.add_source(mw.UniformCurrentSource(size=(dl, dl, 0.5 * dz), center=(0.43 * PLANAR_L, dl, 0.3 * d),
                                                         polarization=component, source_time=pulse, name="src"))
                scene.add_monitor(mw.FieldTimeMonitor("probe", components=(component,),
                                                      position=(0.57 * PLANAR_L, dl, 0.61 * d)))
                dt_estimate = 0.9 / (C_0 * math.sqrt(2 / dl**2 + 1 / dz**2))
                steps = int(math.ceil((args.cycles / f_cont + 3 * pulse.delay) / dt_estimate))
                sim = mw.Simulation.fdtd(scene, frequency=f_cont, run_time=mw.TimeConfig(time_steps=steps),
                                         full_field_dft=False)
                result, stats = _timed_run(sim)
                payload = result.monitor("probe")
                found = find_resonances(payload, (0.9 * f_cont, 1.1 * f_cont), start_time=2.5 * pulse.delay)
                f_fdtd, decay, hi_error = _pick_mode(found, f_cont)
                dt = stats["dt_s"]
                omega_tilde = solver(kx_tilde, eps, d, h)
                f_ref = 2.0 / dt * math.asin(0.5 * omega_tilde * dt) / (2 * math.pi)
                _append(args.output, dict(
                    case="planar", mode=label, eps_r=eps, resolution=n_gap, dz_m=dz, interface_cells=d / dz,
                    offset_index=phase_index, interface_phase=phase, method=method,
                    samples=1 if method == "staircase" else str(_samples_for(args)),
                    f_fdtd_hz=f_fdtd, f_ref_semidiscrete_hz=f_ref, f_ref_continuum_hz=f_cont,
                    rel_error=f_fdtd / f_ref - 1.0, decay_rate=decay, inversion_error=hi_error,
                    envelope=_envelope(payload, component, 2.5 * pulse.delay),
                    modes_found=found.frequency.detach().cpu().numpy().tolist(), **stats))
                del result


# --------------------------------------------------------------------------
# pair: printed coupled pair cross-section as an x cavity
# --------------------------------------------------------------------------
PAIR_HEIGHT = 0.005
PAIR_WIDTH = 0.005
PAIR_OFFSET = 0.005
PAIR_EPS = 2.25
PAIR_DL = 0.0025
PAIR_NCELLS_X = 43


def pair_grid(dx, *, half_y=0.14, half_z=0.13, refine=0.020, coarse=0.0025):
    """Transverse nodes of the printed-line grid: dx inside +-refine, coarse outside."""
    def axis(lo, hi):
        fine = np.arange(-round(refine / dx), round(refine / dx) + 1) * dx
        outer = np.arange(round(lo / coarse), round(hi / coarse) + 1) * coarse
        nodes = np.r_[outer[outer < -refine - 1e-12], fine, outer[outer > refine + 1e-12]]
        return nodes[(nodes >= lo - 1e-12) & (nodes <= hi + 1e-12)]
    return axis(-half_y, half_y), axis(-PAIR_HEIGHT, half_z)


def run_pair(args):
    dx = float(args.resolution) * 1e-3
    y_nodes, z_nodes = pair_grid(dx)
    length = PAIR_NCELLS_X * PAIR_DL
    x_nodes = np.linspace(0.0, length, PAIR_NCELLS_X + 1)
    kx = math.pi / length
    f_guess = C_0 * kx / (2 * math.pi * math.sqrt(1.907))
    for method in args.methods:
        scene = mw.Scene(
            domain=mw.Domain(bounds=((0.0, length), (y_nodes[0], y_nodes[-1]), (z_nodes[0], z_nodes[-1]))),
            grid=mw.GridSpec.custom(x_nodes, y_nodes, z_nodes),
            boundary=mw.BoundarySpec.pec(),
            device="cuda",
            subpixel_samples=subpixel_spec(method, args.samples),
        )
        scene.add_structure(mw.Structure(
            mw.Box(position=(0.5 * length, 0.0, -PAIR_HEIGHT / 2), size=(2 * length, 1.0, PAIR_HEIGHT)),
            mw.Material(eps_r=PAIR_EPS), name="substrate"))
        for name, center in (("positive", PAIR_OFFSET), ("negative", -PAIR_OFFSET)):
            scene.add_structure(mw.Structure(
                mw.PlanarPEC.rectangle(((-length, 2 * length), (center - PAIR_WIDTH / 2, center + PAIR_WIDTH / 2)),
                                       coordinate=0.0, physical_thickness=35e-6),
                mw.Material.pec(), name=name))
        pulse = mw.GaussianPulse(frequency=f_guess, fwidth=0.4 * f_guess)
        for sign, name in ((1.0, "src_p"), (-1.0, "src_n")):
            # Equal (even) excitation under both strips.
            scene.add_source(mw.UniformCurrentSource(size=(PAIR_DL, 1.0e-3, 2.0e-3),
                                                     center=(0.41 * length, sign * PAIR_OFFSET, -2.5e-3),
                                                     polarization="Ez", source_time=pulse, name=name))
        scene.add_monitor(mw.FieldTimeMonitor("probe", components=("Ez",),
                                              position=(0.5 * length + 0.3 * PAIR_DL, PAIR_OFFSET, -1.9e-3)))
        scene.add_monitor(mw.FieldTimeMonitor("probe_n", components=("Ez",),
                                              position=(0.5 * length + 0.3 * PAIR_DL, -PAIR_OFFSET, -1.9e-3)))
        dmin = min(PAIR_DL, float(np.min(np.diff(y_nodes))), float(np.min(np.diff(z_nodes))))
        dt_estimate = 0.9 / (C_0 * math.sqrt(1 / PAIR_DL**2 + 2 / dmin**2))
        steps = int(math.ceil((args.cycles / f_guess + 3 * pulse.delay) / dt_estimate))
        sim = mw.Simulation.fdtd(scene, frequency=f_guess, run_time=mw.TimeConfig(time_steps=steps),
                                 full_field_dft=False)
        result, stats = _timed_run(sim)
        traces = {}
        for name in ("probe", "probe_n"):
            payload = result.monitor(name)
            found = find_resonances(payload, (0.85 * f_guess, 1.15 * f_guess), start_time=2.5 * pulse.delay)
            traces[name] = [(float(f), float(g)) for f, g in zip(found.frequency.cpu().numpy(), found.decay_rate.cpu().numpy())]
        f_even, decay, _ = _pick_mode(find_resonances(result.monitor("probe"), (0.85 * f_guess, 1.15 * f_guess),
                                                      start_time=2.5 * pulse.delay), f_guess)
        _append(args.output, dict(
            case="pair", resolution=float(args.resolution), dx_m=dx, method=method,
            samples=1 if method == "staircase" else args.samples, kx_rad_per_m=kx, cavity_length_m=length,
            longitudinal_dl_m=PAIR_DL, f_even_hz=f_even, decay_rate=decay, modes_probe=traces,
            transverse_nodes=[len(y_nodes), len(z_nodes)], **stats))


def _load_reference_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("line_cross_section_reference", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # dataclasses resolve their module at class creation
    spec.loader.exec_module(module)
    return module


def _richardson(values, ratio=2.0):
    v = np.asarray(values, dtype=float)
    d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
    r = d1 / d2 if d2 else float("inf")
    p = math.log(abs(r), ratio) if r > 0 else float("nan")
    p_used = p if (math.isfinite(p) and 0.5 <= p <= 4) else 1.0
    value = v[-1] + d2 / (ratio**p_used - 1)
    return dict(levels=v.tolist(), value=float(value), order=float(p), uncertainty=float(abs(value - v[-1])))


def _even_beta(R, mesh, frequency):
    fw = R.FullWave(mesh, frequency)
    modes = fw.solve(2 * math.pi * frequency / R.C0 * 1.45, count=4)
    for mode in modes[:3]:
        vp = fw.line_voltage(mode, (R.PAIR_OFFSET, 0.0), (R.PAIR_OFFSET, -R.HEIGHT))
        vn = fw.line_voltage(mode, (-R.PAIR_OFFSET, 0.0), (-R.PAIR_OFFSET, -R.HEIGHT))
        if abs(vp + vn) > abs(vp - vn):
            return mode["beta_rad_per_m"]
    raise RuntimeError("even mode not found")


def pair_reference(args):
    """Lossless even-mode references (CPU): continuum Richardson beta(f) and the
    Yee-grid analog on the FDTD transverse nodes, plus a y/z attribution study."""
    R = _load_reference_module(args.reference_module)
    out = dict(reference_module=str(args.reference_module), frequencies_hz=[], continuum={}, analog={}, attribution={})
    section = R.printed_line_section("differential_pair", thickness=0.0, sigma=0.0)
    base = R.MeshSpec(2e-4, 4e-3, 0.3)
    freqs = [float(f) for f in args.frequencies]
    out["frequencies_hz"] = freqs
    for label, sec in (("substrate", section),):
        out["continuum"][label] = []
        for f in freqs:
            levels = [_even_beta(R, R.Mesh(sec, base.refined(level)), f) for level in (1, 2, 3)]
            out["continuum"][label].append(_richardson(levels))
            print(label, f, out["continuum"][label][-1], flush=True)
    f0 = freqs[len(freqs) // 2]
    for label, sec in (("substrate", section),):
        out["analog"][label] = {}
        for dx_mm in (2.5, 1.25, 0.625, 0.3125):
            y, z = pair_grid(dx_mm * 1e-3)
            out["analog"][label][str(dx_mm)] = _even_beta(R, R.Mesh(sec, nodes=(y, z)), f0)
            print("analog", label, dx_mm, out["analog"][label][str(dx_mm)], flush=True)
    for dy_mm, dz_mm in ((1.25, 1.25), (1.25, 0.3125), (0.3125, 1.25), (0.3125, 0.3125)):
        y, _ = pair_grid(dy_mm * 1e-3)
        _, z = pair_grid(dz_mm * 1e-3)
        out["attribution"][f"dy{dy_mm}_dz{dz_mm}"] = _even_beta(R, R.Mesh(section, nodes=(y, z)), f0)
        print("attribution", dy_mm, dz_mm, out["attribution"][f"dy{dy_mm}_dz{dz_mm}"], flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


# --------------------------------------------------------------------------
# sphere / pec: TFSF scattering with a Huygens projection box
# --------------------------------------------------------------------------
WAVELENGTH = 1.0
FREQUENCY = C_0 / WAVELENGTH


def _gauss_theta(count):
    mu, weights = np.polynomial.legendre.leggauss(count)
    theta = np.arccos(-mu)  # increasing theta
    return theta, weights[::-1]


def _sphere_scene(args, method, *, radius, pec, offset, dx, include_sphere=True):
    tfsf_half = radius + 0.12 * WAVELENGTH
    box_half = tfsf_half + max(4 * dx, 0.06 * WAVELENGTH)
    domain_half = box_half + float(args.padding) * WAVELENGTH
    layers = int(args.pml_layers)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-domain_half, domain_half),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=layers),
        device="cuda",
        subpixel_samples=subpixel_spec(method, args.samples),
    )
    center = tuple(float(o) * dx for o in offset)
    if include_sphere:
        material = mw.Material.pec() if pec else mw.Material(eps_r=float(args.eps))
        scene.add_structure(mw.Structure(geometry=mw.Sphere(position=center, radius=radius), material=material, name="sphere"))
    scene.add_source(mw.PlaneWave(direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                                  source_time=mw.CW(frequency=FREQUENCY),
                                  injection=mw.TFSF(bounds=((-tfsf_half, tfsf_half),) * 3), name="pw"))
    theta_gl, _ = _gauss_theta(int(args.theta_points))
    theta = np.r_[0.0, theta_gl, math.pi]
    phi = np.arange(int(args.phi_points)) * 2 * math.pi / int(args.phi_points)
    if include_sphere:
        scene.add_monitor(mw.AngleProjectionMonitor("far", position=center, size=(2 * box_half,) * 3,
                                                    theta=tuple(theta), phi=tuple(phi), frequencies=(FREQUENCY,),
                                                    radius=1.0e3 * WAVELENGTH, origin=center))
    else:
        scene.add_monitor(mw.PointMonitor("incident", (0.0, 0.0, 0.0), fields=("Ex",)))
    return scene, theta, phi


def _run_time(args):
    return mw.TimeConfig.auto(steady_cycles=int(args.steady_cycles), transient_cycles=int(args.transient_cycles))


def run_sphere(args, *, pec=False):
    n_ppw = float(args.resolution)
    index = 1.0 if pec else math.sqrt(float(args.eps))
    dx = WAVELENGTH / (index * n_ppw)
    size_parameter = float(args.size_parameter)
    radius = size_parameter * WAVELENGTH / (2 * math.pi)
    methods = args.methods
    sim_kw = dict(frequencies=[FREQUENCY], run_time=_run_time(args),
                  spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False)
    # Incident amplitude at the TFSF centre, same grid, no sphere.
    scene0, _, _ = _sphere_scene(args, "staircase", radius=radius, pec=pec, offset=(0, 0, 0), dx=dx, include_sphere=False)
    result0, stats0 = _timed_run(mw.Simulation.fdtd(scene0, **sim_kw))
    incident = abs(complex(torch.as_tensor(result0.monitor("incident")["data"]).reshape(-1)[0].item()))
    del result0
    torch.cuda.empty_cache()
    k = 2 * math.pi / WAVELENGTH
    m = None if pec else index
    q = A.mie_sphere_efficiencies(size_parameter, m, pec=pec)
    c_sca_ref = q["q_sca"] * math.pi * radius**2
    offsets = SPHERE_OFFSETS[: int(args.offsets)]
    for offset_index, offset in enumerate(offsets):
        for method in methods:
            scene, theta, phi = _sphere_scene(args, method, radius=radius, pec=pec, offset=offset, dx=dx)
            result, stats = _timed_run(mw.Simulation.fdtd(scene, **sim_kw))
            flux = float(torch.as_tensor(result.raw_monitor("far")["flux"]).reshape(-1)[0].real)
            projected = result.monitor("far")
            intensity = projected.radiation_intensity[0].detach().cpu().numpy()  # [theta, phi] W/sr
            _, weights = _gauss_theta(int(args.theta_points))
            power_ff = float(np.sum(weights[:, None] * intensity[1:-1, :]) * (2 * math.pi / len(phi)))
            i_inc = incident**2 / (2 * ETA_0)
            e2 = projected.E_intensity[0].detach().cpu().numpy()
            radius_obs = float(projected.radius.reshape(-1)[0])
            sigma = 4 * math.pi * radius_obs**2 * e2 / incident**2
            th, ph = np.meshgrid(theta, phi, indexing="ij")
            sigma_ref = A.mie_sphere_bistatic_rcs(radius, WAVELENGTH, th, ph, m, pec=pec)
            cut = {}
            for label, phi_value in (("E", 0.0), ("H", 0.5 * math.pi)):
                j = int(np.argmin(np.abs(phi - phi_value)))
                cut[label] = float(np.linalg.norm(sigma[:, j] - sigma_ref[:, j]) / np.linalg.norm(sigma_ref[:, j]))
            c_sca_flux = flux / i_inc
            c_sca_ff = power_ff / i_inc
            j0 = int(np.argmin(np.abs(phi)))
            _append(args.output, dict(
                case="pec" if pec else "sphere", eps_r=None if pec else float(args.eps), size_parameter=size_parameter,
                resolution=n_ppw, dx_m=dx, radius_cells=radius / dx, method=method,
                samples=1 if method in ("staircase", "conformal") else args.samples,
                offset_index=offset_index, offset_cells=list(offset), incident_amplitude=incident,
                c_sca_ref=c_sca_ref, c_sca_flux=c_sca_flux, c_sca_farfield=c_sca_ff,
                err_csca_flux=c_sca_flux / c_sca_ref - 1, err_csca_farfield=c_sca_ff / c_sca_ref - 1,
                err_pattern_E=cut["E"], err_pattern_H=cut["H"],
                sigma_back=float(sigma[-1, j0]), sigma_back_ref=float(sigma_ref[-1, j0]),
                err_back=float(sigma[-1, j0] / sigma_ref[-1, j0] - 1),
                sigma_forward=float(sigma[0, j0]), sigma_forward_ref=float(sigma_ref[0, j0]),
                padding_wavelengths=float(args.padding), pml_layers=int(args.pml_layers),
                steady_cycles=int(args.steady_cycles), transient_cycles=int(args.transient_cycles),
                pattern=dict(theta=theta.tolist(), sigma_E=sigma[:, j0].tolist(), sigma_E_ref=sigma_ref[:, j0].tolist()),
                vacuum_wall_s=stats0["wall_s"], **stats))
            del result
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# cylinder: z-invariant, TE and TM
# --------------------------------------------------------------------------
def run_cylinder(args):
    n_ppw = float(args.resolution)
    index = math.sqrt(float(args.eps))
    dx = WAVELENGTH / (index * n_ppw)
    size_parameter = float(args.size_parameter)
    radius = size_parameter * WAVELENGTH / (2 * math.pi)
    tfsf_half = radius + 0.12 * WAVELENGTH
    box_half = tfsf_half + max(4 * dx, 0.06 * WAVELENGTH)
    domain_half = box_half + float(args.padding) * WAVELENGTH
    zlen = 2 * dx
    sim_kw = dict(frequencies=[FREQUENCY], run_time=_run_time(args),
                  spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False)
    offsets = SPHERE_OFFSETS[: int(args.offsets)]
    for polarization in [p.upper() for p in args.modes]:
        e_pol = (0.0, 0.0, 1.0) if polarization == "TM" else (0.0, 1.0, 0.0)

        def build(method, offset, with_cylinder):
            scene = mw.Scene(
                domain=mw.Domain(bounds=((-domain_half, domain_half), (-domain_half, domain_half), (0.0, zlen))),
                grid=mw.GridSpec.uniform(dx),
                boundary=mw.BoundarySpec.faces(default="pml", num_layers=int(args.pml_layers), z="periodic"),
                device="cuda",
                subpixel_samples=subpixel_spec(method, args.samples),
            )
            center = (float(offset[0]) * dx, float(offset[1]) * dx, 0.5 * zlen)
            if with_cylinder:
                scene.add_structure(mw.Structure(
                    geometry=mw.Cylinder(position=center, radius=radius, height=100 * zlen, axis="z"),
                    material=mw.Material(eps_r=float(args.eps)), name="cylinder"))
            scene.add_source(mw.PlaneWave(direction=(1.0, 0.0, 0.0), polarization=e_pol,
                                          source_time=mw.CW(frequency=FREQUENCY),
                                          injection=mw.TFSF(bounds=((-tfsf_half, tfsf_half), (-tfsf_half, tfsf_half),
                                                                    (-zlen, 2 * zlen))), name="pw"))
            if with_cylinder:
                for name, axis, pos, direction in (("xn", "x", -box_half, "-"), ("xp", "x", box_half, "+"),
                                                   ("yn", "y", -box_half, "-"), ("yp", "y", box_half, "+")):
                    p = [center[0], center[1], 0.5 * zlen]
                    size = [2 * box_half, 2 * box_half, zlen]
                    i = "xyz".index(axis)
                    p[i] = pos
                    size[i] = 0.0
                    scene.add_monitor(mw.FinitePlaneMonitor(name, position=tuple(p), size=tuple(size),
                                                            fields=_flux_fields(axis), frequencies=(FREQUENCY,),
                                                            compute_flux=True, normal_direction=direction))
            else:
                scene.add_monitor(mw.PointMonitor("incident", (0.0, 0.0, 0.5 * zlen),
                                                  fields=("Ez" if polarization == "TM" else "Ey",)))
            return scene

        result0, stats0 = _timed_run(mw.Simulation.fdtd(build("staircase", (0, 0), False), **sim_kw))
        incident = abs(complex(torch.as_tensor(result0.monitor("incident")["data"]).reshape(-1)[0].item()))
        del result0
        q = A.mie_cylinder_efficiencies(size_parameter, index, polarization=polarization)
        width_ref = q["q_sca"] * 2 * radius
        for offset_index, offset in enumerate(offsets):
            for method in args.methods:
                result, stats = _timed_run(mw.Simulation.fdtd(build(method, offset, True), **sim_kw))
                power = 0.0
                for name in ("xn", "xp", "yn", "yp"):
                    payload = result.raw_monitor(name)
                    power += float(torch.as_tensor(payload["flux"]).reshape(-1)[0].real)
                width = power / zlen / (incident**2 / (2 * ETA_0))
                _append(args.output, dict(
                    case="cylinder", polarization=polarization, eps_r=float(args.eps), size_parameter=size_parameter,
                    resolution=n_ppw, dx_m=dx, radius_cells=radius / dx, method=method,
                    samples=1 if method == "staircase" else args.samples, offset_index=offset_index,
                    incident_amplitude=incident, width_ref=width_ref, width=width, err_width=width / width_ref - 1,
                    padding_wavelengths=float(args.padding), pml_layers=int(args.pml_layers), **stats))
                del result
                torch.cuda.empty_cache()


def _flux_fields(axis):
    from witwin.maxwell.monitors import required_flux_fields

    return required_flux_fields(axis)


# --------------------------------------------------------------------------
# analytic self-check
# --------------------------------------------------------------------------
def mie_check(args):
    x = 2 * math.pi * 0.525 / 0.6328
    bh = A.mie_sphere_efficiencies(x, 1.55)
    rows = dict(
        bohren_huffman_example=dict(x=x, m=1.55, q_sca=bh["q_sca"], q_back=bh["q_back"], expected_q_sca=3.10543,
                                    expected_q_back=2.92534),
        pec_backscatter_x1=A.mie_sphere_efficiencies(1.0, pec=True)["q_back"],
        rayleigh_sphere=[A.mie_sphere_efficiencies(0.01, 2.0)["q_sca"], A.rayleigh_sphere_q_sca(0.01, 2.0)],
        rayleigh_cylinder_TM=[A.mie_cylinder_efficiencies(0.01, 2.0, polarization="TM")["q_sca"],
                              A.rayleigh_cylinder_q_sca(0.01, 2.0, "TM")],
        rayleigh_cylinder_TE=[A.mie_cylinder_efficiencies(0.01, 2.0, polarization="TE")["q_sca"],
                              A.rayleigh_cylinder_q_sca(0.01, 2.0, "TE")],
    )
    print(json.dumps(rows, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=("planar", "pair", "pair-reference", "sphere", "pec", "cylinder", "mie-check", "report"))
    parser.add_argument("--resolution", type=float, default=20)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--samples-z", type=int, default=None, help="planar: anisotropic (1, 1, S) samples")
    parser.add_argument("--eps", type=float, default=4.0)
    parser.add_argument("--size-parameter", type=float, default=1.5)
    parser.add_argument("--offsets", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--cycles", type=float, default=40)
    parser.add_argument("--padding", type=float, default=0.25, help="vacuum gap to the PML in wavelengths")
    parser.add_argument("--pml-layers", type=int, default=12)
    parser.add_argument("--steady-cycles", type=int, default=12)
    parser.add_argument("--transient-cycles", type=int, default=20)
    parser.add_argument("--theta-points", type=int, default=24)
    parser.add_argument("--phi-points", type=int, default=24)
    parser.add_argument("--output", default="subpixel_runs.jsonl")
    parser.add_argument("--input", nargs="+", default=None)
    parser.add_argument("--plots", default=None)
    parser.add_argument("--reference-module", default="benchmark/line_cross_section_reference.py")
    parser.add_argument("--frequencies", nargs="+", type=float, default=[0.99e9, 1.01e9, 1.03e9])
    args = parser.parse_args(argv)
    if args.case == "mie-check":
        return mie_check(args)
    if args.case == "pair-reference":
        return pair_reference(args)
    if args.case == "report":
        from scripts.dev.fdtd.subpixel_convergence_report import report

        return report(args.input, args.plots)
    if args.methods is None:
        args.methods = list(PEC_METHODS if args.case == "pec" else METHODS)
    if args.case == "planar":
        args.modes = args.modes or ["TM0", "TE0"]
        return run_planar(args)
    if args.case == "pair":
        return run_pair(args)
    if args.case == "sphere":
        return run_sphere(args)
    if args.case == "pec":
        return run_sphere(args, pec=True)
    if args.case == "cylinder":
        args.modes = args.modes or ["TM", "TE"]
        return run_cylinder(args)


if __name__ == "__main__":
    main()
