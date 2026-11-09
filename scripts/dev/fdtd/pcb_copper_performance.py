"""Qualify physical volume-copper controls before equal-error PCB timings.

This uses the public GPU solver. The analytic film transfer matrix is an
independent frequency-domain reference, not an alternate numerical solver.
An unqualified or nonexclusive run is never a performance acceptance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import witwin.maxwell as mw

from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.pcb_legacy_performance import processes
from scripts.dev.fdtd.pcb_candidate_inputs import candidate
from scripts.dev.fdtd.pcb_application_campaign import FIXTURES

C0 = 299792458.
MU0 = 1.25663706212e-6
EPS0 = 1 / (MU0 * C0**2)
ETA0 = MU0 * C0
SIGMA = 5.8e7


def skin_depth(frequency):
    return math.sqrt(1 / (math.pi * frequency * MU0 * SIGMA))


def film_reference(thickness, frequency):
    """Full Maxwell finite-film r/t at the two physical copper faces."""
    s = -2j * math.pi * frequency
    gamma = np.sqrt(s * MU0 * (SIGMA + s * EPS0))
    impedance = s * MU0 / gamma
    a = np.cosh(gamma * thickness)
    b = impedance * np.sinh(gamma * thickness)
    c = np.sinh(gamma * thickness) / impedance
    denominator = 2 * a + b / ETA0 + ETA0 * c
    reflection = (b / ETA0 - ETA0 * c) / denominator
    transmission = 2 / denominator
    absorption = 1 - abs(reflection)**2 - abs(transmission)**2
    return reflection, transmission, absorption


def normal_grid(thickness, frequency, cells_per_skin):
    """Resolve both physical faces, with the same fine spacing outside them.

    Exterior virtual coordinates remove half the copper thickness on each
    side. Thus launch/probe distances are identical to a two-face sheet at 0.
    The last transition cell is merged when it would be anomalously small.
    """
    count = math.ceil(thickness * cells_per_skin / skin_depth(frequency))
    spacing = thickness / count
    distance = [0.]
    step = spacing
    while distance[-1] + step < .004:
        distance.append(distance[-1] + step)
        step = min(1.3 * step, .001)
    if .004 - distance[-1] < .25 * step:
        distance.pop()
    distance.append(.004)
    distance.extend(np.linspace(.005, .025, 21))
    distance = np.asarray(distance)
    inside = np.linspace(-thickness / 2, thickness / 2, count + 1)
    return np.concatenate((-distance[:0:-1] - thickness / 2, inside,
                           distance[1:] + thickness / 2))


def film_scene(kind, thickness, frequency, cells_per_skin, *, device="cuda:0"):
    if kind not in ("volume", "surface"):
        raise ValueError("The physical control kind must be volume or surface.")
    x = (normal_grid(thickness, frequency, cells_per_skin) if kind == "volume"
         else np.linspace(-.025, .025, 51))
    y = np.linspace(-.001, .001, 5)
    z = np.array([-.001, 0., .001])
    source_x = -.015 - (thickness / 2 if kind == "volume" else 0.)
    geometry = mw.Box(position=torch.zeros(3, dtype=torch.float64),
        size=torch.tensor([thickness if kind == "volume" else 0., .08, .08], dtype=torch.float64))
    material = (mw.Material(eps_r=1., sigma_e=SIGMA) if kind == "volume" else
                mw.CopperFoil(thickness=thickness, conductivity=SIGMA, frequency_range=(1e9, 4e9)))
    return mw.Scene(domain=mw.Domain(bounds=((float(x[0]), float(x[-1])), (-.001, .001), (-.001, .001))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.faces(default="pml", num_layers=12, y="pmc", z="pec"),
        device=device, subpixel_samples=mw.SubpixelSpec(samples=(4, 1, 1), averaging="arithmetic"),
        structures=(mw.Structure(name="copper", geometry=geometry, material=material),),
        ports=(mw.LumpedPort("feed", positive=(source_x, 0., .001), negative=(source_x, 0., -.001),
            voltage_path=mw.AxisPath("z"), reference_impedance=ETA0,
            voltage_aperture=mw.Box(position=(source_x, 0., 0.), size=(0., .001, .002)),
            current_surface=mw.Box(position=(source_x, 0., -.0005), size=(.001, .0015, 0.))),),
        metadata={"physical_copper_thickness_m": thickness, "conductivity_s_per_m": SIGMA,
            "reference_planes": "physical volume faces mapped to sheet left/right traces",
            "exterior_mapping": "x - sign(x)*thickness/2 for volume; x for surface"})


def fit_waves(x, values, low, high, k):
    mask = (x >= low - 1e-14) & (x <= high + 1e-14)
    basis = np.stack((np.exp(1j*k*x[mask]), np.exp(-1j*k*x[mask])), axis=-1)
    amplitudes = np.linalg.lstsq(basis, values[mask], rcond=None)[0]
    residual = np.linalg.norm(basis @ amplitudes - values[mask]) / np.linalg.norm(values[mask])
    return amplitudes, float(residual)


def pcb_volume_scene(family, frequency, cells_per_skin, *, device="cuda:0"):
    """Restore the declared physical copper volume from the saved sheet input.

    Normal refinement alone is a workload forecast. Finite-strip edge heat
    convergence remains a separate acceptance gate before any timing comparison.
    """
    from witwin.maxwell.adapters.pcb import import_pcb_scene
    original = json.loads((FIXTURES/f"metahz-scene-pcb-{family}.json").read_text())
    definition = candidate(original, finite_copper=True)
    scene = import_pcb_scene(definition, frequency=frequency, device=device)
    model = definition["conductors"][0]["constitutive_model"]
    thickness = model["thickness_m"]
    bottom = definition["conductors"][0]["bounds_m"][4]
    z = np.asarray(definition["mesh"]["axes_m"]["z"])
    expanded = z + np.where(z > bottom, thickness, 0.)
    fine = normal_grid(thickness, frequency, cells_per_skin)
    fine = fine[(fine >= -.004-thickness/2-1e-15) & (fine <= .004+thickness/2+1e-15)]
    fine = fine + bottom + thickness/2
    expanded = expanded[(expanded < fine[0]) | (expanded > fine[-1])]
    # Dielectric interfaces remain at their original physical positions. Avoid
    # an arbitrarily tiny cell when an arithmetic transition happens to approach
    # an interface; never move a copper face or a dielectric boundary.
    for dielectric in definition["dielectrics"]:
        for boundary in dielectric["bounds_m"][4:6]:
            if fine[0] < boundary < fine[-1]:
                protected = np.isclose(fine, bottom, atol=1e-15, rtol=0) | np.isclose(fine, bottom+thickness, atol=1e-15, rtol=0)
                fine = fine[(abs(fine-boundary) > thickness/(2*math.ceil(thickness*cells_per_skin/skin_depth(frequency)))) | protected]
                fine = np.append(fine, boundary)
    z = np.unique(np.concatenate((expanded, fine)))
    structures = []
    for structure in scene.structures:
        if isinstance(structure.material, mw.CopperFoil):
            position = structure.geometry.position.detach().clone()
            size = structure.geometry.size.detach().clone()
            position[2] = bottom + thickness/2
            size[2] = thickness
            structures.append(mw.Structure(name=structure.name,
                geometry=mw.Box(position=position, size=size), material=mw.Material(eps_r=1., sigma_e=SIGMA)))
        else:
            structures.append(structure)
    ports = []
    contour_half_planes = []
    for port in scene.ports:
        positive, negative = np.array(port.positive, dtype=np.float64), np.array(port.negative, dtype=np.float64)
        positive[2] += thickness
        negative[2] += thickness
        aperture_position = port.voltage_aperture.position.detach().clone()
        aperture_position[2] += thickness
        contour_position = port.current_surface.position.detach().clone()
        contour_size = port.current_surface.size.detach().clone()
        low = float(contour_position[2]-contour_size[2]/2)
        high = float(contour_position[2]+contour_size[2]/2)
        low += thickness if low >= bottom else 0.
        high += thickness if high >= bottom else 0.
        contour_half_planes.extend((low, high))
        contour_position[2], contour_size[2] = (low+high)/2, high-low
        ports.append(mw.LumpedPort(port.name, positive=positive, negative=negative,
            voltage_path=port.voltage_path, reference_impedance=port.reference_impedance,
            voltage_aperture=mw.Box(position=aperture_position, size=port.voltage_aperture.size),
            current_surface=mw.Box(position=contour_position, size=contour_size)))
    # Preserve the physical H-contour planes on the refined Yee half grid.
    # Refine the grid around the plane; do not snap or change the measurement.
    half_step = thickness/math.ceil(thickness*cells_per_skin/skin_depth(frequency))
    for plane in contour_half_planes:
        if min(abs(plane-bottom),abs(plane-bottom-thickness)) <= 2*half_step:
            raise ValueError("A contour half-plane is too close to a protected copper face for this grid.")
        z = z[abs(z-plane) > 1.5*half_step]
        z = np.unique(np.concatenate((z,[plane-half_step/2,plane+half_step/2])))
    axes = [np.asarray(definition["mesh"]["axes_m"][axis]) for axis in "xy"] + [z]
    return scene.clone(domain=mw.Domain(bounds=[(float(axis[0]),float(axis[-1])) for axis in axes]),
        grid=mw.GridSpec.custom(*axes), structures=tuple(structures), ports=tuple(ports),
        subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 4), averaging="arithmetic"),
        metadata={**scene.metadata, "physical_volume_copper_sigma": SIGMA,
            "physical_copper_bounds_m": [bottom,bottom+thickness],
            "qualification": "normal refinement workload only; lateral copper edge and heat convergence pending"})


def forecast_pcb(family, frequency, cells_per_skin, duration):
    scene = pcb_volume_scene(family, frequency, cells_per_skin)
    simulation = mw.Simulation.fdtd(scene, frequency=frequency,
        excitations=mw.PortExcitation("feed" if scene.ports[0].name == "feed" else scene.ports[0].name,
            source_time=mw.GaussianPulse(frequency, fwidth=frequency)),
        run_time=mw.TimeConfig(time_steps=1), full_field_dft=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prepared = simulation.prepare()
    torch.cuda.synchronize()
    solver = prepared.solver
    return {"family": family, "skin_depth_m": skin_depth(frequency), "sigma_s_per_m": SIGMA,
        "grid_shape": [solver.Nx,solver.Ny,solver.Nz], "dt_s": solver.dt,
        "steps": math.ceil(duration/solver.dt), "requested_duration_s": duration,
        "min_spacing_m": [float(np.diff(getattr(solver.scene,axis+"_nodes64")).min()) for axis in "xyz"],
        "prepared_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "prepare_seconds": time.perf_counter()-start,
        "status": "actual native preparation; no field run or heat/lateral convergence qualification"}


def run_film(kind, thickness, frequency, cells_per_skin, duration, *, execute=True):
    scene = film_scene(kind, thickness, frequency, cells_per_skin)
    simulation = mw.Simulation.fdtd(scene, frequency=frequency,
        excitations=mw.PortExcitation("feed", source_time=mw.CW(frequency)),
        run_time=mw.TimeConfig(time_steps=1, transient_cycles=0), full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="hanning"))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    prepared = simulation.prepare()
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter() - start
    solver = prepared.solver
    steps = math.ceil(duration / solver.dt)
    simulation.config.run_time = mw.TimeConfig(time_steps=steps, transient_cycles=0)
    report = {"kind": kind, "thickness_m": thickness, "sigma_s_per_m": SIGMA,
        "frequency_hz": frequency, "skin_depth_m": skin_depth(frequency),
        "cells_per_skin_requested": cells_per_skin, "grid_shape": [solver.Nx, solver.Ny, solver.Nz],
        "min_normal_spacing_m": float(np.diff(solver.scene.x_nodes64).min()),
        "dt_s": solver.dt, "steps": steps, "requested_duration_s": duration,
        "physical_duration_s": steps*solver.dt, "prepare_seconds": prepare_seconds,
        "prepared_allocated_bytes": torch.cuda.memory_allocated(),
        "six_float32_fields_lower_bound_bytes": 24 * solver.Nx * solver.Ny * solver.Nz,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "physical calibration / workload forecast, not exclusive repeated performance evidence"}
    if not execute:
        return report
    x = solver.scene.x_nodes64.copy()
    widths = np.asarray(solver.scene.dx_dual64).copy()
    sigma = solver.sigma_e_Ez[:, solver.Ny//2, 0].detach().cpu().numpy().copy()
    dt = solver.dt
    # Native batching captures the actual RF/field/DFT pipeline, including the
    # device time table. Millions of skin-resolution steps do not use a CPU loop.
    del prepared, solver
    batch = mw.Simulation.batch([simulation], outputs=mw.BatchOutput(fields=("Ex", "Ey", "Ez")),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=1, graph="required", graph_steps=256))
    report["processes_before"] = processes()
    start = time.perf_counter()
    with batch.prepare() as execution:
        result = execution.run().results()[0]
    torch.cuda.synchronize()
    report["prepare_and_run_seconds"] = time.perf_counter() - start
    report["processes_after"] = processes()
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    electric = result.tensor("Ez").detach().cpu().numpy()
    if electric.ndim == 4:
        electric = electric[0]
    profile = electric[:, electric.shape[1]//2, 0]
    virtual = x - np.sign(x)*thickness/2 if kind == "volume" else x
    k = 2/.001*np.arcsin(np.sin(math.pi*frequency*dt)*.001/(C0*dt))
    (a, b), left_residual = fit_waves(virtual, profile, -.014, -.006, k)
    (c, d), right_residual = fit_waves(virtual, profile, .006, .014, k)
    determinant = a*a-d*d
    reflection, transmission = (b*a-c*d)/determinant, (c*a-b*d)/determinant
    expected_r, expected_t, expected_absorption = film_reference(thickness, frequency)
    incoming_flux = (abs(a)**2 + abs(d)**2)/(2*ETA0)
    if kind == "volume":
        heat_density = float(.5*np.sum(sigma*abs(profile)**2*widths))
        report["normal_sigma_integral_s"] = float(np.sum(sigma*widths))
        report["normal_sigma_integral_relative_error"] = abs(float(np.sum(sigma*widths))/(SIGMA*thickness)-1)
    else:
        heat_density = float(result.conductor_losses["copper"]["surface_power"][0])/.002**2
    absorption = heat_density/incoming_flux
    report.update({"reflection": [float(reflection.real), float(reflection.imag)],
        "transmission": [float(transmission.real), float(transmission.imag)],
        "reference_reflection": [float(expected_r.real), float(expected_r.imag)],
        "reference_transmission": [float(expected_t.real), float(expected_t.imag)],
        "reference_absorption": expected_absorption, "measured_heat_absorption": absorption,
        "heat_relative_error": abs(absorption/expected_absorption-1),
        "reflection_absolute_error": float(abs(reflection-expected_r)),
        "fit_residuals": [left_residual, right_residual],
        "transverse_relative_deviation": float(np.max(abs((electric-profile[:,None,None])[
            (virtual >= -.006) & (virtual <= .006)]))/np.max(abs(profile))),
        "status": "calibration measured; review residuals, field uniformity, heat and convergence before acceptance"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=("volume", "surface"), default="volume")
    parser.add_argument("--thickness-um", type=float, choices=(18., 35.), default=18.)
    parser.add_argument("--frequency", type=float, default=2.45e9)
    parser.add_argument("--cells-per-skin", type=int, default=4)
    parser.add_argument("--duration", type=float, default=3e-9)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pcb-family", choices=("flexible", "rigid"))
    args = parser.parse_args()
    if args.cells_per_skin < 1 or args.duration <= 0:
        parser.error("Positive resolution and duration are required.")
    source = package_identity(mw, "witwin-maxwell")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"schema": "pcb-copper-equal-error/1", "source": source,
        "validation_script_sha256": script_hash,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "measurement": (forecast_pcb(args.pcb_family,args.frequency,args.cells_per_skin,args.duration)
            if args.pcb_family else run_film(args.kind, args.thickness_um*1e-6, args.frequency,
                                args.cells_per_skin, args.duration, execute=not args.prepare_only))}
    report["source_unchanged"] = package_identity(mw, "witwin-maxwell")["source_sha256"] == source["source_sha256"]
    report["validation_script_unchanged"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["measurement"], indent=2), flush=True)


if __name__ == "__main__":
    main()
