"""Bind a physical probe-fed patch to an independently built open-source reference."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import witwin.maxwell as mw
from benchmark.scenes.antenna.printed_patch import printed_patch_scene
from scripts.dev.fdtd.pcb_acceptance import package_identity


def serial(value):
    value = np.asarray(value)
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=16e-9)
    parser.add_argument("--lateral-dl", type=float, default=.001)
    parser.add_argument("--substrate-cells", type=int, default=3)
    parser.add_argument("--edge-dl", type=float, default=None)
    parser.add_argument("--pml-layers", type=int, default=8)
    parser.add_argument("--air-padding", type=float, default=0.)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--reference-dt-ratio", type=float, default=.4)
    parser.add_argument("--record-contour-control", action="store_true")
    parser.add_argument("--record-current-balance", action="store_true")
    parser.add_argument("--matched-current-source", action="store_true")
    parser.add_argument("--reference-current-amplitude", type=float, default=1e-3)
    parser.add_argument("--dielectric-subpixel", choices=("native", "z2", "quarter-cell"), default="native")
    parser.add_argument("--material-setup-only", action="store_true",
        help="Export actual native and independent material coefficients without time stepping.")
    parser.add_argument("--dump-reference-material", action="store_true")
    parser.add_argument("--load-resistance", type=float, default=50.)
    parser.add_argument("--pulse-phase", type=float, default=0.)
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--reference-source", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.reference_dt_ratio <= 1:
        raise ValueError("Reference dt ratio must lie in (0, 1].")
    if not math.isfinite(args.load_resistance) or args.load_resistance <= 0:
        raise ValueError('The physical source/load resistance must be finite and positive.')
    if not math.isfinite(args.pulse_phase):
        raise ValueError('Pulse phase must be finite.')
    if args.record_current_balance and not args.record_contour_control:
        raise ValueError('Current balance requires the independently observed magnetic contour.')
    if not math.isfinite(args.reference_current_amplitude) or args.reference_current_amplitude <= 0:
        raise ValueError('The declared reference current amplitude must be finite and positive.')
    args.output = args.output.resolve()
    args.reference_source = args.reference_source.resolve()
    current_helper_path = None
    current_helper_sha256 = None
    if args.matched_current_source:
        from scripts.dev.fdtd import matched_current_reference
        current_helper_path = Path(matched_current_reference.__file__)
        current_helper_sha256 = hashlib.sha256(current_helper_path.read_bytes()).hexdigest()
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    import openEMS as reference_package
    from openEMS.utilities import DFT_time2freq

    args.output.mkdir(parents=True, exist_ok=True)
    source_identity = package_identity(mw, "witwin-maxwell")
    driver_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_path = Path(__import__(printed_patch_scene.__module__, fromlist=["__file__"]).__file__)
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    frequencies = tuple(float(f) for f in np.linspace(2.9e9, 3.5e9, 31))
    scene = printed_patch_scene(lateral_dl=args.lateral_dl, substrate_cells=args.substrate_cells,
                                radiation_monitor=False, frequencies=frequencies,
                                edge_dl=args.edge_dl, pml_layers=args.pml_layers, air_padding=args.air_padding)
    if args.dielectric_subpixel == 'z2':

        scene = scene.clone(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 2), averaging='polarized'))
    elif args.dielectric_subpixel == 'quarter-cell':
        scene = scene.clone(subpixel_samples=mw.SubpixelSpec(samples=(2, 2, 2), averaging='arithmetic'))
    if args.load_resistance != 50.:
        feed = scene.ports[0]
        scene = scene.clone(ports=[mw.LumpedPort(feed.name, positive=feed.positive,
            negative=feed.negative, voltage_path=feed.voltage_path,
            voltage_aperture=feed.voltage_aperture, current_surface=feed.current_surface,
            reference_impedance=args.load_resistance)])
    if args.record_contour_control:
        feed = scene.ports[0]
        control = mw.LumpedPort('contour_control', positive=feed.positive, negative=feed.negative,
            voltage_path=feed.voltage_path, voltage_aperture=feed.voltage_aperture,
            current_surface=feed.current_surface, reference_impedance=feed.reference_impedance)
        scene = scene.clone(ports=[feed, control])
    if args.record_current_balance:
        scene = scene.clone(monitors=[*scene.monitors,
            mw.PlaneMonitor('current_balance_E', axis='z',
                position=(scene.ports[0].positive[2]+scene.ports[0].negative[2])/2, fields=('Ez',))])
    pulse = mw.GaussianPulse(frequency=3.2e9, fwidth=2.2e9, phase=args.pulse_phase)
    config = mw.Simulation.fdtd(scene, frequencies=frequencies,
        excitations=mw.PortExcitation("feed", source_time=pulse),
        spectral_sampler=mw.SpectralSampler(window="none"),
        run_time=mw.TimeConfig(time_steps=1), full_field_dft=False)
    prepared = config.prepare()
    solver = prepared.solver
    steps = math.ceil(args.duration / solver.dt)
    config.config.run_time = mw.TimeConfig(time_steps=steps)
    start = time.perf_counter()
    own = None if args.material_setup_only else prepared.run()
    own_seconds = time.perf_counter() - start
    if own is not None:
        own.save(args.output / "maxwell.pt")
    if args.material_setup_only or args.dump_reference_material:
        np.savez_compressed(args.output / "maxwell-native-materials.npz",
            **{name: getattr(solver, name).detach().cpu().numpy()
               for name in ("eps_Ex", "eps_Ey", "eps_Ez", "sigma_e_Ex", "sigma_e_Ey", "sigma_e_Ez", "cex_decay", "cex_curl", "cey_decay", "cey_curl", "cez_decay", "cez_curl")},
            **{axis+"_nodes": np.asarray(getattr(solver.scene, axis+"_nodes64")) for axis in "xyz"})

    reference_dt = solver.dt * args.reference_dt_ratio
    reference_steps = math.ceil(args.duration / reference_dt)
    external = openEMS(NrTS=reference_steps, EndCriteria=0., TimeStep=reference_dt)
    phase_term = f'+{pulse.phase:.17g}' if pulse.phase else ''
    expression = f"exp(-0.5*((t-{pulse.delay:.17g})/{pulse.sigma_t:.17g})^2)*cos(2*pi*{pulse.frequency:.17g}*(t-{pulse.delay:.17g}){phase_term})"
    external.SetCustomExcite(expression, pulse.frequency, 10e9)
    external.SetBoundaryCond([f"PML_{args.pml_layers}"] * 6)
    csx = ContinuousStructure()
    external.SetCSX(csx)
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1.)
    for axis in "xyz":
        grid.AddLine(axis, np.asarray(getattr(solver.scene, axis + "_nodes64")))
    geometry_records = []
    for structure in scene.structures:
        geometry, material = structure.geometry, structure.material
        if isinstance(geometry, mw.PlanarPEC):
            bounds = np.zeros((3, 2))
            normal = "xyz".index(geometry.normal_axis)
            bounds[normal] = geometry.coordinate
            tangent = [i for i in range(3) if i != normal]
            vertices = geometry.vertices.detach().cpu().numpy()
            bounds[tangent, 0], bounds[tangent, 1] = vertices.min(0), vertices.max(0)
        else:
            center, size = geometry.position.cpu().double().numpy(), geometry.size.cpu().double().numpy()
            bounds = np.column_stack((center - size / 2, center + size / 2))
        prop = csx.AddMetal(structure.name) if material.is_pec else csx.AddMaterial(structure.name, epsilon=float(material.eps_r))
        prop.AddBox(start=bounds[:, 0], stop=bounds[:, 1], priority=10 if material.is_pec else 0)
        geometry_records.append({"name": structure.name, "bounds_m": bounds.tolist(), "pec": bool(material.is_pec)})
    declaration = scene.ports[0]
    aperture = declaration.voltage_aperture
    center, size = aperture.position.cpu().double().numpy(), aperture.size.cpu().double().numpy()
    source_bounds = np.column_stack((center-size/2, center+size/2))
    known_edges = ()
    if args.matched_current_source:
        from scripts.dev.fdtd.matched_current_reference import compile_known_current_edges, install_known_current_sources
        known_edges = compile_known_current_edges(
            tuple(getattr(solver.scene, axis+'_nodes64') for axis in 'xyz'), axis=2,
            path=(declaration.negative[2], declaration.positive[2]),
            aperture_bounds={axis: tuple(source_bounds[axis]) for axis in (0, 1)},
            dt=reference_dt, epsilon=2.2*8.8541878128e-12)
        install_known_current_sources(csx, known_edges, amplitude=args.reference_current_amplitude)
        port = None
    else:
        port = external.AddLumpedPort(1, args.load_resistance, source_bounds[:, 0], source_bounds[:, 1], "z", excite=1., priority=20)
    # Independently record the same finite-area voltage quadrature. The native
    # reference port also retains its standard central voltage/current probes.
    compiled = solver._port_runtimes[0].geometry
    indices = compiled.voltage_indices.cpu().numpy()
    weights = compiled.voltage_weights.cpu().numpy()
    transverse = {}
    for index, weight in zip(indices, weights):
        key = tuple(index[:2])
        transverse[key] = transverse.get(key, 0.) + weight / size[2]
    names, averaging_weights = [], []
    for number, ((i, j), weight) in enumerate(transverse.items()):
        x, y = solver.scene.x_nodes64[i], solver.scene.y_nodes64[j]
        name = f"aperture_voltage_{number}"
        probe = csx.AddProbe(name, p_type=0, weight=1. if args.matched_current_source else -1.)
        probe.AddBox(start=[x, y, source_bounds[2, 0]], stop=[x, y, source_bounds[2, 1]])
        names.append(name)
        averaging_weights.append(weight)
    contour = declaration.current_surface
    cp, cs = contour.position.cpu().double().numpy(), contour.size.cpu().double().numpy()
    current_name = "declared_current_contour"
    current_probe = csx.AddProbe(current_name, p_type=1, weight=1., norm_dir=2)
    current_bounds = np.column_stack((cp-cs/2, cp+cs/2))
    balance_probes = []
    if args.record_current_balance:
        # Read the actual compiled contour indices. Re-decoding a float32 Box
        # center would discard the compiler's bounded-ULP geometry contract.
        rows = [row.detach().cpu().numpy() for row in compiled.current_indices]
        if compiled.axis != 'z' or compiled.current_components != ('Hx', 'Hy', 'Hx', 'Hy'):
            raise ValueError('This patch balance control requires its compiled z-normal contour.')
        iz = int(rows[0][0, 2])
        xlow, xhigh = int(rows[3][0, 0]), int(rows[1][0, 0])
        ylow, yhigh = int(rows[0][0, 1]), int(rows[2][0, 1])
        balance_x, balance_y = np.arange(xlow+1,xhigh+1), np.arange(ylow+1,yhigh+1)
        current_bounds = np.asarray([[solver.scene.x_half64[xlow],solver.scene.x_half64[xhigh]],
            [solver.scene.y_half64[ylow],solver.scene.y_half64[yhigh]],
            [solver.scene.z_half64[iz],solver.scene.z_half64[iz]]])
        for i in balance_x:
            for j in balance_y:
                name = f'balance_normal_E_{i}_{j}'
                probe = csx.AddProbe(name, p_type=0, weight=1. if args.matched_current_source else -1.)
                probe.AddBox(start=[solver.scene.x_nodes64[i], solver.scene.y_nodes64[j], solver.scene.z_nodes64[iz]],
                             stop=[solver.scene.x_nodes64[i], solver.scene.y_nodes64[j], solver.scene.z_nodes64[iz+1]])
                balance_probes.append((name, int(i), int(j),
                    float(solver.scene.dx_dual64[i]*solver.scene.dy_dual64[j])))
    current_probe.AddBox(start=current_bounds[:, 0], stop=current_bounds[:, 1])
    csx.Write2XML(str(args.output / "reference-geometry.xml"))
    configuration_path = args.output / "reference-simulation.xml"
    previous_xml = configuration_path.read_bytes() if configuration_path.exists() else None
    external.Write2XML(str(configuration_path))
    if args.reuse_raw and args.material_setup_only:
        raise ValueError("Material setup requires an independently executed coefficient dump.")
    if args.reuse_raw and previous_xml != configuration_path.read_bytes():
        raise ValueError("Reference configuration changed; existing raw traces cannot be reused.")
    start = time.perf_counter()
    original_directory = Path.cwd()
    try:
        if not args.reuse_raw:
            external.Run(str(args.output / "reference-raw"), cleanup=False, verbose=0, numThreads=args.threads,
                debug_material=args.dump_reference_material or args.material_setup_only,
                setup_only=args.material_setup_only,
                debug_operator=args.dump_reference_material or args.material_setup_only)
    finally:
        os.chdir(original_directory)
    reference_seconds = time.perf_counter() - start
    if args.material_setup_only:
        actual = package_identity(mw, "witwin-maxwell")
        if actual != source_identity or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != driver_sha256:
            raise RuntimeError("Executed source changed during coefficient export.")
        record = {"kind": "actual_native_material_coefficients", "time_steps_executed": 0,
            "maxwell_dielectric_subpixel": args.dielectric_subpixel,
            "fixture_geometry_precision": "declared SI float64 Box dimensions",
            "source_identity": source_identity, "source_identity_unchanged": True,
            "driver_sha256": driver_sha256, "fixture_sha256": fixture_sha256,
            "grid_shape": [len(getattr(solver.scene, axis+"_nodes64")) for axis in "xyz"],
            "reference_setup_seconds": reference_seconds}
        (args.output / "material-setup.json").write_text(json.dumps(record, indent=2)+"\n")
        return
    execution_path = args.output / "reference-execution.json"
    if not args.reuse_raw:
        execution_path.write_text(json.dumps({"wall_seconds": reference_seconds, "steps": reference_steps, "dt_s": reference_dt}, indent=2) + "\n")
    else:
        reference_seconds = json.loads(execution_path.read_text())["wall_seconds"] if execution_path.exists() else None
    if port is not None:
        port.CalcPort(str(args.output / "reference-raw"), frequencies, ref_impedance=50.)
    average_voltage = 0.
    for name, weight in zip(names, averaging_weights):
        data = np.loadtxt(args.output / "reference-raw" / name, comments="%")
        average_voltage += weight * DFT_time2freq(data[:, 0], data[:, 1], frequencies, signal_type="pulse")
    data = np.loadtxt(args.output / "reference-raw" / current_name, comments="%")
    contour_current = DFT_time2freq(data[:, 0], data[:, 1], frequencies, signal_type="pulse")
    if args.matched_current_source:
        # The native observer is minus physical right-hand H circulation.
        contour_current = -contour_current
    own_port = own.port("feed")
    known_current = None
    if args.matched_current_source:
        from scripts.dev.fdtd.matched_current_reference import known_current_spectrum
        data = np.loadtxt(args.output/'reference-raw'/'et', comments='%')
        known_current = known_current_spectrum(data[:, 0], data[:, 1], frequencies,
            dt=reference_dt, amplitude=args.reference_current_amplitude)
        reference_default_z = averaged_z = np.conj(average_voltage)/known_current
        gamma = (averaged_z-50.)/(averaged_z+50.)
    else:
        gamma = np.conj(port.uf_ref / port.uf_inc)
        reference_default_z = np.conj(port.uf_tot/port.if_tot)
        averaged_z = np.conj(average_voltage / port.if_tot)
    own_z = own_port.z_in.cpu().numpy()
    gamma_own = (own_z-50.)/(own_z+50.)
    contour_z = np.conj(average_voltage / contour_current)
    pinning = {}
    for path in (args.reference_source, args.reference_source / "CSXCAD", args.reference_source / "openEMS"):
        pinning[str(path)] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    record = {"schema": "independent-patch-reference/1",
        "maxwell_source": source_identity,
        "maxwell_source_sha256": source_identity["source_sha256"],
        "source_unchanged": source_identity["source_sha256"] == package_identity(mw, "witwin-maxwell")["source_sha256"],
        "fixture_sha256": fixture_sha256,
        "driver_sha256": driver_sha256,
        "driver_unchanged": driver_sha256 == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maxwell_dielectric_subpixel": args.dielectric_subpixel,
            "fixture_geometry_precision": "declared SI float64 Box dimensions",
        "physical_load_resistance_ohm": args.load_resistance,
        "fixture_unchanged": fixture_sha256 == hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "reference": {"pins": pinning, "python": reference_package.__file__},
        "geometry": geometry_records, "grid_nodes_m": {axis: getattr(solver.scene, axis + "_nodes64").tolist() for axis in "xyz"},
        "geometry_configuration": {"lateral_dl": args.lateral_dl, "substrate_cells": args.substrate_cells,
                                   "edge_dl": args.edge_dl, "pml_layers": args.pml_layers, "air_padding": args.air_padding},
        "frequency_hz": frequencies, "steps": steps, "dt_s": solver.dt, "duration_s": steps * solver.dt,
        "reference_steps": reference_steps, "reference_dt_s": reference_dt,
        "reference_duration_s": reference_steps * reference_dt,
        "source_expression": expression, "reference_voltage_sign": "plus E integral ground to probe" if args.matched_current_source else "minus E integral ground to probe", "reference_impedance_ohm": 50.,
        "reference_default_z": serial(reference_default_z), "reference_aperture_z": serial(averaged_z),
        "reference_declared_contour_z": serial(contour_z), "reference_reflection": serial(gamma),
        "maxwell_z": serial(own_port.z_in.cpu().numpy()), "maxwell_reflection": serial(gamma_own),
        "max_complex_reflection_difference": float(abs(gamma-gamma_own).max()),
        "reference_best_match_hz": frequencies[int(abs(gamma).argmin())],
        "maxwell_best_match_hz": frequencies[int(abs(gamma_own).argmin())],
        "maxwell_seconds": own_seconds, "reference_seconds": reference_seconds, "reused_reference_raw": args.reuse_raw,
        "reference_phasor_conversion": "Conjugate native exp(-i omega t) projection into Maxwell exp(+i omega t) convention.",
        "voltage_aperture_weights": averaging_weights, "physical_qualification": "comparison run; convergence and port-contract differences must be reviewed"}
    if args.matched_current_source:
        from dataclasses import asdict
        record['known_current_source'] = {
            'scope': 'Independently derived physical impressed-current support with no local distributed resistor; intrinsic Z is measured V/I_known.',
            'reference_amplitude_a': args.reference_current_amplitude,
            'reference_known_current': serial(known_current),
            'reference_edges': [asdict(edge) for edge in known_edges],
            'reference_H_sign': 'minus physical right-hand H circulation, matching native observer',
            'source_clock': 'et timestamp plus dt/2; verified by independent single-edge Ampere experiments',
            'max_native_feedback_impedance_ratio': float(abs(own_z/args.load_resistance).max()),
            'reference_has_local_resistor': False}
        record['known_current_source']['helper_sha256'] = current_helper_sha256
        record['known_current_source']['helper_unchanged'] = (
            current_helper_sha256 == hashlib.sha256(current_helper_path.read_bytes()).hexdigest())
    if args.record_contour_control:
        observed = own.port('contour_control')
        observed_z = observed.z_in.cpu().numpy()
        observed_gamma = (observed_z-50.)/(observed_z+50.)
        record['maxwell_observer_contour_z'] = serial(observed_z)
        record['maxwell_observer_contour_reflection'] = serial(observed_gamma)
        record['max_observer_contour_reflection_difference'] = float(abs(
            observed_gamma-(contour_z-50)/(contour_z+50)).max())
    if args.record_current_balance:
        # With V = +integral(E_native) in Maxwell and the reference U probe's
        # negative line weight, both compared voltage phasors share this sign.
        # Ampere's law gives I_branch = I_H + d/dt integral(D_native.dA).
        dielectric = 2.2*8.8541878128e-12
        reference_displacement = np.zeros(len(frequencies), dtype=np.complex128)
        own_displacement = np.zeros_like(reference_displacement)
        own_normal = own.raw_monitor('current_balance_E')['components']['Ez']['data'].detach().cpu().numpy()
        for name,i,j,area in balance_probes:
            data = np.loadtxt(args.output/'reference-raw'/name, comments='%')
            voltage = np.conj(DFT_time2freq(data[:, 0], data[:, 1], frequencies, signal_type='pulse'))
            reference_displacement += dielectric*area*voltage/solver.scene.dz_primal64[iz]
            own_displacement += dielectric*area*own_normal[:, i, j]
        reference_derivative = -2j*np.sin(np.pi*np.asarray(frequencies)*reference_dt)/reference_dt
        own_derivative = -2j*np.sin(np.pi*np.asarray(frequencies)*solver.dt)/solver.dt
        reference_branch_current = np.conj(contour_current)+reference_derivative*reference_displacement
        own_recovered_current = observed.current.detach().cpu().numpy()+own_derivative*own_displacement
        reference_branch_z = np.conj(average_voltage)/reference_branch_current
        reference_branch_gamma = (reference_branch_z-50)/(reference_branch_z+50)
        record['ampere_current_balance'] = {
            'scope': 'Independent normal-E flux across the entire declared native H contour; no fitted capacitance or source renormalization.',
            'electric_probe_count': len(balance_probes), 'normal_cell_bounds_m': solver.scene.z_nodes64[iz:iz+2].tolist(),
            'physical_area_m2': sum(row[3] for row in balance_probes),
            'maxwell_D_flux': serial(own_displacement), 'reference_D_flux': serial(reference_displacement),
            'maxwell_recovered_branch_current': serial(own_recovered_current),
            'maxwell_native_branch_current': serial(own_port.current.detach().cpu().numpy()),
            'maxwell_current_balance_relative_l2': float(np.linalg.norm(own_recovered_current-own_port.current.detach().cpu().numpy())/np.linalg.norm(own_port.current.detach().cpu().numpy())),
            'reference_recovered_branch_current': serial(reference_branch_current),
            'reference_recovered_branch_z': serial(reference_branch_z),
            'reference_recovered_branch_reflection': serial(reference_branch_gamma),
            'max_recovered_branch_reflection_difference': float(abs(gamma_own-reference_branch_gamma).max()),
            'qualification': 'Diagnostic current-definition control; native time/area balance and external convergence are evaluated independently.'}
        if known_current is not None:
            record['ampere_current_balance']['reference_known_current_balance_relative_l2'] = float(
                np.linalg.norm(reference_branch_current-known_current)/np.linalg.norm(known_current))
    (args.output / "report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if (not record["source_unchanged"] or not record["fixture_unchanged"] or not record['driver_unchanged']
            or not record.get('known_current_source', {}).get('helper_unchanged', True)):
        raise RuntimeError("Source changed during the reference campaign; retain diagnostic data but do not qualify them.")
    print(json.dumps({k: record[k] for k in ("max_complex_reflection_difference", "reference_best_match_hz", "maxwell_best_match_hz", "maxwell_seconds", "reference_seconds")}), flush=True)


if __name__ == "__main__":
    main()
