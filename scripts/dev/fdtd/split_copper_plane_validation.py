"""Qualified-scope audit and explicitly scheduled CUDA split-plane execution.

The optional host mode assembles harmonic algebra only. Native time-domain
acceptance is populated solely by actual PyTorch CUDA field advancement.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from scripts.dev.fdtd.split_copper_plane import (
    SplitPlaneCUDA, compile_split_plane, frequency_oracle,
    independent_film_reference, compile_local_cn_support, local_cn_frequency_oracle,
)
from witwin.maxwell.copper_foil import CopperFoil, MU_0


GATES = dict(reflection_absolute_error_max=.002,
             transmission_relative_error_max=.03,
             absorption_relative_error_max=.03)


def source_identity():
    root = Path(__file__).resolve().parents[3]
    suffixes = {'.py', '.cu', '.cpp', '.h', '.cuh'}
    paths = sorted(path for path in (root/'witwin/maxwell').rglob('*')
                   if path.is_file() and path.suffix in suffixes)
    paths += [Path(__file__).resolve(), Path(__file__).with_name('split_copper_plane.py').resolve(),
              root/'tests/materials/surface_impedance/test_split_copper_plane.py']
    control = root/'split-cut-host-validation-control.py'
    if control.exists():
        paths.append(control)
    for relative in ('split-local-cn-host-control.py',
                     'tests/materials/surface_impedance/test_split_copper_local_cn.py'):
        path = root/relative
        if path.exists():
            paths.append(path)
    identities = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths}
    import witwin.core
    core = Path(witwin.core.__file__).resolve().parent
    for path in sorted(path for path in core.rglob('*') if path.is_file() and path.suffix in suffixes):
        identities['installed_witwin_core/'+str(path.relative_to(core))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identities


def imported_package_identity():
    import importlib.metadata
    import witwin.core
    import witwin.maxwell
    return dict(maxwell_import_file=str(Path(witwin.maxwell.__file__).resolve()),
        core_import_file=str(Path(witwin.core.__file__).resolve()),
        core_distribution_version=importlib.metadata.version('witwin'),
        signed_source_suffixes=['.py', '.cu', '.cpp', '.h', '.cuh'])


def scalar_metrics(reflection, transmission, absorption, reference):
    r, t, a = reference
    errors = dict(reflection_absolute_error=abs(reflection-r),
        transmission_relative_error=abs(transmission-t)/abs(t),
        absorption_relative_error=abs(absorption-a)/a)
    return dict(reflection=[reflection.real, reflection.imag],
        transmission=[transmission.real, transmission.imag], absorption=absorption,
        **errors, original_plane_gates_pass=all(errors[key] <= GATES[key+'_max'] for key in errors))


def run_cuda(geometry, material, dt, frequency, args, implicit_bounds=None):
    if os.environ.get('WITWIN_RUN_SPLIT_CUT_CUDA') != '1':
        raise RuntimeError('Actual field advance requires WITWIN_RUN_SPLIT_CUT_CUDA=1 in an authorized GPU window.')
    runtime = SplitPlaneCUDA(geometry, material, dt, frequency,
        scheme=args.scheme.replace('-', '_'), implicit_bounds=implicit_bounds)
    ramp_end = args.ramp_cycles/frequency
    measure_start_time = (args.ramp_cycles+args.settle_cycles)/frequency
    total_time = measure_start_time+args.measurement_cycles/frequency
    count = math.ceil(total_time/dt)
    start = math.ceil(measure_start_time/dt)
    device = runtime.electric.device
    times = torch.arange(count+1, dtype=torch.float64, device=device)*dt
    phase = cmath.exp(1j*runtime.k*(float(geometry.electric_positions[0])-geometry.cut))
    progress = (times/ramp_end).clamp(0, 1)
    envelope = progress**3*(10-15*progress+6*progress**2)
    source = envelope*torch.exp(-2j*math.pi*frequency*times)*phase
    midpoint_source = (source[:-1]+source[1:])/2
    weight = torch.hann_window(count-start, periodic=False, dtype=torch.float64, device=device)
    weight = weight/weight.sum()
    phase_weight = weight*torch.exp(2j*math.pi*frequency*times[start+1:])
    phasor = torch.zeros_like(runtime.electric)
    magnetic_phasor = torch.zeros_like(runtime.magnetic)
    magnetic_clock_phase = torch.ones(len(runtime.magnetic), dtype=torch.complex128, device=device)
    magnetic_clock_phase[runtime.explicit] = cmath.exp(-1j*math.pi*frequency*dt)
    magnetic_clock_offset = torch.zeros(len(runtime.magnetic), dtype=torch.float64)
    magnetic_clock_offset[runtime.explicit.cpu()] = -dt/2
    state_phasor = torch.zeros_like(runtime.state)
    torch.cuda.synchronize()
    wall_start = time.monotonic()
    with torch.no_grad():
        initial_energy = runtime.energy().clone()
        for step in range(count):
            runtime.step(midpoint_source[step])
            if step >= start:
                phasor.add_(runtime.electric*phase_weight[step-start])
                # Explicit exterior H is a Yee half step behind E; the implicit
                # CN patch H is synchronized to E. Keep their actual clocks.
                magnetic_phasor.add_(runtime.magnetic*phase_weight[step-start]*magnetic_clock_phase)
                state_phasor.add_(runtime.state*phase_weight[step-start])
            if (step+1) % max(1, count//10) == 0:
                print(json.dumps(dict(stage='native_cuda_field_advance', spacing_min_m=float(geometry.magnetic_lengths.min()),
                    cut_m=geometry.cut, completed_steps=step+1, total_steps=count)), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-wall_start
    xlow = float(geometry.electric_positions[0])-geometry.cut
    xhigh = float(geometry.electric_positions[-1])-geometry.cut
    reflection = (complex(phasor[0].cpu())-phase)*cmath.exp(1j*runtime.k*xlow)
    transmission = complex(phasor[-1].cpu())*cmath.exp(-1j*runtime.k*xhigh)
    absorption = 1-abs(reflection)**2-abs(transmission)**2
    trace_e = phasor[runtime.trace]
    current = runtime.algebra['D'].to(trace_e.dtype)@trace_e + runtime.algebra['B'].T.to(trace_e.dtype)@state_phasor
    heat_absorption = float(torch.vdot(trace_e, current).real.cpu())/runtime.boundary_y
    energy_change = runtime.energy()-initial_energy
    balance = energy_change+runtime.cumulative_heat+runtime.cumulative_boundary_loss-runtime.cumulative_input_work
    balance_relative = float((balance.abs()/runtime.cumulative_input_work.abs()).cpu())
    return dict(mode='research_only_1d_pytorch_cuda_time_domain_without_Scene_3d_integration',
        **scalar_metrics(reflection, transmission, absorption,
            independent_film_reference(material.thickness, material.conductivity, frequency)),
        heat_absorption=heat_absorption, energy_work_relative_residual=balance_relative,
        timesteps=count, dt_s=dt, measured_samples=count-start,
        wall_seconds_including_python_dispatch=elapsed,
        timing_qualification='functional shared-device evidence; no performance acceptance',
        device_name=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        state_dtype=str(runtime.state.dtype), field_dtype=str(runtime.electric.dtype),
        stepping_scheme=args.scheme, magnetic_clock_offsets_relative_E_s=magnetic_clock_offset.tolist(),
        electric_phasors=torch.view_as_real(phasor.cpu()).tolist(),
        magnetic_phasors_at_declared_physical_clocks=torch.view_as_real(magnetic_phasor.cpu()).tolist(),
        skin_state_phasors=torch.view_as_real(state_phasor.cpu()).tolist(),
        surface_electric_phasors=torch.view_as_real(trace_e.cpu()).tolist(),
        surface_inward_current_phasors=torch.view_as_real(current.cpu()).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-audit', action='store_true', help='Geometry and harmonic algebra only; no CPU time stepping.')
    parser.add_argument('--scheme', choices=('explicit', 'local-cn'), default='explicit')
    parser.add_argument('--courant', type=float)
    parser.add_argument('--offsets', type=float, nargs='+', default=(0., .3, .7))
    parser.add_argument('--spacings', type=float, nargs='+', default=(.005, .0025))
    parser.add_argument('--frequency', type=float, default=1e9)
    parser.add_argument('--half-length', type=float, default=.2)
    parser.add_argument('--skin-depths', type=float, default=1.)
    parser.add_argument('--ramp-cycles', type=float, default=8.)
    parser.add_argument('--settle-cycles', type=float, default=32.)
    parser.add_argument('--measurement-cycles', type=float, default=16.)
    args = parser.parse_args()
    if args.courant is None:
        args.courant = .9 if args.scheme == 'local-cn' else .4
    if args.output.exists():
        raise ValueError('Refusing to replace an existing validation record.')
    if min(args.spacings) <= 0 or not all(0 <= offset <= 1 for offset in args.offsets):
        raise ValueError('Spacings must be positive and physical cut offsets lie in [0, 1].')
    if min(args.ramp_cycles, args.settle_cycles, args.measurement_cycles) <= 0:
        raise ValueError('The predeclared ramp, settling, and measurement windows must be positive.')
    sigma = 5.8e7
    thickness = args.skin_depths*math.sqrt(2/(2*math.pi*args.frequency*MU_0*sigma))
    foil = CopperFoil(thickness=thickness, conductivity=sigma,
                      frequency_range=(args.frequency/2, args.frequency*2))
    reference = independent_film_reference(thickness, sigma, args.frequency)
    before = source_identity()
    report = dict(scope='1D true split-cell diagnostic; public SurfaceMesh+Cu preparation remains rejected',
        gates=GATES, parameters=vars(args)|{'output':str(args.output)},
        physical_reference=dict(reflection=[reference[0].real, reference[0].imag],
            transmission=[reference[1].real, reference[1].imag], absorption=reference[2]),
        physical_material=dict(thickness_m=thickness, conductivity_S_per_m=sigma,
            frequency_Hz=args.frequency, frequency_range_Hz=list(foil.frequency_range)),
        imported_packages_before=imported_package_identity(),
        source_sha256_before=before, rows=[], research_time_domain_completed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for spacing in args.spacings:
        half_intervals = round(args.half_length/spacing)
        if half_intervals < 2 or not math.isclose(half_intervals*spacing, args.half_length,
                                                rel_tol=1e-14, abs_tol=0):
            raise ValueError('The requested spacing must cover the same fixed physical half-length exactly.')
        nodes = torch.arange(-half_intervals, half_intervals+1, dtype=torch.float64)*spacing
        for offset in args.offsets:
            geometry = compile_split_plane(nodes, offset*spacing)
            bounds = (-spacing, 2*spacing) if args.scheme == 'local-cn' else None
            dt = (geometry.bulk_stable_dt(args.courant) if args.scheme == 'local-cn'
                  else geometry.stable_dt(args.courant))
            oracle = (local_cn_frequency_oracle(geometry, foil, dt, args.frequency, bounds)
                      if args.scheme == 'local-cn' else frequency_oracle(geometry, foil, dt, args.frequency))
            row = dict(spacing_m=spacing, offset_cells=offset, cut_m=geometry.cut,
                aligned_branch=geometry.aligned, electric_dofs=len(geometry.electric_positions),
                magnetic_dofs=len(geometry.magnetic_positions),
                minimum_magnetic_interval_m=float(geometry.magnetic_lengths.min()),
                geometry=dict(original_nodes_m=geometry.original_nodes.tolist(),
                    electric_positions_m=geometry.electric_positions.tolist(),
                    magnetic_positions_m=geometry.magnetic_positions.tolist(),
                    electric_dual_lengths_m=geometry.electric_lengths.tolist(),
                    magnetic_interval_lengths_m=geometry.magnetic_lengths.tolist(),
                    edge_low=geometry.edge_low.tolist(), edge_high=geometry.edge_high.tolist(),
                    independent_trace_indices=geometry.traces.tolist()),
                dt_s=dt, harmonic_algebra=dict(
                    **scalar_metrics(oracle['reflection'], oracle['transmission'], oracle['absorption'], reference),
                    heat_absorption=oracle['heat_absorption'], harmonic_residual=oracle['harmonic_residual']))
            if bounds is not None:
                support = compile_local_cn_support(geometry, bounds)
                row['compatible_local_cn'] = dict(implicit_bounds_m=list(bounds),
                    electric_dofs=support['electric'].tolist(), magnetic_dofs=support['magnetic'].tolist(),
                    explicit_magnetic_dofs=support['explicit'].tolist(),
                    unregularized_explicit_cut_dt_s=geometry.stable_dt(args.courant),
                    bulk_over_cut_dt=dt/geometry.stable_dt(args.courant),
                    harmonic_matrix_condition=oracle['harmonic_matrix_condition'])
                clocks = torch.zeros(len(geometry.magnetic_lengths), dtype=torch.float64)
                clocks[support['explicit']] = -dt/2
                row['compatible_local_cn']['algorithmic_magnetic_time_offsets_relative_E_s'] = clocks.tolist()
            if not args.cpu_audit:
                row['cuda_research'] = run_cuda(geometry, foil, dt, args.frequency, args, bounds)
            report['rows'].append(row)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(spacing_m=spacing, offset_cells=offset,
                harmonic_algebra=row['harmonic_algebra'],
                cuda_research_original_plane_gates_pass=row.get('cuda_research', {}).get('original_plane_gates_pass'))), flush=True)
    after = source_identity()
    report['source_sha256_after'] = after
    report['imported_packages_after'] = imported_package_identity()
    report['source_unchanged'] = before == after
    report['research_time_domain_completed'] = not args.cpu_audit and len(report['rows']) == len(args.spacings)*len(args.offsets)
    report['research_original_plane_gates_pass'] = (all(row['cuda_research']['original_plane_gates_pass']
        and row['cuda_research']['energy_work_relative_residual'] < 5e-8 for row in report['rows'])
        if report['research_time_domain_completed'] else None)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['source_unchanged']:
        raise RuntimeError('The private validation source changed during execution.')


if __name__ == '__main__':
    main()
