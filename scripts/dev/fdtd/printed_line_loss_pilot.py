"""CPU preparation of one bounded physical line-loss pilot, without launching it."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import witwin.core
import witwin.maxwell as mw

from benchmark.scenes.rf.printed_lines import printed_line_scene
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.printed_line_broadband import configuration, encode, physical_sources
from scripts.dev.fdtd.printed_line_loss_basis import capture_loss_basis
from scripts.dev.fdtd.printed_line_regional_power import regional_monitors
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene


def prepare_pilot(kind, *, separation=.12, metal='copper', sigma=.01):
    """Compile real supports; all runtime timing/memory estimates remain open."""
    if separation not in (.12, .20):
        raise ValueError('The first loss pilot uses fixed 120mm or 200mm source separation.')
    args = SimpleNamespace(kind=kind, dx=.0025, longitudinal_dl=.0025,
        padding=.02 if kind == 'microstrip' else .08, separation=separation,
        axial_half_length=.14, transverse_refinement_radius=.02, sigma=sigma, metal=metal)
    scene = configuration(args, 'cpu')
    ports, groups = physical_sources(scene, kind, args.longitudinal_dl)
    scene = scene.clone(ports=ports)
    prepared = prepare_scene(scene)
    compiled = [compile_port_geometry(prepared, port, device='cpu') for port in prepared.ports]
    bounds = ((-.015, .015), *tuple(scene.domain.bounds[axis] for axis in (1, 2)))
    frequencies = tuple(float(value) for value in torch.linspace(.8e9, 1.2e9, 5, dtype=torch.float64))
    monitors, schema = regional_monitors(scene, bounds, frequencies, observation='planes')
    count = sum(selection['recorded_native_plane_edge_count'] for selection in schema.values())
    paths = {name: Path(inspect.getfile(function)) for name, function in (
        ('fixture', printed_line_scene), ('broadband', configuration),
        ('regional', regional_monitors), ('coherent_basis', capture_loss_basis))}
    paths['preparation'] = Path(__file__)
    core_root = Path(next(iter(witwin.core.__path__))).resolve()
    core_sources = {str(path.relative_to(core_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(core_root.rglob('*'))
        if path.is_file() and path.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'}}
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU pilot preparation unexpectedly initialized CUDA.')
    return {'schema': 'physical-line-loss-pilot-preparation/1',
        'status': 'CPU prepared only; CUDA execution, measured runtime and physical acceptance pending',
        'configuration': vars(args), 'frequencies_hz': frequencies,
        'planned_duration_s': 64e-9, 'checkpoint_target_s': 32e-9,
        'checkpoint_contract': 'Prepare the full 64ns horizon before running. A 32ns checkpoint resumes only that same planned horizon; it is not a completed 32ns Result.',
        'physical_domain_bounds_m': scene.domain.bounds,
        'prepared_grid_shape': [len(getattr(prepared, axis + '_nodes64')) for axis in 'xyz'],
        'native_grid_sha256': {axis: hashlib.sha256(np.asarray(getattr(prepared, axis + '_nodes64')).tobytes()).hexdigest() for axis in 'xyz'},
        'loss_roi_bounds_m': bounds, 'coherent_drive_groups': groups,
        'per_basis_drive_count': len(groups), 'first_execution_limit': 'one material and one length; its complete two-drive scalar basis, then review',
        'source_voltage_dofs': {port.name: len(compilation.voltage_weights) for port, compilation in zip(ports, compiled)},
        'regional_schema': schema, 'regional_monitor_count': len(monitors),
        'native_plane_complex128_output_bytes': count * len(frequencies) * 16,
        'single_frequency_dense_complex128_bytes': sum(torch.tensor(selection['full_shape']).prod().item() for selection in schema.values()) * 16,
        'source': package_identity(mw, 'witwin-maxwell'),
        'actual_core': {'root': str(core_root), 'sources': core_sources,
            'source_sha256': hashlib.sha256(json.dumps(core_sources, sort_keys=True).encode()).hexdigest()},
        'helper_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        'cuda_initialized': torch.cuda.is_initialized(),
        'scope': 'Preparation only. Plane spectral bytes exclude full-volume time fields, material tensors, PML, foil state and flux monitors. No throughput estimate or continuum qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('microstrip', 'stripline', 'cpw', 'differential_pair'), default='stripline')
    parser.add_argument('--separation', type=float, choices=(.12, .20), default=.12)
    parser.add_argument('--metal', choices=('pec', 'copper'), default='copper')
    parser.add_argument('--sigma', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing preparation evidence; choose a fresh output.')
    record = prepare_pilot(args.kind, separation=args.separation, metal=args.metal, sigma=args.sigma)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(encode(record), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
