"""Check actual prepared source polarity/support before root-owned CPU runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import witwin.core
import witwin.maxwell as mw
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.printed_line_broadband import physical_sources
from scripts.dev.fdtd.printed_line_external_reference import reference_line_scene
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xml_source_check(preparation, xml_path, ports, compiled):
    xml = ET.parse(xml_path).getroot()
    resistors = {int(item.get('Name').removeprefix('port_resist_')): item
        for item in xml.iter('LumpedElement') if item.get('Name', '').startswith('port_resist_')}
    excitation = {int(item.get('Name').removeprefix('port_excite_')): item
        for item in xml.iter('Excitation') if item.get('Name', '').startswith('port_excite_')}
    sources = preparation['source_contract']
    if set(resistors) != set(range(1, len(ports) + 1)) or len(sources) != len(ports):
        raise ValueError('Prepared resistor/source support does not cover every actual ground gap.')
    expected_active = {index + 1 for index, source in enumerate(sources) if source['active']}
    if set(excitation) != expected_active:
        raise ValueError('XML excitation set differs from the complete declared coherent drive group.')
    rows = []
    for index, (source, port, geometry) in enumerate(zip(sources, ports, compiled), 1):
        resistor = resistors[index]
        axis = 'xyz'.index(source['voltage_axis'])
        if float(resistor.get('R')) != 50. or int(resistor.get('Direction')) != axis:
            raise ValueError('Actual passive resistor or orientation changed.')
        box = resistor.find('Primitives/Box')
        points = np.array([[float(box.find(name).get(axis.upper())) for axis in 'xyz'] for name in ('P1', 'P2')])
        expected = np.array([source['start_m'], source['stop_m']])
        if not np.array_equal(points, expected): raise ValueError('Actual XML source/load box differs from declared physical support.')
        center, size = [value.detach().cpu().numpy() for value in (port.voltage_aperture.position, port.voltage_aperture.size)]
        native_bounds = np.stack((center - size / 2, center + size / 2))
        bounds = np.stack((points.min(0), points.max(0)))
        delta = float(np.max(abs(native_bounds[:, 1:] - bounds[:, 1:])))
        tolerance = 32 * np.spacing(max(float(np.max(abs(bounds[:, 1:]))), .001))
        if delta > tolerance or geometry.voltage_component != 'E' + source['voltage_axis']:
            raise ValueError('Native/reference transverse physical source control differs.')
        native_sign = int(torch.sign(geometry.voltage_weights.sum()))
        reference_sign = -int(np.sign(expected[1, axis] - expected[0, axis]))
        if source['active']:
            active = excitation[index]
            vector = np.array(list(map(float, active.get('Excite').split(','))))
            required = np.zeros(3); required[axis] = reference_sign
            if not np.array_equal(vector, required) or float(active.get('Delay')) != 0. or active.get('Enabled') != '1':
                raise ValueError('Actual relative source polarity, coherence or delay is invalid.')
        rows.append({'native_port': port.name, 'axis': source['voltage_axis'],
            'native_voltage_row_sign': native_sign, 'reference_electric_source_sign': reference_sign,
            'reference_to_native_positive_drive_orientation': reference_sign / native_sign,
            'transverse_support_max_delta_m': delta, 'active': source['active'],
            'reference_axial_width_m': float(bounds[1, 0] - bounds[0, 0]),
            'native_axial_width_m': float(size[0])})
    if any(row['reference_to_native_positive_drive_orientation'] != -1 for row in rows):
        raise ValueError('Source orientation is not the declared common unfitted -1 conversion.')
    if preparation['kind'] in ('stripline', 'cpw'):
        if len(rows) != 4 or preparation['drive_groups'] != [[0, 1], [2, 3]]:
            raise ValueError('Balanced three-conductor reference requires both ground gaps at both ends.')
        for left, right in ((0, 1), (2, 3)):
            if rows[left]['reference_electric_source_sign'] != -rows[right]['reference_electric_source_sign']:
                raise ValueError('Ground-exchange source symmetry is broken.')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--coarse-report', type=Path, required=True)
    parser.add_argument('--kind', required=True)
    parser.add_argument('--dx', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    coarse = json.loads(args.coarse_report.read_text())
    files = sorted(args.prepared.glob('drive-*/preparation.json'))
    expected_count = 4 if args.kind == 'differential_pair' else 2
    if len(files) != expected_count: raise ValueError('Incomplete prepared independent drive basis.')
    scene_args = SimpleNamespace(kind=args.kind, dx=args.dx, longitudinal_dl=.0025,
        transverse_refinement_radius=.02, padding=.08, separation=.4, axial_half_length=.3, sigma=0.)
    scene = reference_line_scene(scene_args)
    ports, groups = physical_sources(scene, args.kind, .0025)
    prepared = prepare_scene(scene.clone(ports=ports))
    compiled = [compile_port_geometry(prepared, port, device='cpu') for port in ports]
    identity = package_identity(mw, 'witwin-maxwell')
    core_root = Path(next(iter(witwin.core.__path__))).resolve()
    core_files = {str(path.relative_to(core_root)): sha(path) for path in sorted(core_root.rglob('*'))
        if path.is_file() and path.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'}}
    records = []
    for path in files:
        p = json.loads(path.read_text()); xml = path.with_name('simulation.xml')
        if p['maxwell_source_sha256'] != identity['source_sha256'] or p['geometry'] != coarse['geometry']:
            raise ValueError('Prepared numerical source or physical material geometry differs.')
        nodes = {axis: np.asarray(p['grid_nodes_m'][axis]) for axis in 'xyz'}
        old = {axis: np.asarray(coarse['grid_nodes_m'][axis]) for axis in 'xyz'}
        if not np.array_equal(nodes['x'], old['x']): raise ValueError('Axial grid changed under transverse refinement.')
        exterior = {axis: np.array_equal(nodes[axis][abs(nodes[axis]) > .020000000001], old[axis][abs(old[axis]) > .020000000001]) for axis in 'yz'}
        if not all(exterior.values()): raise ValueError('Actual vacuum/PML exterior grid moved under refinement.')
        if args.kind == 'differential_pair' and p['source_contract'] != coarse['runs'][p['active_drive']]['source_contract']:
            raise ValueError('Coupled-pair physical source/termination changed under refinement.')
        source_rows = xml_source_check(p, xml, ports, compiled)
        nyquist_steps = math.floor(1 / (2 * 1.8e9 * p['dt_s']))
        sample_interval = max(1, nyquist_steps // 4)
        samples_bound = math.ceil(p['steps'] / sample_interval) + 2
        streams_bound = len(p['probe_contract']) + 2 * len(p['source_contract'])
        records.append({'preparation': str(path), 'preparation_sha256': sha(path), 'xml_sha256': sha(xml),
            'active_drive': p['active_drive'], 'source_rows': source_rows,
            'axial_grid_bitwise_equal_to_coarse': True, 'exterior_grid_bitwise_equal_to_coarse': exterior,
            'native_shape': [len(nodes[axis]) for axis in 'xyz'], 'actual_dt_s': p['dt_s'],
            'steps': p['steps'], 'native_probe_count': len(p['probe_contract']),
            'primary_probe_sample_interval_steps': sample_interval, 'sample_count_upper': samples_bound,
            'ascii_output_upper_bytes': streams_bound * (samples_bound * 4 * 24 + 2048)})
    total = sum(record['ascii_output_upper_bytes'] for record in records) + 512 * 1024 ** 2
    if total >= 5 * 1024 ** 3: raise ValueError('Bounded line-reference disk estimate exceeds 5 GiB.')
    if torch.cuda.is_initialized(): raise RuntimeError('CPU preflight initialized CUDA.')
    record = {'schema': 'independent-reference-next-physical-preflight/1', 'kind': args.kind,
        'status': 'CPU preparation only: no new field solve or mode suppression claimed',
        'source': identity, 'actual_core_root': str(core_root),
        'actual_core_sha256': hashlib.sha256(json.dumps(core_files, sort_keys=True).encode()).hexdigest(),
        'coarse_reference_sha256': sha(args.coarse_report), 'drive_records': records,
        'total_ascii_output_upper_bytes_including_512MiB_margin': total,
        'expected_ground_difference_control': 'stripline/CPW actual raw even-to-target-odd magnetic norm <=1e-3 at every frequency and drive; still requires field execution',
        'source_scope': 'Exact transverse gap/symmetry and passive supports; common known -1 source orientation, zero-width external vs5mm native axial support outside ROI. No incident-field amplitude/phase or S matching.',
        'cuda_initialized': False, 'driver_sha256': sha(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'kind': args.kind, 'drives': len(records), 'disk_upper_bytes': total}), flush=True)


if __name__ == '__main__': main()
