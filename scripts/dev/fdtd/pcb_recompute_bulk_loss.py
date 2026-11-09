"""Recompute static bulk work from retained native application fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import witwin.maxwell as mw
from witwin.maxwell.adapters.pcb import import_pcb_scene
from scripts.dev.fdtd.pcb_acceptance import package_identity
from scripts.dev.fdtd.pcb_application_campaign import power_observations


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--campaign', type=Path)
    inputs.add_argument('--surface-report', type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow-unbound-input-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Keep prior reduction evidence; select a new output path.')
    identity = package_identity(mw, 'witwin-maxwell')
    if args.surface_report:
        original = json.loads(args.surface_report.read_text())
        path = Path(original['input'])
        input_bound = 'input_sha256' in original
        if not input_bound and not args.allow_unbound_input_diagnostic:
            raise ValueError('Original run did not bind its input hash; explicitly request a diagnostic-only reduction.')
        if input_bound and digest(path) != original['input_sha256']:
            raise ValueError('The original enclosing-surface input changed.')
        definition = json.loads(path.read_text())
        scene = import_pcb_scene(definition, frequency=2.45e9, device=args.device)
        scene = scene.clone(monitors=power_observations(definition, scene, 2.45e9))
        result_path = args.surface_report.parent/'result.pt'
        result_hash = digest(result_path)
        result = mw.Result.load(result_path, scene=scene, map_location=args.device)
        bulk = float(result.power_loss('bulk_loss').conduction[0])
        port = result.port(scene.ports[0].name)
        port_values = {key: {'real': value.real.detach().cpu().tolist(),
                             'imag': value.imag.detach().cpu().tolist()}
            for key, value in (('voltage', port.voltage), ('current', port.current),
                               ('reflection', port.reflection_coefficient))}
        if digest(result_path) != result_hash:
            raise RuntimeError('The retained enclosing-surface result changed.')
        records = []
        for row in original['records']:
            records.append({'original': row, 'native_bulk_conduction_w': bulk,
                'relative_power_closure': abs(row['radiated_w']+bulk+row['copper_loss_w']-row['accepted_w'])/row['accepted_w'],
                'relative_flux_closure': abs(row['surface_flux_w']+bulk+row['copper_loss_w']-row['accepted_w'])/row['accepted_w']})
        report = {'schema': 'retained-surface-native-bulk-loss/1', 'reduction_source': identity,
            'original_report': str(args.surface_report.resolve()), 'original_report_sha256': digest(args.surface_report),
            'original_field_source': original['source'], 'original_source_unchanged': original['source_unchanged'],
            'original_input_hash_recorded': input_bound,
            'qualification': 'reduction only; original physical convergence remains separate' if input_bound else 'diagnostic only; original input hash was not recorded',
            'input_sha256': digest(path), 'result_sha256': result_hash, 'records': records,
            'field_reexecution': False,
            'retained_port': port_values,
            'source_unchanged': identity['source_sha256'] == package_identity(mw, 'witwin-maxwell')['source_sha256']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not report['source_unchanged']:
            raise RuntimeError('Reduction source changed; this record cannot qualify.')
        print(json.dumps({'output': str(args.output), 'records': records}), flush=True)
        return
    if args.input_dir is None:
        parser.error('--campaign requires --input-dir')
    original = json.loads(args.campaign.read_text())
    records = []
    for row in original['runs']:
        path = args.input_dir/f"metahz-scene-{row['family']}.json"
        if digest(path) != row['input_sha256']:
            raise ValueError('The original application input changed.')
        definition = json.loads(path.read_text())
        scene = import_pcb_scene(definition, frequency=row['frequency_hz'], device=args.device,
            selected_conductors=row['selected_conductor_ids'])
        scene = scene.clone(monitors=power_observations(definition, scene, row['frequency_hz']))
        outputs = []
        for label in ('direct', 'adapter'):
            result_path = args.campaign.parent/f"{row['family']}-{label}.pt"
            result_hash = digest(result_path)
            result = mw.Result.load(result_path, scene=scene, map_location=args.device)
            loss = result.power_loss('bulk_loss').conduction
            outputs.append(loss.detach().cpu())
            if digest(result_path) != result_hash:
                raise RuntimeError('The retained result changed during reduction.')
            del result
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        before = row['power']
        accepted = torch.as_tensor(before['accepted_w'], dtype=torch.float64)
        radiation = torch.as_tensor(before['radiated_w'], dtype=torch.float64)
        copper = torch.as_tensor(before['copper_w'], dtype=torch.float64)
        closure = (radiation+outputs[0]+copper-accepted).abs()/accepted
        records.append({'family': row['family'], 'input_sha256': row['input_sha256'],
            'original_power': before, 'native_bulk_conduction_w': outputs[0].tolist(),
            'relative_power_closure': closure.tolist(), 'direct_adapter_bitwise_equal': True,
            'result_files': {label: {'path': str(args.campaign.parent/f"{row['family']}-{label}.pt"),
                'sha256': digest(args.campaign.parent/f"{row['family']}-{label}.pt")} for label in ('direct', 'adapter')},
            'field_reexecution': False})
        print(json.dumps({key: records[-1][key] for key in
            ('family', 'native_bulk_conduction_w', 'relative_power_closure')}), flush=True)
    report = {'schema': 'retained-native-bulk-loss/1', 'reduction_source': identity,
        'original_campaign': str(args.campaign.resolve()), 'original_campaign_sha256': digest(args.campaign),
        'original_field_source': original['source'], 'records': records,
        'source_unchanged': identity['source_sha256'] == package_identity(mw, 'witwin-maxwell')['source_sha256'],
        'scope': 'Only conductivity sampling in static bulk-work reduction changes; original fields, source, geometry, radiation and conductor spectra are retained.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not report['source_unchanged']:
        raise RuntimeError('Reduction source changed; this record cannot qualify.')


if __name__ == '__main__':
    main()
