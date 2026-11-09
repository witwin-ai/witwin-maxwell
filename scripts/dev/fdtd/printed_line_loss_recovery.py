"""Separate recovery receipt for the completed V3 first public CPW column.

That column saved its executed copper-operator signature but not every loss
basis value. This tool recompiles the realization through the ordinary public
``Simulation.prepare()`` (no field step), proves by SHA-256 that it reproduces
the executed copper-operator signature (and the Norton/material/source
signatures), and only then consumes the saved fields. The executed spectral
normalization and source magnitudes are read from the saved Result, never
re-derived from the declared material. Any mismatch refuses the receipt and
the consumer is not run.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from scripts.dev.fdtd.printed_line_broadband import encode
from scripts.dev.fdtd.printed_line_loss_basis import FoilRealization, LossBasis, _cpu, _equal, _validate_records
from scripts.dev.fdtd.printed_line_loss_bridge import (
    PROVENANCES, SEVEN_GATES, GATE_THRESHOLDS, available_power_source_magnitudes, build_capture,
    consume_public_column, copper_operator_signature, evidence_from_v3_completion, grid_of,
    save_capture, sha256_file, spectral_contract,
)

RECEIPT_SCHEMA = 'printed-line-Cu-realization-recovery-receipt/1'
RAW_FILES = ('completion.json', 'result.pt', 'actual-public-VI.pt', 'checkpoint32ns.pt')


class RecoveryRefused(ValueError):
    """The recompiled realization is not proven to be the executed one."""


def realizations_from_recompiled(entries, records):
    """Real/imaginary copper banks of a recompiled solver, with executed norms."""
    output = []
    for entry in entries:
        if entry['imaginary']:
            continue
        face, component = entry['face'], entry['electric_name']
        partners = [p for p in entries if p['imaginary'] and p['face'] is face and p['electric_name'] == component + '_imag']
        if len(partners) > 1 or any(not _equal(_cpu(entry[k]), _cpu(p[k])) for p in partners
                for k in ('indices', 'area', 'B', 'rates', 'D')):
            raise RecoveryRefused('Recompiled complex copper banks disagree.')
        if face.name not in records:
            raise RecoveryRefused('Executed Result has no record for a recompiled copper owner.')
        traces = [t for t in records[face.name]['bulk_electric_traces'] if t['component'] == component]
        if len(traces) != 1:
            raise RecoveryRefused('Executed Result lacks a unique sampled trace for a recompiled copper bank.')
        executed_norm = torch.as_tensor(traces[0]['sampling']['weight_sums'], dtype=torch.float64).clone()
        output.append(FoilRealization(face.name, component, 'xyz'[face.axis], int(face.normal_index),
            ('real', 'imaginary') if partners else ('real',),
            *(_cpu(entry[k]) for k in ('indices', 'area', 'B', 'rates', 'D')), executed_norm))
    if sum(1 for e in entries if e['imaginary']) != sum(len(r.banks) - 1 for r in output):
        raise RecoveryRefused('Recompiled solver has an orphan imaginary copper bank.')
    return tuple(output)


def recovery_receipt(context, result, completion, *, recompiled_entries, recompiled_operator,
                     recompiled_clock, recompiled_grid, resistances):
    """Pure proof step; raises RecoveryRefused unless every hash matches."""
    evidence = evidence_from_v3_completion(completion)
    executed = evidence['operator']
    if recompiled_clock != evidence['clock']:
        raise RecoveryRefused('Recompiled timestep/horizon differs from the executed clock.')
    mismatched = sorted(k for k in ('Norton', 'material', 'sources', 'copper') if recompiled_operator.get(k) != executed.get(k))
    if mismatched:
        raise RecoveryRefused(f'Recompiled operator signatures differ from execution: {mismatched}.')
    records = _cpu(result.conductor_losses)
    realizations = realizations_from_recompiled(recompiled_entries, records)
    copper = copper_operator_signature(realizations)
    if copper != executed['copper']:
        raise RecoveryRefused('Recompiled realization hash does not match the executed copper-operator signature.')
    contract = spectral_contract(identity=evidence['identity'], grid=recompiled_grid, operator=executed,
        context=context, total_steps=evidence['clock']['total_steps'])
    basis = LossBasis(tuple(float(f) for f in result.frequencies), evidence['clock']['actual_dt_s'],
        contract, realizations, records)
    try:
        _validate_records(basis)
    except ValueError as error:
        raise RecoveryRefused(f'Recovered basis fails the original loss-basis validator: {error}') from error
    sources = available_power_source_magnitudes(result, resistances)
    capture = build_capture(basis, sources, provenance=PROVENANCES[1], operator=executed, clock=evidence['clock'],
        identity=evidence['identity'], kind='cpw', length_index=0, drive_column=0)
    receipt = dict(schema=RECEIPT_SCHEMA, matched=True, executed_operator=executed, recompiled_operator=recompiled_operator,
        recompiled_realization_copper_signature=copper, executed_clock=evidence['clock'], recompiled_clock=recompiled_clock,
        binding_sha256=capture['manifest']['binding_sha256'], realization_sha256=capture['manifest']['realization_sha256'],
        executed_values_from_saved_Result=['spectral_norm (Cu sampling weight_sums)', 'conductor records/two-sided fields',
            'source magnitude sqrt(8 R available_power)'],
        recompiled_values_hash_proved=['indices', 'area', 'B', 'rates', 'D'],
        reconstructed_from_declared_material=False, new_field_steps=0)
    return receipt, capture, evidence


def _pins(run_dir, pin_file):
    expected = json.loads(Path(pin_file).read_text())['after']
    output = {}
    for name in RAW_FILES:
        path = str(Path(run_dir) / name)
        if path not in expected:
            raise RecoveryRefused(f'No archived executed SHA-256 pin for {name}.')
        actual = sha256_file(path)
        if actual != expected[path]:
            raise RecoveryRefused(f'Executed raw file {name} changed since its archived audit.')
        output[path] = actual
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--pins', type=Path, required=True, help='Archived independent audit JSON with after-SHA pins.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        return _main(args, started)
    except RecoveryRefused as error:
        (args.output / 'recovery-refused.json').write_text(json.dumps(dict(schema=RECEIPT_SCHEMA, matched=False,
            reason=str(error), consumer_run=False), indent=2) + '\n')
        raise


def _main(args, started):
    import witwin.core
    import witwin.maxwell as mw
    from scripts.dev.fdtd.pcb_acceptance import package_identity
    from scripts.dev.fdtd.printed_line_loss_execution import make_simulation
    from scripts.dev.fdtd.printed_line_public_execution import actual_clock, build_public_column, column_case, operator_signature, source_basis
    pins_before = _pins(args.run_dir, args.pins)
    completion = json.loads((args.run_dir / 'completion.json').read_text())
    current = dict(maxwell=package_identity(mw, 'witwin-maxwell')['source_sha256'], core=package_identity(witwin.core, 'witwin')['source_sha256'])
    executed_identity = evidence_from_v3_completion(completion)['identity']
    if current != executed_identity:
        raise RecoveryRefused('Recompilation must use the executed numerical package/core identity.')
    case = column_case(0)
    context = build_public_column('cpw', case)
    result = mw.Result.load(args.run_dir / 'result.pt', scene=context['observed'], map_location='cpu')
    # Ordinary public preparation only; no field is advanced or re-run.
    torch.cuda.reset_peak_memory_stats()
    simulation = make_simulation(context['observed'], context['groups'], source_basis(2)[:, 0], case,
        graph=completion['column']['graph'], steps=1)
    prepared = simulation.prepare()
    solver = prepared.solver
    entries = [{k: entry[k] for k in ('face', 'electric_name', 'imaginary', 'indices', 'area', 'B', 'rates', 'D')}
        for entry in solver._copper_foils]
    recompiled = dict(operator=operator_signature(solver), clock=actual_clock(float(solver.dt), case),
        grid=grid_of(solver.scene), resistances={r.port.name: float(r.lumped.resistance) for r in solver._port_runtimes},
        entries=[{k: (_cpu(v) if torch.is_tensor(v) else v) for k, v in e.items()} for e in entries])
    gpu = dict(device=torch.cuda.get_device_name(solver.device), uuid=str(torch.cuda.get_device_properties(solver.device).uuid),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        field_steps_advanced=0)
    del entries, solver, prepared, simulation
    gc.collect()
    torch.cuda.empty_cache()
    receipt, capture, evidence = recovery_receipt(context, result, completion, recompiled_entries=recompiled['entries'],
        recompiled_operator=recompiled['operator'], recompiled_clock=recompiled['clock'],
        recompiled_grid=recompiled['grid'], resistances=recompiled['resistances'])
    receipt.update(raw_pins=pins_before, recompile_GPU=gpu, executed_identity=executed_identity,
        recompile_identity=current, run_dir=str(args.run_dir))
    manifest = save_capture(args.output, capture)
    (args.output / 'recovery-receipt.json').write_text(json.dumps(encode(receipt), indent=2, allow_nan=False) + '\n')
    # Hash proof passed: consume the saved executed fields on CPU only.
    saved = torch.load(args.run_dir / 'actual-public-VI.pt', map_location='cpu', weights_only=False)
    column = consume_public_column(context, result, capture, evidence, saved['VI'], saved['queries'])
    pending = {gate: 'requires every drive column of this family/length; only CPW L0 D0 executed'
        for gate in SEVEN_GATES if gate not in column['per_column_gate_values']}
    pins_after = _pins(args.run_dir, args.pins)
    report = dict(column, recovery_receipt_binding_sha256=manifest['binding_sha256'], basis_level_gates_pending=pending,
        seven_gate_thresholds=GATE_THRESHOLDS, raw_files_unchanged=pins_before == pins_after,
        new_field_steps=0, wall_s=time.monotonic() - started)
    (args.output / 'first-column-consumer.json').write_text(json.dumps(encode(report), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(binding=manifest['binding_sha256'], per_column=report['per_column_gate_values'],
        pending=list(pending), replay=column['coherent_loss']['max_relative_replay_error']), default=str))
    return report


if __name__ == '__main__':
    main()
