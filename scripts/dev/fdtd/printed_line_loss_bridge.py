"""Executed copper-loss realization bridge for public printed-line columns.

Direct capture calls the original ``capture_loss_basis`` on the live executed
solver. Every captured ``FoilRealization`` (native indices, area, B, rates, D
and the executed spectral normalization), the Result's conductor records and
the actual Norton source spectra are bound by one SHA-256 to the executed
copper-operator signature. Nothing is reconstructed from the declared material.

The consumer validates provenance, clock, coherent loss and tail fields, then
calls the unchanged shared reducers: the original signed SI transfer helpers
(``physical_transfer``/``signed_peak_power``/``channels``), the original
``CoherentObservations``/``coherent_conductor_records`` and
``regional_plane_power``. Gate names and thresholds are the seven of
``reduce_pair``. Its records are new public-workflow records; they are never
written in, or relabelled as, the old ``reduce_pair`` execution format.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.dev.fdtd.negative_vi_complete_basis import channels
from scripts.dev.fdtd.printed_line_broadband import encode
from scripts.dev.fdtd.printed_line_loss_basis import (
    FoilRealization, LossBasis, SpectralContract, _cpu, _validate_records,
    capture_loss_basis, coherent_conductor_records, film_area,
)
from scripts.dev.fdtd import printed_line_loss_execution as _execution
from scripts.dev.fdtd.printed_line_loss_execution import tensor_signature
from scripts.dev.fdtd.printed_line_regional_power import regional_plane_power
from scripts.dev.fdtd.printed_line_two_length import physical_transfer, real_basis, signed_peak_power, two_length_modes
from witwin.maxwell.fdtd.copper_foil import copper_spectral_power


CAPTURE_SCHEMA = 'printed-line-executed-Cu-realization/1'
COLUMN_SCHEMA = 'public-line-column-loss-consumer/1'
BASIS_SCHEMA = 'public-line-original-seven-gates/1'
TWO_LENGTH_SCHEMA = 'public-line-independent-two-length/1'
CAPTURE_FILE = 'executed-Cu-realization.pt'
MANIFEST_FILE = 'executed-Cu-realization.json'
PROVENANCES = ('direct_executed_capture', 'recompiled_signature_matched_recovery')
SOURCE_REPRESENTATIONS = ('complex_thevenin_peak_voltage', 'magnitude_from_executed_available_power')
# The seven original per-length functional gates of reduce_pair. The CPW/pair
# current740 reducer used the same seven with the same thresholds.
SEVEN_GATES = ('source_identity_unchanged', 'same_actual_port_operator', 'state_condition',
    'coherent_solve', 'source_band_coverage', 'tail_decay', 'coherent_regional_power_balance')
GATE_THRESHOLDS = dict(state_condition_number_max=1e4, solve_residual_max=1e-10,
    source_band_min_relative=1e-3, tail_peak_relative_max=1e-3, line_power_balance_relative=.03)
TAIL_INTERVAL = 32
TAIL_WINDOW_S = 2e-9
# Float64 replay of the executed float64 film-work operator: a few dozen
# rounded operations per edge, far below any physical tolerance and far above
# reduction-order roundoff. Bit-level tampering is rejected by the SHA-256
# digests, not by this replay; the replay rejects relative changes of B/rates/
# D/area well above 1e-12 (tests use 1e-9).
LOSS_REPLAY_RELATIVE = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(encode(value), sort_keys=True, allow_nan=False)


def copper_operator_signature(realizations):
    """Exactly the executed ``operator_signature(solver)['copper']`` digest.

    The runtime list holds one real entry per owner/component, immediately
    followed by its imaginary twin for complex fields; the original capture
    already proved both banks share indices/area/B/rates/D.
    """
    entries = []
    for realization in realizations:
        for bank in realization.banks:
            suffix = '' if bank == 'real' else '_imag'
            entries.append(dict(face=realization.face, component=realization.component + suffix,
                indices=realization.indices, area=realization.area, B=realization.B,
                rates=realization.rates, D=realization.D))
    return tensor_signature(entries)


def realization_digest(basis):
    return tensor_signature(dict(frequencies=basis.frequencies, dt=basis.dt,
        contract=asdict(basis.execution_contract),
        realizations=[{key: getattr(r, key) for key in FoilRealization.__dataclass_fields__}
            for r in basis.realizations]))


def tail_monitors(context):
    tails = [m for m in context['observed'].monitors if m.name.startswith('full64_tail_')]
    if not tails or [m.name for m in tails] != [f'full64_tail_{j}' for j in range(len(tails))]:
        raise ValueError('The predeclared ordered tail probes are missing.')
    return tails


def observation_signature(context):
    plan = context['plan']
    return tensor_signature(dict(rows=plan.rows, expected_points=plan.expected_points,
        queries={name: asdict(term) for name, term in plan.queries.items()},
        native_rows=plan.native_rows, frequencies=plan.frequencies,
        regional_schema=context['regional_schema'], regional_bounds=context['regional_bounds'],
        groups=context['groups'], tails=[dict(name=m.name, components=m.components,
            position=m.position, interval=m.interval) for m in tail_monitors(context)]))


def grid_of(prepared_scene):
    return {axis: getattr(prepared_scene, axis + '_nodes64') for axis in 'xyz'}


def spectral_contract(*, identity, grid, operator, context, total_steps):
    return SpectralContract(tensor_signature(dict(maxwell=identity['maxwell'], core=identity['core'])),
        tensor_signature(grid), tensor_signature(dict(material=operator['material'], port_operator=operator['Norton'])),
        observation_signature(context), int(total_steps), 'none')


def executed_source_spectra(solver, result, frequencies):
    """The executed complex Norton drive spectra, as the original run_drive."""
    output = {}
    for runtime in solver._port_runtimes:
        if tuple(float(f) for f in runtime.frequencies) != tuple(frequencies):
            raise ValueError('Actual source-spectrum frequency axis changed.')
        voltage, _ = runtime.drive_accumulator.phasors(
            normalization='peak', integration_dt=solver._spectral_integration_dt,
            reference_frequency=solver._spectral_reference_frequency)
        output[runtime.port.name] = dict(representation=SOURCE_REPRESENTATIONS[0],
            thevenin_peak_voltage_V=voltage.detach().cpu().clone(),
            resistance_ohm=float(runtime.lumped.resistance),
            available_power_W=result.port(runtime.port.name).available_power.detach().cpu().clone())
    return output


def available_power_source_magnitudes(result, resistances):
    """Recovery only: |V| from the executed available power |V|^2/(8R).

    The resistances come from the recompiled Norton runtimes whose signature
    equals the executed one. The phase is unknown and never fabricated.
    """
    output = {}
    for name, resistance in resistances.items():
        power = result.port(name).available_power.detach().cpu().clone()
        output[name] = dict(representation=SOURCE_REPRESENTATIONS[1],
            thevenin_peak_voltage_magnitude_V=torch.sqrt(8. * resistance * power.double()),
            resistance_ohm=float(resistance), available_power_W=power)
    return output


def _binding(manifest):
    core = {key: value for key, value in manifest.items() if key not in ('binding_sha256', 'capture_file_sha256')}
    return hashlib.sha256(_canonical(core).encode()).hexdigest()


def build_capture(basis, sources, *, provenance, operator, clock, identity, kind, length_index, drive_column):
    """Bind one executed realization to the executed copper-operator signature."""
    if provenance not in PROVENANCES:
        raise ValueError('Unknown realization provenance.')
    if not isinstance(basis, LossBasis) or not basis.realizations:
        raise ValueError('A copper column must carry its actual nonempty realization.')
    _validate_records(basis)
    copper = copper_operator_signature(basis.realizations)
    if copper != operator['copper']:
        raise ValueError('The realization does not reproduce the executed copper-operator signature.')
    if basis.execution_contract.planned_steps != clock['total_steps'] or basis.dt != clock['actual_dt_s']:
        raise ValueError('The realization clock differs from the executed clock.')
    if set(sources) == set() or any(entry.get('representation') not in SOURCE_REPRESENTATIONS for entry in sources.values()):
        raise ValueError('Executed source spectra must declare their representation.')
    manifest = dict(schema=CAPTURE_SCHEMA, provenance=provenance, kind=kind, length_index=length_index,
        drive_column=drive_column, clock=clock, executed_operator=dict(operator), identity=dict(identity),
        executed_copper_operator_signature=copper, realization_sha256=realization_digest(basis),
        conductor_records_sha256=tensor_signature(basis.conductor_records),
        source_spectra_sha256=tensor_signature(sources), contract=asdict(basis.execution_contract),
        realizations=[dict(face=r.face, component=r.component, banks=r.banks, edges=len(r.indices),
            states=len(r.rates)) for r in basis.realizations],
        reconstructed_from_declared_material=False)
    manifest['binding_sha256'] = _binding(manifest)
    return dict(basis=basis, source_spectra=sources, manifest=manifest)


def capture_executed_realization(result, solver, *, operator, clock, context, identity, kind, length_index, drive_column):
    """Direct path: read the live executed solver/Result before release."""
    if result.solver is not solver:
        raise ValueError('Capture the realization from the solver that produced this Result.')
    contract = spectral_contract(identity=identity, grid=grid_of(solver.scene), operator=operator,
        context=context, total_steps=clock['total_steps'])
    basis = capture_loss_basis(result, contract)
    sources = executed_source_spectra(solver, result, basis.frequencies)
    return build_capture(basis, sources, provenance=PROVENANCES[0], operator=operator, clock=clock,
        identity=identity, kind=kind, length_index=length_index, drive_column=drive_column)


def save_capture(folder, capture):
    folder = Path(folder)
    path = folder / CAPTURE_FILE
    if path.exists() or (folder / MANIFEST_FILE).exists():
        raise FileExistsError('A captured realization is never overwritten.')
    torch.save(dict(schema=CAPTURE_SCHEMA, basis=capture['basis'], source_spectra=capture['source_spectra'],
        binding_sha256=capture['manifest']['binding_sha256']), path)
    manifest = dict(capture['manifest'], capture_file_sha256=sha256_file(path))
    (folder / MANIFEST_FILE).write_text(json.dumps(encode(manifest), indent=2, allow_nan=False) + '\n')
    return manifest


def load_capture(folder):
    """Load and re-prove every digest; tampering anywhere is rejected."""
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    if manifest.get('schema') != CAPTURE_SCHEMA or manifest.get('provenance') not in PROVENANCES:
        raise ValueError('Not an executed copper-realization capture.')
    if sha256_file(folder / CAPTURE_FILE) != manifest['capture_file_sha256']:
        raise ValueError('The captured realization file changed after capture.')
    if _binding(manifest) != manifest['binding_sha256']:
        raise ValueError('The realization manifest changed after capture.')
    payload = torch.load(folder / CAPTURE_FILE, map_location='cpu', weights_only=False)
    basis, sources = payload['basis'], payload['source_spectra']
    if payload.get('schema') != CAPTURE_SCHEMA or payload.get('binding_sha256') != manifest['binding_sha256']:
        raise ValueError('Capture payload is bound to a different manifest.')
    _validate_records(basis)
    checks = dict(executed_copper_operator_signature=copper_operator_signature(basis.realizations),
        realization_sha256=realization_digest(basis), conductor_records_sha256=tensor_signature(basis.conductor_records),
        source_spectra_sha256=tensor_signature(sources))
    for key, value in checks.items():
        if value != manifest[key]:
            raise ValueError(f'Captured realization digest {key} does not match its binding.')
    if manifest['executed_copper_operator_signature'] != manifest['executed_operator']['copper']:
        raise ValueError('Capture is not tied to its executed copper-operator signature.')
    if encode(asdict(basis.execution_contract)) != manifest['contract']:
        raise ValueError('Capture contract changed.')
    return dict(basis=basis, source_spectra=sources, manifest=manifest)


def evidence_from_public_execution(record):
    """Normalize a new ``public-printed-line-column/1`` execution.json."""
    if record.get('schema') != 'public-printed-line-column/1' or record.get('functional_passed') is not True:
        raise ValueError('Only a completed public-printed-line-column/1 record is admitted.')
    before, after = record['source_identity_before'], record['source_identity_after']
    return dict(record_schema=record['schema'], kind=record['kind'], length_index=record['length_index'],
        drive_column=record['drive_column'], real_basis=[float(x) for x in record['real_basis']],
        clock=record['clock'], operator=record['actual_operator'], identity=before,
        source_identity_unchanged=before == after, graph=record['graph'],
        capture_binding_sha256=record['executed_Cu_realization']['binding_sha256'])


def evidence_from_v3_completion(record):
    """Normalize the historical V3 one-column completion.json (recovery only)."""
    if record.get('qualified') is not True or record.get('ordinary_prepared_run_returned_complete_Result') is not True:
        raise ValueError('Only the qualified complete V3 column is admitted.')
    column = record['column']
    if (column['kind'], column['length_index'], column['drive_column']) != ('cpw', 0, 0):
        raise ValueError('The V3 completion is the single CPW L0 D0 column.')
    clock = {key: record['clock'][key] for key in ('actual_dt_s', 'total_steps', 'checkpoint_step',
        'requested_duration_s', 'requested_checkpoint_time_s', 'actual_duration_s', 'actual_checkpoint_time_s')}
    return dict(record_schema='historical-V3-one-column-completion', kind='cpw', length_index=0, drive_column=0,
        real_basis=[float(x) for x in real_basis(2)[:, 0]], clock=clock, operator=record['actual_operator'],
        identity=dict(maxwell=record['actual_package']['source_sha256'], core=record['actual_core']['source_sha256']),
        source_identity_unchanged=record['source_before'] == record['source_after'], graph=column['graph'],
        capture_binding_sha256=None)


def _relative(actual, expected):
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    scale = float(expected.abs().max())
    return float((actual - expected).abs().max()) / scale if scale > 0 else (0. if float(actual.abs().max()) == 0 else math.inf)


def check_provenance(context, capture, evidence):
    from witwin.maxwell.scene import prepare_scene
    manifest, basis = capture['manifest'], capture['basis']
    if (manifest['kind'], manifest['length_index'], manifest['drive_column']) != (evidence['kind'], evidence['length_index'], evidence['drive_column']):
        raise ValueError('Capture and execution name different physical columns.')
    if manifest['executed_operator'] != evidence['operator'] or manifest['identity'] != evidence['identity']:
        raise ValueError('Capture is not bound to this executed operator/source identity.')
    if copper_operator_signature(basis.realizations) != evidence['operator']['copper']:
        raise ValueError('Realization does not reproduce the executed copper-operator signature.')
    if evidence['capture_binding_sha256'] is not None and evidence['capture_binding_sha256'] != manifest['binding_sha256']:
        raise ValueError('Execution record names a different realization capture.')
    expected = spectral_contract(identity=evidence['identity'], grid=grid_of(prepare_scene(context['observed'])),
        operator=evidence['operator'], context=context, total_steps=evidence['clock']['total_steps'])
    if basis.execution_contract != expected:
        raise ValueError('Capture contract differs from this source/grid/material/observation declaration.')
    if not evidence['source_identity_unchanged']:
        raise ValueError('Executed source identity changed during the run.')
    return dict(binding_sha256=manifest['binding_sha256'], provenance=manifest['provenance'],
        executed_copper_operator_signature=manifest['executed_copper_operator_signature'],
        record_schema=evidence['record_schema'], contract=asdict(basis.execution_contract))


def check_clock(context, result, capture, evidence):
    case, clock, basis = context['case'], evidence['clock'], capture['basis']
    dt, total = clock['actual_dt_s'], clock['total_steps']
    if _execution_clock(dt, case) != clock or capture['manifest']['clock'] != clock:
        raise ValueError('Executed clock is not the complete physical horizon of this case.')
    stats = result.stats()
    if (stats['time_steps'] != total or stats['steps_run'] != total or stats['dt'] != dt
        or stats.get('normalize_source', False) or str(getattr(stats.get('dft_window'), 'value', stats.get('dft_window'))) != 'none'):
        raise ValueError('Result does not hold the complete raw executed horizon.')
    frequencies = tuple(float(f) for f in result.frequencies)
    if frequencies != tuple(case.frequencies) or basis.frequencies != frequencies or basis.dt != dt:
        raise ValueError('Result/realization frequency axis or timestep differ.')
    records = result.conductor_losses
    sampled = 0
    for realization in basis.realizations:
        trace = next(t for t in records[realization.face]['bulk_electric_traces'] if t['component'] == realization.component)
        sampling = trace['sampling']
        start = torch.as_tensor(sampling['start_steps'], dtype=torch.float64)
        weights = torch.as_tensor(sampling['weight_sums'], dtype=torch.float64)
        if (sampling['planned_steps'] != total or float(sampling['completed_steps']) != total
            or str(getattr(sampling['window'], 'value', sampling['window'])) != 'none' or sampling['dt'] != dt
            or sampling['normalize_source'] or tuple(float(f) for f in sampling['frequencies']) != frequencies
            or not torch.equal(weights, total - start)
            or not torch.equal(weights, realization.spectral_norm.to(torch.float64))):
            raise ValueError('Executed Cu sampling clock or normalization differs from the realization.')
        sampled += 1
    direct = capture['manifest']['provenance'] == PROVENANCES[0]
    return dict(total_steps=total, dt_s=dt, checkpoint_step=clock['checkpoint_step'], Cu_sampling_traces=sampled,
        observer_samples=stats.get('observer_samples'),
        capture_vs_Result_normalization='verified (live solver capture vs saved Result)' if direct
            else 'identical by construction (recovery reads the norm from this Result); only the clock rule N-start is independent')


def _execution_clock(dt, case):
    from scripts.dev.fdtd.printed_line_public_execution import actual_clock
    return actual_clock(dt, case)


def check_coherent_loss(result, capture):
    """Replay the Result's film work through the captured realization only."""
    basis = capture['basis']
    records = _cpu(result.conductor_losses)
    if tensor_signature(records) != tensor_signature(basis.conductor_records):
        raise ValueError('Captured conductor records are not this Result\'s executed records.')
    frequencies = torch.tensor(basis.frequencies, dtype=torch.float64)
    worst = 0.
    totals = {}
    for realization in basis.realizations:
        record = records[realization.face]
        bulk = next(t for t in record['bulk_electric_traces'] if t['component'] == realization.component)
        film = next(t for t in record['surface_power_traces'] if t['component'] == realization.component)
        total, edge = copper_spectral_power(bulk['electric_fields'], frequencies, basis.dt, realization.B,
            realization.rates, realization.D, film_area(film, realization), return_edge_power=True)
        error = _relative(edge, film['power'])
        worst = max(worst, error)
        if not error <= LOSS_REPLAY_RELATIVE:
            raise ValueError('Executed Result film work is not reproduced by the captured realization.')
        totals[realization.face] = totals.get(realization.face, 0) + total
    coherent = coherent_conductor_records([basis], torch.ones((len(basis.frequencies), 1), dtype=torch.complex128))
    for face, record in records.items():
        for candidate in (totals[face], coherent[face]['surface_power']):
            error = _relative(candidate, record['surface_power'])
            worst = max(worst, error)
            if not error <= LOSS_REPLAY_RELATIVE:
                raise ValueError('Coherent single-column film work differs from the executed record.')
    direct = capture['manifest']['provenance'] == PROVENANCES[0]
    return dict(realizations=len(basis.realizations), max_relative_replay_error=worst,
        records_vs_Result='verified (records captured before save vs reloaded Result)' if direct
            else 'identical by construction (recovery takes records from this Result); the B/rates/D film-work replay is the independent check',
        replay_bound=LOSS_REPLAY_RELATIVE, executed_surface_power_W={k: v['surface_power'] for k, v in records.items()})


def check_tails(context, result, clock):
    """Bounded original-point tail field: complete trace, last 2 ns over peak."""
    total, dt = clock['total_steps'], clock['actual_dt_s']
    expected = len(range(0, total, TAIL_INTERVAL))
    records = []
    for monitor in tail_monitors(context):
        if monitor.interval != TAIL_INTERVAL:
            raise ValueError('Tail probe interval changed.')
        payload = result.monitor(monitor.name)
        field = torch.as_tensor(payload['field']).detach().cpu().numpy()
        times = torch.as_tensor(payload['t']).detach().cpu().numpy()
        if (field.shape != times.shape or len(times) != expected or not np.isfinite(field).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > total * dt):
            raise ValueError('Tail trace does not cover its complete executed interval.')
        peak = float(abs(field).max())
        if not peak > 0:
            raise ValueError('A tail probe recorded no field.')
        selection = times >= total * dt - TAIL_WINDOW_S
        ratio = float(abs(field[selection]).max() / peak) if bool(selection.any()) else None
        records.append(dict(name=monitor.name, component=monitor.components[0], position_m=monitor.position,
            samples=len(times), last_time_s=float(times[-1]), peak_abs=peak, last2ns_peak_relative=ratio))
    return records


def check_sources(context, result, capture):
    sources = capture['source_spectra']
    names = [name for group in context['groups'] for name in group]
    if set(sources) != set(names) or len(set(names)) != len(names):
        raise ValueError('Every executed physical port needs its executed source spectrum.')
    magnitudes = {}
    for name, entry in sources.items():
        resistance = entry['resistance_ohm']
        power = result.port(name).available_power.detach().cpu()
        if not torch.equal(entry['available_power_W'], power) or resistance != 50.:
            raise ValueError('Source spectra are not bound to this Result\'s available power.')
        if entry['representation'] == SOURCE_REPRESENTATIONS[0]:
            voltage = entry['thevenin_peak_voltage_V']
            replay = voltage.abs().square() / (8. * resistance)
            if _relative(replay, power) > 8 * torch.finfo(power.dtype).eps:
                raise ValueError('Complex source spectrum does not produce the executed available power.')
            magnitude = voltage.abs().double()
        else:
            magnitude = entry['thevenin_peak_voltage_magnitude_V']
            if not torch.equal(magnitude, torch.sqrt(8. * resistance * power.double())):
                raise ValueError('Source magnitude is not the executed available-power magnitude.')
        if magnitude.shape != (len(context['case'].frequencies),) or not bool(torch.isfinite(magnitude).all()):
            raise ValueError('Source spectrum must cover every recorded frequency.')
        magnitudes[name] = magnitude
    for group in context['groups']:
        kinds = {sources[name]['representation'] for name in group}
        if len(kinds) != 1:
            raise ValueError('A balanced source group mixes source representations.')
        if kinds == {SOURCE_REPRESENTATIONS[0]}:
            # Signed complex drive, exactly as the original family reducers.
            reference = sources[group[0]]['thevenin_peak_voltage_V']
            if any(not torch.equal(reference, sources[name]['thevenin_peak_voltage_V']) for name in group[1:]):
                raise ValueError('A balanced physical source group lost its common signed complex drive.')
        elif any(not torch.equal(magnitudes[group[0]], magnitudes[name]) for name in group[1:]):
            raise ValueError('A balanced physical source group lost its common drive magnitude.')
    return {name: float(value.min() / value.max()) if float(value.max()) > 0 else 0. for name, value in magnitudes.items()}


def balanced_group_comparison(capture):
    kinds = {entry['representation'] for entry in capture['source_spectra'].values()}
    return ('signed complex drive, exact equality' if kinds == {SOURCE_REPRESENTATIONS[0]}
        else 'magnitude only (recovery: phase not saved by the historical column)')


def consume_public_column(context, result, capture, evidence, VI, queries):
    """Strictly admit one public column; return per-column gate inputs."""
    from scripts.dev.fdtd.printed_line_public_vi_checks import independent_public_audit
    provenance = check_provenance(context, capture, evidence)
    clock = check_clock(context, result, capture, evidence)
    loss = check_coherent_loss(result, capture)
    tails = check_tails(context, result, evidence['clock'])
    band = check_sources(context, result, capture)
    if not torch.is_tensor(VI) or VI.dtype != torch.complex128 or VI.shape != (len(context['plan'].rows), 2, len(context['case'].frequencies)):
        raise ValueError('Public signed SI V/I must keep its complete station/row/frequency axes.')
    audit = independent_public_audit(context, result, VI, queries, total=evidence['clock']['total_steps'])
    basis = capture['basis']
    frequencies = len(basis.frequencies)
    single = CoherentObservations([result], [basis], np.ones((frequencies, 1)))
    raw_power = regional_plane_power(single, context['observed'], context['regional_schema'], context['regional_bounds'])
    del single
    tail_values = [record['last2ns_peak_relative'] for record in tails]
    column = dict(schema=COLUMN_SCHEMA, kind=evidence['kind'], length_index=evidence['length_index'],
        drive_column=evidence['drive_column'], real_basis=evidence['real_basis'], provenance=provenance,
        clock=clock, coherent_loss=loss, tails=tails, source_band_min_relative=band,
        public_VI_independent_audit=dict(primary_checks=audit['primary_checks'],
            constitutive_side_checks=audit['constitutive_side_checks'], VI_component_checks=audit['VI_component_checks']),
        single_column_regional_power_diagnostic=raw_power,
        balanced_source_group_comparison=balanced_group_comparison(capture),
        executed_operator=evidence['operator'], executed_identity=evidence['identity'], case=asdict(context['case']),
        port_operator_sha256=evidence['operator']['Norton'], source_identity_unchanged=evidence['source_identity_unchanged'],
        per_column_gate_values=dict(
            source_identity_unchanged=dict(value=evidence['source_identity_unchanged'], passed=evidence['source_identity_unchanged']),
            source_band_coverage=dict(value=min(band.values()), threshold=GATE_THRESHOLDS['source_band_min_relative'],
                passed=all(v >= GATE_THRESHOLDS['source_band_min_relative'] for v in band.values())),
            tail_decay=dict(value=max(v for v in tail_values if v is not None) if any(v is not None for v in tail_values) else None,
                threshold=GATE_THRESHOLDS['tail_peak_relative_max'],
                passed=bool(tail_values) and all(v is not None and v <= GATE_THRESHOLDS['tail_peak_relative_max'] for v in tail_values))),
        single_column_power_closure_is_gate=False,
        physical_mode_attenuation_accuracy_qualified=False)
    return column


def column_summary(column, *, VI, result, basis):
    """Inputs the basis reducer needs, in memory only."""
    return dict(kind=column['kind'], length_index=column['length_index'], drive_column=column['drive_column'],
        real_basis=column['real_basis'], clock={k: column['clock'][k] for k in ('total_steps', 'dt_s', 'checkpoint_step')},
        VI=VI, result=result, basis=basis,
        port_operator_sha256=column['port_operator_sha256'], source_identity_unchanged=column['source_identity_unchanged'],
        executed_operator=column['executed_operator'], executed_identity=column['executed_identity'],
        grid_sha256=column['provenance']['contract']['grid_sha256'], case=column['case'],
        source_band_ratios=list(column['source_band_min_relative'].values()),
        tail_values=[t['last2ns_peak_relative'] for t in column['tails']],
        binding_sha256=column['provenance']['binding_sha256'])


# Module-level reducers are looked up at call time (tests may substitute I/O).
CoherentObservations = _execution.CoherentObservations


def reduce_public_basis(kind, columns, *, cpu_scene, regional_schema, regional_bounds):
    """Original seven gates for one complete independent drive basis.

    Per frequency this follows the CPW/pair complete-basis reducer: full signed
    SI states, original ``physical_transfer`` across station pairs, positive
    signed-power transfer eigenstates normalized to 1 W, coefficients solved at
    the left station, then coherent fields/heads through the original shared
    heat/flux reducer. Failures are recorded, never suppressed.
    """
    n = 1 if kind == 'cpw' else 2 if kind == 'differential_pair' else None
    if n is None or len(columns) != 2 * n:
        raise ValueError('A complete scalar (2) or coupled (4) independent drive basis is required.')
    count = 2 * n
    basis_rows = real_basis(count)
    for index, column in enumerate(columns):
        if column['length_index'] not in (0, 1):
            raise ValueError('Basis columns must name one of the two original physical lengths.')
        if (column['kind'] != kind or column['drive_column'] != index
            or column['length_index'] != columns[0]['length_index']
            or not np.array_equal(np.asarray(column['real_basis']), basis_rows[:, index])):
            raise ValueError('Columns must be the ordered original real basis of one family/length.')
        if column['clock'] != columns[0]['clock']:
            raise ValueError('Basis columns differ in executed clock.')
    if len({column['binding_sha256'] for column in columns}) != count:
        raise ValueError('Every basis column needs its own executed realization capture.')
    # Explicit cross-column pre-check, as reduce_pair: mismatches are recorded
    # gate failures and the coherent combination is not attempted.
    def common(key):
        return len({json.dumps(encode(column.get(key)), sort_keys=True) for column in columns}) == 1
    identity_common = common('executed_identity')
    realization_common = all(len({column['executed_operator'][k] for column in columns}) == 1 for k in ('material', 'copper'))
    norton_common = len({column['executed_operator']['Norton'] for column in columns}) == 1
    geometry_common = common('grid_sha256') and common('case')
    # Norton differences are the original port-operator gate; material/Cu,
    # identity or geometry differences make a coherent field sum meaningless.
    precheck = dict(executed_identity_common=identity_common, material_and_copper_common=realization_common,
        grid_and_case_common=geometry_common, Norton_common=norton_common)
    coherent_admissible = identity_common and realization_common and geometry_common
    VI = [np.asarray(column['VI'].detach().cpu().numpy() if torch.is_tensor(column['VI']) else column['VI'], dtype=np.complex128)
        for column in columns]
    voltage, current = zip(*(channels(kind, state) for state in VI))
    V = np.stack(voltage, axis=-1)
    I = np.stack(current, axis=-1)
    frequencies = tuple(columns[0]['basis'].frequencies)
    records = []
    coefficients = [[] for _ in range(n)]
    failures = []
    for f, frequency in enumerate(frequencies):
        scaled = np.concatenate((V[:, :, f], 50 * I[:, :, f]), axis=1)
        condition = [float(np.linalg.cond(x)) for x in scaled]
        rank = [int(np.linalg.matrix_rank(x)) for x in scaled]
        record = dict(frequency_hz=frequency, state_condition_number_50ohm=condition, physical_state_rank=rank)
        try:
            transfers = [dict(station_pair=(a, b), **physical_transfer(V[[a, b], :, f], I[[a, b], :, f]))
                for a, b in ((0, 1), (1, 2), (0, 2))]
        except ValueError as error:
            record.update(transfer_failure=str(error), solve_residual=[math.inf])
            failures.append(dict(frequency_hz=frequency, stage='physical_transfer', message=str(error)))
            records.append(record)
            continue
        values, vectors = np.linalg.eig(transfers[-1]['matrix'])
        powers = signed_peak_power(vectors[:n], vectors[n:] / 50.)
        residuals = [t['reconstruction_relative'] for t in transfers]
        modes = []
        if np.any(abs(powers) < 1e-12) or np.count_nonzero(powers > 0) != n:
            failures.append(dict(frequency_hz=frequency, stage='positive_power_states',
                message='Complete transfer does not resolve n positive- and n negative-power states.'))
            residuals.append(math.inf)
        else:
            for number, index in enumerate(np.flatnonzero(powers > 0)):
                vector = vectors[:, index] / np.sqrt(powers[index])
                coefficient = np.linalg.solve(scaled[0], vector)
                residual = float(np.linalg.norm(scaled[0] @ coefficient - vector) / np.linalg.norm(vector))
                residuals.append(residual)
                coefficients[number].append(coefficient)
                modes.append(dict(eigenvalue=values[index], scaled_V_and_50I_state=vector,
                    coherent_coefficients=coefficient, solve_residual=residual))
        record.update(transfers=[dict(station_pair=t['station_pair'], matrix=t['matrix'],
            reconstruction_relative=t['reconstruction_relative']) for t in transfers],
            all_eigenvalues=values, signed_eigenstate_power_W=powers,
            positive_power_states=modes, solve_residual=residuals)
        records.append(record)
    powers = []
    complete = all(len(values) == len(frequencies) for values in coefficients) and coherent_admissible
    if complete:
        results = [column['result'] for column in columns]
        bases = [column['basis'] for column in columns]
        for number, values in enumerate(coefficients):
            observation = CoherentObservations(results, bases, np.asarray(values))
            power = regional_plane_power(observation, cpu_scene, regional_schema, regional_bounds)
            powers.append(dict(positive_power_state=number, coherent_coefficients=np.asarray(values), shared_power=power))
            del observation
    closures = [x for item in powers for x in item['shared_power']['balance_relative_to_entering']]
    ratios = [ratio for column in columns for ratio in column['source_band_ratios']]
    tails = [value for column in columns for value in column['tail_values']]
    gates = {
        'source_identity_unchanged': identity_common and all(column['source_identity_unchanged'] for column in columns),
        'same_actual_port_operator': norton_common and realization_common and geometry_common
            and len({column['port_operator_sha256'] for column in columns}) == 1,
        'state_condition': all(max(r['state_condition_number_50ohm']) <= GATE_THRESHOLDS['state_condition_number_max'] for r in records),
        'coherent_solve': all(max(r['solve_residual']) <= GATE_THRESHOLDS['solve_residual_max'] for r in records),
        'source_band_coverage': all(ratio >= GATE_THRESHOLDS['source_band_min_relative'] for ratio in ratios),
        'tail_decay': bool(tails) and all(value is not None and value <= GATE_THRESHOLDS['tail_peak_relative_max'] for value in tails),
        'coherent_regional_power_balance': complete and bool(closures) and all(value is not None and value <= GATE_THRESHOLDS['line_power_balance_relative'] for value in closures)}
    if tuple(gates) != SEVEN_GATES:
        raise AssertionError('The original seven gates are fixed.')
    values = dict(
        source_identity_unchanged=gates['source_identity_unchanged'],
        same_actual_port_operator=sorted({column['port_operator_sha256'] for column in columns}),
        state_condition=max(max(r['state_condition_number_50ohm']) for r in records),
        coherent_solve=max(max(r['solve_residual']) for r in records),
        source_band_coverage=min(ratios), tail_decay=max(v for v in tails if v is not None) if any(v is not None for v in tails) else None,
        coherent_regional_power_balance=max((x for x in closures if x is not None), default=None))
    return dict(schema=BASIS_SCHEMA, kind=kind, length_index=columns[0]['length_index'], drive_columns=count,
        frequency_records=records, coherent_regional_power=powers, gate_values=values,
        thresholds=GATE_THRESHOLDS, original_seven_gates=gates, original_seven_gates_passed=all(gates.values()),
        reducer_failures=failures, cross_column_precheck=precheck, capture_bindings=[column['binding_sha256'] for column in columns],
        case=columns[0].get('case'), executed_identity=columns[0].get('executed_identity'),
        executed_operators=[column.get('executed_operator') for column in columns], grid_sha256=columns[0].get('grid_sha256'),
        clock=columns[0]['clock'], reference_stations_m=(columns[0].get('case') or {}).get('positions'),
        physical_release_qualified=False,
        scope='Functional basis/coherent power gates only; mode, mesh, external reference, Z/beta/S and attenuation accuracy remain separate.')


def _first_operator(report):
    operators = report['executed_operators']
    if any(operator != operators[0] for operator in operators[1:]):
        raise ValueError('A basis report holds differing executed operators across drives.')
    return operators[0]


def reduce_public_two_lengths(kind, short_path, long_path, *, current_identity, overrides=None):
    """Original independent two-length extraction with its cross-length contract.

    As review-two-lengths: same postprocess and executed identity, identical
    case except separation/positions, equal material/Cu/source signatures with
    intentionally translated (different) global Norton stamps, equal grid,
    dt and clock, and reference stations equal to the case positions.
    """
    from scripts.dev.fdtd.printed_line_public_execution import convergence_case
    from scripts.dev.fdtd.printed_line_two_length import decode

    def column_case(index):
        return convergence_case(index, **(overrides or {}))
    paths = [Path(short_path), Path(long_path)]
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError('Two distinct basis reports are required.')
    pins = {str(path): sha256_file(path) for path in paths}
    reports = [decode(json.loads(path.read_text())) for path in paths]
    for length_index, report in enumerate(reports):
        if report.get('schema') != BASIS_SCHEMA or report['kind'] != kind or report['length_index'] != length_index:
            raise ValueError('Two ordered independent physical lengths of one family are required.')
        if not report['original_seven_gates_passed']:
            raise ValueError('Both lengths must pass the original seven gates first.')
        if report['postprocess_identity'] != current_identity:
            raise ValueError('Basis report was reduced by a different postprocess source identity.')
        expected = column_case(length_index)
        if report['case'] != encode(asdict(expected)):
            raise ValueError('Basis report case is not the original fixed case for its length.')
        if list(report['reference_stations_m']) != list(expected.positions):
            raise ValueError('Reference stations differ from the case positions.')
    short, long = reports
    if short['postprocess_identity'] != long['postprocess_identity'] or short['executed_identity'] != long['executed_identity']:
        raise ValueError('Lengths differ in postprocess or executed source identity.')
    for key in set(short['case']) | set(long['case']):
        if key not in ('separation', 'positions') and short['case'].get(key) != long['case'].get(key):
            raise ValueError(f'Independent lengths differ in physical case input {key}.')
    operators = [_first_operator(report) for report in reports]
    for key in ('material', 'copper', 'sources'):
        if operators[0][key] != operators[1][key]:
            raise ValueError(f'Independent lengths differ in executed {key} operator.')
    if operators[0]['Norton'] == operators[1]['Norton']:
        raise ValueError('Distinct source spans must translate the global Norton stamps.')
    if short['grid_sha256'] != long['grid_sha256'] or short['clock'] != long['clock']:
        raise ValueError('Independent lengths differ in grid, dt or clock.')
    lengths = [report['reference_stations_m'][-1] - report['reference_stations_m'][0] for report in reports]
    if lengths != [column_case(j).positions[-1] - column_case(j).positions[0] for j in (0, 1)] or not 0 < lengths[0] < lengths[1]:
        raise ValueError('Reference lengths differ from the fixed cases.')
    records, failures = [], []
    for left, right in zip(short['frequency_records'], long['frequency_records']):
        frequency = left['frequency_hz']
        if right['frequency_hz'] != frequency:
            raise ValueError('Length reports differ in frequency order.')
        transfers = [np.asarray(next(t for t in report['transfers'] if tuple(t['station_pair']) == (0, 2))['matrix'], dtype=np.complex128)
            for report in (left, right)]
        row = dict(frequency_hz=frequency)
        try:
            measured = two_length_modes(*transfers, lengths=lengths, frequency=frequency, maximum_index=1.5)
            row.update(independent_two_length=measured, operator_extraction_completed=True)
            if any(mode['alpha_np_per_m'] < 0 for mode in measured['forward_modes']):
                failures.append(dict(frequency_hz=frequency, gate='passive_signed_forward_attenuation',
                    actual=[mode['alpha_np_per_m'] for mode in measured['forward_modes']]))
        except ValueError as error:
            row.update(operator_extraction_completed=False, operator_failure=str(error))
            failures.append(dict(frequency_hz=frequency, gate='independent_operator_extraction', message=str(error)))
        records.append(row)
    if {str(path): sha256_file(path) for path in paths} != pins:
        raise ValueError('A basis report changed during the two-length reduction.')
    return dict(schema=TWO_LENGTH_SCHEMA, kind=kind, physical_reference_lengths_m=lengths, input_report_sha256=pins,
        Norton_global_stamps_differ=True, material_Cu_sources_grid_clock_equal=True,
        frequency_records=records, failed_physical_gates=failures, predeclared_maximum_index=1.5,
        attenuation_or_mode_release_qualified=False)


def load_public_column(context, folder):
    """Load one new-workflow column folder from disk (CPU)."""
    import witwin.maxwell as mw
    folder = Path(folder)
    record = json.loads((folder / 'execution.json').read_text())
    evidence = evidence_from_public_execution(record)
    for name, digest in record['saved_files_sha256'].items():
        if sha256_file(folder / name) != digest:
            raise ValueError(f'Executed raw file {name} changed after execution.')
    capture = load_capture(folder)
    result = mw.Result.load(folder / 'result.pt', scene=context['observed'], map_location='cpu')
    saved = torch.load(folder / 'actual-public-VI.pt', map_location='cpu', weights_only=False)
    return evidence, capture, result, saved['VI'], saved['queries']


RECOVERED_INPUTS = ('completion.json', 'result.pt', 'actual-public-VI.pt')


def load_recovered_column(context, run_dir, recovery_dir, receipt_sha256):
    """Load the completed V3 first column through its matched, pinned receipt."""
    import witwin.maxwell as mw
    run_dir, recovery_dir = Path(run_dir), Path(recovery_dir)
    if sha256_file(recovery_dir / 'recovery-receipt.json') != receipt_sha256:
        raise ValueError('Recovery receipt differs from its pinned SHA-256.')
    receipt = json.loads((recovery_dir / 'recovery-receipt.json').read_text())
    pinned = {Path(path) for path in receipt.get('raw_pins', {})}
    if not pinned or any(run_dir / name not in pinned for name in RECOVERED_INPUTS):
        raise ValueError('Recovery receipt does not pin every loaded executed input.')
    capture = load_capture(recovery_dir)
    if receipt.get('matched') is not True or capture['manifest']['provenance'] != PROVENANCES[1] \
            or receipt['binding_sha256'] != capture['manifest']['binding_sha256']:
        raise ValueError('A recovered column needs its own matched recovery receipt.')
    if (recovery_dir / 'recovery-refused.json').exists():
        raise ValueError('The recovery was refused.')
    for path, digest in receipt['raw_pins'].items():
        if Path(path).parent != run_dir or sha256_file(path) != digest:
            raise ValueError('Recovered raw column changed since its receipt.')
    evidence = evidence_from_v3_completion(json.loads((run_dir / 'completion.json').read_text()))
    result = mw.Result.load(run_dir / 'result.pt', scene=context['observed'], map_location='cpu')
    saved = torch.load(run_dir / 'actual-public-VI.pt', map_location='cpu', weights_only=False)
    return evidence, capture, result, saved['VI'], saved['queries']


def _write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False) + '\n')


def main(argv=None):
    from scripts.dev.fdtd.printed_line_public_execution import build_public_column, convergence_case
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    basis = sub.add_parser('basis', help='Consume every drive column of one family/length and run the seven gates.')
    basis.add_argument('--kind', choices=('cpw', 'differential_pair'), required=True)
    basis.add_argument('--length-index', type=int, choices=(0, 1), required=True)
    basis.add_argument('--columns', type=Path, nargs='*', default=[])
    basis.add_argument('--recovered-column', nargs=3, metavar=('RUN_DIR', 'RECOVERY_DIR', 'RECEIPT_SHA256'),
        help='The completed V3 CPW L0 D0 column with its matched recovery receipt.')
    basis.add_argument('--output', type=Path, required=True)
    pair = sub.add_parser('two-length', help='Independent two-length extraction from two basis reports.')
    pair.add_argument('--kind', choices=('cpw', 'differential_pair'), required=True)
    pair.add_argument('--short', type=Path, required=True)
    pair.add_argument('--long', type=Path, required=True)
    pair.add_argument('--output', type=Path, required=True)
    for command in (basis, pair):
        # Convergence-control cases; defaults keep the original fixed case.
        for flag in ('--dx', '--longitudinal-dl', '--padding', '--sigma'):
            command.add_argument(flag, type=float, default=None)
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ('dx', 'longitudinal_dl', 'padding', 'sigma') if getattr(args, k) is not None}
    if args.command == 'two-length':
        from scripts.dev.fdtd.printed_line_public_execution import source_identity
        value = reduce_public_two_lengths(args.kind, args.short, args.long, current_identity=source_identity(), overrides=overrides)
        _write(args.output, value)
        print(json.dumps(dict(kind=args.kind, failed=value['failed_physical_gates'],
            alpha=[[m['alpha_np_per_m'] for m in r['independent_two_length']['forward_modes']] for r in value['frequency_records'] if r['operator_extraction_completed']]), default=str))
        return value
    context = build_public_column(args.kind, convergence_case(args.length_index, **overrides))
    summaries, columns = [], []
    loaders = [lambda folder=folder: load_public_column(context, folder) for folder in args.columns]
    if args.recovered_column is not None:
        loaders.append(lambda: load_recovered_column(context, *args.recovered_column))
    for load in loaders:
        evidence, capture, result, VI, queries = load()
        if evidence['length_index'] != args.length_index or evidence['kind'] != args.kind:
            raise ValueError('A column does not belong to the requested family/length.')
        column = consume_public_column(context, result, capture, evidence, VI, queries)
        columns.append(column)
        summaries.append(column_summary(column, VI=VI, result=result, basis=capture['basis']))
    order = sorted(range(len(columns)), key=lambda j: columns[j]['drive_column'])
    columns, summaries = [columns[j] for j in order], [summaries[j] for j in order]
    report = reduce_public_basis(args.kind, summaries, cpu_scene=context['observed'],
        regional_schema=context['regional_schema'], regional_bounds=context['regional_bounds'])
    report['columns'] = columns
    from scripts.dev.fdtd.printed_line_public_execution import source_identity
    report['postprocess_identity'] = source_identity()
    _write(args.output, report)
    print(json.dumps(dict(kind=args.kind, length_index=args.length_index, gates=report['original_seven_gates'],
        values=report['gate_values']), default=str))
    return report


if __name__ == '__main__':
    main()
