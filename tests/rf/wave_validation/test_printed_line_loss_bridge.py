"""CPU contracts of the executed Cu-loss realization bridge and seven gates.

Synthetic fixtures are not field receipts. They exercise the real capture,
digest, consumer checks, the unchanged original reduce_pair (with only its
file/field I/O substituted) and the restored original transfer helpers.
"""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
import ast, hashlib, json, math
import numpy as np, pytest, torch

import scripts.dev.fdtd.printed_line_loss_bridge as bridge
import scripts.dev.fdtd.printed_line_loss_execution as execution
from scripts.dev.fdtd.printed_line_broadband import encode
from scripts.dev.fdtd.printed_line_loss_basis import FoilRealization, LossBasis
from scripts.dev.fdtd.printed_line_loss_recovery import RecoveryRefused, recovery_receipt
from scripts.dev.fdtd.printed_line_public_execution import actual_clock, build_public_column, column_case, operator_signature
from scripts.dev.fdtd.printed_line_two_length import real_basis
from witwin.maxwell.fdtd.copper_foil import copper_spectral_power

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / 'docs/assessments/printed-lines-2026-10-28'
FREQ = (.8e9, .9e9, 1e9, 1.1e9, 1.2e9)
DT = 2.7797007933179227e-12
CLOCK = actual_clock(DT, column_case(0))
N = CLOCK['total_steps']
IDENTITY = dict(maxwell='a' * 64, core='b' * 64)


def compact_case():
    return replace(column_case(0), axial_half_length=.06, separation=.06,
        padding=.01, positions=(-.015, 0., .015), duration=4e-9, checkpoint_time=2e-9)


@pytest.fixture
def archived_sources():
    if not DOCS.exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-28')


@pytest.fixture(scope='module')
def contour_helpers():
    # The printed-line builders load their contour helpers from this evidence directory.
    if not (ROOT/'docs/assessments/printed-lines-2026-10-22').exists():
        pytest.skip('repository evidence not in this checkout: docs/assessments/printed-lines-2026-10-22')


@pytest.fixture(scope='module')
def context(contour_helpers):
    return build_public_column('cpw', compact_case())


def make_entry(face, component, seed, *, k=5, m=3, imaginary=False):
    g = torch.Generator().manual_seed(seed)
    B = torch.rand((m, 2), generator=g, dtype=torch.float64) * 3e4
    return dict(face=face, electric_name=component + ('_imag' if imaginary else ''), imaginary=imaginary,
        indices=torch.tensor([[2 + j, 3 + seed, 5] for j in range(k)], dtype=torch.int64),
        area=torch.rand(k, generator=g, dtype=torch.float64) * 1e-6 + 1e-6, B=B,
        rates=torch.rand(m, generator=g, dtype=torch.float64) * 1e10 + 1e9,
        D=torch.tensor([[2., -.5], [-.5, 2.]], dtype=torch.float64) * 1e-3)


def fake_solver(entries):
    names = [p + c for p in ('eps_', 'sigma_e_') for c in ('Ex', 'Ey', 'Ez')]
    names += [p + c for p in ('mu_', 'sigma_m_') for c in ('Hx', 'Hy', 'Hz')]
    names += [f'c{c.lower()}_{k}' for c in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz') for k in ('curl', 'decay')]
    return NS(_copper_foils=entries, _port_runtimes=[], _electric_source_terms=[], _magnetic_source_terms=[],
        _compiled_sources=[], **{n: torch.ones(2) for n in names})


def records_for(entries, seed=7):
    g = torch.Generator().manual_seed(seed)
    records = {}
    frequencies = torch.tensor(FREQ, dtype=torch.float64)
    for entry in (e for e in entries if not e['imaginary']):
        k = len(entry['indices'])
        fields = torch.complex(torch.randn((5, 2, k), generator=g, dtype=torch.float64),
            torch.randn((5, 2, k), generator=g, dtype=torch.float64))
        total, edge = copper_spectral_power(fields, frequencies, DT, entry['B'], entry['rates'], entry['D'], entry['area'], return_edge_power=True)
        record = records.setdefault(entry['face'].name, dict(frequencies=frequencies.clone(), spectral_valid=torch.ones(5, dtype=torch.bool),
            coordinate_frame='cartesian', periodic_axes=(), surface_power=torch.zeros(5, dtype=torch.float64),
            surface_power_traces=[], bulk_electric_traces=[], dissipated_energy=0., stored_energy=0.))
        record['surface_power'] = record['surface_power'] + total
        record['surface_power_traces'].append(dict(component=entry['electric_name'], indices=entry['indices'].clone(), power=edge,
            area=entry['area'].clone(), normal_axis='z', normal_coordinate=.01))
        record['bulk_electric_traces'].append(dict(component=entry['electric_name'], indices=entry['indices'].clone(),
            normal_axis='z', normal_coordinate=.01, electric_fields=fields,
            sampling=dict(owner=entry['face'].name, component=entry['electric_name'], planned_steps=N,
                completed_steps=torch.tensor(float(N), dtype=torch.float64), window='none', dt=DT, normalize_source=False,
                frequencies=frequencies.clone(), start_steps=torch.zeros(5, dtype=torch.float64),
                weight_sums=torch.full((5,), float(N), dtype=torch.float64))))
    return records


def fake_result(records, *, ports=None, monitors=None, steps=N):
    ports = ports or {}
    return NS(conductor_losses=records, frequencies=FREQ, port=lambda name: ports[name],
        monitor=lambda name, **_: monitors[name],
        stats=lambda: dict(time_steps=steps, steps_run=steps, dt=DT, normalize_source=False, dft_window='none'))


def basis_and_operator(context, *, imaginary=False):
    signal, ground = NS(name='signal', axis=2, normal_index=5), NS(name='ground', axis=2, normal_index=5)
    entries = []
    for seed, (face, component) in enumerate(((signal, 'Ex'), (signal, 'Ey'), (ground, 'Ey'))):
        entries.append(make_entry(face, component, seed))
        if imaginary:
            twin = dict(entries[-1], electric_name=component + '_imag', imaginary=True)
            entries.append(twin)
    operator = dict(Norton='1' * 64, material='2' * 64, sources='3' * 64, copper=operator_signature(fake_solver(entries))['copper'])
    records = records_for(entries)
    from witwin.maxwell.scene import prepare_scene
    contract = bridge.spectral_contract(identity=IDENTITY, grid=bridge.grid_of(prepare_scene(context['observed'])),
        operator=operator, context=context, total_steps=N)
    realizations = []
    for entry in (e for e in entries if not e['imaginary']):
        realizations.append(FoilRealization(entry['face'].name, entry['electric_name'], 'z', 5,
            ('real', 'imaginary') if imaginary else ('real',),
            *(entry[k].clone() for k in ('indices', 'area', 'B', 'rates', 'D')), torch.full((5,), float(N), dtype=torch.float64)))
    return entries, operator, records, LossBasis(FREQ, DT, contract, tuple(realizations), records)


def complex_sources(context, records, scale=1.):
    sources, ports = {}, {}
    for number, group in enumerate(context['groups']):
        voltage = torch.tensor([1 + .1j * f for f in range(5)], dtype=torch.complex128) * (1 + number) * scale
        for name in group:
            power = voltage.abs().square() / (8. * 50.)
            ports[name] = NS(available_power=power.clone())
            sources[name] = dict(representation=bridge.SOURCE_REPRESENTATIONS[0], thevenin_peak_voltage_V=voltage.clone(),
                resistance_ohm=50., available_power_W=power.clone())
    return sources, ports


def evidence(executed, **override):
    value = dict(record_schema='public-printed-line-column/1', kind='cpw', length_index=0, drive_column=0,
        real_basis=[1., .25], clock=dict(CLOCK), operator=dict(executed), identity=dict(IDENTITY),
        source_identity_unchanged=True, graph=True, capture_binding_sha256=None)
    value.update(override)
    return value


def make_capture(context, **kw):
    entries, operator, records, basis = basis_and_operator(context, **kw)
    sources, ports = complex_sources(context, records)
    capture = bridge.build_capture(basis, sources, provenance=bridge.PROVENANCES[0], operator=operator, clock=dict(CLOCK),
        identity=IDENTITY, kind='cpw', length_index=0, drive_column=0)
    return entries, operator, records, capture, ports


@pytest.mark.parametrize('imaginary', [False, True])
def test_realization_digest_reproduces_executed_operator_copper_signature(context, imaginary):
    entries, operator, _, capture, _ = make_capture(context, imaginary=imaginary)
    assert bridge.copper_operator_signature(capture['basis'].realizations) == operator['copper']
    tampered = replace(capture['basis'].realizations[0], B=capture['basis'].realizations[0].B * (1 + 2 ** -52))
    assert bridge.copper_operator_signature((tampered, *capture['basis'].realizations[1:])) != operator['copper']


def test_capture_refuses_realization_not_matching_executed_signature(context):
    _, operator, records, basis = basis_and_operator(context)
    sources, _ = complex_sources(context, records)
    with pytest.raises(ValueError, match='copper-operator signature'):
        bridge.build_capture(basis, sources, provenance=bridge.PROVENANCES[0], operator=dict(operator, copper='0' * 64),
            clock=dict(CLOCK), identity=IDENTITY, kind='cpw', length_index=0, drive_column=0)
    with pytest.raises(ValueError, match='clock'):
        bridge.build_capture(basis, sources, provenance=bridge.PROVENANCES[0], operator=operator,
            clock=dict(CLOCK, total_steps=N + 1), identity=IDENTITY, kind='cpw', length_index=0, drive_column=0)


def test_saved_capture_round_trip_and_every_tamper_is_rejected(context, tmp_path):
    _, _, _, capture, _ = make_capture(context)
    folder = tmp_path / 'ok'; folder.mkdir()
    manifest = bridge.save_capture(folder, capture)
    loaded = bridge.load_capture(folder)
    assert loaded['manifest']['binding_sha256'] == manifest['binding_sha256']
    with pytest.raises(FileExistsError):
        bridge.save_capture(folder, capture)
    # 1. Tampered realization with a consistently rewritten file hash.
    bad = tmp_path / 'bad_B'; bad.mkdir()
    bridge.save_capture(bad, capture)
    payload = torch.load(bad / bridge.CAPTURE_FILE, weights_only=False)
    first = payload['basis'].realizations[0]
    payload['basis'] = replace(payload['basis'], realizations=(replace(first, rates=first.rates * (1 + 1e-12)), *payload['basis'].realizations[1:]))
    torch.save(payload, bad / bridge.CAPTURE_FILE)
    with pytest.raises(ValueError, match='file changed'):
        bridge.load_capture(bad)
    meta = json.loads((bad / bridge.MANIFEST_FILE).read_text()); meta['capture_file_sha256'] = bridge.sha256_file(bad / bridge.CAPTURE_FILE)
    (bad / bridge.MANIFEST_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='digest'):
        bridge.load_capture(bad)
    # 2. Rewritten manifest digest without a new binding.
    meta['realization_sha256'] = '0' * 64
    (bad / bridge.MANIFEST_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='manifest changed'):
        bridge.load_capture(bad)
    # 3. Rewritten executed operator and recomputed binding: digest proof fails.
    meta = json.loads((folder / bridge.MANIFEST_FILE).read_text())
    meta['executed_operator']['copper'] = meta['executed_copper_operator_signature'] = '0' * 64
    meta['binding_sha256'] = bridge._binding(meta)
    other = tmp_path / 'bad_operator'; other.mkdir()
    payload = torch.load(folder / bridge.CAPTURE_FILE, weights_only=False); payload['binding_sha256'] = meta['binding_sha256']
    torch.save(payload, other / bridge.CAPTURE_FILE); meta['capture_file_sha256'] = bridge.sha256_file(other / bridge.CAPTURE_FILE)
    (other / bridge.MANIFEST_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='executed_copper_operator_signature'):
        bridge.load_capture(other)


def test_provenance_binds_column_operator_identity_and_contract(context):
    _, operator, _, capture, _ = make_capture(context)
    assert bridge.check_provenance(context, capture, evidence(operator))['provenance'] == bridge.PROVENANCES[0]
    for override, message in ((dict(drive_column=1), 'different physical columns'),
            (dict(operator=dict(operator, Norton='9' * 64)), 'executed operator'),
            (dict(identity=dict(IDENTITY, core='c' * 64)), 'executed operator'),
            (dict(capture_binding_sha256='0' * 64), 'different realization'),
            (dict(source_identity_unchanged=False), 'source identity changed')):
        with pytest.raises(ValueError, match=message):
            bridge.check_provenance(context, capture, evidence(operator, **override))
    shifted = dict(capture, basis=replace(capture['basis'], execution_contract=replace(capture['basis'].execution_contract, observation_sha256='0' * 64)))
    with pytest.raises(ValueError, match='contract'):
        bridge.check_provenance(context, shifted, evidence(operator))


def test_clock_mismatches_are_rejected(context):
    _, operator, records, capture, _ = make_capture(context)
    ctx = dict(context, case=column_case(0))
    assert bridge.check_clock(ctx, fake_result(records), capture, evidence(operator))['total_steps'] == N
    with pytest.raises(ValueError, match='complete physical horizon'):
        bridge.check_clock(ctx, fake_result(records), capture, evidence(operator, clock=dict(CLOCK, total_steps=N - 1)))
    with pytest.raises(ValueError, match='complete raw executed horizon'):
        bridge.check_clock(ctx, fake_result(records, steps=N - 1), capture, evidence(operator))
    for key, value in (('completed_steps', torch.tensor(float(N - 1), dtype=torch.float64)), ('window', 'hanning'),
            ('weight_sums', torch.full((5,), float(N - 3), dtype=torch.float64)), ('dt', DT * (1 + 1e-15))):
        broken = {face: dict(record, bulk_electric_traces=[dict(t, sampling=dict(t['sampling'])) for t in record['bulk_electric_traces']])
            for face, record in records.items()}
        broken['signal']['bulk_electric_traces'][0]['sampling'][key] = value
        with pytest.raises(ValueError, match='sampling clock'):
            bridge.check_clock(ctx, fake_result(broken), capture, evidence(operator))


def test_coherent_loss_replay_binds_realization_to_executed_records(context):
    _, operator, records, capture, _ = make_capture(context)
    proof = bridge.check_coherent_loss(fake_result(records), capture)
    assert proof['max_relative_replay_error'] <= bridge.LOSS_REPLAY_RELATIVE
    for field, factor in (('B', 1 + 1e-9), ('rates', 1 + 1e-9), ('D', 1 + 1e-9), ('area', 1 + 1e-9)):
        first = capture['basis'].realizations[0]
        tampered = replace(capture['basis'], realizations=(replace(first, **{field: getattr(first, field) * factor}), *capture['basis'].realizations[1:]))
        with pytest.raises(ValueError, match='not reproduced'):
            bridge.check_coherent_loss(fake_result(records), dict(capture, basis=tampered))
    edited = {face: dict(record, surface_power=record['surface_power'] * 1.001) for face, record in records.items()}
    with pytest.raises(ValueError, match='executed records'):
        bridge.check_coherent_loss(fake_result(edited), capture)


def tail_payload(peak=2., last=1e-4, samples=None):
    count = len(range(0, N, 32)) if samples is None else samples
    times = torch.arange(count, dtype=torch.float64) * 32 * DT
    field = torch.full((count,), last, dtype=torch.float64); field[10] = peak
    return dict(t=times, field=field)


def test_tail_field_is_complete_and_uses_original_last2ns_rule(context):
    ctx = dict(context)
    names = [m.name for m in bridge.tail_monitors(context)]
    good = {name: tail_payload() for name in names}
    records = bridge.check_tails(ctx, fake_result({}, monitors=good), CLOCK)
    assert [r['last2ns_peak_relative'] for r in records] == [5e-5] * len(names)
    for broken, message in ((tail_payload(samples=719), 'complete executed interval'), (tail_payload(peak=0., last=0.), 'no field'),
            (dict(tail_payload(), field=torch.full((720,), math.nan)), 'complete executed interval')):
        with pytest.raises(ValueError, match=message):
            bridge.check_tails(ctx, fake_result({}, monitors=dict(good, **{names[0]: broken})), CLOCK)


def test_source_spectra_bind_to_executed_available_power(context):
    _, operator, records, capture, ports = make_capture(context)
    ratios = bridge.check_sources(context, fake_result(records, ports=ports), capture)
    assert set(ratios) == {n for g in context['groups'] for n in g}
    assert all(r == pytest.approx(1 / math.sqrt(1 + .16)) for r in ratios.values())
    edited = dict(ports); name = next(iter(edited)); edited[name] = NS(available_power=ports[name].available_power * 1.01)
    with pytest.raises(ValueError, match='available power'):
        bridge.check_sources(context, fake_result(records, ports=edited), capture)
    magnitudes = bridge.available_power_source_magnitudes(fake_result(records, ports=ports), {n: 50. for n in ports})
    assert bridge.check_sources(context, fake_result(records, ports=ports), dict(capture, source_spectra=magnitudes))


# ---------------------------------------------------------------- seven gates

@pytest.mark.usefixtures('archived_sources')
def test_restored_original_helpers_are_byte_identical_to_archived_sources():
    pairs = (('scripts/dev/fdtd/printed_line_two_length.py', DOCS / 'two-length-g5-preparation/private-source-text/scripts/dev/fdtd/printed_line_two_length.py.txt'),
        ('scripts/dev/fdtd/negative_vi_complete_basis.py', DOCS / 'complete-basis-prefix6-current740/code/scripts/dev/fdtd/negative_vi_complete_basis.py.txt'))
    for maintained, archived in pairs:
        assert (ROOT / maintained).read_bytes() == archived.read_bytes()


def _literal_dict(path, function=None, target=None, keyword=None):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and target and any(getattr(t, 'id', None) == target for t in node.targets):
            return {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
        if isinstance(node, ast.Dict) and keyword and any(isinstance(k, ast.Constant) and k.value == keyword for k in node.keys):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == keyword:
                    return {kk.value: ast.literal_eval(vv) for kk, vv in zip(v.keys, v.values)}


@pytest.mark.usefixtures('archived_sources')
def test_seven_gate_names_and_thresholds_are_the_original_ones():
    source = Path(execution.__file__)
    release = _literal_dict(source, keyword='release_gates')
    archived = _literal_dict(DOCS / 'coupled-full64-current740/code/scripts/dev/fdtd/negative_vi_full64.py.txt', target='GATES')
    T = bridge.GATE_THRESHOLDS
    assert (T['state_condition_number_max'], T['solve_residual_max'], T['tail_peak_relative_max'], T['line_power_balance_relative']) == (
        release['state_condition_number_max'], release['solve_residual_max'], release['tail_peak_relative_max'], release['line_power_balance_relative'])
    assert (T['state_condition_number_max'], T['solve_residual_max'], T['tail_peak_relative_max'], T['line_power_balance_relative'], T['source_band_min_relative']) == (
        archived['physical_state_condition_max'], archived['coherent_solve_residual_max'], archived['tail_last2ns_peak_relative'],
        archived['physical_power_balance_relative'], archived['source_band_min_relative'])
    tree = ast.parse(source.read_text())
    reduce_pair = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reduce_pair')
    gates = next(n for n in ast.walk(reduce_pair) if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'gate_results')
    assert tuple(k.value for k in gates.value.keys) == bridge.SEVEN_GATES


def analytic_states(positions, *, alpha=.4, Z0=50 * (1.05 + .03j), drives=((1., .3 + .2j), (.4 - .1j, 1.))):
    z = np.asarray(positions)
    out = []
    for a, b in drives:
        state = np.zeros((len(z), 2, 5), dtype=np.complex128)
        for f, frequency in enumerate(FREQ):
            gamma = 2 * np.pi * frequency * 1.5 / 299792458. + 1j * alpha
            state[:, 0, f] = a * np.exp(1j * gamma * z) + b * np.exp(-1j * gamma * z)
            state[:, 1, f] = (a * np.exp(1j * gamma * z) - b * np.exp(-1j * gamma * z)) / Z0
        out.append(state)
    return out


class FakeCoherent:
    seen = []

    def __init__(self, results, bases, coefficients):
        FakeCoherent.seen.append(np.array(coefficients, copy=True))
        self.coefficients = np.asarray(coefficients)


def fake_power(scale):
    def power(observation, scene, schema, bounds):
        return dict(balance_relative_to_entering=[scale * 1e-3 * float(np.abs(c).sum()) for c in observation.coefficients])
    return power


@pytest.mark.usefixtures('contour_helpers')
@pytest.mark.parametrize('variant', ['pass', 'tail', 'band', 'operator', 'power'])
def test_seven_gates_and_coefficients_identical_to_unchanged_reduce_pair(tmp_path, monkeypatch, variant):
    case = execution.PilotCase(positions=(-.015, 0., .015))
    states = analytic_states(case.positions)
    sources = [{p: np.array([1., .9, .8, .7, .6 if not (variant == 'band' and d == 1) else 1e-4], dtype=np.complex128) * (1 + .2j)
        for p in ('left', 'right')} for d in range(2)]
    tails = [[2e-4, 3e-4], [4e-4, 2e-3 if variant == 'tail' else 5e-4]]
    operators = ['op', 'op2' if variant == 'operator' else 'op']
    scale = 50. if variant == 'power' else 1.
    folders = []
    for d in range(2):
        folder = tmp_path / f'drive{d}'; folder.mkdir(); folders.append(folder)
        files = {}
        for name in ('basis.pt', 'result.pt', 'final-fields.pt', 'input.json'):
            (folder / name).write_bytes(name.encode()); files[name] = hashlib.sha256(name.encode()).hexdigest()
        report = dict(case=encode(asdict(case)), amplitudes=list(execution.DRIVE_BASIS[d]), source_before={'s': 1}, source_after={'s': 1},
            all_source_identity_unchanged=True, test_step_override=None, actual_dt_s=DT, planned_steps=math.ceil(case.duration / DT),
            saved_files_sha256=files, physical_state=states[d], port_operator_sha256=operators[d],
            actual_source_spectra={p: {'thevenin_peak_voltage': v} for p, v in sources[d].items()},
            tails={f't{j}': {'last2ns_peak_relative': v} for j, v in enumerate(tails[d])}, checkpoint=None)
        (folder / 'execution.json').write_text(json.dumps(encode(report)))
    for module in (execution, bridge):
        monkeypatch.setattr(module, 'CoherentObservations', FakeCoherent)
        monkeypatch.setattr(module, 'regional_plane_power', fake_power(scale))
    monkeypatch.setattr(execution, 'build_case', lambda c: ('scene', None, 'bounds', 'schema', None, None, None))
    monkeypatch.setattr(execution, 'torch', NS(load=lambda *a, **k: 'basis'))
    monkeypatch.setattr(execution, 'mw', NS(Result=NS(load=lambda *a, **k: 'result')))
    monkeypatch.setattr(execution, 'analytic_homogeneous_line', lambda *a, **k: {})
    FakeCoherent.seen = []
    original = execution.reduce_pair(case, folders)
    original_coefficients = FakeCoherent.seen.pop()
    columns = [dict(kind='cpw', length_index=0, drive_column=d, real_basis=real_basis(2)[:, d], clock=CLOCK,
        VI=torch.tensor(states[d]), result='result', basis=NS(frequencies=FREQ), port_operator_sha256=operators[d],
        source_identity_unchanged=True, binding_sha256=f'b{d}', **column_meta(Norton=operators[d]),
        source_band_ratios=[float(np.abs(v).min() / np.abs(v).max()) for v in sources[d].values()], tail_values=tails[d])
        for d in range(2)]
    ours = bridge.reduce_public_basis('cpw', columns, cpu_scene='scene', regional_schema='schema', regional_bounds='bounds')
    assert ours['original_seven_gates'] == original['pilot_functional_gates']
    assert list(ours['original_seven_gates']) == list(bridge.SEVEN_GATES)
    assert np.array_equal(FakeCoherent.seen.pop(), original_coefficients)
    for mine, theirs in zip(ours['frequency_records'], original['frequency_records']):
        assert np.array_equal(mine['positive_power_states'][0]['coherent_coefficients'], theirs['coherent_coefficients'])
    assert ours['original_seven_gates_passed'] == (variant == 'pass') == original['pilot_functional_passed']


def column_meta(**operator):
    return dict(executed_operator=dict(dict(Norton='n', material='m', copper='c', sources='s'), **operator),
        executed_identity=dict(IDENTITY), grid_sha256='g', case=dict(separation=.4))


def test_incomplete_basis_or_reused_capture_is_rejected():
    states = analytic_states((-.015, 0., .015))
    columns = [dict(kind='cpw', length_index=0, drive_column=d, real_basis=real_basis(2)[:, d], clock=CLOCK, VI=states[d],
        binding_sha256='same', port_operator_sha256='op', source_identity_unchanged=True, source_band_ratios=[1.], tail_values=[0.],
        **column_meta()) for d in range(2)]
    with pytest.raises(ValueError, match='original physical lengths'):
        bridge.reduce_public_basis('cpw', [dict(c, length_index=None) for c in columns], cpu_scene=None, regional_schema=None, regional_bounds=None)
    with pytest.raises(ValueError, match='complete'):
        bridge.reduce_public_basis('cpw', columns[:1], cpu_scene=None, regional_schema=None, regional_bounds=None)
    with pytest.raises(ValueError, match='own executed realization'):
        bridge.reduce_public_basis('cpw', columns, cpu_scene=None, regional_schema=None, regional_bounds=None)
    columns[1] = dict(columns[1], real_basis=[1., .25])
    with pytest.raises(ValueError, match='ordered original real basis'):
        bridge.reduce_public_basis('cpw', columns, cpu_scene=None, regional_schema=None, regional_bounds=None)


def two_length_reports(tmp_path, edits=None):
    edits = edits or {}
    from scripts.dev.fdtd.printed_line_two_length import physical_transfer
    paths = []
    for length in (0, 1):
        case = column_case(length)
        states = analytic_states(case.positions, alpha=.3)
        V = np.stack([x[:, 0:1, :] for x in states], -1); I = np.stack([x[:, 1:2, :] for x in states], -1)
        records = [dict(frequency_hz=f, transfers=[dict(station_pair=(0, 2), matrix=physical_transfer(V[[0, 2], :, i], I[[0, 2], :, i])['matrix'])])
            for i, f in enumerate(FREQ)]
        operator = dict(Norton=f'n{length}', material='m', copper='c', sources='s')
        report = dict(schema=bridge.BASIS_SCHEMA, kind='cpw', length_index=length, original_seven_gates_passed=True, frequency_records=records,
            postprocess_identity=dict(IDENTITY), executed_identity=dict(IDENTITY), case=asdict(case), executed_operators=[operator, operator],
            grid_sha256='g', clock=dict(total_steps=N, dt_s=DT, checkpoint_step=CLOCK['checkpoint_step']), reference_stations_m=case.positions)
        for key, value in edits.get(length, {}).items():
            report[key] = value(report) if callable(value) else value
        path = tmp_path / f'L{length}.json'; path.write_text(json.dumps(encode(report))); paths.append(path)
    return paths


def test_independent_two_length_extraction_recovers_analytic_line(tmp_path):
    paths = two_length_reports(tmp_path)
    out = bridge.reduce_public_two_lengths('cpw', *paths, current_identity=IDENTITY)
    assert not out['failed_physical_gates'] and set(out['input_report_sha256']) == {str(p) for p in paths}
    for i, row in enumerate(out['frequency_records']):
        mode = row['independent_two_length']['forward_modes'][0]
        assert mode['alpha_np_per_m'] == pytest.approx(.3, rel=1e-9)
        assert mode['beta_rad_per_m'] == pytest.approx(2 * np.pi * FREQ[i] * 1.5 / 299792458., rel=1e-9)


@pytest.mark.parametrize('length,edit,message', [
    (0, dict(original_seven_gates_passed=False), 'seven gates'),
    (1, dict(postprocess_identity=dict(IDENTITY, core='c' * 64)), 'postprocess'),
    (1, dict(executed_identity=dict(IDENTITY, maxwell='c' * 64)), 'identity'),
    (1, dict(case=lambda r: dict(r['case'], sigma=.02)), 'original fixed case'),
    (1, dict(executed_operators=lambda r: [dict(o, material='x') for o in r['executed_operators']]), 'material'),
    (1, dict(executed_operators=lambda r: [dict(o, copper='x') for o in r['executed_operators']]), 'copper'),
    (1, dict(executed_operators=lambda r: [dict(o, sources='x') for o in r['executed_operators']]), 'sources'),
    (1, dict(executed_operators=lambda r: [dict(o, Norton='n0') for o in r['executed_operators']]), 'Norton'),
    (1, dict(executed_operators=lambda r: [r['executed_operators'][0], dict(r['executed_operators'][0], copper='y')]), 'differing executed operators'),
    (1, dict(grid_sha256='h'), 'grid'),
    (1, dict(clock=lambda r: dict(r['clock'], total_steps=N + 1)), 'clock'),
    (1, dict(reference_stations_m=lambda r: [r['reference_stations_m'][0] + 1e-9, *r['reference_stations_m'][1:]]), 'stations'),
    (1, dict(length_index=0), 'ordered'),
])
def test_two_length_contract_mismatch_is_rejected(tmp_path, length, edit, message):
    paths = two_length_reports(tmp_path, {length: edit})
    with pytest.raises(ValueError, match=message):
        bridge.reduce_public_two_lengths('cpw', *paths, current_identity=IDENTITY)


def coupled_states(positions, drives):
    """Two decoupled channel lines mapped back to the original coupled rows."""
    z = np.asarray(positions); out = []
    for amplitudes in drives:
        raw = np.zeros((3, 2, 2, 5), dtype=np.complex128)
        for f, frequency in enumerate(FREQ):
            k0 = 2 * np.pi * frequency / 299792458.
            channel = []
            for (a, b), (index, alpha, Z) in zip(amplitudes, ((1.5, .3, 60.), (1.7, .5, 35.))):
                g = k0 * index + 1j * alpha
                channel.append((a * np.exp(1j * g * z) + b * np.exp(-1j * g * z), (a * np.exp(1j * g * z) - b * np.exp(-1j * g * z)) / Z))
            (V1, I1), (V2, I2) = channel
            raw[:, 1, 0, f] = V2; raw[:, 0, 0, f] = V1 + .5 * V2
            raw[:, 0, 1, f] = I1; raw[:, 1, 1, f] = I2 + .5 * I1
        out.append(raw.reshape(6, 2, 5))
    return out


def test_differential_pair_four_column_basis_uses_channel_transform(monkeypatch):
    from scripts.dev.fdtd.negative_vi_complete_basis import channels
    positions = column_case(0).positions
    drives = (((1, .2), (.3, .1j)), ((.2j, 1), (.1, .4)), ((.3, .1), (1, .2j)), ((.1, .3), (.2, 1)))
    states = coupled_states(positions, drives)
    V, I = channels('differential_pair', states[0])
    assert V.shape == (3, 2, 5)
    monkeypatch.setattr(bridge, 'CoherentObservations', FakeCoherent)
    monkeypatch.setattr(bridge, 'regional_plane_power', fake_power(1.))
    FakeCoherent.seen = []
    columns = [dict(kind='differential_pair', length_index=0, drive_column=d, real_basis=real_basis(4)[:, d], clock=CLOCK,
        VI=torch.tensor(states[d]), result='r', basis=NS(frequencies=FREQ), port_operator_sha256='n', source_identity_unchanged=True,
        binding_sha256=f'b{d}', source_band_ratios=[.9], tail_values=[1e-4], **column_meta()) for d in range(4)]
    out = bridge.reduce_public_basis('differential_pair', columns, cpu_scene=None, regional_schema=None, regional_bounds=None)
    assert out['original_seven_gates_passed'] and len(FakeCoherent.seen) == 2
    L = positions[2] - positions[0]
    for f, record in enumerate(out['frequency_records']):
        k0 = 2 * np.pi * FREQ[f] / 299792458.
        expected = sorted(np.exp(1j * (k0 * n + 1j * a) * L) for n, a in ((1.5, .3), (1.7, .5)))
        got = sorted(m['eigenvalue'] for m in record['positive_power_states'])
        np.testing.assert_allclose(got, expected, rtol=1e-9)
    mismatched = [dict(c, **column_meta(copper='x')) if c['drive_column'] == 3 else c for c in columns]
    FakeCoherent.seen = []
    out = bridge.reduce_public_basis('differential_pair', mismatched, cpu_scene=None, regional_schema=None, regional_bounds=None)
    assert not out['original_seven_gates']['same_actual_port_operator'] and not out['original_seven_gates']['coherent_regional_power_balance']
    assert FakeCoherent.seen == [] and not out['cross_column_precheck']['material_and_copper_common']


@pytest.mark.parametrize('rotation', [-1, 1j, -1j])
def test_balanced_group_requires_identical_signed_complex_drive(context, rotation):
    _, operator, records, capture, ports = make_capture(context)
    group = context['groups'][0]
    sources = {k: dict(v) for k, v in capture['source_spectra'].items()}
    sources[group[1]]['thevenin_peak_voltage_V'] = sources[group[1]]['thevenin_peak_voltage_V'] * rotation
    with pytest.raises(ValueError, match='signed complex drive'):
        bridge.check_sources(context, fake_result(records, ports=ports), dict(capture, source_spectra=sources))
    magnitudes = bridge.available_power_source_magnitudes(fake_result(records, ports=ports), {n: 50. for n in ports})
    assert bridge.balanced_group_comparison(dict(capture, source_spectra=magnitudes)).startswith('magnitude only')
    assert bridge.balanced_group_comparison(capture).startswith('signed complex')


# ---------------------------------------------------------------- recovery

def v3_completion(operator):
    return dict(qualified=True, ordinary_prepared_run_returned_complete_Result=True,
        column=dict(kind='cpw', length_index=0, drive_column=0, graph=True),
        clock=dict(CLOCK, physical_advances=N, primary_preparations=2, primary_solve_invocations=2), actual_operator=operator,
        actual_package=dict(source_sha256=IDENTITY['maxwell']), actual_core=dict(source_sha256=IDENTITY['core']),
        source_before={'f': '1'}, source_after={'f': '1'})


@pytest.mark.parametrize('failure', [None, 'B', 'rates', 'D', 'area', 'indices', 'norton', 'clock', 'orphan'])
def test_recovery_receipt_requires_hash_match_else_refuses(context, failure):
    entries, operator, records, basis = basis_and_operator(context)
    _, ports = complex_sources(context, records)
    result = fake_result(records, ports=ports)
    recompiled = [dict(e) for e in entries]
    recompiled_operator, clock = dict(operator), dict(CLOCK)
    if failure in ('B', 'rates', 'D', 'area'):
        recompiled[1][failure] = recompiled[1][failure] * (1 + 2 ** -50)
        # The operator report still claims the executed copper digest; only
        # the extracted realization itself can prove the match.
    elif failure == 'indices':
        recompiled[0]['indices'] = recompiled[0]['indices'].flip(0)
    elif failure == 'norton':
        recompiled_operator['Norton'] = '0' * 64
    elif failure == 'clock':
        clock = actual_clock(DT * 1.0001, column_case(0))
    elif failure == 'orphan':
        recompiled.append(dict(recompiled[0], electric_name='Ez_imag', imaginary=True))
    kwargs = dict(recompiled_entries=recompiled, recompiled_operator=recompiled_operator, recompiled_clock=clock,
        recompiled_grid={'x': np.zeros(2)}, resistances={n: 50. for n in ports})
    if failure is None:
        receipt, capture, _ = recovery_receipt(context, result, v3_completion(operator), **kwargs)
        assert receipt['matched'] and capture['manifest']['provenance'] == bridge.PROVENANCES[1]
        assert capture['manifest']['executed_copper_operator_signature'] == operator['copper']
        assert not capture['manifest']['reconstructed_from_declared_material']
        assert set(next(iter(capture['source_spectra'].values()))) >= {'thevenin_peak_voltage_magnitude_V'}
        assert bridge.check_coherent_loss(result, capture)['max_relative_replay_error'] <= bridge.LOSS_REPLAY_RELATIVE
    else:
        with pytest.raises(RecoveryRefused):
            recovery_receipt(context, result, v3_completion(operator), **kwargs)


def test_public_runner_captures_realization_before_release_and_records_binding():
    import scripts.dev.fdtd.printed_line_public_execution as public
    tree = ast.parse(Path(public.__file__).read_text())
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_public_column')
    calls = [n.func.id for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.index('capture_executed_realization') < calls.index('save_capture')
    text = ast.unparse(body)
    assert "executed_Cu_realization" in text and "'executed-Cu-realization.pt'" in text
    assert 'reduce_pair' not in text and "'physical-loss-report.json'" not in text


def test_recovered_column_loader_requires_its_matched_receipt_and_pins(context, tmp_path):
    entries, operator, records, _ = basis_and_operator(context)
    _, ports = complex_sources(context, records)
    receipt, capture, _ = recovery_receipt(context, fake_result(records, ports=ports), v3_completion(operator),
        recompiled_entries=entries, recompiled_operator=operator, recompiled_clock=dict(CLOCK),
        recompiled_grid={'x': np.zeros(2)}, resistances={n: 50. for n in ports})
    run = tmp_path / 'run'; run.mkdir()
    pins = {}
    for name in ('completion.json', 'result.pt', 'actual-public-VI.pt'):
        (run / name).write_bytes(name.encode()); pins[str(run / name)] = hashlib.sha256(name.encode()).hexdigest()
    folder = tmp_path / 'recovery'; folder.mkdir()
    bridge.save_capture(folder, capture)
    receipt['raw_pins'] = pins

    def write(target, value):
        (target / 'recovery-receipt.json').write_text(json.dumps(encode(value)))
        return bridge.sha256_file(target / 'recovery-receipt.json')
    sha = write(folder, receipt)
    with pytest.raises(ValueError, match='pinned SHA-256'):
        bridge.load_recovered_column(context, run, folder, '0' * 64)
    for edit, message in ((dict(raw_pins={}), 'pin every loaded'),
            (dict(raw_pins={k: v for k, v in pins.items() if not k.endswith('VI.pt')}), 'pin every loaded'),
            (dict(matched=False), 'matched recovery receipt'), (dict(binding_sha256='0' * 64), 'matched recovery receipt'),
            (dict(raw_pins=dict(pins, **{str(run / 'result.pt'): '0' * 64})), 'changed since its receipt'),
            (dict(raw_pins=dict(pins, **{str(tmp_path / 'other.pt'): '0' * 64})), 'changed since its receipt')):
        sha = write(folder, dict(receipt, **edit))
        with pytest.raises(ValueError, match=message):
            bridge.load_recovered_column(context, run, folder, sha)
    direct = tmp_path / 'direct'; direct.mkdir()
    _, _, _, direct_capture, _ = make_capture(context)
    bridge.save_capture(direct, direct_capture)
    sha = write(direct, dict(receipt, binding_sha256=direct_capture['manifest']['binding_sha256']))
    with pytest.raises(ValueError, match='matched recovery receipt'):
        bridge.load_recovered_column(context, run, direct, sha)
