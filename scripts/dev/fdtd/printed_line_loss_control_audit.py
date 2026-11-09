"""Independently audit a saved declared material-control pilot without field replay."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.printed_line_loss_execution import PilotCase, DRIVE_BASIS, build_case, CoherentObservations, execution_identity
from scripts.dev.fdtd.printed_line_regional_power import regional_plane_power
from scripts.dev.fdtd.printed_line_broadband import encode


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def decode(value):
    if isinstance(value, dict):
        if set(value) == {'real', 'imag'}: return complex(value['real'], value['imag'])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list): return [decode(item) for item in value]
    return value

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-identity', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    before = {str(path): sha(path) for path in sorted(args.input.rglob('*')) if path.is_file()}
    completion = json.loads((args.input / 'completion-audit.json').read_text())
    for name, expected in completion['output_sha256'].items():
        if sha(args.input / name) != expected: raise ValueError(f'Actual output changed: {name}')
    identity = execution_identity()
    if completion['source_before'] != identity or completion['source_after'] != identity:
        raise ValueError('Audit must import the actual executed numerical/core/helper archive.')
    source = json.loads(args.source_identity.read_text()); archive = Path(source['archive'])
    for name, expected in source['archive_files'].items():
        if sha(archive / name) != expected: raise ValueError(f'Frozen source changed: {name}')
    report = decode(json.loads((args.input / 'physical-loss-report.json').read_text()))
    values=dict(report['case'])
    for name in ('frequencies','positions'):values[name]=tuple(values[name])
    case=PilotCase(**values);scene, groups, bounds, schema, rows, expected_probes, tails = build_case(case)
    extracted = decode(json.loads((args.input / 'actual-real-source-basis-extraction.json').read_text()))
    if len(report['frequency_records']) != len(case.frequencies) or tuple(v['frequency_hz'] for v in report['frequency_records']) != case.frequencies:
        raise ValueError('Complete actual frequency axis required.')
    results = []; bases = []; drives = []; reconstructed_states = []; source_tables = []
    for column in range(2):
        folder = args.input / f'drive{column}'
        actual = decode(json.loads((folder / 'execution.json').read_text()))
        assert actual['source_before'] == actual['source_after'] == identity
        assert actual['amplitudes'] == list(DRIVE_BASIS[column]) and actual['test_step_override'] is None
        for name, expected in actual['saved_files_sha256'].items(): assert sha(folder / name) == expected
        dt = actual['actual_dt_s']; count = actual['planned_steps']; split = math.floor(case.checkpoint_time / dt)
        assert count == math.ceil(case.duration / dt)
        checkpoint = mw.FDTDResumeCheckpoint.load(folder / 'checkpoint.pt', map_location='cpu')
        assert checkpoint.step == checkpoint.physics.step == split
        assert checkpoint.total_steps == count and checkpoint.dt == dt
        checkpoint_steps = {key: value.tolist() for key, value in checkpoint.physics.tensors.items()
            if key.startswith('copper_foil_') and (key.endswith('_spectral_step') or key.endswith('_spectral_norm'))}
        assert len(checkpoint_steps) == (12 if case.metal == 'copper' else 0)
        assert all(np.all(np.asarray(value) == split) for value in checkpoint_steps.values())
        fields = torch.load(folder / 'final-fields.pt', weights_only=False, map_location='cpu')
        assert set(fields) == set(('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'))
        assert all(bool(torch.isfinite(value).all()) for value in fields.values())
        assert sum(float(abs(value).max()) for value in fields.values()) > 0
        basis = torch.load(folder / 'basis.pt', weights_only=False, map_location='cpu'); bases.append(basis)
        assert basis.dt == dt and basis.execution_contract.planned_steps == count and basis.frequencies == case.frequencies
        if case.metal == 'pec':assert not basis.realizations and not basis.conductor_records
        else:assert len(basis.realizations) == 6
        assert all(torch.equal(v.spectral_norm, torch.full_like(v.spectral_norm, count)) and v.banks == ('real',) for v in basis.realizations)
        result = mw.Result.load(folder / 'result.pt', scene=scene, map_location='cpu'); results.append(result)
        native = {}
        for name, (component, index) in expected_probes.items():
            payload = result.monitor(name); assert tuple(payload['field_indices'][component]) == index
            assert tuple(payload['frequencies']) == case.frequencies
            native[name] = payload['components'][component].numpy().astype(np.complex128).reshape(len(case.frequencies))
            assert np.array_equal(native[name], np.asarray(actual['native_phasors'][name]))
        state = np.asarray([[sum(native[name] * weight for name, _, weight in row) for row in pair] for pair in rows])
        assert np.array_equal(state, np.asarray(actual['physical_state'])); reconstructed_states.append(state)
        tail_ratios = {}
        for monitor in tails:
            payload = result.monitor(monitor.name); field = np.asarray(payload['field']); times = np.asarray(payload['t'])
            selected = times >= count * dt - 2e-9
            ratio = float(abs(field[selected]).max() / abs(field).max())
            assert ratio == actual['tails'][monitor.name]['last2ns_peak_relative']
            tail_ratios[monitor.name] = ratio
        source_tables.append(actual['actual_source_spectra'])
        drives.append(dict(column=column,actual_dt_s=dt,actual_steps=count,actual_duration_s=count*dt,
            checkpoint_step=split,checkpoint_total_steps=checkpoint.total_steps,checkpoint_copper_spectral_counters=checkpoint_steps,
            final_field_shapes={name:list(value.shape) for name,value in fields.items()},
            final_field_peak={name:float(abs(value).max()) for name,value in fields.items()},
            native_probe_count=len(native),native_VI_reconstruction_exact=True,tail_last2ns_peak_relative=tail_ratios,
            field_graph=actual['actual_field_graph'],observer_graph=actual['actual_observer_graph'],
            actual_port_operator_sha256=actual['port_operator_sha256'],seconds_including_checkpoint_prepare=actual['seconds_including_checkpoint_prepare']))
        del checkpoint, fields
    assert drives[0]['actual_port_operator_sha256'] == drives[1]['actual_port_operator_sha256']
    assert bases[0].execution_contract == bases[1].execution_contract
    realization_checks = []
    for a, b in zip(bases[0].realizations, bases[1].realizations):
        assert (a.face,a.component) == (b.face,b.component)
        assert all(torch.equal(getattr(a,key),getattr(b,key)) for key in ('indices','area','B','rates','D','spectral_norm'))
        realization_checks.append(dict(face=a.face,component=a.component,edge_count=len(a.indices),rates_count=len(a.rates),
            B_shape=list(a.B.shape),D_shape=list(a.D.shape),positive_area=bool((a.area>0).all()),
            finite_coefficients=all(bool(torch.isfinite(getattr(a,k)).all()) for k in ('B','rates','D')),same_actual_two_drive_realization=True))
    inverse = np.linalg.inv(np.asarray(DRIVE_BASIS).T); source_checks = {}
    for name in source_tables[0]:
        values = np.asarray([table[name]['thevenin_peak_voltage'] for table in source_tables])
        pure = inverse.T @ values; group = next(i for i,names in enumerate(groups) if name in names)
        cancellation = float(np.max(abs(pure[1-group])/abs(pure[group])))
        np.testing.assert_allclose(pure,extracted['actual_source_DFT_extraction'][name]['pure_left_right_voltage'],rtol=2e-15,atol=1e-30)
        assert cancellation <= 1e-10 and all(table[name]['resistance_ohm']==50 for table in source_tables)
        source_checks[name]=dict(group=group,inactive_to_active_max=cancellation,
            band_min_to_peak=[float(abs(v).min()/abs(v).max()) for v in values],real_inverse_coefficients=inverse.tolist())
    states=np.asarray(reconstructed_states); transfer_checks=[]
    for fi, frequency in enumerate(case.frequencies):
        l=case.positions.index(-.015); r=case.positions.index(.015)
        x0=states[:,l,:,fi].T.copy();x0[1]*=50
        x1=states[:,r,:,fi].T.copy();x1[1]*=50
        transfer=np.linalg.solve(x0.T,x1.T).T;values,vectors=np.linalg.eig(transfer)
        branch=[i for i,value in enumerate(values) if np.angle(value)>0];assert len(branch)==1
        j=branch[0]; z=50*vectors[0,j]/vectors[1,j]
        alpha=-np.log(abs(values[j]))/.03; beta=np.angle(values[j])/.03
        saved=report['frequency_records'][fi]
        np.testing.assert_allclose([z,alpha,beta],[saved['characteristic_impedance_ohm'],saved['alpha_np_per_m'],saved['beta_rad_per_m']],rtol=1e-12,atol=1e-12)
        coefficients=np.asarray(saved['coherent_coefficients']); target=np.asarray(saved['forward_eigenvector_scaled'])
        residual=float(np.linalg.norm(x0@coefficients-target)/np.linalg.norm(target))
        mode_power=float(.5*np.real(target[0]*np.conj(target[1]/50)))
        np.testing.assert_allclose(mode_power,1.,rtol=2e-14,atol=0)
        transfer_checks.append(dict(frequency_hz=frequency,Z_ohm=z,alpha_np_per_m=alpha,beta_rad_per_m=beta,
            state_condition_50ohm=[float(np.linalg.cond(x0)),float(np.linalg.cond(x1))],actual_VI_transfer_reconstruction_passed=True,
            coherent_solve_residual=residual,forward_state_reference_power_w=mode_power))
    coefficients=np.array([value['coherent_coefficients'] for value in report['frequency_records']],dtype=np.complex128,copy=True)
    power=regional_plane_power(CoherentObservations(results,bases,coefficients),scene,schema,bounds)
    reduction_checks = {}
    saved_power = report['coherent_regional_power']
    entering = np.asarray(saved_power['entering_net_face_power_w'])
    heat = np.asarray(saved_power['bulk_conduction_w']) + np.asarray(saved_power['copper_film_w'])
    # Reanalysis uses one CPU thread; the executed reduction used four. Retain
    # bitwise verdicts and errors; only a float64 roundoff budget is admissible.
    for key in ('bulk_conduction_w','copper_film_w','net_outward_flux_w','balance_relative_to_entering','balance_relative_to_heat'):
        a, b = np.asarray(power[key]), np.asarray(saved_power[key])
        scale = entering.copy()
        if key == 'balance_relative_to_entering': scale /= entering
        if key == 'balance_relative_to_heat': scale /= heat
        budget = 64 * np.finfo(np.float64).eps * scale
        error = abs(a - b)
        assert np.all(error <= budget), (key, error, budget)
        reduction_checks[key] = dict(bitwise_equal=bool(np.array_equal(a,b)),absolute_errors=error,
            max_absolute_error=float(error.max()),float64_roundoff_budget=budget,passed=True)
    face_errors = []
    for fi, (a,b) in enumerate(zip(power['native_signed_face_flux_w'],saved_power['native_signed_face_flux_w'])):
        assert a.keys() == b.keys() and len(a) == 6
        errors = {key:a[key]-b[key] for key in a}
        assert max(map(abs,errors.values())) <= 64*np.finfo(np.float64).eps*entering[fi]
        face_errors.append(errors)
    assert all(sha(Path(path))==expected for path,expected in before.items())
    assert execution_identity()==identity and not torch.cuda.is_initialized()
    args.output.write_text(json.dumps(encode(dict(schema='actual-material-control-independent-CPU-audit/1',declared_case=case.__dict__,
        input_files_sha256=before,all_original_inputs_unchanged=True,executed_source_core_helpers=identity,
        actual_field_drives_replayed=0,CUDA_initialized=False,drives=drives,actual_realization_checks=realization_checks,
        actual_source_DFT_checks=source_checks,independent_VI_transfer_checks=transfer_checks,
        shared_power_reduction_within_float64_roundoff=True,power_reduction_checks=reduction_checks,signed_face_errors=face_errors,coherent_regional_power=power,
        actual_functional_gates=report['pilot_functional_gates'],actual_functional_passed=report['pilot_functional_passed'],
        failed_functional_gates=[key for key,value in report['pilot_functional_gates'].items() if not value],
        physical_release_qualified=False,unqualified=['Second physical line length / resolvedalpha','Independent Cu/sigma controls','Source/time/padding and mesh convergence','Full P5 matrix'],
        driver_sha256=sha(Path(__file__)))),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(actual_drives=2,raw_files_verified=len(before),functional=report['pilot_functional_passed'],physical_release_qualified=False)),flush=True)
if __name__=='__main__':main()
