"""Summarize every preregistered FMM case without fitting incomplete series."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_mom_fmm_scaling as runner

RADIAL_BRIDGE = None

METRICS = {'retained_bytes': ('retained', 'planned_bytes')}
for phase in ('setup', 'forward', 'adjoint', 'differentiable_forward', 'backward'):
    for metric in ('seconds', 'peak_allocated_bytes', 'peak_reserved_bytes'):
        METRICS[phase+'_'+metric] = (phase, metric)

for metric in ('peak_allocated_bytes', 'peak_reserved_bytes'):
    METRICS['whole_interval_'+metric] = ('whole_setup_actions_backward_allocator', metric)


def value_at(record, path):
    value = record
    for key in path:
        value = value[key]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Measured resources must be finite and positive')
    return value


def log_fit(points):
    """OLS over every individual repetition at every preregistered size."""
    x, y = zip(*[(math.log(n), math.log(value)) for n, value in points])
    mx, my = statistics.mean(x), statistics.mean(y)
    variance = sum((a-mx)**2 for a in x)
    if variance == 0:
        raise ValueError('Scaling fit requires distinct sizes')
    slope = sum((a-mx)*(b-my) for a, b in zip(x, y))/variance
    intercept = my-slope*mx
    residual = math.sqrt(statistics.mean((b-intercept-slope*a)**2 for a, b in zip(x, y)))
    return dict(exponent=slope, log_intercept=intercept, log_residual_rms=residual,
                samples=len(x), sizes=len(set(x)))


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    plan, cases = manifest['plan'], manifest['cases']
    if plan != runner.PLAN:
        raise ValueError('Manifest differs from the preregistered current scaling plan')
    if plan['table_storage'] == 'radial_series_shared_radius_v1':
        if RADIAL_BRIDGE is None:
            raise ValueError('Radial summary requires the verified original order bridge')
        import hashlib
        if manifest['closed_order_manifest_sha256'] != hashlib.sha256(RADIAL_BRIDGE[0].read_bytes()).hexdigest():
            raise ValueError('Radial summary bridge differs from the actual cohort')
    expected_cases = [[family, level, repeat] for family in plan['families']
                      for level in plan['subdivisions'] for repeat in range(plan['repetitions'])]
    if cases != expected_cases:
        raise ValueError('Manifest cases do not cover the declared plan')
    records, missing, failures = {}, [], []
    for index, (family, level, repeat) in enumerate(cases):
        path = directory/f'case-{index:02d}.json'
        if not path.exists():
            missing.append(index)
            continue
        record = json.loads(path.read_text())
        if record.get('case') != index or any(record.get(k) != manifest[k] for k in ('identity', 'revision', 'source_hash')):
            raise ValueError('Case identity differs from manifest')
        if record.get('status') in ('failed', 'process_failed'):
            failures.append(dict(case=index, family=family, subdivision=level, repetition=repeat,
                status=record['status'], error=record.get('error'), returncode=record.get('returncode')))
            continue
        if record.get('status') != 'measured' or record.get('source_hash') != manifest['source_hash']:
            raise ValueError('Measured case requires matching source hash and status')
        if [record.get('family'), record.get('subdivision'), record.get('repetition')] != cases[index]:
            raise ValueError('Case coordinates differ from manifest')
        if type(record.get('edges')) is not int or record['edges'] < 1:
            raise ValueError('Case requires positive edge count')
        if record.get('whole_setup_actions_backward_peak_measured') is not True or record.get('native_actions_gate_passed') is not True:
            raise ValueError('Measured case requires whole-interval peaks and passing native actions')
        if (plan['table_storage'] == 'radial_series_shared_radius_v1'
                and (record.get('accuracy') is not None or record.get('all_coefficients_reference_measured') is not False
                     or record.get('reference_policy') != 'not_recomputed_old_reports_are_historical_only')):
            raise ValueError('Radial cohort cannot relabel historical accuracy as a new measurement')
        selection = record.get('order_selection')
        order = runner.validate_order_selection(selection, family, level)
        parameters = selection['parameters']
        if (record.get('selected_order') != order or record.get('edges') != parameters['edges']
                or record.get('frequency') != parameters['frequency']
                or any(record.get(key) != parameters[key] for key in
                       ('kernel_bound_rule', 'table_storage', 'near_storage', 'shift_storage'))):
            raise ValueError('Measured case differs from its physical order selection')
        if repeat > 0:
            reused = runner.reused_case_order(directory/f'case-{index-repeat:02d}.json', index,
                manifest['identity'], manifest['revision'], manifest['source_hash'])
            if selection != reused:
                raise ValueError('Repeated case order selection differs from its first repetition')
        elif selection.get('source_kind') == 'fresh_current_order_plan':
            order_plan = record.get('order_plan')
            if not isinstance(order_plan, dict) or order_plan.get('selected_order') != order:
                raise ValueError('Fresh order selection requires its actual constructor order plan')
        elif selection.get('source_kind') not in ('closed_original_admission', 'closed_original_representation_bridge'):
            raise ValueError('First case must use a closed admission or a current order plan')
        if repeat == 0 and plan['table_storage'] == 'radial_series_shared_radius_v1':
            if selection != RADIAL_BRIDGE[1]['selections'][f'{family}:{level}']:
                raise ValueError('Current radial order differs from its closed original bridge')
        if selection.get('source_path') is not None:
            import hashlib
            if hashlib.sha256(Path(selection['source_path']).read_bytes()).hexdigest() != selection.get('source_sha256'):
                raise ValueError('Measured case order source SHA differs')
        tolerance = plan['native_action_tolerance']
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance != 1e-11:
            raise ValueError('Manifest must preserve the original native action tolerance')
        for key in ('adjoint_relative_error', 'current_gradient_relative_error'):
            error = record.get(key)
            if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error < tolerance:
                raise ValueError('Measured case failed the native action error gate')
        for metric in METRICS.values():
            value_at(record, metric)
        interval = record['whole_setup_actions_backward_allocator']
        warmups = record.get('warmups')
        if not isinstance(warmups, list) or len(warmups) != plan['action_warmups']:
            raise ValueError('Measured case must include every declared warmup')
        phases = [record[key] for key in ('setup', 'forward', 'adjoint', 'differentiable_forward', 'backward')]
        for warmup in warmups:
            phases.extend((warmup['forward'], warmup['adjoint']))
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            if interval[key] < max(value_at(phase, (key,)) for phase in phases):
                raise ValueError('Whole-interval peak omits a measured phase')
        records[index] = record
    families = {}
    for family in plan['families']:
        indices = [i for i, case in enumerate(cases) if case[0] == family]
        complete = all(i in records for i in indices)
        points = []
        for level in plan['subdivisions']:
            samples = [records[i] for i in indices if i in records and cases[i][1] == level]
            if not samples:
                continue
            sizes = {r['edges'] for r in samples}
            if len(sizes) != 1:
                raise ValueError('Repeated case edge counts differ')
            points.append(dict(subdivision=level, edges=sizes.pop(), repetitions=len(samples),
                selected_orders=[r['selected_order'] for r in samples],
                metrics={name: dict(min=min(values), median=statistics.median(values), max=max(values))
                    for name, path in METRICS.items()
                    for values in [[value_at(r, path) for r in samples]]}))
        fits = None
        if complete:
            if len({r['edges'] for i, r in records.items() if i in indices}) != len(plan['subdivisions']):
                raise ValueError('Declared subdivisions must have distinct measured sizes')
            fits = {name: log_fit([(records[i]['edges'], value_at(records[i], path)) for i in indices])
                    for name, path in METRICS.items()}
        families[family] = dict(complete=complete, points=points, all_point_fits=fits,
            fit_scope='all_declared_sizes_and_individual_repetitions' if complete else None)
    accuracy = {str(i): r.get('accuracy') for i, r in records.items() if r.get('accuracy') is not None}
    return dict(version=('fmm_radial_series_resource_summary_v4'
                        if plan['table_storage'] == 'radial_series_shared_radius_v1'
                        else 'fmm_admission_reuse_resource_summary_v3'), identity=manifest['identity'],
        expected_cases=len(cases), measured_cases=len(records), missing_cases=missing,
        failed_cases=failures, families=families, available_accuracy=accuracy,
        complete=len(records) == len(cases), complexity_qualification=False,
        physical_qualification=False, limitations=plan['pending']+[
            'shared GPU pilot; uncontrolled timing interference',
            'empirical slopes alone do not establish asymptotic complexity',
            'whole interval excludes solver, teardown and later original reference controls'])


def main():
    global RADIAL_BRIDGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--radial-order-bridge', type=Path)
    args = parser.parse_args()
    if args.radial_order_bridge is not None:
        runner.configure_radial_series()
        RADIAL_BRIDGE = (args.radial_order_bridge, runner.load_radial_order_bridge(args.radial_order_bridge))
    text = json.dumps(summarize(args.directory), indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
