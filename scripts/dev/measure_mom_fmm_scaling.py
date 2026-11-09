"""Predeclared FMM resource experiments; each case runs in an isolated process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = {
    'version': 'scalar_fmm_scaling_admission_reuse_v3',
    'families': ['fixed_ka', 'growing_ka'],
    'subdivisions': [0, 1, 2, 3],
    'repetitions': 3,
    'radius': .01,
    'centers': [-.5, .5],
    'base_frequency': 3e8,
    'growing_frequency_factor': 2,
    'integration_order': 4,
    'singular_order': 3,
    'order_min': 6,
    'order_max': 12,
    'green_absolute_budget': 1e-6,
    'gradient_absolute_budget': 1e-5,
    'table_storage': 'compact_shared_displacement_v1',
    'shift_storage': 'separable_v1',
    'near_storage': 'scalar_reciprocal_dk_v1',
    'kernel_bound_rules': {'fixed_ka': 'positive_tail_min_v1',
                           'growing_ka': 'direct_first_wave_min_v1'},
    'required_closed_admission_subdivisions': [3],
    'reuse_first_repetition_order': True,
    'm2l_row_batch': 16,
    'native_action_tolerance': 1e-11,
    'leaf_size': 30,
    'separation_ratio': .25,
    'action_retained_budget_bytes': 2*1024**3,
    'table_budget_bytes': 1024**3,
    'near_budget_bytes': 1024**3,
    'projection_budget_bytes': 1024**3,
    'action_warmups': 1,
    'backward_timing': 'cold_first_backward',
    'rhs_count': 2,
    'seed': 1729,
    'accuracy_tolerance': 1e-8,
    'full_operator_check_max_edges': 120,
    'qualification': False,
    'pending': ['all-size reference quadrature convergence', 'solve resources',
                'host memory accounting', 'H2 and dense crossover',
                'full physical acceptance', 'high-frequency implementation'],
}
CASES = [(family, level, repeat) for family in PLAN['families']
         for level in PLAN['subdivisions'] for repeat in range(PLAN['repetitions'])]


def configure_radial_series():
    """Change representation only; retain every physical case and gate."""
    PLAN.update(version='scalar_fmm_radial_series_resource_v4',
                table_storage='radial_series_shared_radius_v1')


def load_radial_order_bridge(path):
    """Bind closed original orders without relabeling their source or accuracy."""
    bundle = json.loads(Path(path).read_text())
    if (bundle.get('version') != 'scalar_fmm_radial_order_bridge_v1'
            or bundle.get('plan_hash') != plan_hash()
            or bundle.get('production_source_hash') != production_source_hash()
            or PLAN['table_storage'] != 'radial_series_shared_radius_v1'
            or bundle.get('production_qualification') is not False):
        raise ValueError('Radial bridge differs from current plan/source')
    def read_binding(binding):
        data = Path(binding['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != binding['sha256']:
            raise ValueError('Closed original bridge SHA differs')
        return json.loads(data)
    original = read_binding(bundle['original_manifest'])
    before = read_binding(bundle['original_input'])
    receipt = read_binding(bundle['original_receipt'])
    terminal = read_binding(bundle['original_terminal'])
    original_plan = dict(PLAN, version='scalar_fmm_scaling_admission_reuse_v3',
                         table_storage='compact_shared_displacement_v1')
    if (original.get('plan') != original_plan
            or original.get('cases') != [list(case) for case in CASES]
            or before.get('plan') != original_plan
            or before.get('cases') != original['cases']
            or any(before.get(k) != original.get(k) for k in ('identity', 'source_hash'))
            or before.get('baseline') != original.get('revision')
            or receipt.get('returncode') != 0 or receipt.get('sources_unchanged') is not True
            or before.get('source_before') != receipt.get('source_after')
            or terminal.get('status') != 'CLOSED' or terminal.get('handle_drained') is not True
            or terminal.get('returncode') != 0 or terminal.get('actual_records') != len(CASES)
            or terminal.get('outside_ps_returncode') != 1):
        raise ValueError('Original series must be closed with identical physical plan')
    for name, digest in before['source_before'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Original frozen dependency changed')
    old_root = Path(before['provider'])
    old_production = hashlib.sha256()
    for source in sorted((old_root/'witwin/maxwell').rglob('*.py')):
        old_production.update(str(source.relative_to(old_root)).encode())
        old_production.update(source.read_bytes())
    old_production_hash = old_production.hexdigest()
    if set(bundle['records']) != {str(i) for i in range(len(CASES))}:
        raise ValueError('Bridge requires every original repetition, including failures')
    selections = {}
    original_plan_hash = hashlib.sha256(json.dumps(original_plan, sort_keys=True).encode()).hexdigest()
    for index, (family, level, repeat) in enumerate(CASES):
        binding = bundle['records'][str(index)]
        record = read_binding(binding)
        validate_worker_result(record, original['identity'], original['revision'], original['source_hash'], index)
        parameters = dict(case_parameters(family, level), table_storage='compact_shared_displacement_v1')
        selection = record.get('order_selection', {})
        order = record.get('selected_order')
        if (record.get('status') != 'measured' or record.get('native_actions_gate_passed') is not True
                or [record.get(k) for k in ('family', 'subdivision', 'repetition')] != [family, level, repeat]
                or type(order) is not int or not PLAN['order_min'] <= order <= PLAN['order_max']
                or selection.get('parameters') != parameters or selection.get('selected_order') != order
                or selection.get('version') != 'scalar_fmm_reused_order_selection_v1'
                or selection.get('plan_hash') != original_plan_hash
                or selection.get('production_source_hash') != old_production_hash
                or selection.get('production_qualification') is not False
                or record.get('edges') != parameters['edges'] or record.get('frequency') != parameters['frequency']
                or any(record.get(k) != parameters[k] for k in ('kernel_bound_rule', 'near_storage', 'table_storage', 'shift_storage'))):
            raise ValueError('Original order differs from its physical case or gates')
        validate_native_actions(record.get('adjoint_relative_error'), record.get('current_gradient_relative_error'))
        # Keep the complete old selection, including its old production hash,
        # separate from the current selection. Current constructors still admit
        # the selected order's native couplings and all storage budgets.
        key = f'{family}:{level}'
        if repeat == 0:
            original_source = selection.get('source_path')
            if original_source is not None:
                bound = read_binding(dict(path=original_source, sha256=selection['source_sha256']))
                original_order_plan = bound.get('order_plan')
            else:
                original_order_plan = record.get('order_plan')
            if level in PLAN['required_closed_admission_subdivisions'] or original_source is None:
                import math
                if (not isinstance(original_order_plan, dict)
                        or original_order_plan.get('selected_order') != order
                        or original_order_plan.get('kernel_bound_rule') != parameters['kernel_bound_rule']
                        or original_order_plan.get('kernel_budget') != parameters['kernel_budget']
                        or not original_order_plan.get('attempts')):
                    raise ValueError('Original order requires its actual kernel admission')
                final = original_order_plan['attempts'][-1]
                if (final.get('order') != order or final.get('kernel_pass') is not True
                        or final.get('unavailable_bounds') != 0):
                    raise ValueError('Original kernel admission failed')
                for field, limit in (('green_absolute_max', PLAN['green_absolute_budget']),
                                     ('gradient_component_absolute_max', PLAN['gradient_absolute_budget'])):
                    value = final.get(field)
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                        raise ValueError('Original kernel admission exceeds the unchanged budget')
            elif order != PLAN['order_min'] or bound.get('native_actions_gate_passed') is not True:
                raise ValueError('Minimum-order source requires its original native gate')
            selections[key] = dict(version='scalar_fmm_reused_order_selection_v1',
                parameters=case_parameters(family, level), plan_hash=plan_hash(),
                production_source_hash=production_source_hash(), selected_order=order,
                source_kind='closed_original_representation_bridge',
                source_path=str(Path(binding['path']).resolve()), source_sha256=binding['sha256'],
                original_order_selection=selection, original_series_revision=original['revision'],
                bridge_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                production_qualification=False)
        elif selections[key]['selected_order'] != order:
            raise ValueError('Original repetition orders differ')
        if selection.get('source_path') is not None:
            read_binding(dict(path=selection['source_path'], sha256=selection['source_sha256']))
    return dict(selections=selections)


def plan_hash():
    return hashlib.sha256(json.dumps(PLAN, sort_keys=True).encode()).hexdigest()


def production_source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'witwin/maxwell').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def case_parameters(family, level):
    if family not in PLAN['families'] or type(level) is not int or level not in PLAN['subdivisions']:
        raise ValueError('Order selection must refer to a declared case')
    return dict(family=family, subdivision=level, edges=60*4**level,
        frequency=PLAN['base_frequency']*(PLAN['growing_frequency_factor']**level
                                         if family == 'growing_ka' else 1),
        radius=PLAN['radius'], centers=PLAN['centers'], epsilon_r=2., mu_r=1.,
        integration_order=PLAN['integration_order'], singular_order=PLAN['singular_order'],
        leaf_size=PLAN['leaf_size'], separation_ratio=PLAN['separation_ratio'],
        kernel_bound_rule=PLAN['kernel_bound_rules'][family],
        kernel_budget=dict(green_absolute=PLAN['green_absolute_budget'],
                           gradient_component_absolute=PLAN['gradient_absolute_budget']),
        table_storage=PLAN['table_storage'], near_storage=PLAN['near_storage'],
        shift_storage=PLAN['shift_storage'])


def validate_order_selection(selection, family, level):
    if (not isinstance(selection, dict)
            or selection.get('version') != 'scalar_fmm_reused_order_selection_v1'
            or selection.get('parameters') != case_parameters(family, level)
            or selection.get('plan_hash') != plan_hash()
            or selection.get('production_source_hash') != production_source_hash()):
        raise ValueError('Order selection differs from frozen plan, production source or physical case')
    order = selection.get('selected_order')
    if type(order) is not int or not PLAN['order_min'] <= order <= PLAN['order_max']:
        raise ValueError('Reused order must preserve the declared integer order range')
    if selection.get('production_qualification') is not False:
        raise ValueError('Configured order reuse must not grant production qualification')
    return order


def load_closed_order_manifest(path):
    bundle = json.loads(Path(path).read_text())
    if (not isinstance(bundle, dict) or bundle.get('version') != 'scalar_fmm_closed_order_manifest_v1'
            or bundle.get('plan_hash') != plan_hash()
            or bundle.get('production_source_hash') != production_source_hash()
            or not isinstance(bundle.get('selections'), dict)):
        raise ValueError('Closed order manifest differs from current frozen plan/source')
    for family in PLAN['families']:
        for level in PLAN['required_closed_admission_subdivisions']:
            if f'{family}:{level}' not in bundle['selections']:
                raise ValueError('Complete series requires every previously admitted large case')
    for key, selection in bundle['selections'].items():
        if not isinstance(selection, dict) or not isinstance(selection.get('parameters'), dict):
            raise ValueError('Closed order selection requires physical parameters')
        parameters = selection['parameters']
        family, level = parameters.get('family'), parameters.get('subdivision')
        if key != f'{family}:{level}':
            raise ValueError('Closed order manifest key differs from physical case')
        order = validate_order_selection(selection, family, level)
        if selection.get('source_kind') != 'closed_original_admission':
            raise ValueError('Closed manifest requires an explicitly curated original admission')
        source = Path(selection['source_path'])
        raw_bytes = source.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != selection['source_sha256']:
            raise ValueError('Closed order source receipt SHA differs')
        raw = json.loads(raw_bytes)
        if raw.get('order_admitted') is True:
            plan = raw.get('order_plan')
            if not isinstance(plan, dict) or not isinstance(plan.get('attempts'), list) or not plan['attempts']:
                raise ValueError('Closed admission requires complete order-plan attempts')
            final = plan['attempts'][-1]
            if not isinstance(final, dict):
                raise ValueError('Closed admission requires a final kernel gate')
            if (raw.get('family') != family or raw.get('frequency_Hz') != parameters['frequency']
                    or raw.get('edges') != parameters['edges'] or plan.get('selected_order') != order
                    or raw.get('kernel_bound_rule') != parameters['kernel_bound_rule']
                    or plan.get('kernel_bound_rule') != parameters['kernel_bound_rule']
                    or plan.get('kernel_budget') != parameters['kernel_budget']
                    or final.get('order') != order or final.get('kernel_pass') is not True
                    or final.get('unavailable_bounds') != 0):
                raise ValueError('Closed original admission does not match physical case or kernel gates')
            import math
            for field, limit in (('green_absolute_max', PLAN['green_absolute_budget']),
                                 ('gradient_component_absolute_max', PLAN['gradient_absolute_budget'])):
                value = final.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                    raise ValueError('Closed original admission failed its kernel gate')
        else:
            # A completed minimum-order workflow smoke already passed native
            # coupling admission; no earlier candidate order exists to search.
            if (order != PLAN['order_min'] or raw.get('status') != 'measured'
                    or raw.get('family') != family or raw.get('subdivision') != level
                    or raw.get('frequency') != parameters['frequency'] or raw.get('edges') != parameters['edges']
                    or raw.get('selected_order') != order
                    or raw.get('kernel_bound_rule') != parameters['kernel_bound_rule']
                    or raw.get('table_storage') != parameters['table_storage']
                    or raw.get('near_storage') != parameters['near_storage']
                    or raw.get('shift_storage') != parameters['shift_storage']
                    or raw.get('native_actions_gate_passed') is not True):
                raise ValueError('Closed minimum-order measurement does not match configured case')
            validate_native_actions(raw.get('adjoint_relative_error'), raw.get('current_gradient_relative_error'))
    return bundle


def reused_case_order(path, index, identity, revision, source_hash):
    family, level, repeat = CASES[index]
    record = json.loads(Path(path).read_text())
    first = CASES.index((family, level, 0))
    validate_worker_result(record, identity, revision, source_hash, first)
    if (repeat < 1 or record.get('status') != 'measured'
            or [record.get('family'), record.get('subdivision'), record.get('repetition')] != [family, level, 0]
            or record.get('native_actions_gate_passed') is not True):
        raise ValueError('Order reuse requires a completed first repetition of the same physical case')
    validate_native_actions(record.get('adjoint_relative_error'), record.get('current_gradient_relative_error'))
    selection = dict(record['order_selection'])
    order = validate_order_selection(selection, family, level)
    parameters = selection['parameters']
    if (record.get('selected_order') != order or record.get('edges') != parameters['edges']
            or record.get('frequency') != parameters['frequency']
            or any(record.get(key) != parameters[key] for key in
                   ('kernel_bound_rule', 'table_storage', 'near_storage', 'shift_storage'))):
        raise ValueError('First repetition differs from its physical order selection')
    selection.update(source_kind='current_first_repetition', source_path=str(Path(path).resolve()),
                     source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return selection


_WHOLE_PEAK = {}


def capture_peak():
    """Preserve absolute process peaks before every phase reset, including gaps."""
    import torch
    for field, query in (('peak_allocated_bytes', torch.cuda.max_memory_allocated),
                         ('peak_reserved_bytes', torch.cuda.max_memory_reserved)):
        _WHOLE_PEAK[field] = max(_WHOLE_PEAK.get(field, 0), query())


def measured(function):
    import torch
    torch.cuda.synchronize()
    capture_peak()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    capture_peak()
    return result, dict(seconds=time.perf_counter()-start,
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        end_allocated_bytes=torch.cuda.memory_allocated())


def validate_native_actions(adjoint_error, gradient_error):
    import math
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not 0 <= value < PLAN['native_action_tolerance']
           for value in (adjoint_error, gradient_error)):
        raise ValueError('Native adjoint/current gradient failed the preregistered gate')


def finish_native_interval(applied, adjoint_value, output, gradient, *,
                           adjoint_error, gradient_error, failure_state_path=None, context=None):
    """Close peaks before preserving and raising an unchanged native gate."""
    import math
    import torch
    finite = bool(torch.isfinite(applied).all() & torch.isfinite(adjoint_value).all()
                  & torch.isfinite(output).all() & torch.isfinite(gradient).all())
    torch.cuda.synchronize()
    capture_peak()
    _WHOLE_PEAK.update(end_allocated_bytes=torch.cuda.memory_allocated(),
                       end_reserved_bytes=torch.cuda.memory_reserved())
    interval = _WHOLE_PEAK.copy()
    try:
        validate_native_actions(adjoint_error, gradient_error)
        if not finite:
            raise ValueError('Measured native actions must be finite')
    except ValueError as exc:
        if failure_state_path is not None:
            path = Path(failure_state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Serialization follows the measured interval and occurs only for
            # an actual failed native gate. Preserve original device tensors.
            torch.save(dict(context=context, applied=applied, adjoint_value=adjoint_value,
                differentiable_output=output, gradient=gradient, allocator=interval,
                adjoint_error=adjoint_error, gradient_error=gradient_error, finite=finite), path)
            exc.receipt = dict(native_state_path=str(path.resolve()),
                native_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                allocator=interval, finite=finite,
                adjoint_error=adjoint_error if math.isfinite(adjoint_error) else None,
                gradient_error=gradient_error if math.isfinite(gradient_error) else None,
                failed_original_gate=str(exc), production_qualification=False)
        raise
    return interval


def run_case(index, *, order_selection=None, failure_state_path=None):
    import torch
    import witwin.core as core
    import witwin.maxwell as mw
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.operator import IntegrationConfig
    from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
    from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
    from witwin.maxwell.mom.fmm_capacity import multipole_action_tensors, storage_bytes
    from witwin.maxwell.mom.operator_error import measure_original_operator_error
    if not torch.cuda.is_available():
        raise ValueError('Actual CUDA resource measurements are required')
    if not Path(core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('Resource measurements require the installed Core dependency')
    family, level, repeat = CASES[index]
    frequency = case_parameters(family, level)['frequency']
    selected_order = None if order_selection is None else validate_order_selection(order_selection, family, level)
    if level in PLAN['required_closed_admission_subdivisions'] and selected_order is None:
        raise ValueError('Previously admitted large cases require SHA-bound order reuse')
    kernel_rule = PLAN['kernel_bound_rules'][family]
    def build():
        exterior = compile_medium(mw.Material(), frequency)
        interior = compile_medium(mw.Material(eps_r=2.), frequency)
        bodies = [(i, str(i), RWGBasis(sphere_mesh(radius=PLAN['radius'],
            center=(x, 0., 0.), subdivisions=level, device='cuda', dtype=torch.float64)), interior)
            for i, x in enumerate(PLAN['centers'])]
        system = SurfaceSystem.create(bodies)
        provider = ScalarSystemBlocks(system, exterior, IntegrationConfig(
            order=PLAN['integration_order'], singular_order=PLAN['singular_order']))
        return MultipoleSystemActions(provider, order=PLAN['order_min'] if selected_order is None else selected_order,
            max_order=PLAN['order_max'] if selected_order is None else None, kernel_bound_rule=kernel_rule,
            kernel_error_budget=mw.KernelErrorBudget(PLAN['green_absolute_budget'], PLAN['gradient_absolute_budget']),
            leaf_size=PLAN['leaf_size'], separation_ratio=PLAN['separation_ratio'],
            **{key: PLAN[key] for key in ('action_retained_budget_bytes', 'table_budget_bytes',
                'near_budget_bytes', 'projection_budget_bytes', 'table_storage', 'near_storage',
                'shift_storage', 'm2l_row_batch')})
    # CUDA initialization is excluded; every setup/action allocation and every
    # inter-phase gap through final finite checks belongs to the same interval.
    torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _WHOLE_PEAK.clear()
    _WHOLE_PEAK.update(baseline_allocated_bytes=torch.cuda.memory_allocated(),
        baseline_reserved_bytes=torch.cuda.memory_reserved(),
        scope='process allocator before construction through final finite/current checks; all warmups, differentiable forward and peak-reset gaps included; no solve, teardown or subsequent original reference check')
    op, setup = measured(build)
    actual_order = op.order_plan['selected_order'] if selected_order is None else selected_order
    selection = dict(version='scalar_fmm_reused_order_selection_v1',
        parameters=case_parameters(family, level), plan_hash=plan_hash(),
        production_source_hash=production_source_hash(), selected_order=actual_order,
        source_kind='fresh_current_order_plan', source_path=None, source_sha256=None,
        production_qualification=False) if order_selection is None else dict(order_selection)
    n = op.shape[0]
    owned = storage_bytes(multipole_action_tensors(op))
    if owned != op.capacity_plan['planned_bytes']:
        raise ValueError('Actual owned storage differs from the structural plan')
    generator = torch.Generator(device='cuda').manual_seed(PLAN['seed'])
    x = torch.randn((PLAN['rhs_count'], n), generator=generator, device='cuda', dtype=op.dtype)
    y = torch.randn(x.shape, generator=generator, device='cuda', dtype=op.dtype)
    warmups = []
    for _ in range(PLAN['action_warmups']):
        value, forward_warmup = measured(lambda: op.apply(x))
        del value
        value, adjoint_warmup = measured(lambda: op.adjoint(y))
        del value
        warmups.append(dict(forward=forward_warmup, adjoint=adjoint_warmup))
    applied, forward = measured(lambda: op.apply(x))
    adjoint_value, adjoint = measured(lambda: op.adjoint(y))
    lhs = (applied.conj()*y).sum()
    rhs = (x.conj()*adjoint_value).sum()
    adjoint_error = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-300))
    variable = x.detach().requires_grad_(True)
    output, differentiable_forward = measured(lambda: op.apply(variable))
    gradient, backward = measured(lambda: torch.autograd.grad(output, variable, y)[0])
    gradient_error = float(torch.linalg.vector_norm(gradient-adjoint_value)
                           /torch.linalg.vector_norm(adjoint_value))
    interval = finish_native_interval(applied, adjoint_value, output, gradient,
        adjoint_error=adjoint_error, gradient_error=gradient_error,
        failure_state_path=failure_state_path,
        context=dict(parameters=case_parameters(family, level), order_selection=selection,
                     retained=op.capacity_plan, x=x, y=y, setup=setup, forward=forward,
                     adjoint=adjoint, differentiable_forward=differentiable_forward,
                     backward=backward, warmups=warmups))
    dense_bytes = n*n*torch.empty((), dtype=op.dtype).element_size()
    # The independent original check is outside the resource interval; it must
    # never inflate or replace the setup/actions/backward measurement scope.
    accuracy = None
    if (PLAN['table_storage'] != 'radial_series_shared_radius_v1'
            and n//2 <= PLAN['full_operator_check_max_edges'] and repeat == 0):
        accuracy = measure_original_operator_error(op, edge_batch=32, tolerance=PLAN['accuracy_tolerance'])
    if selection['source_path'] is not None:
        if hashlib.sha256(Path(selection['source_path']).read_bytes()).hexdigest() != selection['source_sha256']:
            raise ValueError('Reused order source changed during measurement')
    return dict(order_selection=selection, order_plan=op.order_plan, case=index, family=family, subdivision=level, repetition=repeat,
        edges=n//2, frequency=frequency, body_ka=abs(op.provider.exterior.k)*PLAN['radius'],
        selected_order=actual_order, setup=setup, forward=forward,
        adjoint=adjoint, backward=backward, warmups=warmups,
        differentiable_forward=differentiable_forward,
        adjoint_relative_error=adjoint_error, current_gradient_relative_error=gradient_error,
        native_actions_gate_passed=True, retained=op.capacity_plan,
        actual_owned_retained_bytes=owned, whole_setup_actions_backward_allocator=interval,
        whole_setup_actions_backward_peak_measured=True,
        same_N_full_dense_bytes=dense_bytes,
        measured_allocated_peak_below_same_N_dense=interval['peak_allocated_bytes']<dense_bytes,
        measured_reserved_peak_below_same_N_dense=interval['peak_reserved_bytes']<dense_bytes,
        whole_peak_qualified=False, memory_crossover_qualified=False,
        accuracy=accuracy, all_coefficients_reference_measured=accuracy is not None,
        reference_policy=('not_recomputed_old_reports_are_historical_only'
                          if PLAN['table_storage'] == 'radial_series_shared_radius_v1'
                          else 'original_small_size_check'),
        reference_quadrature_converged=False, table_storage=PLAN['table_storage'],
        shift_storage=PLAN['shift_storage'], near_storage=PLAN['near_storage'],
        kernel_bound_rule=kernel_rule,
        pid=os.getpid(), python=sys.executable, core=str(Path(core.__file__).resolve()),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, complexity_qualification=False,
        physical_qualification=False, status='measured')


def source_identity(order_manifest=None):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source = hashlib.sha256()
    for path in sorted((ROOT/'witwin').rglob('*.py')) + [Path(__file__).resolve()]:
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(path.read_bytes())
    source_hash = source.hexdigest()
    identity = hashlib.sha256(json.dumps(dict(plan=PLAN, revision=revision, source_hash=source_hash,
        closed_order_manifest_sha256=None if order_manifest is None else hashlib.sha256(Path(order_manifest).read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()
    return identity, revision, source_hash


def validate_worker_result(result, identity, revision, source_hash, case):
    expected = dict(identity=identity, revision=revision, source_hash=source_hash, case=case)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('Worker result identity differs from the frozen manifest')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=int, choices=range(len(CASES)))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--order-admissions', type=Path)
    parser.add_argument('--radial-order-bridge', type=Path)
    parser.add_argument('--reuse-case', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and args.case is None:
        parser.error('Worker requires a case')
    if args.reuse_case is not None and not args.worker:
        parser.error('First-repetition reuse is a private worker option')
    if args.radial_order_bridge is not None and args.order_admissions is not None:
        parser.error('Use one explicitly versioned order source')
    if args.radial_order_bridge is not None:
        configure_radial_series()
    if args.case is None and args.order_admissions is None and args.radial_order_bridge is None:
        parser.error('Complete series requires the closed original order-admission manifest')
    admissions = args.radial_order_bridge or args.order_admissions
    bundle = (load_radial_order_bridge(args.radial_order_bridge) if args.radial_order_bridge is not None
              else None if admissions is None else load_closed_order_manifest(admissions))
    def current_identity():
        return source_identity(admissions)
    identity, revision, source_hash = current_identity()
    if args.worker:
        try:
            family, level, _ = CASES[args.case]
            selection = (None if bundle is None else bundle['selections'].get(f'{family}:{level}'))
            if args.reuse_case is not None:
                selection = reused_case_order(args.reuse_case, args.case, identity, revision, source_hash)
            kwargs = {} if selection is None else dict(order_selection=selection)
            if args.radial_order_bridge is not None:
                kwargs['failure_state_path'] = args.output/f'case-{args.case:02d}-failed-native-state.pt'
            result = run_case(args.case, **kwargs)
            if current_identity() != (identity, revision, source_hash):
                raise ValueError('Source identity changed during worker execution')
        except Exception as exc:
            result = dict(case=args.case, status='failed', error_type=type(exc).__name__,
                error=str(exc), receipt=getattr(exc, 'receipt', None), complexity_qualification=False)
        print(json.dumps(dict(identity=identity, revision=revision, source_hash=source_hash, **result), allow_nan=False), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'manifest.json'
    expected = dict(identity=identity, revision=revision, source_hash=source_hash, plan=PLAN, cases=CASES,
        closed_order_manifest_sha256=None if admissions is None else hashlib.sha256(admissions.read_bytes()).hexdigest())
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(expected)):
            raise ValueError('Output manifest differs from this revision or predeclared plan')
    else:
        manifest.write_text(json.dumps(expected, indent=2)+'\n')
    for index in ([args.case] if args.case is not None else range(len(CASES))):
        destination = args.output/f'case-{index:02d}.json'
        if destination.exists():
            prior = json.loads(destination.read_text())
            validate_worker_result(prior, identity, revision, source_hash, index)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--case', str(index), '--output', str(args.output)]
        if args.order_admissions is not None:
            command += ['--order-admissions', str(args.order_admissions.resolve())]
        if args.radial_order_bridge is not None:
            command += ['--radial-order-bridge', str(args.radial_order_bridge.resolve())]
        family, level, repeat = CASES[index]
        if repeat > 0:
            first_path = args.output/f'case-{CASES.index((family, level, 0)):02d}.json'
            if first_path.exists():
                prior = json.loads(first_path.read_text())
                validate_worker_result(prior, identity, revision, source_hash, CASES.index((family, level, 0)))
                if prior.get('status') != 'measured':
                    failure = dict(identity=identity, revision=revision, source_hash=source_hash, case=index,
                        status='failed', error='First repetition of identical admission/setup failed; not replayed',
                        not_launched=True, complexity_qualification=False)
                    destination.write_text(json.dumps(failure, indent=2)+'\n')
                    continue
                command += ['--reuse-case', str(first_path.resolve())]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (args.output/f'case-{index:02d}.log').write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            result = dict(identity=identity, revision=revision, source_hash=source_hash, case=index, status='process_failed',
                          returncode=completed.returncode, complexity_qualification=False)
        else:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            validate_worker_result(result, identity, revision, source_hash, index)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, destination)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
