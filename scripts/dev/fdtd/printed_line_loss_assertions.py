"""Conditioned diagnostics for finite-precision physical superposition.

Relative thresholds are unchanged. A mixed output zero is not the scale of
its nonzero input basis; signed net flux is not the entering power scale.
The original output-relative/pointwise verdicts remain explicit diagnostics.
"""
from __future__ import annotations
import math

import numpy as np
import torch


def superposition_metrics(combined, direct, basis, coefficients, *, rtol=1e-4, atol=1e-11, physical_weights=None):
    a, b = [torch.as_tensor(value, dtype=torch.complex128, device='cpu') for value in (combined, direct)]
    fields = [torch.as_tensor(value, dtype=torch.complex128, device='cpu') for value in basis]
    weights = torch.as_tensor(coefficients, dtype=torch.complex128, device='cpu')
    if (not fields or a.numel() == 0 or b.shape != a.shape or weights.shape != (len(fields),)
        or any(field.shape != a.shape for field in fields)
        or any(not bool(torch.isfinite(value).all()) for value in (a, b, weights, *fields))):
        raise ValueError('Superposition metrics require complete finite matching actual field arrays and coefficients.')
    difference = abs(a - b)
    volumes = torch.ones_like(a.real) if physical_weights is None else torch.as_tensor(physical_weights, dtype=torch.float64)
    if volumes.shape != a.shape or not bool(torch.isfinite(volumes).all()) or not bool((volumes > 0).all()):
        raise ValueError('Physical field norm weights must be finite positive native dual volumes.')
    square_root = volumes.sqrt()
    l2_error, inf_error = float((difference * square_root).norm()), float(difference.max())
    l2_scale = sum(float(abs(weight)) * float((field * square_root).norm()) for weight, field in zip(weights, fields))
    inf_scale = sum(float(abs(weight)) * float(abs(field).max()) for weight, field in zip(weights, fields))
    l2_absolute = atol * math.sqrt(float(volumes.sum()))
    legacy_budget = atol + rtol * abs(b)
    failures = difference > legacy_budget
    output_norm = float((a * square_root).norm())
    return {'rtol': rtol, 'atol_per_native_dof': atol,
        'basis_scale_definition': 'sum_j |c_j| ||actual_basis_j||, computed independently in weightedL2 and Linfinity for the supplied actual observation vector',
        'native_dof_count': a.numel(), 'l2_absolute_error': l2_error, 'linfinity_absolute_error': inf_error,
        'l2_actual_basis_scale': l2_scale, 'linfinity_actual_basis_scale': inf_scale,
        'l2_relative_to_actual_basis': l2_error / l2_scale if l2_scale else None,
        'linfinity_relative_to_actual_basis': inf_error / inf_scale if inf_scale else None,
        'mixed_output_condition_number_l2': l2_scale / output_norm if output_norm else None,
        'l2_budget': l2_absolute + rtol * l2_scale, 'linfinity_budget': atol + rtol * inf_scale,
        'passed': l2_error <= l2_absolute + rtol * l2_scale and inf_error <= atol + rtol * inf_scale,
        'original_pointwise_passed': not bool(failures.any()),
        'original_pointwise_failure_count': int(failures.sum()),
        'original_pointwise_failure_indices_first20': torch.nonzero(failures).tolist()[:20],
        'direct_output_peak_abs': float(abs(b).max())}


def physical_electric_vectors(results, cpu_scene, schema, frequency):
    """Actual full E-vector samples with physical native-Yee volume weights.

    Standard native dual cells are clipped to the physical Domain. On foil
    edges the single native representative is replaced by the two actual side
    traces, weighted by their physical half volumes. The norm is integral |E|²
    (V² m), not a separately claimed instantaneous stored material energy.
    """
    import witwin.maxwell as mw
    from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
    from witwin.maxwell.scene import prepare_scene
    from scripts.dev.fdtd.printed_line_regional_power import _native_plane_field
    if str(cpu_scene.device) != 'cpu' or getattr(cpu_scene, 'coordinates', None) is not None:
        raise ValueError('Field-norm inference requires the actual Cartesian CPU scene.')
    bounds = tuple(tuple(value) for value in cpu_scene.domain.bounds)
    monitor = mw.PowerLossMonitor('physical_field_norm_geometry',
        position=tuple((a + b) / 2 for a, b in bounds), size=tuple(b - a for a, b in bounds),
        quadrature='physical_intersection')
    compiled = compile_power_loss_monitor(prepare_scene(cpu_scene), monitor)
    dense_weights = {}
    for component, mask in compiled.component_masks.items():
        values = torch.zeros(mask.shape, dtype=torch.float64)
        values[mask] = compiled.component_volumes[component].double()
        needed = set(torch.nonzero(mask.any(0).any(0)).flatten().tolist())
        actual = {plane['normal_index'] for plane in schema[component]['planes']}
        if not needed.issubset(actual):
            raise ValueError('Recorded native planes do not cover every physical field-norm degree of freedom.')
        dense_weights[component] = values
    records = [result.conductor_losses for result in results]
    blocks = [[] for _ in results]; weight_blocks = []
    claimed = {component: torch.zeros_like(values, dtype=torch.bool) for component, values in dense_weights.items()}
    for name, record in records[0].items():
        if record['coordinate_frame'] != 'cartesian' or record['periodic_axes']:
            raise ValueError('Bounded full-vector diagnostic does not cross mapped or periodic seams.')
        fi = tuple(float(f) for f in record['frequencies']).index(frequency)
        surfaces = {trace['component']: trace for trace in record['surface_power_traces']}
        for trace in record['bulk_electric_traces']:
            component = trace['component']; indices = trace['indices'].long()
            selected = dense_weights[component][tuple(indices.T)] > 0
            index = indices[selected]
            if not len(index): continue
            if bool(claimed[component][tuple(index.T)].any()):
                raise ValueError('A physical foil trace was counted twice in the field norm.')
            claimed[component][tuple(index.T)] = True
            volume = dense_weights[component][tuple(index.T)]
            all_widths = torch.broadcast_to(trace['normal_half_widths'].double(), (2, len(indices)))
            widths = all_widths[:, selected]
            area = surfaces[component]['area'][selected].double()
            fraction = volume / (area * widths.sum(0))
            if bool((fraction > 1 + 64 * torch.finfo(torch.float64).eps).any()):
                raise ValueError('Foil/native field volume geometry is inconsistent.')
            normal = 'xyz'.index(trace['normal_axis']); coordinate = trace['normal_coordinate']
            if (bool((coordinate - widths[0] < bounds[normal][0]).any())
                or bool((coordinate + widths[1] > bounds[normal][1]).any())):
                raise ValueError('This bounded foil field norm requires complete normal half-cells inside Domain.')
            side_weights = widths * area[None] * fraction[None]
            weight_blocks.append(side_weights.reshape(-1))
            for block, owner in zip(blocks, records):
                other = next(value for value in owner[name]['bulk_electric_traces'] if value['component'] == component)
                if not torch.equal(other['indices'], trace['indices']) or not torch.equal(other['normal_half_widths'], trace['normal_half_widths']):
                    raise ValueError('Actual split-trace geometry changed between field bases.')
                block.append(other['electric_fields'][fi][:, selected].to(torch.complex128).reshape(-1))
    for component, weights in dense_weights.items():
        selected = (weights > 0) & ~claimed[component]
        weight_blocks.append(weights[selected])
        for block, result in zip(blocks, results):
            field = _native_plane_field(result, component, schema[component], frequency)[0]
            block.append(field[selected])
    volume = torch.cat(weight_blocks)
    vectors = [torch.cat(block) for block in blocks]
    return vectors, volume, {'bounds_m': bounds, 'norm_definition': 'sqrt(sum_native_nonfoil V_d |E_d|² + sum_foil_sides A_d h_side |E_side|²)',
        'field_units': 'V/m', 'physical_weights_units': 'm^3', 'squared_L2_units': 'V^2 m',
        'Linfinity_units': 'V/m', 'native_and_side_dof_count': len(volume),
        'component_volume_sum_m3': float(volume.sum()),
        'scope': 'Full physical E-vector observation norm; individual weak-component relative errors are retained separately and are not relabelled passing.'}


def flux_metrics(combined, direct, *, rtol=3e-5, atol=1e-17):
    frequencies = tuple(direct['frequencies_hz'])
    if (not frequencies or tuple(combined['frequencies_hz']) != frequencies
        or len(combined['native_signed_face_flux_w']) != len(frequencies)
        or len(direct['native_signed_face_flux_w']) != len(frequencies)):
        raise ValueError('Signed flux comparison requires complete matching frequency/face records.')
    records = []
    for index, frequency in enumerate(frequencies):
        a, b = [value['native_signed_face_flux_w'][index] for value in (combined, direct)]
        if len(b) != 6 or a.keys() != b.keys() or not np.isfinite([*a.values(), *b.values()]).all():
            raise ValueError('Every actual signed closed-surface face must be retained and finite.')
        actual, reference = sum(a.values()), sum(b.values())
        for value, signed in ((combined, a), (direct, b)):
            declared = float(value['net_outward_flux_w'][index])
            if abs(declared - sum(signed.values())) > 8 * np.finfo(float).eps * sum(map(abs, signed.values())) + 1e-30:
                raise ValueError('Declared signed net power is inconsistent with its actual face records.')
        entering = sum(max(-power, 0.) for power in b.values())
        if entering <= 0: raise ValueError('Relative power control requires resolved positive entering reference face power.')
        errors = {name: a[name] - b[name] for name in b}
        net_error = abs(actual - reference)
        face_l1_error = sum(map(abs, errors.values()))
        records.append({'frequency_hz': frequency, 'combined_signed_faces_w': a, 'direct_signed_faces_w': b,
            'signed_face_errors_w': errors, 'direct_entering_face_power_w': entering,
            'combined_net_power_w': actual, 'direct_net_power_w': reference,
            'net_absolute_error_w': net_error, 'signed_face_error_l1_w': face_l1_error,
            'net_error_relative_to_entering': net_error / entering,
            'face_l1_error_relative_to_entering': face_l1_error / entering,
            'passed': net_error <= atol + rtol * entering and face_l1_error <= len(b) * atol + rtol * entering,
            'original_net_relative_error': net_error / abs(reference) if reference else None,
            'original_net_relative_passed': net_error <= atol + rtol * abs(reference)})
    return {'rtol': rtol, 'atol_per_face_w': atol,
        'scale_definition': 'direct entering net-face power=sum_faces max(-direct signed integrated face flux,0); no fitted amplitude or phase',
        'frequency_records': records, 'passed': all(record['passed'] for record in records),
        'original_net_relative_passed': all(record['original_net_relative_passed'] for record in records)}


def _native_flux_geometry(reference, actual):
    """Exact physical geometry for a bounded Cartesian native-product face."""
    from witwin.maxwell.fdtd.plane_product import _PAIRS, _COORDS
    if reference.get('native_product_disabled',False)or actual.get('native_product_disabled',False):
        raise ValueError('This bounded contract requires actual Cartesian native-product flux.')
    if reference['axis'] != actual['axis'] or reference.get('normal_direction','+') != actual.get('normal_direction','+'):
        raise ValueError('Actual source-basis flux orientations differ.')
    axis=reference['axis'];names=_COORDS[axis]
    def same(a,b):
        a=torch.as_tensor(a,device='cpu');b=torch.as_tensor(b,device='cpu')
        return a.shape==b.shape and torch.equal(a,b)
    for name in names:
        if not same(reference[name],actual[name]) or not same(reference['cell_widths'][name],actual['cell_widths'][name]):
            raise ValueError('Actual source-basis flux areas/coordinates differ.')
    left=reference.get('native_components',reference.get('components',{}));right=actual.get('native_components',actual.get('components',{}))
    required={v for e,h,_,_ in _PAIRS[axis]for v in(e,h)}
    if not required.issubset(left)or not required.issubset(right):raise ValueError('Complete native product fields are required.')
    for component in required:
        if any(not same(a,b)for a,b in zip(left[component]['coords'],right[component]['coords'])):
            raise ValueError('Actual source-basis native coordinates differ.')
        if torch.as_tensor(left[component]['data']).shape!=torch.as_tensor(right[component]['data']).shape:
            raise ValueError('Actual source-basis native field shapes differ.')
    a,b=reference.get('native_flux_heads',()),actual.get('native_flux_heads',())
    if len(a)!=len(b):raise ValueError('Actual source-basis copper side supports differ.')
    for x,y in zip(a,b):
        if x['component']!=y['component']or x['averaging_axis']!=y['averaging_axis']or not same(x['indices'],y['indices']):
            raise ValueError('Actual source-basis copper endpoint geometry differs.')
        if any(torch.as_tensor(x[side]).shape!=torch.as_tensor(y[side]).shape for side in('negative','positive')):
            raise ValueError('Actual source-basis copper phasor shapes differ.')


def _native_flux_absolute_weights(payload,e,h,average):
    """Transpose absolute interpolation and physical endpoint/area weights."""
    from witwin.maxwell.fdtd.plane_product import _COORDS
    components=payload.get('native_components',payload.get('components',{}));a,b=components[e],components[h]
    def geometry(value):return torch.as_tensor(value,dtype=torch.float64,device='cpu')
    coords=tuple(geometry(value)for value in a['coords'])
    if any(not torch.equal(x,geometry(y))for x,y in zip(coords,b['coords'])):
        raise ValueError('Native product E/H endpoints must coincide.')
    names=_COORDS[payload['axis']];targets=[geometry(payload[name])for name in names]
    widths=[geometry(payload['cell_widths'][name])for name in names]
    if any(x.ndim!=1 or w.shape!=x.shape or not bool(torch.all(torch.isfinite(w)&(w>0)))for x,w in zip(targets,widths)):
        raise ValueError('Resolved physical native flux areas are required.')
    other=1-average;source=coords[other];target=targets[other]
    interpolation=torch.zeros((len(target),len(source)),dtype=torch.float64)
    if len(source)==1:interpolation[:,0]=1
    else:
        if not bool(torch.all(torch.diff(source)>0)):raise ValueError('Native flux coordinates must increase.')
        high=torch.searchsorted(source.contiguous(),target.contiguous()).clamp(1,len(source)-1);low=high-1
        fraction=(target-source[low])/(source[high]-source[low]);row=torch.arange(len(target))
        interpolation[row,low]=1-fraction;interpolation[row,high]+=fraction
    area=widths[0][:,None]*widths[1][None,:]
    physical=(area if average==0 else area.T)@interpolation.abs()
    nodes=coords[average]
    if len(nodes)<2 or not bool(torch.all(torch.diff(nodes)>0)):raise ValueError('Native flux requires distinct primal endpoints.')
    cell=(torch.searchsorted(nodes.contiguous(),targets[average].contiguous(),right=True)-1).clamp(0,len(nodes)-2)
    fraction=(targets[average]-nodes[cell])/(nodes[cell+1]-nodes[cell])
    if not bool(torch.all((fraction>=0)&(fraction<=1))):raise ValueError('Native flux target lies outside its physical primal cell.')
    lo=torch.zeros((len(nodes)-1,len(source)),dtype=torch.float64);hi=torch.zeros_like(lo)
    lo.index_add_(0,cell,physical*(1-fraction[:,None]));hi.index_add_(0,cell,physical*fraction[:,None])
    weight=torch.zeros((len(nodes),len(source)),dtype=torch.float64);weight[:-1]+=lo;weight[1:]+=hi
    sides=[];claimed=set()
    for number,head in enumerate(payload.get('native_flux_heads',())):
        if head['component']!=e:continue
        indices=torch.as_tensor(head['indices'],dtype=torch.long,device='cpu')
        if head['averaging_axis']!=average or indices.ndim!=2 or indices.shape[1]!=2:
            raise ValueError('Physical copper endpoint addresses differ.')
        keys=[tuple(index)for index in indices.tolist()]
        if len(set(keys))!=len(keys)or claimed.intersection(keys):raise ValueError('Duplicate physical copper endpoint.')
        claimed.update(keys);i,j=indices[:,average],indices[:,other]
        if bool((i<=0).any())or bool((i>=len(nodes)-1).any()):raise ValueError('Copper endpoints require both physical half cells.')
        negative,positive=hi[i-1,j],lo[i,j];weight[i,j]-=negative+positive
        sides.append((number,indices,negative,positive))
    if bool((weight<0).any()):raise ValueError('Negative residual endpoint bound weight.')
    return weight if average==0 else weight.T,sides


def actual_basis_bilinear_flux_metrics(combined,direct,bases,coefficients,*,monitor_name='line_roi_flux',rtol=3e-5,atol=1e-17):
    """Bounded source-superposition error relative to actual input fields, in W.

    The scale contains ONLY actual independent basis fields, fixed coefficients,
    physical areas, native endpoint and absolute interpolation weights, including
    both Cu sides. It is neither incident power nor abs(signed integrated flux).
    The original entering-relative metric/default/verdict remains unchanged and
    is retained separately; no relative weak-output accuracy is implied.
    """
    from witwin.maxwell.fdtd.plane_product import _PAIRS
    from witwin.maxwell.fdtd.quadrature import plane_normal_poynting
    frequencies=tuple(direct.frequencies);c=torch.as_tensor(coefficients,dtype=torch.complex128,device='cpu')
    if not frequencies or not bases or c.shape!=(len(frequencies),len(bases))or not bool(torch.isfinite(c).all()):
        raise ValueError('Complete finite declared source coefficients are required.')
    if any(tuple(value.frequencies)!=frequencies or value.stats().get('normalize_source',False)for value in(combined,direct,*bases)):
        raise ValueError('Actual basis/outputs must retain common raw spectral normalization.')
    expected={axis+'_'+side for axis in'xyz'for side in('neg','pos')};records=[]
    power=[dict(frequencies_hz=frequencies,native_signed_face_flux_w=[],net_outward_flux_w=[])for _ in range(2)]
    def field(value):
        value=torch.as_tensor(value,device='cpu').to(torch.complex128)
        if not bool(torch.isfinite(value).all()):raise ValueError('Actual source phasors must be finite.')
        return value
    for fi,frequency in enumerate(frequencies):
        measured=[value.monitor(monitor_name,frequency=frequency)['faces']for value in(combined,direct)]
        inputs=[value.monitor(monitor_name,frequency=frequency)['faces']for value in bases]
        if any(set(value)!=expected for value in(*measured,*inputs)):
            raise ValueError('All six physical signed faces are required.')
        scales={};signed=[{},{}]
        for name in sorted(expected):
            geometry=inputs[0][name]
            for payload in [value[name]for value in(*inputs[1:],*measured)]:_native_flux_geometry(geometry,payload)
            if geometry['axis']!=name[0]or geometry.get('normal_direction','+')!=('-'if name.endswith('neg')else'+'):
                raise ValueError('Closed-face physical orientation is inconsistent.')
            scale=0.
            for e,h,average,_ in _PAIRS[geometry['axis']]:
                weight,sides=_native_flux_absolute_weights(geometry,e,h,average)
                entries=[value[name].get('native_components',value[name].get('components',{}))for value in inputs]
                E=sum(abs(c[fi,j])*abs(field(entry[e]['data']))for j,entry in enumerate(entries))
                H=sum(abs(c[fi,j])*abs(field(entry[h]['data']))for j,entry in enumerate(entries))
                scale+=.5*float((weight*E*H).sum())
                for number,indices,negative,positive in sides:
                    ii,jj=indices[:,0],indices[:,1]
                    for side,weights in(('negative',negative),('positive',positive)):
                        endpoint=sum(abs(c[fi,j])*abs(field(value[name]['native_flux_heads'][number][side]))for j,value in enumerate(inputs))
                        scale+=.5*float((weights*endpoint*H[...,ii,jj]).sum())
            if not math.isfinite(scale)or scale<0:raise ValueError('Actual bilinear input scale is unresolved.')
            scales[name]=scale
            for output,faces in zip(signed,measured):
                density,area=plane_normal_poynting(faces[name]);output[name]=float((density*area).sum())
        total=sum(scales.values())
        if total<=0:raise ValueError('Positive actual source-basis bilinear W scale required; no floor is applied.')
        errors={name:signed[0][name]-signed[1][name]for name in expected};net=abs(sum(errors.values()));l1=sum(map(abs,errors.values()))
        records.append(dict(frequency_hz=frequency,actual_input_basis_bilinear_W_scale=total,per_face_input_bilinear_W_scales=scales,
            combined_signed_faces_w=signed[0],direct_signed_faces_w=signed[1],signed_face_errors_w=errors,
            net_absolute_error_w=net,signed_face_error_l1_w=l1,net_error_relative_to_actual_basis=net/total,
            face_l1_error_relative_to_actual_basis=l1/total,passed=net<=atol+rtol*total and l1<=6*atol+rtol*total))
        for value,faces in zip(power,signed):value['native_signed_face_flux_w'].append(faces);value['net_outward_flux_w'].append(sum(faces.values()))
    return dict(rtol=rtol,atol_per_face_w=atol,scale_units='W',
        scale_definition='0.5 absolute native endpoint/dual-Cu/area/interpolation operator applied to (sum|c_j||E_j|)(sum|c_k||H_k|); actual independent basis only, not incident or abs net signed power.',
        frequency_records=records,passed=all(record['passed']for record in records),
        original_entering_power_metrics=flux_metrics(*power,rtol=rtol,atol=atol),
        scope='Input-conditioned bounded source-superposition functional error. Original weak-output relative precision verdict remains separate; no attenuation qualification.')
