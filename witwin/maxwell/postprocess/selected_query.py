"""Selected native FDTD samples, sharing the physical Yee interpolation stencil."""
from __future__ import annotations
import math
import torch

_COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def selections(result, components, frequencies):
    if not isinstance(components, (tuple, list)) or not components or any(c not in _COMPONENTS for c in components) or len(set(components)) != len(components):
        raise ValueError('components must select unique nonempty native E/H component names.')
    recorded = tuple(float(f) for f in result.frequencies)
    selected = recorded if frequencies is None else tuple(frequencies)
    if not selected or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) for f in selected) or len(set(selected)) != len(selected):
        raise ValueError('Select unique finite recorded frequencies in the requested output order.')
    if len(set(recorded)) != len(recorded) or any(f not in recorded for f in selected):
        raise ValueError('Every selected frequency must uniquely match a recorded frequency.')
    return tuple(components), tuple(recorded.index(f) for f in selected)


def validate_sampling(result, face, frequency_index):
    """Require actual producer evidence; never estimate a complex head multiplier."""
    sampling = face.get('sampling')
    required = {'schema', 'dt', 'frequencies', 'start_steps', 'planned_steps', 'completed_steps',
                'window', 'weight_sums', 'phasor_factor', 'normalize_source',
                'electric_phase_origin_step', 'electric_time_offset_steps', 'phasor_convention',
                'owner', 'component', 'normal_axis', 'normal_index', 'energy_representation'}
    if not isinstance(sampling, dict) or not required.issubset(sampling):
        raise ValueError('Strict copper queries require retained actual per-face sampling provenance; old snapshots cannot supply it.')
    tensor_keys=('frequencies','start_steps','completed_steps','weight_sums')
    if (any(not isinstance(sampling[key],torch.Tensor) or sampling[key].dtype!=torch.float64 for key in tensor_keys)
        or sampling['completed_steps'].ndim!=0 or type(sampling['planned_steps']) is not int
        or type(sampling['electric_phase_origin_step']) is not int
        or isinstance(sampling['dt'],bool) or not isinstance(sampling['dt'],(int,float))):
        raise ValueError('Strict copper sampling requires its actual typed float64 clock and frequency/norm records.')
    stats = result.stats()
    steps = stats.get('time_steps')
    if (type(steps) is not int or steps <= 0 or stats.get('steps_run') != steps
        or sampling['schema'] != 'native-copper-electric-sampling/1'
        or sampling['planned_steps'] != steps or sampling['window'] != 'none'
        or sampling['normalize_source'] is not False or stats.get('normalize_source') is not False
        or stats.get('dft_window') != 'none' or sampling['dt'] != stats.get('dt')
        or not math.isfinite(float(sampling['dt'])) or sampling['dt'] <= 0
        or sampling['electric_phase_origin_step'] != 1 or sampling['electric_time_offset_steps'] != 0.
        or sampling['phasor_convention'] != 'exp(-iwt)'
        or sampling['energy_representation'] != 'physical_real_field'
        or sampling['owner'] != face['name'] or sampling['component'] != 'E'+'xyz'[face['component']]
        or sampling['normal_axis'] != 'xyz'[face['axis']] or sampling['normal_index'] != face['normal']):
        raise ValueError('Strict copper and native observations require one complete raw none-window electric clock.')
    frequencies = torch.as_tensor(sampling['frequencies'], dtype=torch.float64)
    starts = torch.as_tensor(sampling['start_steps'], device=frequencies.device)
    norm = torch.as_tensor(sampling['weight_sums'], device=frequencies.device)
    completed = torch.as_tensor(sampling['completed_steps'], device=frequencies.device)
    expected = frequencies.new_tensor(tuple(float(f) for f in result.frequencies))
    representation = stats.get('time_plan', {}).get('source_representation')
    expected_factor = 1. if representation == 'analytic_cw' else 2. if representation == 'real_projected' else None
    if (frequencies.shape != expected.shape or not torch.equal(frequencies, expected)
        or starts.shape != expected.shape or norm.shape != expected.shape
        or completed.numel() != 1 or not bool(torch.isfinite(completed).all())
        or float(completed) != steps or not bool((starts == 0).all())
        or not bool((norm == steps).all()) or sampling['phasor_factor'] != expected_factor
        or expected_factor is None or not 0 <= frequency_index < len(expected)):
        raise ValueError('Copper sampling frequencies, actual count/norm or phase representation differ from native observations.')
    if face.get('electric_fields') is None or face['electric_fields'].ndim != 2 or face['electric_fields'].shape[0] != 2:
        raise ValueError('Strict copper queries require both actual retained side heads.')
    if face['electric_fields'].dtype!=torch.complex128 or not bool(torch.isfinite(face['electric_fields']).all()):
        raise ValueError('Strict copper queries require actual finite complex128 constitutive side heads; native primitive precision is separate.')
    support=face['indices'];compiled=face['compiled_indices'].to(device=support.device)
    if (support.dtype!=torch.long or support.ndim!=2 or support.shape[1]!=3
        or not bool((support[:,face['axis']]==face['normal']).all())):
        raise ValueError('Copper query indices must retain actual integer edge ownership and normal.')
    from ..fdtd.coords import component_coords
    shape=tuple(len(axis) for axis in component_coords(result.prepared_scene,'E'+'xyz'[face['component']]))
    if any(bool(((support[:,axis]<0)|(support[:,axis]>=shape[axis])).any()) for axis in range(3)):
        raise ValueError('Copper query support lies outside actual component coordinates.')
    flatten=lambda index:(index[:,0]*shape[1]+index[:,1])*shape[2]+index[:,2]
    stored=flatten(support);expected_support=flatten(compiled)
    if len(torch.unique(stored))!=len(stored) or not bool(torch.isin(stored,expected_support).all()):
        raise ValueError('Retained copper edges differ from the actual compiled owner support.')



def _frequency_index(payload, frequency):
    frequencies = tuple(payload.get('frequencies', (payload.get('frequency'),)))
    matches = [i for i, f in enumerate(frequencies) if f == frequency]
    return (matches[0], len(frequencies)) if len(matches) == 1 else (None, len(frequencies))


def native_sampler(result, points, component, axes, frequency_index):
    """Gather retained native fields, Point and exact single-plane components."""
    shape = tuple(len(axis) for axis in axes)
    device = points.device
    nfreq = len(result.frequencies)
    frequency = float(result.frequencies[frequency_index])
    dense = result._fields.get(component.upper())
    # Real dense fields are final instantaneous snapshots, not spectra. Match
    # the existing query backend by seeking retained complex monitor samples.
    if isinstance(dense, torch.Tensor) and not dense.is_complex():
        dense = None
    if dense is not None:
        if not isinstance(dense, torch.Tensor) or not dense.is_complex():
            raise ValueError('Selected queries require complex retained spectral fields.')
        if nfreq == 1 and tuple(dense.shape) == shape:
            values = dense.to(device=device)
        elif tuple(dense.shape) == (nfreq, *shape):
            values = dense[frequency_index].to(device=device)
        else:
            raise ValueError('Retained spectral field shape differs from its actual frequency/Yee axes.')
        counts = result.stats().get('dft_sample_counts', ())
        if len(counts) != nfreq or counts[frequency_index] != result.stats().get('time_steps'):
            raise ValueError('Selected full-field queries require their actual complete spectral sample count.')
        def full(indices, *, required=True):
            indices = indices.to(device=device)
            return values[tuple(indices.T)], torch.ones(len(indices), device=device, dtype=torch.bool)
        return full
    ids, chunks = [], []
    expected_count = result.stats().get('time_steps')
    for payload in result._monitors.values():
        if not isinstance(payload, dict):
            continue
        if payload.get('kind') not in ('point','plane'):
            continue
        component_payload = payload.get('components', {}).get(component)
        if component_payload is None:
            continue
        if payload.get('kind') == 'plane':
            native_entry = payload.get('native_components', payload.get('components', {})).get(component)
            # Derived collocated or normally interpolated presentations are
            # not native degrees of freedom. They neither fill a missing
            # stencil nor invalidate independently retained native samples.
            if (isinstance(native_entry,dict) and
                (native_entry.get('collocated',False)
                 or tuple(native_entry.get('plane_weights',())) != (1.,)
                 or len(tuple(native_entry.get('plane_indices',()))) != 1)):
                continue
        fi, count = _frequency_index(payload, frequency)
        if fi is None:
            continue
        samples = torch.as_tensor(payload.get('samples', ())).reshape(-1)
        if len(samples) != count or float(samples[fi]) != expected_count:
            raise ValueError('Retained native samples do not share the declared complete spectral horizon.')
        if payload.get('kind') == 'point':
            index = payload.get('field_indices', {}).get(component)
            if index is None or len(index) != 3 or any(not 0 <= int(i) < shape[a] for a, i in enumerate(index)):
                raise ValueError('Retained PointMonitor has no valid actual native component index.')
            value = torch.as_tensor(component_payload, device=device)
            if value.numel() != count:
                raise ValueError('Retained PointMonitor spectrum differs from its frequency count.')
            ids.append(torch.tensor([(int(index[0])*shape[1]+int(index[1]))*shape[2]+int(index[2])], device=device))
            chunks.append(value.reshape(-1)[fi:fi+1])
        elif payload.get('kind') == 'plane':
            entry = payload.get('native_components', payload.get('components', {})).get(component)
            if (not isinstance(entry, dict) or entry.get('collocated', False)
                or tuple(entry.get('plane_weights', ())) != (1.,)
                or len(tuple(entry.get('plane_indices', ()))) != 1 or payload.get('axis') not in ('x','y','z')):
                raise ValueError('Selected queries require an uncollocated single actual native plane, without normal interpolation.')
            normal = 'xyz'.index(payload['axis']);tangents = [a for a in range(3) if a != normal]
            normal_index = int(entry['plane_indices'][0])
            if not 0 <= normal_index < shape[normal]:
                raise ValueError('Native plane index is outside the actual component grid.')
            coords = entry.get('coords', ())
            if len(coords) != 2:
                raise ValueError('Native plane must retain both exact tangential coordinate arrays.')
            locations = []
            for axis, coordinates in zip(tangents, coords):
                coordinate = torch.as_tensor(coordinates, device=device, dtype=torch.float64)
                native = axes[axis].to(device=device, dtype=torch.float64)
                location = torch.searchsorted(native.contiguous(), coordinate.contiguous())
                if (coordinate.ndim != 1 or not len(coordinate) or bool((location >= len(native)).any())
                    or not torch.equal(native[location], coordinate)
                    or bool((coordinate[1:] <= coordinate[:-1]).any())):
                    raise ValueError('Retained native plane coordinates must match distinct actual Yee nodes exactly.')
                locations.append(location)
            value = torch.as_tensor(entry.get('data'), device=device)
            plane_shape = tuple(len(x) for x in locations)
            if count == 1 and tuple(value.shape) == plane_shape:
                selected = value
            elif tuple(value.shape) == (count, *plane_shape):
                selected = value[fi]
            else:
                raise ValueError('Native plane shape does not match its frequencies and component coordinates.')
            first, second = torch.meshgrid(*locations, indexing='ij')
            index = [None]*3;index[normal] = torch.full_like(first, normal_index)
            index[tangents[0]], index[tangents[1]] = first, second
            ids.append(((index[0]*shape[1]+index[1])*shape[2]+index[2]).reshape(-1))
            chunks.append(selected.reshape(-1))
    if not chunks:
        def missing(indices, *, required=True):
            raise ValueError(f'No retained spectral native {component} samples cover the selected frequency.')
        return missing
    if any(not value.is_complex() for value in chunks):
        raise ValueError('Selected queries require complex native spectral samples.')
    sample_ids, order = torch.sort(torch.cat(ids));sample_values = torch.cat(chunks)[order]
    duplicates = sample_ids[1:] == sample_ids[:-1]
    if bool(duplicates.any()) and not torch.equal(sample_values[1:][duplicates], sample_values[:-1][duplicates]):
        raise ValueError('Overlapping retained native samples conflict; no averaging or fitted scale is permitted.')
    def sparse(indices, *, required=True):
        indices = indices.to(device=device)
        flat = (indices[:,0]*shape[1]+indices[:,1])*shape[2]+indices[:,2]
        position = torch.searchsorted(sample_ids, flat).clamp_max(len(sample_ids)-1)
        present = sample_ids[position] == flat
        if required and not bool(present.all()):
            raise ValueError(f'Missing {component} interpolation stencil samples in retained native data.')
        return sample_values[position], present
    return sparse


def query_selected(result, points, *, components, frequencies, trace_side, owner, representation, point_roundoff=0.):
    from ..fdtd.coords import component_coords
    from ..fdtd.plane_product import _require_uniform_nonmagnetic
    from .yee_interpolation import copper_query_faces, interpolation_stencil
    from ..result import _interpolate_fdtd_component
    if points.requires_grad:
        raise NotImplementedError('Selected native coordinate or shape gradients are not yet qualified; use fixed query geometry for field/head VJPs.')
    if points.dtype != torch.float64:
        raise ValueError('Selected native geometry currently requires float64 points; default six-field queries retain their existing dtype support.')
    components, indices = selections(result, components, frequencies)
    if representation not in ('constitutive', 'published_native'):
        raise ValueError('Selected queries require constitutive or published_native representation.')
    if owner is not None and (not isinstance(owner, str) or not owner):
        raise ValueError('An explicit copper owner must be a nonempty structure name.')
    stats = result.stats()
    source_representation = stats.get('time_plan', {}).get('source_representation')
    if source_representation == 'analytic_cw':
        raise NotImplementedError('Selected native analytic complex time-bank queries need separately qualified bank clocks.')
    if source_representation != 'real_projected':
        raise ValueError('Selected native queries require an explicit retained real_projected time representation.')
    if getattr(result.scene, 'coordinates', None) is not None:
        raise NotImplementedError('Selected native queries in mapped coordinates are not yet qualified.')
    if any(result.scene.boundary.axis_kind(a) in ('periodic','bloch') for a in 'xyz'):
        raise NotImplementedError('Selected native periodic or Bloch seam queries are not yet qualified.')
    if stats.get('normalize_source') is True:
        raise NotImplementedError('Selected native physical queries do not yet undo recorded source normalization.')
    if stats.get('normalize_source') is not False:
        raise ValueError('Selected native queries require an explicit retained source-normalization flag.')
    if stats.get('dft_window') in ('hanning','ramp'):
        raise NotImplementedError('Selected native weighted-window queries need qualified common observer and copper clock records.')
    if stats.get('dft_window') != 'none':
        raise ValueError('Selected native queries require an explicit recognized retained spectral window.')
    if representation == 'published_native' and (owner is None or trace_side != 'negative'):
        raise ValueError('Published copper primitives require an explicit owner and negative trace_side.')
    values = {component: [] for component in components}
    for fi in indices:
        faces = copper_query_faces(result, device=points.device, frequency_index=fi)
        for component in components:
            if owner is not None and len([face for face in faces if face['name']==owner and component=='E'+'xyz'[face['component']]])!=1:
                raise ValueError('Each requested copper owner/component must have one unique retained tangential trace.')
            axes = [torch.as_tensor(axis, device=points.device, dtype=points.dtype) for axis in component_coords(result.prepared_scene, component)]
            stencil = interpolation_stencil(result, points, component, axes, trace_side=trace_side, point_roundoff=point_roundoff, faces=faces)
            owned = torch.zeros(len(points), dtype=torch.bool, device=points.device)
            for locations, weights, overrides in stencil:
                active = weights != 0
                for selected, positive, face in overrides:
                    used = selected & active
                    if not bool(used.any()):
                        continue
                    if owner is None or face['name'] != owner or component != 'E'+'xyz'[face['component']]:
                        raise ValueError('Every active copper trace term requires its exact explicit owner and tangential component.')
                    if trace_side not in ('negative','positive'):
                        raise ValueError('Strict copper samples require an explicit negative or positive trace_side.')
                    axis = face['axis'];plane = axes[axis][face['normal']]
                    if not bool((points[used,axis] == plane).all()):
                        raise ValueError('The selected owned copper operation requires points on its exact native plane.')
                    validate_sampling(result, face, fi)
                    owned |= used
                if owner is not None:
                    matched = torch.zeros(len(points), dtype=torch.bool, device=points.device)
                    for selected, _, face in overrides:
                        if face['name'] == owner: matched |= selected
                    if bool((active & ~matched).any()):
                        raise ValueError('Every nonzero owned trace stencil term must belong to the declared copper owner.')
            if owner is not None:
                _require_uniform_nonmagnetic(result.scene)
                if not bool(owned.all()):
                    raise ValueError('The requested owner has no active tangential trace at every query point.')
                bounds = result.scene.domain.bounds
                if any(bool(((points[:,a] <= bounds[a][0]) | (points[:,a] >= bounds[a][1])).any()) for a in range(3)):
                    raise ValueError('Owned copper query points must be inside the nonPML physical domain.')
            sampled = _interpolate_fdtd_component(result, points, component, trace_side=trace_side,
                point_roundoff=point_roundoff, copper_faces=faces,
                selected_frequency_index=fi, selected_representation=representation)
            values[component].append(sampled)
    return {component: torch.stack(samples) for component, samples in values.items()}
