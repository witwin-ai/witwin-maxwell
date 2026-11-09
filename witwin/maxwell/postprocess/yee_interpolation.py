"""Side-preserving interpolation of the native split copper Yee unknowns.

Tangential E has an independent recorded value on each face. Tangential H and
normal E live on exterior half nodes: within the interface interval we use the
nearest two samples on the requested side, including linear extrapolation to
that side's boundary. This reconstruction is second order for smooth one-sided
fields; it does not impose an exact continuum surface-charge jump.
"""
from __future__ import annotations

import torch


def copper_query_faces(result, *, device, frequency_index=0):
    """Join physical face geometry with the actual (PEC-contact filtered) traces."""
    from ..compiler.copper_foil import compile_copper_faces
    compiled = {(face.name, 'E'+'xyz'[face.component]): face
                for face in compile_copper_faces(result.prepared_scene, device=device)}
    faces = []
    for name, record in result.conductor_losses.items():
        for trace in record.get('bulk_electric_traces', ()):
            face = compiled.get((name, trace['component']))
            if face is None:
                raise ValueError('Retained copper trace support does not match the query Scene.')
            # Only retained traces need the field frequency; geometry-only
            # planning (no recorded traces) has no frequency to select.
            requested_frequency = result.frequencies[frequency_index]
            frequencies = torch.as_tensor(record['frequencies'], device=device)
            matches = torch.nonzero(frequencies == requested_frequency).flatten()
            if len(matches) != 1:
                raise ValueError('Copper query traces do not contain the recorded field frequency.')
            record_frequency_index = int(matches[0])
            if 'spectral_valid' in record and not bool(record['spectral_valid'][record_frequency_index]):
                raise ValueError('Copper query traces contain no spectral samples.')
            retained=trace.get('electric_fields')
            if (not isinstance(retained,torch.Tensor) or retained.ndim!=3
                or retained.shape[0]!=len(frequencies) or retained.shape[1]!=2
                or retained.shape[2]!=len(trace['indices'])):
                raise ValueError('Retained copper query spectra require both side heads at every declared frequency and owned edge.')
            faces.append(dict(axis=face.axis, normal=face.normal_index, component=face.component,
                              indices=trace['indices'].to(device=device),
                              electric_fields=trace['electric_fields'][record_frequency_index],
                              name=name, sampling=trace.get('sampling'), compiled_indices=face.indices))
    # A missing snapshot must never silently turn a split interface into ordinary
    # continuous interpolation. Geometry is retained for the precise query guard.
    available = {(face['name'], face['component']) for face in faces}
    for (name, component), face in compiled.items():
        if (name, face.component) not in available:
            faces.append(dict(axis=face.axis, normal=face.normal_index, component=face.component,
                              indices=face.indices, electric_fields=None, name=name))
    return faces


def _membership(indices, support, shape, *, normal):
    tangent = [axis for axis in range(3) if axis != normal]
    def flatten(values):
        return values[:, tangent[0]]*shape[tangent[1]]+values[:, tangent[1]]
    ordered = torch.unique(flatten(support), sorted=True)
    if not len(ordered):
        return torch.zeros(len(indices), device=indices.device, dtype=torch.bool)
    query = flatten(indices)
    position = torch.searchsorted(ordered, query).clamp_max(len(ordered)-1)
    return ordered[position] == query


def interface_side(points, face, trace_side, point_roundoff=0.):
    normal = face['axis']
    # Face normal coordinates are primal nodes, independently of the queried
    # component's own half/node staggering.
    plane = face['plane']
    distance = points[:, normal]-plane
    scale = torch.maximum(points[:, normal].abs(), plane.abs()).clamp_min(torch.finfo(points.dtype).tiny)
    tolerance = 2*torch.finfo(points.dtype).eps*scale + point_roundoff
    exact = distance.abs() <= tolerance
    positive = distance > tolerance
    if trace_side == 'positive':
        positive = positive | exact
    return positive, exact


def interpolation_stencil(result, points, component, axes, *, trace_side=None, point_roundoff=0., faces=None):
    """Eight weighted native samples per point, with optional side-trace values."""
    if trace_side not in (None, 'negative', 'positive'):
        raise ValueError("FDTD trace_side must be 'negative', 'positive', or None.")
    if points.device.type == 'cuda' and not points.requires_grad:
        # Index/branch planning is static geometry, not a field operation. Keep
        # these small decisions on the host to avoid a device synchronization
        # for every face and corner. Field gathers and the live side spectra
        # remain on device and retain their ordinary PyTorch derivatives.
        if faces is None:
            faces = copper_query_faces(result, device='cpu')
        host_faces = [dict(face, indices=face['indices'].detach().cpu()) for face in faces]
        planned = interpolation_stencil(result, points.detach().cpu(), component,
            [axis.detach().cpu() for axis in axes], trace_side=trace_side,
            point_roundoff=point_roundoff, faces=host_faces)
        return [(indices.to(points.device), weight.to(points.device),
                 [(selected.to(points.device), positive.to(points.device), face)
                  for selected, positive, face in overrides])
                for indices, weight, overrides in planned]
    lower, fractions = [], []
    for dim, axis in enumerate(axes):
        position = points[:, dim].contiguous()
        if bool(((position < axis[0]) | (position > axis[-1])).any()):
            raise ValueError('Query lies outside the stored Yee component domain')
        index = torch.searchsorted(axis, position, right=True).clamp(1, len(axis)-1)-1
        lower.append(index)
        fractions.append((position-axis[index])/(axis[index+1]-axis[index]))
    if faces is None:
        faces = copper_query_faces(result, device=points.device)
    for face in faces:
        nodes = getattr(result.prepared_scene, 'xyz'[face['axis']]+'_nodes64')
        face['plane'] = points.new_tensor(nodes[face['normal']])
    shape = tuple(len(axis) for axis in axes)
    corners = []
    for corner in range(8):
        indices = torch.stack([lower[dim]+((corner>>dim)&1) for dim in range(3)], -1)
        factors = [fractions[dim] if (corner>>dim)&1 else 1-fractions[dim] for dim in range(3)]
        overrides = []
        altered = []
        for face in faces:
            normal, own = face['axis'], 'xyz'.index(component[1].lower())
            positive, exact = interface_side(points, face, trace_side, point_roundoff)
            support = face['indices']
            if component.startswith('E') and own == normal:
                # Charge lives on the primal surface vertices bounding the
                # tangential electric edges, including their two endpoints.
                upper = support.clone()
                upper[:, face['component']] += 1
                support = torch.cat((support, upper))
            elif ((component.startswith('E') and own != face['component']) or
                  (component.startswith('H') and (own == normal or own == face['component']))):
                continue
            active = _membership(indices, support, shape, normal=normal)
            for tangent in range(3):
                if tangent != normal:
                    active = active & (factors[tangent] != 0)
            if not bool(active.any()):
                continue
            if trace_side is None and bool((active & exact).any()):
                raise ValueError("A query on a CopperFoil interface requires trace_side='negative' or 'positive'.")
            if component.startswith('E') and own != normal:
                plane_index = indices[:, normal] == face['normal']
                selected = active & plane_index
                if bool(selected.any()):
                    if face['electric_fields'] is None:
                        raise ValueError('Copper field queries require retained two-sided electric spectra.')
                    overrides.append((selected, positive, face))
                continue
            # H_t and E_n are staggered along the sheet normal. Do not blend
            # their independent exterior half samples across the interface.
            n = face['normal']
            axis = axes[normal]
            interval = active & (points[:, normal] > axis[n-1]) & (points[:, normal] < axis[n])
            if not bool(interval.any()):
                continue
            for previous_normal, previous_plane, previous_mask in altered:
                if (normal, n) != (previous_normal, previous_plane) and bool((interval & previous_mask).any()):
                    raise NotImplementedError('Copper query stencil crosses ambiguous intersecting interfaces.')
            insufficient = ((~positive) & (n < 2)) | (positive & (n+1 >= len(axis)))
            if bool((interval & insufficient).any()):
                raise ValueError('Copper field reconstruction requires two retained half nodes on the requested side.')
            pair_lower = torch.where(positive, n, n-2).clamp(0, len(axis)-2)
            first, second = axis[pair_lower], axis[pair_lower+1]
            for other in faces:
                if other['axis'] != normal or other['normal'] == n:
                    continue
                other_support = other['indices']
                if component.startswith('E') and own == normal:
                    upper = other_support.clone()
                    upper[:, other['component']] += 1
                    other_support = torch.cat((other_support, upper))
                elif ((component.startswith('E') and own != other['component']) or
                      (component.startswith('H') and own == other['component'])):
                    continue
                crossing = (other['plane'] > torch.minimum(first, points[:, normal])) & (other['plane'] < torch.maximum(second, points[:, normal]))
                crossing = crossing & _membership(indices, other_support, shape, normal=normal)
                if bool((interval & crossing).any()):
                    raise ValueError('Insufficient same-side half nodes between neighboring copper interfaces.')
            fraction = (points[:, normal]-first)/(second-first)
            indices[:, normal] = torch.where(interval, pair_lower+((corner>>normal)&1), indices[:, normal])
            normal_factor = fraction if (corner>>normal)&1 else 1-fraction
            factors[normal] = torch.where(interval, normal_factor, factors[normal])
            altered.append((normal, n, interval))
        weight = factors[0]*factors[1]*factors[2]
        corners.append((indices, weight, overrides))
    return corners


def normalized_trace_values(face, sample_native, *, normalized):
    """Use the same spectral scale/phase as retained native E, with a live VJP."""
    native, present = sample_native(face['indices'], required=False)
    original = face['electric_fields'].to(native.device)
    left = original[0]
    denominator = left[present].abs().square().sum()
    if bool(denominator > 0):
        factor = (left[present].conj()*native[present]).sum()/denominator
        mismatch = (native[present]-factor*left[present]).abs().square().sum()
        scale = native[present].abs().square().sum().clamp_min(torch.finfo(torch.float64).tiny)
        if bool(mismatch/scale > 1e-7):
            raise ValueError('Electric fields do not match the retained foil spectral schedule and normalization.')
    elif normalized and bool(original.abs().any()):
        raise ValueError('Normalized copper queries require a nonzero retained left-trace sample to determine the common spectral scale.')
    else:
        factor = original.new_tensor(1.)
    return original*factor
