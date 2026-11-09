"""Fixed physical V/I quadrature expanded into actual native Yee point samples.

The reference curve, its quadrature points and SI integral weights are fixed.
Only the interpolation stencil changes with the propagation mesh. This does not
qualify the separate reference-curve quadrature error as continuum truth.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.postprocess.yee_interpolation import interpolation_stencil


def fixed_physical_probe_monitors(scene, reference_scene, reference_monitors, reference_rows):
    """Use the shared side-aware interpolation planner; never synthesize fields."""
    if any(scene.boundary.axis_kind(axis) in ('periodic', 'bloch') for axis in 'xyz'):
        raise ValueError('Fixed physical V/I planning does not cross periodic or Bloch seams.')
    if getattr(scene, 'coordinates', None) is not None:
        raise ValueError('Fixed physical V/I planning requires Cartesian coordinates.')
    prepared = prepare_scene(scene)
    reference = {monitor.name: monitor for monitor in reference_monitors}
    geometries = [dict(axis=face.axis, normal=face.normal_index, component=face.component,
        indices=face.indices, electric_fields=None, name=face.name)
        for face in compile_copper_faces(prepared, device='cpu')]
    sheets = []
    for structure in scene.structures:
        geometry = structure.geometry
        if isinstance(geometry, mw.PlanarPEC):
            vertices = geometry.vertices.detach().cpu().numpy()
            lower, upper = vertices.min(0), vertices.max(0)
            rectangle = {(float(a), float(b)) for a in (lower[0], upper[0]) for b in (lower[1], upper[1])}
            if geometry.holes or len(vertices) != 4 or set(map(tuple, vertices)) != rectangle:
                raise ValueError('This bounded fixed-V/I planner requires rectangular planar PEC footprints.')
            sheets.append(('xyz'.index(geometry.normal_axis), geometry.coordinate, lower, upper))
        elif bool(getattr(structure.material, 'is_copper_foil', False)):
            size = geometry.size.detach().cpu().numpy(); center = geometry.position.detach().cpu().numpy()
            normal = int(np.flatnonzero(size == 0)[0]); tangents = [i for i in range(3) if i != normal]
            sheets.append((normal, float(center[normal]), center[tangents]-size[tangents]/2, center[tangents]+size[tangents]/2))
    monitors = []; expected = {}; keys = {}; rows = []; audit = []
    context = SimpleNamespace(prepared_scene=prepared)
    requested = {}
    for pair in reference_rows:
        for row in pair:
            for name, component, _ in row:
                requested.setdefault(component, {})[reference[name].position] = None
    planned = {}
    for component, positions in requested.items():
        positions = list(positions)
        axes = [torch.as_tensor(axis, dtype=torch.float64) for axis in component_coords(prepared, component)]
        points = torch.tensor(positions, dtype=torch.float64)
        stencil = interpolation_stencil(context, points, component, axes, faces=geometries)
        for number, position in enumerate(positions):
            expanded = []
            native_points = [np.array([float(axis[i]) for axis,i in zip(axes,indices[number])])
                for indices, weights, _ in stencil if float(weights[number]) != 0]
            lower, upper = np.min(native_points,axis=0), np.max(native_points,axis=0)
            for normal, plane, footprint_lower, footprint_upper in sheets:
                tangent = [i for i in range(3) if i != normal]
                intersects = np.all(upper[tangent] >= footprint_lower) and np.all(lower[tangent] <= footprint_upper)
                if intersects and lower[normal] < plane < upper[normal]:
                    raise ValueError('Fixed physical V/I interpolation stencil crosses a conductor interface.')
            for indices, weights, overrides in stencil:
                if overrides:
                    raise ValueError('Fixed physical V/I samples cannot substitute an unrecorded copper electric side trace.')
                weight = float(weights[number])
                if weight == 0: continue
                index = tuple(int(i) for i in indices[number])
                key = (component, *index)
                if key not in keys:
                    name = f'fixed_physical_{len(keys)}'; keys[key] = name
                    coordinates = tuple(float(axis[i]) for axis, i in zip(axes, index))
                    monitors.append(mw.PointMonitor(name, coordinates, fields=(component,)))
                    expected[name] = (component, index)
                expanded.append((keys[key], component, weight))
            planned[component, position] = expanded
    def sample(component, position, integral_weight):
        return [(name, component, integral_weight * weight)
            for name, _, weight in planned[component, position]]
    for voltage, current in reference_rows:
        station_rows = []
        for kind, original in (('voltage', voltage), ('current', current)):
            expanded = []
            for name, component, weight in original:
                position = reference[name].position
                expanded.extend(sample(component, position, weight))
                audit.append(dict(kind=kind, component=component, physical_position_m=position,
                    integral_weight_SI=float(weight)))
            station_rows.append(expanded)
        rows.append(tuple(station_rows))
    assert all(np.isfinite(weight) for pair in rows for row in pair for _, _, weight in row)
    return monitors, rows, expected, dict(reference_quadrature=audit,
        reference_domain_bounds_m=reference_scene.domain.bounds,
        scope='Fixed physical reference quadrature; actual native samples and shared side-aware interpolation. Curve quadrature convergence is separate from propagation-mesh convergence.')
