"""Exact scalar bulk edges, window reductions and polarized reconstruction."""
from itertools import product

import torch

from .build import build_extension
from .geometry import native_geometry
from .primitives import sampling_geometry


def reconstruct(scene, arithmetic, inverse, coords=None, axis=None):
    from ..subpixel import _field_gradients

    # Match the node reconstruction's periodic seam and interior stencils.
    field = arithmetic['x'] if axis is None else arithmetic
    gradients = list(_field_gradients(scene, field, coords=coords, periodic_nodes=axis is None).values())
    extension = build_extension()
    if axis is None:
        return {a: extension.reconstruct(arithmetic[a], inverse[a], gradients, i, True)
                for i, a in enumerate('xyz')}
    return extension.reconstruct(arithmetic, inverse, gradients, 'xyz'.index(axis), False)


def kottke_component(scene, component, bases, samples, background, structures, geometry, shifts, *, smooth):
    from .. import materials as m

    electric = component[0] == "E"
    axis = component[1].lower()
    values = [m._static_structure_material(s)[1 if electric else 3][axis]*background for s in structures]
    shape = tuple(len(bases[a]) for a in "xyz")
    # Retain the all-background shortcut before any distance/window work.
    if all(value == background for value in values):
        return torch.full(shape, background, device=scene.device, dtype=bases["x"].dtype)
    coords = m._edge_component_coords(bases, (0., 0., 0.))
    points = torch.stack(coords, -1).reshape(-1, 3)
    widths = []
    stagger = m._EDGE_STAGGER_AXES[component]
    for i, a in enumerate("xyz"):
        if not (scene.grid.is_custom or scene.grid.is_auto):
            width = points.new_full((len(bases[a]),), m._uniform_axis_spacing(scene, a))
        else:
            if i in stagger:
                width = getattr(scene, a).diff().to(points.dtype)
            else:
                width = torch.as_tensor(getattr(scene, f"d{a}_dual64"), device=scene.device, dtype=points.dtype)
            # Global-axis widths: a partial selection is a contiguous subset of the global Yee axis.
            base = m._edge_axis_base_1d(scene, a, i in stagger)
            position = torch.searchsorted(base, bases[a])
            if not bool((position < len(base)).all() and torch.equal(base[position.clamp(max=len(base) - 1)], bases[a])):
                raise ValueError(f"Edge-sample coordinates along {a} are not a subset of the global Yee axis.")
            width = width[position]
        view = [1, 1, 1]
        view[i] = -1
        widths.append(width.reshape(view).expand(shape).reshape(-1))
    widths = torch.stack(widths, -1)
    band = .5*torch.linalg.vector_norm(widths, dim=-1)+1e-6
    extension = build_extension()

    def image_shifts(geometry):
        options = shifts[id(geometry)]
        if options is None:
            options = tuple((-float(hi-lo), 0., float(hi-lo))
                            if scene.boundary.axis_kind(a) in ("periodic", "bloch") else (0.,)
                            for a, (lo, hi) in zip("xyz", scene.domain.bounds))
        return options

    def distance(j, query):
        result = query.new_full((len(query),), float("inf"))
        for shift in product(*image_shifts(structures[j].geometry)):
            shifted = query-query.new_tensor(shift)
            if native_geometry(structures[j].geometry):
                sdf = extension.distance(geometry[j:j+1], shifted)[0]
            else:
                sdf = sampling_geometry(structures[j].geometry).signed_distance(
                    *(shifted[:, i:i+1, None] for i in range(3))).reshape(-1)
            result = torch.minimum(result, sdf)
        return result

    distances = torch.stack([distance(j, points) for j in range(len(structures))])
    value_tensor = torch.tensor(values, device=scene.device, dtype=points.dtype)
    material, near = extension.bulk(distances, value_tensor, band, background, smooth)
    indices = near.nonzero()[:, 0]
    # Keep the batch-4 empty-band rule and ascending 4096-cell chunk ownership.
    if indices.numel() == 0:
        return material.reshape(shape)
    lattice = [(torch.arange(n, device=scene.device, dtype=points.dtype)+.5)/n-.5 for n in samples]
    offsets = torch.stack(torch.meshgrid(*lattice, indexing="ij"), -1).reshape(-1, 3)
    for selected in indices.split(4096):
        local_offsets = offsets[None]*widths[selected, None]
        query = (points[selected, None]+local_offsets).reshape(-1, 3)
        sample = query.new_full((len(query),), background)
        for j, structure in enumerate(structures):
            if smooth:
                spacing = tuple(float(widths[selected, i].min())/samples[i] for i in range(3))
                fill = query.new_zeros(len(query))
                for shift in product(*image_shifts(structure.geometry)):
                    shifted = query-query.new_tensor(shift)
                    # Keep the coverage primitive and its local minimum spacing
                    # in interface windows; O3b owns their packed quadrature.
                    fill = torch.maximum(fill, m._subsample_fill_fraction(
                        sampling_geometry(structure.geometry), *(shifted[:, i:i+1, None] for i in range(3)), spacing).reshape(-1))
                sample = fill*value_tensor[j]+(1-fill)*sample
            else:
                sdf = distances[j]
                inside = (sdf[selected]<0)[:, None].expand(-1, len(offsets)).clone()
                straddle = sdf[selected].abs()<band[selected]
                if bool(straddle.any()):
                    q = query.reshape(len(selected), -1, 3)[straddle].reshape(-1, 3)
                    inside[straddle] = (distance(j, q)<0).reshape(-1, len(offsets))
                sample = torch.where(inside.reshape(-1), value_tensor[j], sample)
        material[selected] = extension.window(sample.reshape(len(selected), -1), local_offsets, "xyz".index(axis))
    return material.reshape(shape)
