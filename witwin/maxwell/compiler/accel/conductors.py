"""Conductor distance and edge-fill pieces; occupancy keeps torch tanh."""
from itertools import product

import torch

from .build import build_extension
from .dispatch import UNCHANGED, _fp32_geometry
from .geometry import MAX_TREE_NODES, qualified_geometry, tree_size
from .primitives import sampling_geometry


def _qualified(structures):
    # Keep the existing fp32 scope until conductor occupancy/edge parity is gated.
    return all(qualified_geometry(s.geometry) and _fp32_geometry(s.geometry)
               and tree_size(s.geometry) <= MAX_TREE_NODES for s in structures)


def try_occupancy(scene, *, magnetic=False):
    from .. import materials as m

    structures = m.pmc_structures(scene) if magnetic else m.volume_pec_structures(scene)
    if not _qualified(structures):
        return UNCHANGED
    occupancy = None
    beta = m._pec_geometry_beta(scene)
    for structure in structures:
        sample = m._geometry_occupancy(scene, sampling_geometry(structure.geometry),
                                       beta=beta, half_weight_boundary=True)
        occupancy = sample if occupancy is None else torch.maximum(occupancy, sample)
    return occupancy.detach() if magnetic and occupancy is not None else occupancy


def try_pec_edge_fill(scene):
    from .. import materials as m

    structures = m.volume_pec_structures(scene)
    if not _qualified(structures):
        return UNCHANGED
    m.validate_legacy_thin_pec(scene)
    union = None
    tolerance = 1.e-5 * min(scene.grid.min_spacing)
    shifts = m._pec_periodic_shift_options(scene)
    for structure in structures:
        geometry = sampling_geometry(structure.geometry)
        for sx, sy, sz in product(*shifts):
            sample = geometry.signed_distance(scene.X-sx, scene.Y-sy, scene.Z-sz)
            sample = torch.where(sample.abs() <= tolerance, sample-sample.detach(), sample)
            union = sample if union is None else torch.minimum(union, sample)
    layout = m.compile_planar_pec(scene)
    if union is None:
        return None if layout is None else layout["edges"]
    fills = dict(zip(("Ex", "Ey", "Ez"), build_extension().conductor_edges(union.contiguous())))
    if layout is not None:
        fills = {name: torch.maximum(fill, layout["edges"][name]) for name, fill in fills.items()}
    return fills
