"""Preallocation storage plan for periodic FMM, including translated geometry."""
import math
from dataclasses import replace
import torch
from .cluster import sphere_block_partition
from .direct_capacity import storage_bytes
from .fmm_storage import m2l_storage_bytes
from .fmm_periodic_error import periodic_image_partition
from .fmm_shift import shift_storage_bytes


def _provider_cache(provider):
    return (provider.edge_stops, *(t for geometry in provider.geometry for t in geometry.tensors()))


def _translated_geometry(provider):
    tensors = [provider.system.face_edges, *_provider_cache(provider)]
    for body in provider.system.bodies:
        basis = body.basis
        tensors.extend((body.mesh.vertices, basis.edges, basis.face_edges, basis.face_signs,
            basis.plus_faces, basis.minus_faces, basis.plus_vertices, basis.minus_vertices, basis.edge_lengths))
    return tensors


def plan_periodic_storage(provider, trees, lattice, *, order, shells, separation_ratio,
                          budget_bytes, table_storage='dense', ewald=False,shift_storage='dense'):
    """Plan retained arrays before moments, image geometry, near blocks or M2L."""
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError('Periodic action retained budget must be a positive integer')
    if table_storage == 'fft_v1':
        raise NotImplementedError('Periodic FFT capacity planning requires separately admitted screened coefficients')
    if table_storage == 'fft_ewald_v1':
        if not ewald:
            raise NotImplementedError('Ewald FFT capacity requires actual screened Ewald coefficients')
        from .ewald_nearfield import EwaldSystemBlocks
        if not isinstance(provider, EwaldSystemBlocks) or provider.kernel.lattice is not lattice:
            raise ValueError('Ewald FFT capacity requires matching original Ewald provider and lattice')
        if shells != provider.kernel.real_shells:
            raise ValueError('Ewald FFT capacity must use the actual configured Ewald real cutoff')
        m2l_storage_bytes(order, provider.system.dtype.itemsize, 'fft_v1')
    provider.assert_current()
    lattice.assert_current()
    trees = tuple(trees)
    if any(tree.expansion_centers is not None for tree in trees):
        raise NotImplementedError('Dyadic expansion centers currently require free space')
    root, *interiors = trees
    real = root.lower.element_size()
    pair_bytes = m2l_storage_bytes(order, real, 'fft_v1' if table_storage == 'fft_ewald_v1' else table_storage)
    count = math.comb(order+3, 3)
    categories = dict(trees=storage_bytes(t for tree in trees for t in tree.tensors()),
        projections=0, m2l=m2l_storage_bytes(order, real, 'compact') if ewald and table_storage != 'fft_ewald_v1' else pair_bytes, near=0, body_indices=sum(16*t.size for t in interiors),
        image_metadata=0 if ewald else 3*real, provider_cache=storage_bytes(_provider_cache(provider)), translated_geometry=0)

    def projection(tree):
        return (4*tree.node_count*real+8*tree.size+4*tree.size*count*2*real+
                shift_storage_bytes(order,tree.node_count,real,shift_storage))

    def regional(test, source):
        rows, blocks = set(), 0
        ts = (test.stops-test.starts).cpu().tolist()
        ss = (source.stops-source.starts).cpu().tolist()
        for aa, bb, flags in sphere_block_partition(test, source, separation_ratio=separation_ratio, pair_batch=32):
            for a, b, far in zip(aa.tolist(), bb.tolist(), flags.tolist()):
                if far:
                    categories['m2l'] += pair_bytes
                else:
                    rows.add(a)
                    blocks += 1
                    categories['near'] += 4*ts[a]*ss[b]*2*real+16*(ts[a]+ss[b])
        if blocks:
            categories['near'] += 8*(2*len(rows)+1+3*blocks)

    for tree in trees:
        tree.assert_current()
        categories['projections'] += projection(tree)
    for tree in interiors:
        regional(tree, tree)
    centers = root.lower*.5+root.upper*.5
    radius = torch.linalg.vector_norm((root.upper-root.lower)*.5, dim=-1)[0]
    image_count, translated_count = 0, 0
    for indices, _, far in periodic_image_partition(lattice, centers[0]-centers[0],
            2*float(radius), shells, separation_ratio, 32):
        for index in indices[~far]:
            image_count += 1
            source = root
            if bool((index != 0).any()):
                translation = lattice.translations(index)
                source = replace(root, lower=root.lower+translation, upper=root.upper+translation)
                categories['trees'] += 6*root.node_count*real
                categories['projections'] += projection(source)
                translated_count += 1
            regional(root, source)
    categories['image_metadata'] += image_count*(lattice.dimension*8+2*real)
    faces, edges = provider.system.total_faces, provider.system.total_edges
    vertices = sum(len(b.mesh.vertices) for b in provider.system.bodies)
    # Faces are shared with the input mesh; all other translated basis/system
    # arrays and prepared triangle geometry are independently retained.
    per_image_geometry = (3*vertices+edges+16*faces)*real+(6*edges+9*faces+len(interiors))*8
    categories['translated_geometry'] = translated_count*per_image_geometry
    provider.assert_current()
    lattice.assert_current()
    total = sum(categories.values())
    record=dict(version='periodic_multipole_retained_capacity_v1', categories=categories,
        table_storage=table_storage, root_kernel='ewald' if ewald else 'finite_images', planned_bytes=total, budget_bytes=budget_bytes,
        admitted=total <= budget_bytes, near_images=image_count, translated_images=translated_count,
        scope='retained_action_arrays_and_provider_caches_including_translated_geometry',
        peak_memory_qualified=False, excludes=['input meshes and system maps', 'shared lattice arrays',
            'Riesz metric and solver state', 'setup/action/backward workspace', 'allocator and host metadata'])
    if shift_storage!='dense':record.update(version='periodic_multipole_retained_capacity_separable_v1',shift_storage=shift_storage)
    return record


def periodic_action_tensors(actions):
    """Enumerate the same retained scope; storage identity deduplicates aliases."""
    exterior = actions.exterior
    remote = exterior.remote
    from .fmm_ewald_fft import EwaldFFTTerminalM2L
    root_tensors = remote.tensors() if isinstance(remote, EwaldFFTTerminalM2L) else (remote.near_indices, *remote.tables())
    result = [*actions.indices, *_provider_cache(actions.provider), exterior.phases, *root_tensors]
    if hasattr(remote, "displacement"):
        result.append(remote.displacement)
    projections = [exterior.test, *actions.projections]
    for region in (*exterior.regions, *actions.interiors):
        projections.extend((region.test, region.source))
        for _, _, coupling in region.far:
            result.extend(coupling.tensors())
        if region.near is not None:
            result.extend(region.near.tensors())
    for projection in projections:
        result.extend(projection.tree.tensors())
        result.extend(projection.tensors())
        if projection.basis is not actions.provider and hasattr(projection.basis, 'system'):
            result.extend(_translated_geometry(projection.basis))
    return tuple(result)
