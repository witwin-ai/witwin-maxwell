"""Structural admission for all action-owned retained multipole tensors."""
import math
from .cluster import sphere_block_partition, support_spheres
from .direct_capacity import storage_bytes
from .fmm_storage import m2l_storage_bytes, shared_displacement_catalog, signed_axis_displacement_catalog, signed_axis_catalog_rule, radial_series_pair_catalog, radial_series_pair_storage_bytes, radial_series_catalog_rule
from .fmm_shift import shift_storage_bytes
from .packed_near import bounded_original_near_cache_plan


class MultipoleCapacityError(MemoryError):
    def __init__(self, receipt):
        super().__init__('Multipole action retained capacity exceeded before numerical expansion setup')
        self.receipt = receipt


def plan_multipole_storage(trees, *, order, separation_ratio, budget_bytes, table_storage='dense',shift_storage='dense',near_storage='full_matrix',near_coefficient_workspace_budget_bytes=128*1024**2,
                           near_coefficient_cache_budget_bytes=128*1024**2):
    if type(order) is not int or order < 0 or type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError('Multipole storage requires nonnegative order and positive integer budget')
    if near_storage not in ('full_matrix', 'scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1'):
        raise ValueError('Unsupported FMM near storage')
    if (type(near_coefficient_workspace_budget_bytes) is not int
            or near_coefficient_workspace_budget_bytes < 1):
        raise ValueError('Near coefficient workspace budget must be a positive integer')
    if (near_storage not in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
            and near_coefficient_workspace_budget_bytes != 128*1024**2):
        raise ValueError('Explicit near coefficient workspace requires streamed reciprocal storage')
    if (type(near_coefficient_cache_budget_bytes) is not int
            or near_coefficient_cache_budget_bytes < 0):
        raise ValueError('Near coefficient cache budget must be a nonnegative integer')
    if (near_storage != 'scalar_reciprocal_cached_dk_v1'
            and near_coefficient_cache_budget_bytes != 128*1024**2):
        raise ValueError('Explicit near coefficient cache budget requires cached reciprocal storage')
    trees = tuple(trees)
    if not trees:
        raise ValueError('Multipole storage requires global exterior and interior trees')
    if table_storage in ('fft_v1','radial_series_shared_radius_v1'):
        for tree in trees:
            m2l_storage_bytes(order, tree.lower.element_size(), table_storage)
    count = math.comb(order+3, 3)
    categories = dict(trees=storage_bytes(t for tree in trees for t in tree.tensors()),
        centers_radii=0, leaf_maps=0, leaf_moments=0, shifts=0, m2l=0, near=0,
        body_indices=sum(16*tree.size for tree in trees[1:]))
    cache_regions = []
    for tree in trees:
        tree.assert_current()
        real = tree.lower.element_size()
        if near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1', 'scalar_reciprocal_streamed_dk_v1', 'scalar_reciprocal_cached_dk_v1') and real != 8:
            raise ValueError('Scalar D/K FMM capacity requires native float64 trees')
        itemsize = 2*real
        categories['centers_radii'] += 4*tree.node_count*real
        categories['leaf_maps'] += 8*tree.size
        categories['leaf_moments'] += 4*tree.size*count*itemsize
        categories['shifts'] += shift_storage_bytes(order,tree.node_count,real,shift_storage)
        sizes = (tree.stops-tree.starts).cpu().tolist()
        near_rows, near_count = set(), 0
        shared_pairs=[]
        cache_pairs=[]
        for aa, bb, far in sphere_block_partition(tree, separation_ratio=separation_ratio, pair_batch=32):
            for a, b, admissible in zip(aa.tolist(), bb.tolist(), far.tolist()):
                if admissible:
                    if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
                        shared_pairs.append((a,b))
                    else:
                        categories['m2l'] += m2l_storage_bytes(order, real, table_storage)
                else:
                    if near_storage in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1') and a>b:
                        continue
                    near_count += 1
                    near_rows.add(a)
                    if near_storage in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
                        if 3*2*sizes[a]*sizes[b]*16 > near_coefficient_workspace_budget_bytes:
                            raise MemoryError('Original near coefficient workspace exceeded before moment setup')
                        categories['near'] += 16*(sizes[a]+sizes[b])
                        if near_storage == 'scalar_reciprocal_cached_dk_v1':cache_pairs.append((a,b))
                    else:
                        categories['near'] += (2 if near_storage in ('scalar_dk_v1', 'scalar_reciprocal_dk_v1') else 4)*sizes[a]*sizes[b]*itemsize+16*(sizes[a]+sizes[b])
        if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
            if table_storage == 'radial_series_shared_radius_v1':
                categories['m2l']+=radial_series_pair_storage_bytes(support_spheres(tree)[0],shared_pairs,order)
            else:
                representatives,_=(signed_axis_displacement_catalog if table_storage == 'compact_shared_signed_axis_v1'
                    else shared_displacement_catalog)(support_spheres(tree)[0],shared_pairs)
                categories['m2l']+=len(representatives)*m2l_storage_bytes(order,real,table_storage)
        if near_count:
            categories['near'] += 8*(2*len(near_rows)+1+3*near_count)
        if near_storage == 'scalar_reciprocal_cached_dk_v1':
            # Match the original reciprocal factory's off-diagonal-first
            # canonical ordering and shape-group insertion exactly. The shared
            # host primitive then selects diagonal-first native prefixes.
            grouped = {}
            for a,b in sorted(cache_pairs,key=lambda pair:(pair[0]==pair[1],*pair)):
                shape = sizes[a],sizes[b]
                counts = grouped.setdefault(shape,[0,0])
                counts[0] += 1
                counts[1] += int(a!=b)
            shapes = tuple((count,*shape) for shape,(count,_) in grouped.items())
            reverse = tuple(mirrored for _,mirrored in grouped.values())
            segments,native = (bounded_original_near_cache_plan(shapes,reverse,
                near_coefficient_cache_budget_bytes) if shapes else ((),0))
            categories['near'] += native
            cache_regions.append(dict(retained_kernel_bytes=native,
                cache_planned_blocks=sum(stop-first for _,first,stop,_ in segments),
                cache_segments=[list(segment) for segment in segments]))
        tree.assert_current()
    total = sum(categories.values())
    record={'version': 'multipole_action_retained_capacity_v1', 'table_storage': table_storage, 'categories': categories,
            'planned_bytes': total, 'budget_bytes': budget_bytes, 'admitted': total <= budget_bytes,
            'scope': 'all_action_owned_retained_tensors', 'peak_memory_qualified': False,
            'excludes': ['provider geometry and input system', 'Riesz metric and solver state',
                        'setup/action/backward workspace', 'allocator cache/workspace and host metadata']}
    if shift_storage!='dense':record.update(version='multipole_action_retained_capacity_separable_v1',shift_storage=shift_storage)
    if near_storage != 'full_matrix':record.update(version='multipole_action_retained_capacity_scalar_dk_v1',near_storage=near_storage)
    if near_storage == 'scalar_reciprocal_dk_v1':
        record.update(version='multipole_action_retained_capacity_reciprocal_scalar_dk_v1',
            reciprocal_near_rule='one_off_diagonal_direction_full_diagonal')
    if table_storage in ('compact_shared_displacement_v1','compact_shared_signed_axis_v1','radial_series_shared_radius_v1'):
        record.update(version='multipole_action_retained_capacity_shared_compact_v1',
            coefficient_sharing_rule='exact_native_displacement_equal_or_opposite_within_medium_order')
    if near_storage == 'scalar_reciprocal_streamed_dk_v1':
        record.update(version='multipole_action_retained_capacity_streamed_reciprocal_dk_v1',
            reciprocal_near_rule='one_off_diagonal_direction_full_diagonal',
            retained_near_kernel_bytes=0,near_construction='recompute_original_per_action',
            near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
            near_coefficient_workspace_scope='three bounded original D/K arrays; quadrature and EM workspaces excluded')
    if table_storage == 'compact_shared_signed_axis_v1':
        record.update(version='multipole_action_retained_capacity_signed_axis_compact_v1',
            coefficient_sharing_rule=signed_axis_catalog_rule,
            physical_pairs_retained=True,production_qualification=False)
    if near_storage == 'scalar_reciprocal_cached_dk_v1':
        record.update(version='multipole_action_retained_capacity_cached_reciprocal_dk_v1',
            reciprocal_near_rule='one_off_diagonal_direction_full_diagonal',
            retained_near_kernel_bytes=sum(region['retained_kernel_bytes'] for region in cache_regions),
            near_construction='bounded_validated_original_first_use_cache',
            near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
            near_coefficient_workspace_scope='three bounded original D/K arrays; quadrature and EM workspaces excluded',
            near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes,
            coefficient_cache_scope='per_regional_owner_native_payload; aggregate included in retained admission',
            regional_coefficient_cache=cache_regions,production_qualification=False)
    if table_storage == 'radial_series_shared_radius_v1':
        record.update(version='multipole_action_retained_capacity_radial_series_v1',
            coefficient_sharing_rule=radial_series_catalog_rule,physical_pairs_retained=True,
            per_physical_displacement_bytes=24,full_cartesian_jets_retained=False,production_qualification=False)
    return record


def plan_directional_reciprocal_near_storage(tree, near_nodes, *, budget_bytes,
        near_storage='scalar_reciprocal_dk_v1',near_coefficient_workspace_budget_bytes=128*1024**2,
                     near_coefficient_cache_budget_bytes=128*1024**2):
    """Count canonical scalar near storage before maps or original integration.

    The physical partition must be closed under test/source exchange. This
    preflight uses integer topology; the packed factory separately validates
    complete CUDA J/M maps before reducing the full logical layout.
    """
    if type(budget_bytes) is not int or budget_bytes<1:
        raise ValueError('Directional reciprocal near budget must be a positive integer')
    if near_storage not in ('scalar_reciprocal_dk_v1','scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1'):
        raise ValueError('Directional reciprocal capacity requires its explicit near policy')
    if (type(near_coefficient_workspace_budget_bytes) is not int
            or near_coefficient_workspace_budget_bytes<1):
        raise ValueError('Directional near coefficient workspace must be a positive integer')
    streamed=near_storage in ('scalar_reciprocal_streamed_dk_v1','scalar_reciprocal_cached_dk_v1')
    if not streamed and near_coefficient_workspace_budget_bytes!=128*1024**2:
        raise ValueError('Explicit directional coefficient workspace requires streamed reciprocal storage')

    if (type(near_coefficient_cache_budget_bytes) is not int
            or near_coefficient_cache_budget_bytes < 0):
        raise ValueError('Directional near coefficient cache budget must be a nonnegative integer')
    if (near_storage != 'scalar_reciprocal_cached_dk_v1'
            and near_coefficient_cache_budget_bytes != 128*1024**2):
        raise ValueError('Explicit directional coefficient cache budget requires cached reciprocal storage')
    tree.assert_current()
    if tree.lower.element_size()!=8:
        raise ValueError('Directional reciprocal near requires native float64 geometry')
    if not isinstance(near_nodes,(tuple,list)):
        raise ValueError('Directional reciprocal near requires explicit node pairs')
    pairs=[]
    children=tree.children.cpu().tolist()
    sizes=(tree.stops-tree.starts).cpu().tolist()
    for pair in near_nodes:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:
            raise ValueError('Directional reciprocal near requires explicit node pairs')
        a,b=pair
        if any(type(n) is not int or not 0<=n<tree.node_count for n in (a,b)):
            raise ValueError('Directional reciprocal near node is outside its physical tree')
        if any(any(c>=0 for c in children[n]) for n in (a,b)):
            raise ValueError('Directional reciprocal near requires terminal physical leaves')
        pairs.append((a,b))
    unique=set(pairs)
    if len(unique)!=len(pairs) or any((b,a) not in unique for a,b in pairs):
        raise ValueError('Directional reciprocal near requires unique complete reverse pairs')
    canonical=[(a,b) for a,b in pairs if a<=b]
    if streamed and any(3*2*sizes[a]*sizes[b]*16>near_coefficient_workspace_budget_bytes for a,b in canonical):
        raise MemoryError('Directional original near coefficient workspace exceeded before maps or integration')
    retained=(8*(2*len({a for a,_ in canonical})+1+3*len(canonical))
        +sum((0 if streamed else 32*sizes[a]*sizes[b])+16*(sizes[a]+sizes[b]) for a,b in canonical)) if canonical else 0
    segments, native = (), 0
    if near_storage == 'scalar_reciprocal_cached_dk_v1':
        # Match original canonical group ordering before selecting native prefixes.
        grouped = {}
        for a,b in sorted(canonical, key=lambda pair: (pair[0]==pair[1], *pair)):
            counts = grouped.setdefault((sizes[a],sizes[b]), [0,0])
            counts[0] += 1
            counts[1] += int(a!=b)
        shapes = tuple((count,*shape) for shape,(count,_) in grouped.items())
        reverse = tuple(mirrored for _,mirrored in grouped.values())
        segments,native = (bounded_original_near_cache_plan(shapes,reverse,
            near_coefficient_cache_budget_bytes) if shapes else ((),0))
        retained += native
    tree.assert_current()
    if retained>budget_bytes:
        raise MemoryError('Directional reciprocal near retained budget exceeded before maps or original integration')
    record=dict(version='directional_reciprocal_scalar_near_capacity_v1',
        logical_blocks=len(pairs),stored_blocks=len(canonical),retained_bytes=retained,
        budget_bytes=budget_bytes,peak_memory_qualified=False,production_qualification=False)
    if streamed:
        record.update(version='directional_streamed_reciprocal_scalar_near_capacity_v1',
            near_storage=near_storage,retained_near_kernel_bytes=0,
            near_coefficient_workspace_budget_bytes=near_coefficient_workspace_budget_bytes,
            coefficient_workspace_scope='three bounded D/K arrays; quadrature and EM workspaces excluded')
    if near_storage == 'scalar_reciprocal_cached_dk_v1':
        record.update(version='directional_cached_reciprocal_scalar_near_capacity_v1',
            retained_near_kernel_bytes=native,
            near_coefficient_cache_budget_bytes=near_coefficient_cache_budget_bytes,
            cache_planned_blocks=sum(stop-first for _,first,stop,_ in segments),
            cache_segments=[list(segment) for segment in segments],
            cache_scope='per_regional_owner_native_payload; host/events/quadrature excluded')
    return record


def multipole_action_tensors(actions):
    result = list(actions.indices)
    for region in (actions.exterior, *actions.interiors):
        result.extend(region.test.tree.tensors())
        result.extend(region.test.tensors())
        for _, _, coupling in region.far:
            result.extend(coupling.tensors())
        if region.near is not None:
            result.extend(region.near.tensors())
    return tuple(result)


def directional_retained_tensor_groups(actions):
    """Enumerate native directional action dependencies without copying them.

    Component owners deliberately exclude caller storage. This inventory also
    includes those dependencies, automatic plans and shared physical inputs.
    It covers retained action state, not setup peaks or solver/backward state.
    """
    from .fmm_system import DirectionalSystemActions
    from .fmm_directional_moments import (NestedDirectionalRWGProjection,
        SparseDirectionalRWGProjection, DirectionalRWGDAGProjection)
    from .nearfield import ScalarSystemBlocks
    if not isinstance(actions, DirectionalSystemActions):
        raise TypeError('Directional retained inventory requires a complete directional system')
    actions.assert_current()
    groups = dict(shared_inputs=list(actions.provider.input_tensors()),
        prepared_geometry=list(actions.provider.geometry_tensors()), trees=[],
        automatic_region_plans=[], automatic_dag_plans=[], projections=[],
        projection_dependencies=[], m2l=[], original_near=[], embeddings=list(actions.tensors()))
    projections = {}
    for region in (actions.exterior, *actions.interiors):
        groups['original_near'].extend(region.tensors())
        for owner in (region.test, region.source, *region.far.targets, *region.far.origins):
            projections[id(owner)] = owner
        for coupling in region.far.couplings:
            groups['m2l'].extend(coupling.tensors())
    for owner in projections.values():
        groups['trees'].extend(owner.tree.tensors())
        groups['projections'].extend(owner.tensors())
        if isinstance(owner, DirectionalRWGDAGProjection) and owner.plan_guard is not None:
            plan = owner.plan_guard
            groups['automatic_region_plans'].extend(plan.region_plan.tensors())
            groups['automatic_dag_plans'].extend((plan.carriers, plan.path_products, plan._statistics))
        if isinstance(owner, (SparseDirectionalRWGProjection, DirectionalRWGDAGProjection)):
            groups['projection_dependencies'].append(owner._caller_carriers)
        elif isinstance(owner, NestedDirectionalRWGProjection) and owner.layout is not None:
            groups['projection_dependencies'].extend(owner.layout.tensors())
        if owner.geometry is not None and not isinstance(owner.geometry, ScalarSystemBlocks):
            groups['prepared_geometry'].extend(owner.geometry.tensors())
        for records in owner.leaves.values():
            for _, leaf in records:
                groups['prepared_geometry'].extend(leaf.geometry.tensors())
    actions.assert_current()
    return {name: tuple(values) for name, values in groups.items()}


def directional_retained_inventory(actions):
    """Count each actual native storage once across the complete action graph.

    Category attribution follows the declared group order; standalone category
    inventories can overlap. Full backing storage is counted for tensor views.
    No workspace estimate is added to retained bytes or treated as a peak.
    """
    groups = directional_retained_tensor_groups(actions)
    seen = set()
    categories = {}
    for name, values in groups.items():
        unique = {}
        for value in values:
            storage = value.untyped_storage()
            unique[value.device, storage.data_ptr()] = storage.nbytes()
        exclusive = {key: size for key, size in unique.items() if key not in seen}
        categories[name] = dict(standalone_storage_bytes=sum(unique.values()),
            attributed_storage_bytes=sum(exclusive.values()),
            standalone_storages=len(unique), attributed_storages=len(exclusive))
        seen.update(unique)
    total = sum(category['attributed_storage_bytes'] for category in categories.values())
    return dict(version='directional_retained_native_dependencies_v1',
        scope='complete_directional_action_retained_native_dependencies_including_shared_inputs',
        categories=categories, unique_storages=len(seen), retained_native_bytes=total,
        dense_complex128_matrix_bytes=16*math.prod(actions.shape),
        storage_identity='device_and_backing_storage_pointer', full_view_backing_storage_counted=True,
        automatic_planner_borrowed_storage_included=True,
        excludes=('setup/action/backward workspace and caller currents', 'Riesz metric and solver state',
            'allocator cache and backend library workspace', 'host objects and integer topology'),
        peak_memory_qualified=False, complexity_qualification=False,
        crossover_qualified=False, production_qualification=False)
