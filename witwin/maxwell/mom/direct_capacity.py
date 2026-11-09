"""Preallocation retained-action accounting for direct scalar interpolation.

This accounts for action-owned tensors. It does not admit setup workspaces,
provider geometry, Riesz/solver/backward state or allocator/host overhead.
"""
import math
from .cluster import support_block_partition


class DirectCapacityError(MemoryError):
    def __init__(self, receipt):
        super().__init__('Direct H2 action retained budget exceeded before interpolation allocation')
        self.receipt = receipt


def storage_bytes(tensors):
    storages = {}
    for value in tensors:
        storage = value.untyped_storage()
        storages[(value.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def plan_direct_storage(trees, *, order, admissibility, budget_bytes):
    """Count the global exterior and disjoint body-interior action layout."""
    if type(order) is not int or order < 1 or type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError('Direct action order and retained budget must be positive integers')
    trees = tuple(trees)
    if not trees:
        raise ValueError('Direct storage planning requires regional trees')
    counts = dict(trees=storage_bytes(t for tree in trees for t in tree.tensors()),
                  boxes=0, leaf_moments=0, transfers=0, kernels=0, near=0,
                  body_indices=sum(16*tree.size for tree in trees[1:]))
    for tree in trees:
        tree.assert_current()
        real = tree.lower.element_size()
        complex_bytes = 2*real
        active = (((tree.upper-tree.lower)*.5) > 0).cpu().tolist()
        axes = [[order if enabled else 1 for enabled in row] for row in active]
        nodes = [math.prod(row) for row in axes]
        sizes = (tree.stops-tree.starts).cpu().tolist()
        children = tree.children.cpu().tolist()
        for i, row in enumerate(children):
            counts['boxes'] += real*(12+3*nodes[i]+2*sum(axes[i]))
            live = [child for child in row if child >= 0]
            if live:
                counts['transfers'] += sum(complex_bytes*nodes[i]*nodes[c] for c in live)
            else:
                counts['leaf_moments'] += 8*sizes[i]+4*complex_bytes*sizes[i]*nodes[i]
        near_rows, near_count = set(), 0
        for aa, bb, far in support_block_partition(tree, tree, admissibility=admissibility, pair_batch=32):
            for a, b, is_far in zip(aa.cpu().tolist(), bb.cpu().tolist(), far.cpu().tolist()):
                if is_far:
                    counts['kernels'] += 4*complex_bytes*nodes[a]*nodes[b]
                else:
                    near_count += 1
                    near_rows.add(a)
                    counts['near'] += 4*complex_bytes*sizes[a]*sizes[b]+16*(sizes[a]+sizes[b])
        if near_count:
            counts['near'] += 8*(2*len(near_rows)+1+3*near_count)
    total = sum(counts.values())
    return {'version': 'direct_action_retained_plan_v1', 'categories': counts,
            'planned_bytes': total, 'budget_bytes': budget_bytes, 'admitted': total <= budget_bytes,
            'scope': 'action_owned_retained_tensors', 'peak_memory_qualified': False,
            'excludes': ['provider geometry and shared inputs', 'Riesz metric and solver state',
                         'setup quadrature and repacking workspace', 'action and backward workspace',
                         'allocator cache/workspace and host metadata']}


def direct_action_tensors(actions):
    """Enumerate owned arrays, excluding shared provider geometry and mesh data."""
    result = list(actions.indices)
    for region in (actions.exterior, *actions.interiors):
        projection = region.test
        result.extend(projection.tree.tensors())
        for box in projection.boxes:
            result.extend(box.tensors())
        result.extend(projection.transfers.values())
        for leaf in projection.leaves.values():
            result.extend(leaf.tensors())
        for _, _, kernel in region.far:
            result.extend((kernel.green, kernel.gradient))
        if region.near is not None:
            result.extend(region.near.tensors())
    return tuple(result)
