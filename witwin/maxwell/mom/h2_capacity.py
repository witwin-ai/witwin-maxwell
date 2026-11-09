"""Structural retained-storage admission before H2 conversion allocations.

This bound covers owned H2 tensors for any accepted nested ranks. It excludes
conversion/SVD workspace and shared physical inputs; it is not an OOM guarantee.
"""
import torch
from .lowrank import LowRankBlock


class H2CapacityError(MemoryError):
    def __init__(self, receipt):
        super().__init__('H2 owned retained-storage bound exceeds its budget before conversion')
        self.receipt = receipt


def plan_h2_storage(tree, records, *, components, dtype, max_rank, budget_bytes):
    """Bound leaf/transfer/coupling storage from topology and input factor ranks.

    Numerical ranks may be smaller. Conservative rejection is intentional;
    the receipt is a structural upper bound, not a prediction of the final rank.
    No seeds, bases, couplings or balanced scales are allocated here.
    """
    tree.assert_current()
    if (type(components) is not int or components < 1 or type(max_rank) is not int or max_rank < 1
            or type(budget_bytes) is not int or budget_bytes < 1
            or dtype not in (torch.complex64, torch.complex128)):
        raise ValueError('H2 storage planning requires positive integer limits and complex precision')
    starts, stops = tree.starts.cpu().tolist(), tree.stops.cpu().tolist()
    children = tuple(tuple(c for c in row if c >= 0) for row in tree.children.cpu().tolist())
    widths = [components*(stop-start) for start, stop in zip(starts, stops)]
    seeds = [[0]*tree.node_count for _ in range(2)]
    pairs = []
    seen = set()
    for a, b, block in records:
        if (type(a) is not int or type(b) is not int or not 0 <= a < tree.node_count
                or not 0 <= b < tree.node_count or (a, b) in seen):
            raise ValueError('H2 storage planning requires unique valid node pairs')
        if not isinstance(block, LowRankBlock):
            raise TypeError('H2 storage planning requires fixed low-rank blocks')
        block.assert_current()
        left, right = block.left, block.right
        if (left.ndim != 2 or right.ndim != 2 or left.shape[0] != widths[a]
                or right.shape[1] != widths[b] or left.shape[1] != right.shape[0]
                or left.device != tree.device or right.device != tree.device
                or left.dtype != dtype or right.dtype != dtype or left.requires_grad or right.requires_grad):
            raise ValueError('H2 planning factors must match fixed node spaces and precision')
        seen.add((a, b))
        pairs.append((a, b))
        seeds[0][a] += left.shape[1]
        seeds[1][b] += left.shape[1]
    complex_bytes = 16 if dtype == torch.complex128 else 8
    basis_plans = []
    for local_seeds in seeds:
        columns = local_seeds.copy()
        for node in range(tree.node_count):
            for child in children[node]:
                columns[child] += columns[node]
        ranks = [0]*tree.node_count
        for node in reversed(range(tree.node_count)):
            ambient = sum(ranks[c] for c in children[node]) if children[node] else widths[node]
            ranks[node] = min(max_rank, columns[node], ambient)
        leaf_values = sum(widths[node]*ranks[node]*complex_bytes for node in range(tree.node_count) if not children[node])
        leaf_indices = sum(widths[node]*8 for node in range(tree.node_count) if not children[node])
        transfer_values = sum(ranks[node]*ranks[child]*complex_bytes for node in range(tree.node_count) for child in children[node])
        basis_plans.append({'rank_bounds': ranks, 'leaf_values_bytes': leaf_values,
                            'leaf_indices_bytes': leaf_indices, 'transfer_values_bytes': transfer_values,
                            'retained_bytes_bound': leaf_values+leaf_indices+transfer_values})
    coupling_bytes = sum(basis_plans[0]['rank_bounds'][a]*basis_plans[1]['rank_bounds'][b]*complex_bytes for a, b in pairs)
    scale_bytes = 2*widths[0]*(complex_bytes//2)
    bound = sum(plan['retained_bytes_bound'] for plan in basis_plans)+coupling_bytes+scale_bytes
    return {'version': 'h2_owned_retained_bound_v1', 'scope': 'h2_owned_retained_tensors',
            'test_basis': basis_plans[0], 'trial_basis': basis_plans[1],
            'coupling_bytes_bound': coupling_bytes, 'scale_bytes': scale_bytes,
            'retained_bytes_bound': bound, 'budget_bytes': budget_bytes, 'admitted': bound <= budget_bytes,
            'conservative_rank_bound': True, 'peak_memory_qualified': False,
            'excludes': ['shared tree/near/metric/geometry/input tensors', 'source H factors during conversion',
                         'seed and ancestor workspaces', 'CUDA SVD/allocator workspace', 'solver/backward state',
                         'host control/metadata objects']}
