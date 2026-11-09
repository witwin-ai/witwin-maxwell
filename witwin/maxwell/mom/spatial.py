"""CUDA bounding-volume pruning for geometric validation.

``BoxTree`` sorts axis-aligned boxes along a Morton curve and groups them
eight at a time into a fixed-branching hierarchy. Traversals descend all node
pairs level by level and discard a pair only when its node boxes prove that no
contained item pair can satisfy the caller's predicate, so candidate sets are
exact supersets. Callers re-apply their original exact predicates to every
returned item pair; pruning never changes an accept/reject decision.
"""
from __future__ import annotations

import math

import torch

_BRANCH = 8
_PAIR_CHUNK = 1 << 22


def _spread_bits(value):
    value = value & 0x1FFFFF
    value = (value | value << 32) & 0x1F00000000FFFF
    value = (value | value << 16) & 0x1F0000FF0000FF
    value = (value | value << 8) & 0x100F00F00F00F00F
    value = (value | value << 4) & 0x10C30C30C30C30C3
    value = (value | value << 2) & 0x1249249249249249
    return value


def _morton_order(centres):
    low = centres.amin(0)
    span = (centres.amax(0)-low).clamp_min(torch.finfo(centres.dtype).tiny)
    grid = ((centres-low)/span*(2**21-1)).round().to(torch.int64)
    code = _spread_bits(grid[:, 0]) | _spread_bits(grid[:, 1]) << 1 | _spread_bits(grid[:, 2]) << 2
    return torch.argsort(code, stable=True)


def _group(count, device):
    """Indices 0..count-1 padded with -1 to rows of eight."""
    rows = math.ceil(count/_BRANCH)
    index = torch.full((rows*_BRANCH,), -1, device=device, dtype=torch.int64)
    index[:count] = torch.arange(count, device=device)
    return index.reshape(rows, _BRANCH)


def _reduce(values, members, fill, op):
    gathered = values[members.clamp_min(0)]
    gathered = torch.where((members >= 0)[..., None], gathered, torch.full_like(gathered, fill))
    return op(gathered, dim=1).values


class BoxTree:
    """Fixed-branching hierarchy over boxes [N,3] with one representative point each."""

    def __init__(self, lower, upper, representatives=None):
        if (lower.ndim != 2 or lower.shape[1] != 3 or upper.shape != lower.shape or not len(lower)
                or lower.device.type != 'cuda'):
            raise ValueError('BoxTree requires nonempty matching CUDA [N,3] boxes')
        device = lower.device
        self.size = len(lower)
        order = _morton_order((lower+upper)/2)
        items = order[_group(self.size, device)]
        items = torch.where(_group(self.size, device) >= 0, items, -1)
        self.items = items
        self.item_lower, self.item_upper = lower, upper
        reps = (lower+upper)/2 if representatives is None else representatives
        node_lower = _reduce(lower, items, math.inf, torch.min)
        node_upper = _reduce(upper, items, -math.inf, torch.max)
        node_rep = reps[items[:, 0]]
        # levels[0] are leaves; children index the level below.
        self.levels = [dict(lower=node_lower, upper=node_upper, rep=node_rep, children=None)]
        while len(self.levels[-1]['lower']) > 1:
            below = self.levels[-1]
            children = _group(len(below['lower']), device)
            self.levels.append(dict(lower=_reduce(below['lower'], children, math.inf, torch.min),
                                    upper=_reduce(below['upper'], children, -math.inf, torch.max),
                                    rep=below['rep'][children[:, 0]], children=children))

    @property
    def height(self):
        return len(self.levels)-1


def _expand(tree, level, nodes):
    if level == 0:
        return nodes[:, None]
    return tree.levels[level]['children'][nodes]


def _pairs(first, second, same, diagonal, strict):
    """Product of child rows; for self traversal keep each unordered pair once."""
    a = first[:, :, None].expand(-1, first.shape[1], second.shape[1])
    b = second[:, None, :].expand(-1, first.shape[1], second.shape[1])
    keep = (a >= 0) & (b >= 0)
    if same:
        position = torch.arange(first.shape[1], device=first.device)
        lower_triangle = (position[:, None] < position[None]) if strict else (position[:, None] <= position[None])
        keep &= ~diagonal[:, None, None] | lower_triangle
    return a[keep], b[keep]


def _traverse(tree_a, tree_b, node_keep, item_keep, *, on_level=None):
    """Yield item index pairs surviving node and item tests, in bounded chunks."""
    same = tree_b is None
    tree_b = tree_a if same else tree_b
    la, lb = tree_a.height, tree_b.height
    device = tree_a.items.device
    frontier = (torch.zeros(1, device=device, dtype=torch.int64), torch.zeros(1, device=device, dtype=torch.int64))
    while la > 0 or lb > 0:
        na, nb = la-1 if la > 0 else 0, lb-1 if lb > 0 else 0
        survivors_a, survivors_b = [], []
        width = (_BRANCH if la > 0 else 1)*(_BRANCH if lb > 0 else 1)
        step = max(1, _PAIR_CHUNK//width)
        for start in range(0, len(frontier[0]), step):
            fa, fb = frontier[0][start:start+step], frontier[1][start:start+step]
            ca, cb = _expand(tree_a, la, fa), _expand(tree_b, lb, fb)
            a, b = _pairs(ca, cb, same, fa == fb, strict=False)
            keep = node_keep(tree_a.levels[na], a, tree_b.levels[nb], b)
            survivors_a.append(a[keep])
            survivors_b.append(b[keep])
        frontier = (torch.cat(survivors_a), torch.cat(survivors_b))
        la, lb = na, nb
        if on_level is not None:
            on_level(tree_a.levels[la], frontier[0], tree_b.levels[lb], frontier[1])
        if not len(frontier[0]):
            return
    step = max(1, _PAIR_CHUNK//(_BRANCH*_BRANCH))
    for start in range(0, len(frontier[0]), step):
        fa, fb = frontier[0][start:start+step], frontier[1][start:start+step]
        a, b = _pairs(tree_a.items[fa], tree_b.items[fb], same, fa == fb, strict=True)
        keep = item_keep(a, b)
        if bool(keep.any()):
            a, b = a[keep], b[keep]
            yield (torch.minimum(a, b), torch.maximum(a, b)) if same else (a, b)


def _boxes_overlap(lower_a, upper_a, lower_b, upper_b, tolerance):
    return ((lower_a <= upper_b+tolerance) & (upper_a >= lower_b-tolerance)).all(-1)


def overlapping_pairs(tree_a, tolerance, tree_b=None):
    """All item pairs whose boxes overlap within ``tolerance`` on every axis.

    Self traversal (``tree_b is None``) returns each unordered pair once as
    (smaller index, larger index). This is a superset generator: callers
    should pass a slightly inflated tolerance and re-test exactly.
    """
    other = tree_a if tree_b is None else tree_b

    def node_keep(level_a, a, level_b, b):
        return _boxes_overlap(level_a['lower'][a], level_a['upper'][a],
                              level_b['lower'][b], level_b['upper'][b], tolerance)

    def item_keep(a, b):
        return _boxes_overlap(tree_a.item_lower[a], tree_a.item_upper[a],
                              other.item_lower[b], other.item_upper[b], tolerance)
    yield from _traverse(tree_a, tree_b, node_keep, item_keep)


def _box_gap(lower_a, upper_a, lower_b, upper_b):
    gap = torch.maximum(lower_b-upper_a, lower_a-upper_b).clamp_min(0)
    return torch.linalg.vector_norm(gap, dim=-1)


def minimum_item_distance(tree_a, tree_b, distance, *, relative_margin=1e-9, absolute_margin=0.):
    """Exact minimum of ``distance(a, b)`` over all item pairs, by branch and bound.

    Representative points supply an upper bound U on the minimum (a point of
    each item lies in its box). A node or item pair is discarded only when its
    box gap exceeds U(1+relative_margin)+absolute_margin, so the minimizing
    pair is always evaluated by the caller's exact ``distance``.
    """
    state = {'upper': math.inf, 'minimum': math.inf}

    def cutoff():
        return state['upper']*(1+relative_margin)+absolute_margin

    def on_level(level_a, a, level_b, b):
        if len(a):
            reps = torch.linalg.vector_norm(level_a['rep'][a]-level_b['rep'][b], dim=-1)
            state['upper'] = min(state['upper'], float(reps.min()))

    def node_keep(level_a, a, level_b, b):
        if not len(a):
            return torch.zeros(0, dtype=torch.bool, device=a.device)
        if not math.isfinite(state['upper']):
            on_level(level_a, a, level_b, b)
        return _box_gap(level_a['lower'][a], level_a['upper'][a],
                        level_b['lower'][b], level_b['upper'][b]) <= cutoff()

    def item_keep(a, b):
        return _box_gap(tree_a.item_lower[a], tree_a.item_upper[a],
                        tree_b.item_lower[b], tree_b.item_upper[b]) <= cutoff()
    evaluated = 0
    for a, b in _traverse(tree_a, tree_b, node_keep, item_keep, on_level=on_level):
        for start in range(0, len(a), _PAIR_CHUNK):
            values = distance(a[start:start+_PAIR_CHUNK], b[start:start+_PAIR_CHUNK])
            evaluated += len(values)
            if len(values):
                smallest = float(values.min())
                state['minimum'] = min(state['minimum'], smallest)
                state['upper'] = min(state['upper'], smallest)
    return state['minimum'], evaluated
