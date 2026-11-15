"""Graded z segments for vertical Green tables at dielectric interfaces.

The tabulated vertical remainder is the total kernel minus the extracted static
(q -> infinity) images. At a dielectric interface the next term of the
reflection expansion, O(k^2/q^2) exp(-q d), leaves a d*log(d) singularity in the
image distance d = |2 z_i - z - z'| (and, across the interface, in the
transmitted direct distance). One Chebyshev polynomial per layer then converges
only algebraically, like k^2 K^-4, which fails the 1e-6 z gate at mm-wave
frequencies. Geometric sub-segments that shrink toward each dielectric interface
confine the singular corner to ever smaller pieces; inside a piece of size delta
the singular term is delta*(x log x + x log delta), so its interpolation error
falls like delta^2 while every other piece pair sees an analytic kernel.
PEC interfaces have exact images and are never graded.
"""

import math

import torch

from ...errors import ConvergenceError


def dielectric_interfaces(stack):
    """Interface heights with a dielectric on both sides."""
    regions = (stack.bottom, *stack.layers, stack.top)
    return tuple(z for i, z in enumerate(stack.interfaces())
                 if regions[i] != "pec" and regions[i + 1] != "pec")


def graded_segments(segments, stack, levels, ratio):
    """Split each segment by `levels` geometric cuts toward its dielectric ends.

    A graded end at e gets cuts at distances L*ratio**-i (i=1..levels) from e,
    L the segment length; with both ends graded the cuts use L/2. Interior
    segment boundaries remain one-sided exactly like interface boundaries.
    """
    graded = set(dielectric_interfaces(stack))
    result = []
    for a, b in segments:
        ends = (a in graded, b in graded)
        length = (b - a) / (2 if all(ends) else 1)
        cuts = {a, b}
        for i in range(1, levels + 1):
            step = length * ratio ** -i
            if ends[0]:
                cuts.add(a + step)
            if ends[1]:
                cuts.add(b - step)
        cuts = sorted(cuts)
        result.extend(zip(cuts[:-1], cuts[1:]))
    return result


def graded_transform(table, sources, observers, segment_index, *, rho=None, components=4):
    """VerticalGreenTable.transform with a truncation per z-pair bucket.

    Every remainder path that is not extracted exactly touches an interface, so
    its spectrum decays at least like exp(-q (eta_s + eta_o)), eta the distance
    to the nearest interface. The ungraded rule 32/min(eta) is the maximum over
    pairs of 64/(eta_s + eta_o) (attained by the closest self pair); graded tables
    apply that bound per pair, rounded up to power-of-two buckets below the
    ungraded global truncation, and never below the horizontal truncation.
    """
    p = table.parent
    edges = p.edges.cpu().tolist()
    heights = torch.cat((sources, observers)).cpu().tolist()
    eta = [min(abs(z - e) for e in edges) for z in heights]
    count = sources.numel()
    buckets = {}
    for i in range(count):
        total = eta[i] + eta[count + i]
        need = table.truncation if total == 0 else min(table.truncation, 64 / total)
        level = 0
        while level < 64 and table.truncation * 2.0 ** -(level + 1) >= max(need, p.truncation):
            level += 1
        buckets.setdefault(level, []).append(i)
    out, tail = None, None
    for level in sorted(buckets):
        index = torch.tensor(buckets[level], dtype=torch.int64, device=sources.device)
        truncation = max(p.truncation, table.truncation * 2.0 ** -level)
        part = table.transform(sources[index].contiguous(), observers[index].contiguous(),
                               segment_index[index].contiguous(), rho=rho, components=components,
                               truncation=truncation)
        if out is None:
            out = part.new_empty((part.shape[0], count, *part.shape[2:]))
        out[:, index] = part
        if rho is None:
            # The tail estimate is per [frequency, pair, component], like the values.
            estimate = table.tail_error_estimate
            if tail is None:
                tail = estimate.new_empty((estimate.shape[0], count, *estimate.shape[2:]))
            tail[:, index] = estimate
    if rho is None:
        table.tail_error_estimate = tail
    return out


Z_GATE = 1e-6
GRADED_Z_NODES = 12
GRADING_RATIO = 4
MAX_GRADING_LEVELS = 5


def admitted_vertical_table(parent, stack, via_spans, *, z_nodes, extract_distance,
                            graded_nodes=GRADED_Z_NODES, extra_levels=0, defer_grading=False):
    """Adaptive z refinement: one measured-error loop, fail-closed.

    Step 0 builds the ungraded table exactly as before grading existed and
    measures its z admission error e0. If e0 <= Z_GATE the table is returned
    unchanged, so every table that admitted before keeps its bits (an a-priori
    error model cannot promise this: any conservative predictor grades some
    tables that admit today). Step 0 costs one ungraded build, 6-25% of the
    graded build it precedes at mm-wave.
    Otherwise the loop rebuilds with `graded_nodes` nodes per piece and ratio-4
    geometric pieces toward each dielectric interface. The interface corner
    error scales like delta^2 K^-4 (delta the smallest piece), so the first level
    is predicted from e0: ceil(log16(e0 (K0/K)^4 / Z_GATE)); each failure adds one
    level. At MAX_GRADING_LEVELS a failing table raises ConvergenceError (no silent path).
    The chosen levels and both errors are recorded in `table.grading_record`
    (network metadata "vertical_z_grading").
    The first level does not credit the K0 -> K node increase: at K0 = 8 the
    ungraded error is not in the K^-4 regime (via stack, 60-200 GHz: the
    ungraded -> level-1 drop is 9x, not the 81x of (12/8)^4 * 16), so the
    credit predicted one level low. K0 = K (grounded bridges) is unchanged.
    defer_grading (R4-2): a multi-frequency parent whose step 0 fails returns
    None instead of grading; the network streams single-frequency tables
    (frequency_groups.py), so a graded table never stores several frequencies.
    graded_nodes/extra_levels only build converged references (tests): a positive
    extra_levels rebuilds extra_levels levels beyond the level the loop reached
    (0 for an admitted ungraded table), capped at MAX_GRADING_LEVELS.
    """
    from .vertical import VerticalGreenTable

    table = VerticalGreenTable.build(parent, stack, via_spans, z_nodes=z_nodes,
                                     extract_distance=extract_distance, z_gate=False)
    error = table.z_interpolation_error.item()
    nodes = max(z_nodes, graded_nodes)
    if error > Z_GATE and dielectric_interfaces(stack):
        if defer_grading and parent.frequencies.numel() > 1:
            return None
        predicted = error / Z_GATE
        predicted_levels = max(1, math.ceil(math.log(predicted) / math.log(GRADING_RATIO ** 2))) if predicted > 1 else 1
        levels = min(MAX_GRADING_LEVELS, predicted_levels)
        while True:
            del table
            table = _graded_table(parent, stack, via_spans, nodes, extract_distance, levels, predicted_levels, z_nodes, error)
            if table.z_interpolation_error.item() <= Z_GATE:
                break
            if levels == MAX_GRADING_LEVELS:
                raise ConvergenceError(
                    f"Vertical z interpolation admission failed: {table.z_interpolation_error.item():.6g} > 1e-6 "
                    f"after {levels} grading levels (ratio {GRADING_RATIO}, {nodes} nodes per piece; "
                    f"ungraded error {error:.6g})")
            levels += 1
    else:
        table.raise_z_failure()
    if not extra_levels or not dielectric_interfaces(stack):
        return table
    reached = 0 if table.grading is None else table.grading[0]
    record = table.grading_record
    del table
    reference = _graded_table(parent, stack, via_spans, nodes, extract_distance,
                              min(MAX_GRADING_LEVELS, reached + extra_levels),
                              reached if record is None else record["predicted_levels"], z_nodes, error)
    reference.raise_z_failure()
    return reference


def _graded_table(parent, stack, via_spans, nodes, extract_distance, levels, predicted_levels, z_nodes, error):
    from .vertical import VerticalGreenTable

    table = VerticalGreenTable.build(parent, stack, via_spans, z_nodes=nodes, extract_distance=extract_distance,
                                     grading=(levels, GRADING_RATIO), z_gate=False)
    table.grading_record = {"levels": levels, "predicted_levels": predicted_levels, "ratio": GRADING_RATIO,
                            "nodes_per_piece": nodes, "pieces": table.segments.shape[0],
                            "ungraded_z_nodes": z_nodes, "ungraded_z_error": error,
                            "z_error": table.z_interpolation_error.item()}
    return table


def grading_metadata(table):
    """Network metadata entry, present only for graded vertical tables."""
    vertical = table.vertical
    record = None if vertical is None else vertical.grading_record
    return {} if record is None else {"vertical_z_grading": record}


def mesh_interfaces(stack, table):
    """Wall-mesh z breakpoints: graded piece boundaries join the stack interfaces.

    Assembly integrates every wall triangle inside one vertical segment, so the
    mesher must cut walls at each graded piece boundary. Ungraded tables return
    the stack interfaces unchanged (same mesh, same bits).
    """
    vertical = table.vertical
    if vertical is None or vertical.grading is None:
        return stack.interfaces()
    return tuple(sorted(set(stack.interfaces()) | set(vertical.segments.reshape(-1).tolist())))
