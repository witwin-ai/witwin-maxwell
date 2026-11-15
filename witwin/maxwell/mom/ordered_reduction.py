"""Fixed-order reductions for bounded, monotonically indexed cell batches."""
import torch


def add_ordered_rows(target, rows, contributions):
    """Add sorted row contributions without competing CUDA atomic writes.

    Callers stream monotonically ordered query/cell rows. Only segment lengths
    cross to the host for scheduling; values and differentiable sums stay on GPU.
    Segment workspace is bounded by the batch, including partial cell segments.
    The returned target and retained autograd history are additional storage.
    """
    if not len(rows):
        return target
    unique, counts = torch.unique_consecutive(rows, return_counts=True)
    parts = contributions.split(counts.tolist(), dim=0)
    reduced = torch.stack([part.sum(dim=0) for part in parts])
    return target.index_copy(0, unique, target[unique] + reduced)


class _RunSums(torch.autograd.Function):
    """Sequential sum of each sorted run at its first position, zero elsewhere.

    The backward gathers each contribution's gradient from its run start, so
    derivatives of every order stay plain differentiable indexing.
    """

    @staticmethod
    def forward(ctx, values, offsets, run_start):
        ctx.save_for_backward(run_start)
        return torch.segment_reduce(values, 'sum', offsets=offsets, axis=0, unsafe=True)

    @staticmethod
    def backward(ctx, grad):
        run_start, = ctx.saved_tensors
        return grad.index_select(0, run_start), None, None


def ordered_index_add_(target, dim, index, source):
    """In-place ``target.index_add_(dim, index, source)`` with a fixed summation order.

    CUDA ``index_add_``/``scatter_add_`` resolve repeated destinations with float
    atomics, so three or more contributions to one entry round differently from
    run to run. Here contributions are stably sorted by destination and each
    destination is summed sequentially in original contribution order
    (segmented sum). The total sits at the first sorted contribution of its run
    and every other sorted contribution becomes an exact zero, so the final
    scatter adds one nonzero value per destination and is order independent.
    No host synchronization is needed; workspace is proportional to ``source``.
    """
    if not len(index):
        return target
    rows = target.movedim(dim, 0)
    order = torch.argsort(index, stable=True)
    sorted_index = index.index_select(0, order)
    # Segment p spans [run_end(p-1), run_end(p)): the whole run at its first
    # position and empty elsewhere. Derived from the sorted destinations, so
    # the host-synchronizing validation of segment_reduce is redundant.
    run_end = torch.searchsorted(sorted_index, sorted_index, right=True)
    offsets = torch.cat((run_end.new_zeros(1), run_end))
    run_start = torch.searchsorted(sorted_index, sorted_index)
    values = source.movedim(dim, 0).index_select(0, order).reshape(len(index), -1)
    if values.is_complex():
        values = torch.view_as_real(values).flatten(1)
    reduced = _RunSums.apply(values, offsets, run_start)
    if target.is_complex():
        reduced = torch.view_as_complex(reduced.reshape(len(index), -1, 2))
    rows.index_put_((sorted_index,), reduced.reshape(len(index), *rows.shape[1:]), accumulate=True)
    return target
