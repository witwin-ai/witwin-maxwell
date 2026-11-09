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
