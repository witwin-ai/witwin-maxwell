from __future__ import annotations


def reduce_sample_counts(shards, attribute: str) -> tuple[int, ...]:
    """Reduce per-shard spectral counters without double-counting spatial tiles."""

    if not shards:
        return ()
    sequences = tuple(
        tuple(int(value) for value in (getattr(shard.solver, attribute, ()) or ()))
        for shard in shards
    )
    width = max((len(sequence) for sequence in sequences), default=0)
    if width == 0:
        return ()
    return tuple(
        max(
            (sequence[index] if index < len(sequence) else 0)
            for sequence in sequences
        )
        for index in range(width)
    )


__all__ = ["reduce_sample_counts"]
