from __future__ import annotations


def reduce_frequency_sample_counts(shards, frequencies) -> tuple[int, ...]:
    """Reduce observer counters by frequency rather than shard-local entry index."""

    requested = tuple(float(frequency) for frequency in frequencies)
    if not requested:
        return ()
    merged = {frequency: 0 for frequency in requested}
    for shard in shards:
        local_frequencies = tuple(
            float(frequency) for frequency in (shard.solver.observer_frequencies or ())
        )
        local_counts = tuple(
            int(count) for count in (shard.solver.observer_sample_counts or ())
        )
        for frequency, count in zip(local_frequencies, local_counts):
            if frequency in merged:
                merged[frequency] = max(merged[frequency], count)
    return tuple(merged[frequency] for frequency in requested)


__all__ = ["reduce_frequency_sample_counts"]
