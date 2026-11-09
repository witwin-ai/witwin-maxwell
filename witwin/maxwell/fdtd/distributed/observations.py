"""Unique quadrature ownership and current field halos for distributed observers."""
from __future__ import annotations

from copy import copy
import math

import numpy as np
import torch


def sampled_monitor_for_shard(monitor, layout, prepared_scene):
    """Assign each global x midpoint to its upper electric-cell bracket owner.

    The right shard stores one low ghost cell, while the left shard stores no
    high ghost cell. Ownership by the upper bracket therefore makes both native
    interpolation neighbors available for every Yee component.
    """
    if monitor.axis == "x" or monitor.sample_spacing is None:
        return monitor
    from ..observers import finite_plane_integration_bounds
    low, high = finite_plane_integration_bounds(prepared_scene, monitor)["x"]
    count = max(2, int(math.ceil((high - low) / monitor.sample_spacing)))
    points = low + (np.arange(count, dtype=np.float64) + .5) * (high - low) / count
    centers = np.asarray(prepared_scene.x_half64, dtype=np.float64)
    upper = np.clip(np.searchsorted(centers, points, side="left"), 0, len(centers) - 1)
    owned = layout.component("Ex").owned_global_slice[0]
    selected = np.flatnonzero((upper >= owned.start) & (upper < owned.stop))
    if not len(selected):
        return None
    local = copy(monitor)
    object.__setattr__(local, "_fdtd_sample_x_indices", selected)
    return local


class DistributedObservationHalos:
    """Refresh all component ghosts after sources/circuits, before quadrature.

    The physical Yee update exchanges only its required curl operands. In
    particular Ex low ghosts and Hx high ghosts are not physical-update receives,
    and tangential E high ghosts still carry the previous electric time. Sampled
    observers must see all six fields at their current staggered times.
    """

    def __init__(self, shards):
        self.shards = tuple(shards)
        self.ready = []
        self.transfers = []
        for shard in self.shards:
            with torch.cuda.device(shard.device):
                self.ready.append(torch.cuda.Event())
        for left, right in zip(self.shards, self.shards[1:]):
            for source, target, components, upper in (
                (right, left, ("Ey", "Ez", "Hx"), True),
                (left, right, ("Ex", "Hy", "Hz"), False),
            ):
                pairs = []
                for component in components:
                    source_layout = source.layout.component(component)
                    target_layout = target.layout.component(component)
                    source_slice = source_layout.owned_local_slice[0]
                    target_slice = target_layout.owned_local_slice[0]
                    source_index = source_slice.start if upper else source_slice.stop - 1
                    target_index = target_slice.stop if upper else target_slice.start - 1
                    source_field = getattr(source.solver, component)
                    target_field = getattr(target.solver, component)
                    pairs.append((target_field[target_index:target_index + 1],
                                  source_field[source_index:source_index + 1]))
                with torch.cuda.device(target.device):
                    done = torch.cuda.Event()
                self.transfers.append((source, target, pairs, done))

    @property
    def bytes_per_step(self):
        return sum(receive.numel()*receive.element_size()
                   for _source,_target,pairs,_done in self.transfers for receive,_send in pairs)

    def exchange(self):
        for shard, event in zip(self.shards, self.ready):
            event.record(shard.compute_stream)
        for source, target, pairs, done in self.transfers:
            with torch.cuda.device(target.device), torch.cuda.stream(target.communication_stream):
                target.communication_stream.wait_event(self.ready[source.rank])
                target.communication_stream.wait_event(self.ready[target.rank])
                for receive, send in pairs:
                    receive.copy_(send, non_blocking=True)
                done.record(target.communication_stream)
        for source, target, _pairs, done in self.transfers:
            # Also protect the sender from recycling the source plane on its
            # next field update before the peer read has completed.
            source.compute_stream.wait_event(done)
            target.compute_stream.wait_event(done)
