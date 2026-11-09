"""Reproducible binary local search using GPU candidate projections."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class BinaryDesignResult:
    result: object
    tiles: torch.Tensor
    objective_history: torch.Tensor
    attempts: int
    accepted: int
    converged: bool


def optimize_binary(response, groups, objective, *, initial=None, mode='pec', epsilon_contrast=1.,
                    max_passes=10, seed=0, batch_size=16):
    """Minimize a scalar CUDA objective; groups contain local design indices.

    Overlapping tile edges use union occupancy, so removing one tile does not
    remove an edge still owned by its neighbor. A complete pass with no accepted
    flip certifies a one-tile local optimum, not global optimality.
    """
    if mode not in ('pec','dielectric') or max_passes < 1 or not groups or isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError("Specify nonempty groups, positive passes and pec/dielectric mode.")
    children = getattr(response,'responses',(response,))
    n = response.design_indices.numel()
    checked = []
    for group in groups:
        if not isinstance(group,torch.Tensor) or group.device != response.device or group.dtype != torch.int64 or group.ndim != 1 or not group.numel():
            raise ValueError("Tile groups must be nonempty int64 CUDA vectors of local indices.")
        if bool(((group<0)|(group>=n)).any()) or group.unique().numel()!=group.numel():
            raise ValueError("Tile indices must be unique and in range within each group.")
        checked.append(group.clone())
    tiles = torch.zeros(len(checked),device=response.device,dtype=torch.bool) if initial is None else initial.clone()
    if tiles.device != response.device or tiles.dtype != torch.bool or tiles.shape != (len(checked),):
        raise ValueError("Initial tiles must be a boolean CUDA vector matching groups.")
    coverage = torch.zeros(n,device=response.device,dtype=torch.int64)
    for active,group in zip(tiles,checked):
        if bool(active):
            coverage[group] += 1
    def candidate(occupancy):
        contrast = torch.stack([occupancy.to(child.dtype)*complex(epsilon_contrast)*child.solver.k0**2
                                if mode == 'dielectric' else torch.zeros(n,device=response.device,dtype=child.dtype)
                                for child in children])
        return response.trial(delta=contrast if hasattr(response,'responses') else contrast[0],
                              pec=occupancy if mode=='pec' else torch.zeros_like(occupancy))
    def score(trial):
        value = objective(trial.projections)
        if not isinstance(value,torch.Tensor) or value.device != response.device or value.numel()!=1 or value.is_complex() or not bool(torch.isfinite(value)):
            raise ValueError("Objective must return a finite real CUDA scalar.")
        return value.detach().reshape(())
    initial_trial = candidate(coverage>0)
    best = score(initial_trial)
    response._accept(initial_trial)
    history = [best]
    generator = torch.Generator(device=response.device).manual_seed(seed)
    attempts = accepted = 0
    converged = False
    for _ in range(max_passes):
        improved = False
        order=torch.randperm(len(checked),generator=generator,device=response.device).cpu().tolist()
        position=0
        while position<len(order):
            chunk=order[position:position+batch_size]
            candidates={}
            if batch_size>1:
                # Rank buckets avoid dummy rows and preserve the established tile order.
                buckets={}
                for index in chunk:
                    buckets.setdefault(checked[index].numel(),[]).append(index)
                try:
                    for bucket in buckets.values():
                        indices=torch.stack([checked[index] for index in bucket])
                        tile_indices=torch.tensor(bucket,device=response.device)
                        changes=1-2*tiles[tile_indices].to(torch.int64)
                        occupancy=(coverage[indices]+changes[:,None])>0
                        contrast=None if mode=='pec' else torch.stack([
                            occupancy.to(child.dtype)*complex(epsilon_contrast)*child.solver.k0**2 for child in children])
                        proposals=response.trial_updates_batch(indices,
                            delta=contrast if hasattr(response,'responses') or contrast is None else contrast[0],
                            pec=occupancy if mode=='pec' else torch.zeros_like(occupancy))
                        candidates.update(zip(bucket,proposals))
                except RuntimeError:
                    # A later invalid proposal must not preempt an earlier valid
                    # improvement. Replay this chunk in its original sequential order.
                    candidates.clear()
            accepted_in_chunk=False
            for index in chunk:
                change=-1 if bool(tiles[index]) else 1
                updated=coverage.clone();updated[checked[index]]+=change
                trial=candidates[index] if index in candidates else candidate(updated>0)
                value=score(trial)
                attempts+=1
                position+=1
                if bool(value<best):
                    response._accept(trial)
                    coverage=updated
                    tiles[index]=~tiles[index]
                    best=value;accepted+=1;improved=True
                    history.append(best)
                    accepted_in_chunk=True
                    # Later proposals were computed at an obsolete generation.
                    break
            if accepted_in_chunk:
                continue
        if not improved:
            converged=True
            break
    result=response._accepted_result()
    return BinaryDesignResult(result,tiles,torch.stack(history),attempts,accepted,converged)
