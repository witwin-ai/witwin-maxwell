"""Detached forward resume with atomic preflight across every physical shard."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import hashlib
import math

import torch

from ..checkpoint import (
    FDTDCheckpointSchema, FDTDCheckpointState, checkpoint_schema,
    preflight_checkpoint_state, restore_checkpoint_state, validate_checkpoint_state,
)
from ..observation_resume import _clone


def is_distributed_solver(solver):
    return hasattr(solver,"shard_layouts") and hasattr(solver,"logical_scene")


def _require_support(solver):
    if solver._nccl:
        raise NotImplementedError("Distributed resume currently requires in-process cuda_p2p transport.")


def _signature(solver):
    from ..resume import (_additive_j_signature, _pulsed_tfsf_signature, _tfsf_timing_signature, _hash_value,
                          _solver_signature, _observer_origin_signature, _without_derived_time_plan_keys,
                          _tfsf_termination_signature)
    partition = tuple((layout.rank,str(layout.device),layout.global_cell_owned.start,
                       layout.global_cell_owned.stop,layout.global_node_owned.start,
                       layout.global_node_owned.stop) for layout in solver.shard_layouts)
    hasher = hashlib.sha256()
    timing = _additive_j_signature(solver)
    timing.update(_pulsed_tfsf_signature(solver))
    timing.update(_tfsf_timing_signature(solver))
    timing.update(_observer_origin_signature(solver))
    timing.update(_tfsf_termination_signature(solver))
    for name in ("sources","ports","networks","thin_wires","monitors"):
        _hash_value(hasher,name,getattr(solver.logical_scene,name))
    # Circuit is a mutable graph builder, whose default repr includes its
    # allocation address. Hash declarative physics, never Python identity.
    for index,circuit in enumerate(solver.logical_scene.circuits):
        for name in ("name","parameters","initial_conditions","config","devices","bindings"):
            _hash_value(hasher,f"circuits[{index}].{name}",getattr(circuit,name))
    _hash_value(hasher,"time_plan",_without_derived_time_plan_keys(getattr(solver,"_time_plan",None)))
    return dict(**timing,kind="distributed_cuda_p2p",partition=partition,logical_fingerprint=hasher.hexdigest(),
                shards=tuple((shard.rank,_solver_signature(shard.solver)) for shard in solver.shards))


def _combined_schema(solver):
    local = tuple((shard.rank,checkpoint_schema(shard.solver)) for shard in solver.shards)
    categories = {}
    for field in fields(FDTDCheckpointSchema):
        if field.name=="version":
            continue
        categories[field.name] = tuple(f"rank{rank}/{name}" for rank,schema in local
                                       for name in getattr(schema,field.name))
    return FDTDCheckpointSchema(version=local[0][1].version,**categories)


def _extra_observations(solver):
    networks = tuple(dict(name=runtime.name,port_energy=runtime.port_energy,
                          absorbed_energy=runtime.absorbed_energy,generated_energy=runtime.generated_energy)
                     for runtime in getattr(solver,"_network_runtimes",()))
    wire = getattr(solver,"_wire_runtime",None)
    wire_monitors = tuple(dict(name=state["compiled"].name,entries=state["entries"],
                              **{key:state[key] for key in ("current_real","current_imag","charge_real","charge_imag")})
                          for state in getattr(wire,"monitor_state",()))
    return dict(networks=networks,wire_monitors=wire_monitors)


def capture_distributed_resume(solver, *, step, total_steps):
    from ..resume import FDTDResumeCheckpoint, capture_resume_checkpoint, _schema_snapshot
    _require_support(solver)
    if not 0<=step<total_steps:
        raise ValueError("Distributed resume checkpoint step must precede the planned end.")
    solver._synchronize_all()
    local = []
    for shard in solver.shards:
        with torch.cuda.device(shard.device),torch.cuda.stream(shard.compute_stream):
            state = capture_resume_checkpoint(shard.solver,step=step,total_steps=total_steps)
            local.append((shard,state,_clone(_extra_observations(shard.solver))))
    schema = _combined_schema(solver)
    tensors = {f"rank{shard.rank}/{name}":value for shard,state,_ in local
               for name,value in state.physics.tensors.items()}
    ordered = {name:tensors[name] for name in schema.state_names}
    auxiliary = dict(shards=tuple(dict(rank=shard.rank,signature=state.signature,
                         physics_schema=_schema_snapshot(state.physics.schema),auxiliary=state.auxiliary,extra=extra)
                         for shard,state,extra in local),
                     shutoff_peak=solver._shutoff_peak.detach().clone(),
                     shutoff_triggered=solver._shutoff_triggered,shutoff_step=solver._shutoff_step)
    solver._synchronize_all()
    return FDTDResumeCheckpoint(step=step,total_steps=total_steps,dt=float(solver.dt),
        physics=FDTDCheckpointState(step=step,schema=schema,tensors=ordered),
        signature=_signature(solver),auxiliary=auxiliary)


def preflight_distributed_resume(solver, checkpoint, *, total_steps):
    from ..resume import (FDTDResumeCheckpoint, _preflight_additive_j_timing,
                          _preflight_pulsed_tfsf_reference, _preflight_tfsf_timing, _preflight_observer_origin,
                          _preflight_tfsf_termination)
    _require_support(solver)
    if not isinstance(checkpoint,FDTDResumeCheckpoint):
        raise TypeError("resume_from must be an FDTDResumeCheckpoint.")
    if checkpoint.total_steps!=total_steps or not 0<=checkpoint.step<total_steps:
        raise ValueError("Distributed resume execution step or planned duration changed.")
    if checkpoint.physics.step!=checkpoint.step or checkpoint.dt!=float(solver.dt):
        raise ValueError("Distributed resume physical step or dt changed.")
    _preflight_additive_j_timing(solver, checkpoint.signature)
    _preflight_pulsed_tfsf_reference(solver, checkpoint.signature)
    _preflight_tfsf_timing(solver, checkpoint.signature)
    _preflight_observer_origin(solver, checkpoint.signature)
    _preflight_tfsf_termination(solver, checkpoint.signature)
    if checkpoint.signature!=_signature(solver):
        raise ValueError("Distributed resume configuration or partition does not match the prepared solver.")
    if not isinstance(checkpoint.auxiliary,Mapping):
        raise ValueError("Distributed resume auxiliary state must be a mapping.")
    if checkpoint.auxiliary.get("shutoff_triggered"):
        raise ValueError("A checkpoint captured after field shutoff cannot be resumed.")
    validate_checkpoint_state(checkpoint.physics)


def _preflight_extra(target, source, path, step, copy_ops, host_updates):
    from ..resume import _require_tensor_compatible
    if isinstance(target,torch.Tensor):
        _require_tensor_compatible(path,source,target)
        copy_ops.append((target,source))
    elif isinstance(target,Mapping):
        if not isinstance(source,Mapping) or set(target)!=set(source):
            raise ValueError(f"Distributed resume {path} layout changed.")
        for key,value in target.items():
            saved = source[key]
            if key in ("phase_cos","phase_sin","window_normalization","sample_count"):
                if not isinstance(saved,(int,float)) or not math.isfinite(saved):
                    raise ValueError(f"Distributed resume {path}.{key} is invalid.")
                if key=="sample_count" and (not isinstance(saved,int) or not 0<=saved<=step):
                    raise ValueError("Distributed wire monitor sample count is invalid.")
                host_updates.append((target,key,saved))
            else:
                _preflight_extra(value,saved,f"{path}.{key}",step,copy_ops,host_updates)
    elif isinstance(target,(tuple,list)):
        if not isinstance(source,(tuple,list)) or len(source)!=len(target):
            raise ValueError(f"Distributed resume {path} count changed.")
        for index,(value,saved) in enumerate(zip(target,source)):
            _preflight_extra(value,saved,f"{path}[{index}]",step,copy_ops,host_updates)
    elif target!=source:
        raise ValueError(f"Distributed resume {path} configuration changed.")


def restore_distributed_resume(solver, checkpoint, *, total_steps):
    from ..resume import (FDTDResumeCheckpoint, _preflight_auxiliary,
                         _schema_from_snapshot, _require_tensor_compatible)
    preflight_distributed_resume(solver,checkpoint,total_steps=total_steps)
    validate_checkpoint_state(checkpoint.physics,expected_schema=_combined_schema(solver))
    payloads = checkpoint.auxiliary.get("shards")
    if not isinstance(payloads,(tuple,list)) or len(payloads)!=len(solver.shards):
        raise ValueError("Distributed resume shard auxiliary layout changed.")
    prepared = []
    for shard,payload in zip(solver.shards,payloads):
        required={"rank","signature","physics_schema","auxiliary","extra"}
        if not isinstance(payload,Mapping) or set(payload)!=required or payload.get("rank")!=shard.rank:
            raise ValueError("Distributed resume shard auxiliary layout or order changed.")
        schema = _schema_from_snapshot(payload["physics_schema"])
        physics = FDTDCheckpointState(step=checkpoint.step,schema=schema,
            tensors={name:checkpoint.physics.tensors[f"rank{shard.rank}/{name}"] for name in schema.state_names})
        state = FDTDResumeCheckpoint(step=checkpoint.step,total_steps=total_steps,dt=checkpoint.dt,
            physics=physics,signature=payload["signature"],auxiliary=payload["auxiliary"])
        from ..resume import preflight_resume_checkpoint
        preflight_resume_checkpoint(shard.solver,state,total_steps=total_steps)
        preflight_checkpoint_state(shard.solver,physics)
        copy_ops,host_updates = _preflight_auxiliary(shard.solver,state)
        _preflight_extra(_extra_observations(shard.solver),payload.get("extra"),
                         f"rank{shard.rank}.extra",checkpoint.step,copy_ops,host_updates)
        prepared.append((shard,state,copy_ops,host_updates))
    peak = checkpoint.auxiliary.get("shutoff_peak")
    _require_tensor_compatible("distributed.shutoff_peak",peak,solver._shutoff_peak)
    # Nothing above mutates physical fields, histories or accumulated spectra.
    for shard,state,copy_ops,host_updates in prepared:
        with torch.cuda.device(shard.device),torch.cuda.stream(shard.compute_stream):
            restore_checkpoint_state(shard.solver,state.physics)
            for target,saved in copy_ops:
                target.copy_(saved)
            for target,key,value in host_updates:
                if isinstance(target,dict):
                    target[key] = value
                else:
                    setattr(target,key,value)
            shard.solver._sync_dft_legacy_state()
            shard.solver._sync_observer_legacy_state()
    solver._shutoff_peak.copy_(peak)
    solver._synchronize_all()
    return checkpoint.step
