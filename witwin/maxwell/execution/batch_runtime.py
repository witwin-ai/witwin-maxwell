"""Bounded cooperative execution of native FDTD field cohorts."""
from __future__ import annotations

import copy
import contextlib
import threading
import time
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from collections.abc import Mapping

import torch
from ..fdtd.cuda.runtime.graph import capture_participant

from .batch import BatchParameters
from .plan import BatchPlanReport
from .records import BatchItem, BatchResult, DistributedFailure, ExecutionRecord, FailureKind


class _BatchCancelled(Exception):
    pass


def _owned(value, device, memo):
    """Export tensors independently while preserving aliases inside one Result."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        memo[id(value)] = value.to(device=device, copy=True)
        return memo[id(value)]
    if isinstance(value, Mapping):
        result = {}
        memo[id(value)] = result
        result.update((key, _owned(item, device, memo)) for key, item in value.items())
        return result
    if isinstance(value, tuple):
        result = tuple(_owned(item, device, memo) for item in value)
        memo[id(value)] = result
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_owned(item, device, memo) for item in value)
        return result
    if type(value).__module__.startswith("witwin.") and hasattr(value, "__dict__") and not isinstance(value, type):
        result = copy.copy(value)
        memo[id(value)] = result
        for name, item in vars(value).items():
            object.__setattr__(result, name, _owned(item, device, memo))
        return result
    # Host coordinate masters and declarative metadata are snapshots too.
    import numpy as np
    if isinstance(value, np.ndarray):
        memo[id(value)] = value.copy()
        return memo[id(value)]
    return value


class PreparedBatch:
    """Reusable native CUDA context, dispatched to a slot coordinator as needed."""

    def __new__(cls, batch):
        if cls is PreparedBatch and (len(batch.execution.devices) > 1 or batch.execution.slots_per_device == 2):
            return object.__new__(ParallelPreparedBatch)
        return object.__new__(cls)

    @capture_participant
    def __init__(self, batch):
        self.batch = batch
        self._record_grad = torch.is_grad_enabled()
        policy = batch.execution
        if len(policy.devices) != 1 or policy.slots_per_device != 1:
            raise NotImplementedError("PreparedBatch currently requires one GPU and one execution slot.")
        if policy.tune:
            raise ValueError("Resolve measured tuning through BatchSimulation.prepare().")
        if batch.outputs.materials or any(name.startswith("H") for name in batch.outputs.fields):
            raise NotImplementedError("Batch material export and magnetic full-field spectra are not implemented yet.")
        if batch.outputs.result_device not in ("local", "cpu", policy.devices[0]):
            if batch.outputs.result_device not in getattr(batch, "_export_pools", {}):
                raise NotImplementedError("Cross-device batch result export requires a destination allocation pool.")
        self._validate_inputs(batch.parameters)
        self.device = policy.devices[0]
        if not torch.cuda.is_available():
            raise RuntimeError("PreparedBatch requires an accessible CUDA device.")
        self.capacity = 1 if policy.batch_size == "auto" else policy.batch_size
        self.capacity = min(self.capacity, len(batch))
        self._gate = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False
        self._slot = None
        self._cohort = None
        self._models = ()
        self._loaded_key = None
        self._loaded_parameters = ()
        self._sources = None
        self._observers = None
        self._graphs = {}
        self._fixed_material_cache = {}
        self._local_material_cache = None
        self._budget = None
        self._capacity_retries = []
        self._records = ()
        self._groups = batch.plan().groups
        self._stream = torch.cuda.Stream(device=self.device)
        self._caller_ready = torch.cuda.Event()
        self._output_ready = torch.cuda.Event()
        account = getattr(batch, "_memory_account", None)
        if account is not None:
            from .pool import BatchMemoryPool
            self._budget = BatchMemoryPool(self.device, account.limit_bytes, policy.reserve_bytes, account=account)
        elif policy.memory_limit_bytes is not None or policy.batch_size == "auto":
            from .pool import BatchMemoryPool
            free, _ = torch.cuda.mem_get_info(self.device)
            requested = policy.memory_limit_bytes
            if isinstance(requested, Mapping):
                requested = requested[self.device]
            limit = max(0, int(free) - policy.reserve_bytes)
            if requested is not None:
                limit = min(limit, requested)
            self._budget = BatchMemoryPool(self.device, limit, policy.reserve_bytes)
        self._enter_stream()
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream), self._allocation_scope(), torch.no_grad():
                first = self._groups[0]["variant_indices"][:self.capacity]
                self._load(first, batch.parameters, self._groups[0]["group_id"])
                if policy.batch_size == "auto":
                    # Calibrate from a real prepared lane. The native pool is
                    # the hard bound; this conservative ratio only selects B.
                    lane_peak = max(1, self._budget.stats()["peak_charged_bytes"])
                    selected = min(policy.max_batch_size, len(batch),
                                   max(1, self._budget.limit_bytes // (2 * lane_peak * getattr(batch, "_budget_slots", 1))))
                    if selected > 1:
                        self.capacity = selected
                        self._slot = None
                        self._loaded_key = None
                        while True:
                            try:
                                first = self._groups[0]["variant_indices"][:self.capacity]
                                self._load(first, batch.parameters, self._groups[0]["group_id"])
                                break
                            except torch.cuda.OutOfMemoryError:
                                if self.capacity == 1:
                                    raise
                                self._reduce_capacity()
                self._stream.synchronize()
        except BaseException:
            self.close()
            raise
        declaration = batch.plan().to_dict()
        graph_mode = "pending" if policy.graph != "off" else "off"
        declaration.update(execution_tier="native_pipeline_prepared", batch_size=self.capacity,
                           graph=graph_mode, slots_per_device=1,
                           memory={} if self._budget is None else self._budget.stats(),
                           capacity_retries=list(self._capacity_retries))
        declaration["warnings"] = [
            "Native field/source/observer pipelines reuse proven local material banks; undeclared geometry changes compile full scenes.",
            "Automatic capacity uses a prepared-lane measurement and bounded OOM reduction; throughput tuning is separate.",
            "Byte limits conservatively charge private allocator backing segments, including cached and retained output storage.",
            "Graph capture is attempted at run time and deferred while another registered batch thread is active.",
            "Only the resident cohort is compiled at prepare; later cohorts are checked before execution.",
            "Input Scene objects must not be mutated during the PreparedBatch lifetime.",
        ]
        for group in declaration["groups"]:
            group["capability_status"] = "native_forward_cohort_validation"
        for device in declaration["devices"]:
            device.update(batch_size=self.capacity, slots=1, graph_mode=graph_mode, graph_steps=0)
        # The execution tier is usable; the complete capacity/share plan is not
        # resolved until every compiled group and allocation has been certified.
        self.report = BatchPlanReport.from_dict(declaration)

    def _allocation_scope(self):
        return self._budget.activate() if self._budget is not None else contextlib.nullcontext()

    def _reduce_capacity(self):
        self._capacity_retries.append(self.capacity)
        self.capacity = max(1, self.capacity // 2)
        self._models = ()
        self._sources = self._observers = self._slot = None
        self._graphs.clear()
        self._loaded_key = None

    def _validate_inputs(self, parameters):
        batch = self.batch
        if parameters is not None:
            if not isinstance(parameters, BatchParameters):
                raise TypeError("parameters must be BatchParameters.")
            if batch.template is None or parameters.schema != batch.parameters.schema:
                raise ValueError("PreparedBatch parameters must preserve the template schema and variant count.")
            batch._validate_parameters(parameters)
        from ..sources import PointDipole
        from ..monitors import PointMonitor, PlaneMonitor, FluxMonitor
        for model in batch.simulations:
            checked = copy.copy(model)
            checked._refresh_trainable_parameters()
            if batch.outputs.result_device == "cpu" and (
                    checked.has_trainable_parameters or parameters is not None and parameters.requires_grad):
                raise ValueError("CPU result export requires a detached forward input.")
            if torch.is_grad_enabled() and (checked.has_trainable_parameters or parameters is not None and parameters.requires_grad) and model.config.shutoff > 0:
                raise ValueError("Batch adjoint requires a fixed time horizon: set shutoff=0.")
            scene = model.scene
            for name in ("ports", "thin_wires", "networks", "lumped_elements", "circuits"):
                if getattr(scene, name, ()):
                    raise NotImplementedError(f"Native batch execution does not yet support Scene.{name}.")
            if model.port_sweep is not None or model.excitations:
                raise NotImplementedError("Batch RF port excitation is not implemented yet.")
            if any(not isinstance(source, PointDipole) for source in scene.resolved_sources()):
                raise NotImplementedError("The current native batch tier supports PointDipole sources.")
            if any(not isinstance(m, (PointMonitor, PlaneMonitor, FluxMonitor)) for m in scene.resolved_monitors()):
                raise NotImplementedError("The current native batch tier supports point, plane and flux monitors.")
            if model.config.enable_plot:
                raise ValueError("Batch execution does not support per-step plotting.")

    def _enter_stream(self):
        with torch.cuda.device(self.device):
            self._caller_ready.record(torch.cuda.current_stream(self.device))
            self._stream.wait_event(self._caller_ready)

    def _load(self, indices, parameters, group_id):
        from ..fdtd.runtime.batch import NativeFieldSlot, _share_static_materials, validate_native_solver
        from ..adjoint_inputs import scene_trainable_material_tensors
        inputs = () if parameters is None else tuple(
            value for kind in ("region_densities", "source_scales")
            for value in getattr(parameters, kind).values())
        models = (self.batch.template,) if self.batch.template is not None else tuple(
            self.batch.simulations[i] for i in indices)
        inputs += tuple(value for model in models for value in (
            model.scene_module.parameters() if model.scene_module is not None
            else scene_trainable_material_tensors(model.scene)))
        key = (tuple(indices), self._record_grad, tuple((id(value), value.requires_grad,
                                    object() if torch.is_inference(value) else value._version)
                                    for value in inputs))
        if parameters is not None:
            key += (tuple((kind, name, id(value)) for kind in ("region_densities", "source_scales")
                          for name, value in getattr(parameters, kind).items()),)
        if any(model.scene_module is not None for model in models):
            # Module-generated material graphs must be fresh after a previous
            # backward has consumed their saved intermediates.
            key += (object(),)
        if key == self._loaded_key:
            return
        self._loaded_key = None
        self._loaded_parameters = ()
        materializer = copy.copy(self.batch)
        materializer.parameters = parameters
        models, solvers = [], []
        # Release the old compiler objects before compiling the next cohort.
        # The slot retains only its persistent fields, coefficients and profiles.
        self._models = ()
        self._sources = None
        self._observers = None
        self._graphs.clear()
        if self._slot is not None:
            self._slot.solvers = ()
        if self._cohort != group_id:
            self._slot = None
            self._local_material_cache = None
        for index in indices:
            free, _ = torch.cuda.mem_get_info(self.device)
            if free <= self.batch.execution.reserve_bytes:
                raise torch.cuda.OutOfMemoryError("Insufficient free device memory above BatchExecution.reserve_bytes.")
            model = copy.copy(self.batch.template or self.batch.simulations[index])
            model.config = copy.deepcopy(model.config)
            with torch.enable_grad():
                scene = materializer.scene(index).clone(device=self.device)
            if self.batch.outputs.monitors != "all":
                selected = set(self.batch.outputs.monitors)
                scene.monitors = [m for m in scene.monitors if m.name in selected]
            model.scene = scene
            model.scene_input = scene
            model.scene_module = None
            with torch.enable_grad():
                from .adjoint import validate_material_channels
                if self._record_grad:
                    validate_material_channels(scene)
                model._batch_source_scales = {}
                if parameters is not None:
                    for source_index, source in enumerate(self.batch.template.scene.resolved_sources()):
                        value = parameters.source_scales.get(source.name)
                        if value is not None and value.requires_grad:
                            scale = value[index].to(self.device)
                            model._batch_source_scales[source_index] = (scale, source.source_time.amplitude)
                model._refresh_trainable_parameters()
                if self._record_grad and (model.has_trainable_parameters or model._batch_source_scales):
                    from ..fdtd.adjoint.bridge import _FDTDGradientBridge
                    bridge = _FDTDGradientBridge(model)
                else:
                    bridge = None
                model._batch_differentiable = bridge is not None
                model._batch_input_guards = inputs
            from ..scene import prepare_scene
            compiled_scene = prepare_scene(scene)
            # Template source scales cannot alter geometry, material sampling,
            # or the grid. Retain one material bank across all such cohorts.
            # Density templates and independent Scene lists have no such proof.
            fixed_materials = (self.batch.template is not None
                               and not parameters.region_densities
                               and not compiled_scene._has_dynamic_materials()
                               and not compiled_scene._has_trainable_geometry())
            if fixed_materials:
                compiled_scene._material_model_cache = self._fixed_material_cache
            elif self.batch.template is None and self.batch.variation is not None and (
                    self.batch.variation.structures or self.batch.variation.material_regions):
                from ..compiler.materials import LocalMaterialCache
                if self._local_material_cache is None:
                    self._local_material_cache = LocalMaterialCache(compiled_scene,
                        self.batch.variation.material_regions,
                        structure_envelopes=self.batch.variation.structures)
                    material_model = self._local_material_cache.base
                else:
                    material_model = self._local_material_cache.compile(compiled_scene)
                compiled_scene._material_model_override = material_model
            elif self.batch.template is not None and parameters.region_densities and not model.has_trainable_parameters:
                from ..compiler.materials import LocalMaterialCache, PerturbationMedium
                if not any(isinstance(s.material, PerturbationMedium) for s in compiled_scene.structures):
                    if self._local_material_cache is None:
                        self._local_material_cache = LocalMaterialCache(compiled_scene, parameters.region_densities)
                        material_model = self._local_material_cache.base
                    else:
                        material_model = self._local_material_cache.compile(compiled_scene)
                    compiled_scene._material_model_override = material_model
            solver = model._build_fdtd_solver_for_scene(compiled_scene, initialize=True)
            solver._batch_source_scale_amplitudes = {
                i: amplitude for i, (_, amplitude) in model._batch_source_scales.items()}
            solver.verbose = False
            validate_native_solver(solver)
            if model._batch_differentiable:
                from ..fdtd.adjoint.dispatch import validate_native_adjoint_preparation
                bridge._validate_supported_configuration(solver)
                validate_native_adjoint_preparation(solver)
            if solvers:
                # Deduplicate before constructing the next lane so the peak
                # does not include a whole cohort of redundant material banks.
                _share_static_materials((solvers[0], solver))
            solvers.append(solver)
            models.append(model)
        if self._slot is None:
            self._slot = NativeFieldSlot(self.capacity, self.device)
        self._slot.bind(solvers)
        self._models = tuple(models)
        # Keep identities alive for as long as the cached cohort uses them.
        # Source amplitudes materialize to Python floats, so the compiled Scene
        # alone does not retain their original tensors. Inference tensors lack
        # version counters and deliberately never reuse a cached compilation.
        self._loaded_parameters = inputs
        self._loaded_key = key
        self._cohort = group_id

    def _solve_cohort(self):
        lifecycle = self._cohort_steps()
        try:
            while True:
                next(lifecycle)
        except StopIteration as done:
            return done.value
        finally:
            lifecycle.close()

    def _cohort_steps(self):
        from ..fdtd.runtime.stepping import solve_steps, _post_source_block, _electric_field_energy
        from ..fdtd.observers import BatchObserverEngine
        from ..fdtd.excitation.temporal import BatchSourceEngine
        slot = self._slot
        slot.reset()
        if self._observers is None:
            self._observers = BatchObserverEngine(slot)
        observers = self._observers
        if self._sources is None:
            self._sources = BatchSourceEngine(slot, max(
                model._resolve_fdtd_time_steps(solver, model.scene)
                for model, solver in zip(self._models, slot.solvers)))
        generators, pending, outputs, checkpoints, strides = [], {}, {}, {}, {}
        try:
            for lane, (model, solver) in enumerate(zip(self._models, slot.solvers)):
                cfg = model.config
                steps = model._resolve_fdtd_time_steps(solver, model.scene)
                solver._cuda_graph_active = False
                iterator = solve_steps(solver, time_steps=steps, dft_frequency=model.frequencies,
                    dft_window=cfg.spectral_sampler.window, full_field_dft=bool(self.batch.outputs.fields),
                    normalize_source=cfg.spectral_sampler.normalize_source, shutoff=cfg.shutoff,
                    shutoff_check_interval=cfg.shutoff_check_interval, cooperative=True,
                    cooperative_observer=(observers, lane), cooperative_driver=True,
                    dft_components=self.batch.outputs.fields)
                generators.append(iterator)
                descriptor = next(iterator)
                if solver.dft_enabled and not descriptor["gpu_dft"]:
                    raise NotImplementedError("Batch full-field spectra require GPU-driven accumulation.")
                pending[lane] = (steps, descriptor["shutoff_min_step"])
            observers.bind()
            from ..fdtd.checkpoint import capture_checkpoint_state
            from ..fdtd.adjoint.core import _checkpoint_stride
            for lane, model in enumerate(self._models):
                if model._batch_differentiable and self._record_grad:
                    checkpoints[lane] = [capture_checkpoint_state(slot.solvers[lane], step=0)]
                    strides[lane] = _checkpoint_stride(model, pending[lane][0])
            yield
            from ..fdtd.cuda.runtime.graph import CudaGraphRunner
            policy = self.batch.execution
            graph_enabled = policy.graph != "off"
            chunk_limit = 32 if policy.graph_steps == "auto" else policy.graph_steps
            spectral_signature = tuple(getattr(solver, "_dft_buffer_generation", 0) for solver in slot.solvers)
            if spectral_signature != getattr(self, "_graph_spectral_signature", None):
                self._graphs.clear()
                self._graph_spectral_signature = spectral_signature
            self._last_graph_replays = 0
            self._last_graph_error = None
            completed = 0
            while pending:
                if self._cancel.is_set():
                    raise _BatchCancelled("Batch cancellation requested.")
                deadline = min(value[0] for value in pending.values())
                for lane in pending:
                    if lane in strides:
                        deadline = min(deadline, (completed//strides[lane]+1)*strides[lane])
                    cfg = self._models[lane].config
                    if cfg.shutoff > 0:
                        deadline = min(deadline, (completed//cfg.shutoff_check_interval+1)*cfg.shutoff_check_interval)
                count = min(chunk_limit if graph_enabled else 1, deadline-completed)
                lanes = tuple(pending)
                def advance_chunk():
                    for _ in range(count):
                        slot.advance()
                        self._sources.advance(observers.step)
                        for lane in lanes:
                            _post_source_block(slot.solvers[lane])
                        observers.advance()
                if graph_enabled:
                    key = (lanes, count)
                    replay = self._graphs.get(key)
                    if replay is None:
                        try:
                            # Capture records GPU work without executing it.
                            # All kernels and buffers are initialized before this
                            # point; no warmup may consume source/observer steps.
                            replay = CudaGraphRunner(device=self.device, warmup_steps=0,
                                memory_pool=self._budget, capture_error_mode="thread_local").capture(advance_chunk)
                            self._graphs[key] = replay
                        except Exception as error:
                            if policy.graph == "required":
                                raise
                            graph_enabled = False
                            self._last_graph_error = str(error)
                    if replay is not None:
                        replay()
                        self._last_graph_replays += 1
                        for lane in lanes:
                            slot.solvers[lane]._cuda_graph_active = True
                    else:
                        advance_chunk()
                else:
                    advance_chunk()
                completed += count
                for lane in tuple(pending):
                    solver = slot.solvers[lane]
                    cfg = self._models[lane].config
                    steps, minimum = pending[lane]
                    if cfg.shutoff > 0 and completed % cfg.shutoff_check_interval == 0:
                        energy = _electric_field_energy(solver)
                        solver._shutoff_peak = torch.maximum(solver._shutoff_peak, energy)
                        if completed - 1 >= minimum and bool((
                                (solver._shutoff_peak > 0) & (energy < cfg.shutoff*solver._shutoff_peak)).item()):
                            solver._shutoff_triggered = True
                            solver._shutoff_step = completed - 1
                    if completed < steps and not solver._shutoff_triggered:
                        if lane in strides and completed % strides[lane] == 0:
                            checkpoints[lane].append(capture_checkpoint_state(solver, step=completed))
                        continue
                    try:
                        generators[lane].send(completed)
                    except StopIteration as done:
                        outputs[lane] = done.value or {}
                        if lane in checkpoints:
                            from .adjoint import BatchAdjointTape
                            # A new bridge is required for every run, even when
                            # the compiled cohort itself is unchanged.
                            from ..fdtd.adjoint.bridge import _FDTDGradientBridge
                            with torch.enable_grad():
                                bridge = _FDTDGradientBridge(self._models[lane])
                            outputs[lane] = BatchAdjointTape(bridge, solver, outputs[lane],
                                checkpoints[lane], steps, bool(self.batch.outputs.fields), self._budget)
                        del pending[lane]
                        slot.active[lane] = 0
                    else:
                        raise RuntimeError("Batch lifecycle did not finalize after external stepping.")
                yield
            self._last_graph_mode = ("mixed" if self._last_graph_error else "graph") if self._last_graph_replays else "eager"
            report = self.report.to_dict()
            report.update(execution_tier=f"native_pipeline_{self._last_graph_mode}",
                          graph=self._last_graph_mode, graph_replays=self._last_graph_replays,
                          graph_fallback_reason=self._last_graph_error)
            report.update(batch_size=self.capacity, capacity_retries=list(self._capacity_retries),
                          memory={} if self._budget is None else self._budget.stats())
            for device in report["devices"]:
                device.update(batch_size=self.capacity, graph_mode=self._last_graph_mode,
                              graph_steps=chunk_limit if self._last_graph_replays else 0)
            self.report = BatchPlanReport.from_dict(report)
            return outputs
        finally:
            for iterator in generators:
                iterator.close()

    def _export(self, lane, raw, index):
        from ..result import Result
        from .adjoint import BatchAdjointTape
        differentiable = isinstance(raw, BatchAdjointTape)
        tape = raw if differentiable else None
        if differentiable:
            raw = raw.attach()
        model = self._models[lane]
        solver = self._slot.solvers[lane]
        device = self.device if self.batch.outputs.result_device == "local" else self.batch.outputs.result_device
        memo = {}
        fields = {name: raw[name] for name in self.batch.outputs.fields if name in raw}
        if set(fields) != set(self.batch.outputs.fields):
            raise RuntimeError("Batch output plan requested unavailable field spectra.")
        destination = getattr(self.batch, "_export_pools", {}).get(device)
        with (destination.activate() if destination is not None else contextlib.nullcontext()), torch.set_grad_enabled(differentiable):
            exported_fields = _owned({name.upper(): value for name, value in fields.items()}, device, memo)
            exported_monitors = _owned(raw.get("observers", {}), device, memo)
        with torch.no_grad():
            scene_snapshot = _owned(model.scene, self.device, {})
        result = Result(method="fdtd", scene=scene_snapshot, frequencies=model.frequencies,
            fields=exported_fields, monitors=exported_monitors,
            metadata={**model.metadata, "batch_index": index, "variant_id": self.batch.variant_ids[index]},
            solver_stats=model._build_fdtd_solver_stats(solver,
                time_steps=model._resolve_fdtd_time_steps(solver, model.scene),
                use_full_field_dft=bool(self.batch.outputs.fields), dft_cfg=model.config.spectral_sampler))
        result._solver_stats["batch_execution_tier"] = f"native_pipeline_{self._last_graph_mode}"
        result._solver_stats["batch_graph_replays"] = self._last_graph_replays
        result._solver_stats["batch_dft_accumulator_bytes"] = sum(
            value.numel() * value.element_size()
            for buffers in solver._dft_batched_fields.values()
            for value in buffers.values() if isinstance(value, torch.Tensor))
        if differentiable:
            result._solver_stats["batch_adjoint"] = "native_checkpoint_per_lane"
            result._solver_stats["batch_adjoint_checkpoint_bytes"] = tape.checkpoint_bytes
            result._solver_stats["batch_adjoint_checkpoints"] = len(tape.bridge._last_checkpoints)
            tape.stats = result._solver_stats
        return result

    @capture_participant
    def iter_results(self, *, parameters=None, order="completion"):
        if order not in ("completion", "input"):
            raise ValueError("order must be 'completion' or 'input'.")
        if self._closed:
            raise RuntimeError("PreparedBatch is closed.")
        parameters = self.batch.parameters if parameters is None else parameters
        self._validate_inputs(parameters)
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("PreparedBatch does not allow concurrent run or iteration.")
        try:
            # close() may have completed while input validation was running.
            if self._closed:
                raise RuntimeError("PreparedBatch is closed.")
            self._cancel.clear()
            self._records = ()
            self._record_grad = torch.is_grad_enabled()
            return BatchResultIterator(self, parameters, order=order)
        except BaseException:
            self._gate.release()
            raise

    def _iterate(self, parameters, order):
        records = {}
        try:
            if order == "input":
                # Contiguous input runs avoid buffering later groups' Results.
                groups = []
                membership = {i: g["group_id"] for g in self._groups for i in g["variant_indices"]}
                for index in range(len(self.batch)):
                    if not groups or groups[-1][0] != membership[index]:
                        groups.append((membership[index], []))
                    groups[-1][1].append(index)
            else:
                groups = [(g["group_id"], g["variant_indices"]) for g in self._groups]
            self._enter_stream()
            for group_id, indices in groups:
                start = 0
                while start < len(indices):
                    cohort = indices[start:start + self.capacity]
                    began = time.perf_counter()
                    failure = None
                    try:
                        if self._cancel.is_set():
                            raise _BatchCancelled("Batch cancellation requested.")
                        with torch.cuda.device(self.device), torch.cuda.stream(self._stream), self._allocation_scope(), torch.no_grad():
                            self._load(cohort, parameters, group_id)
                            outputs = self._solve_cohort()
                    except Exception as exc:
                        if (isinstance(exc, torch.cuda.OutOfMemoryError)
                                and (self.batch.execution.batch_size == "auto" or getattr(self.batch, "_auto_capacity", False))
                                and self.capacity > 1):
                            self._reduce_capacity()
                            continue
                        failure = exc
                        fatal_cuda = any(message in str(exc).lower() for message in (
                            "illegal memory access", "device-side assert", "unspecified launch failure"))
                        if self.batch.execution.fail_fast or isinstance(exc, _BatchCancelled) or fatal_cuda:
                            self._cancel.set()
                        # A failed solve must never be reused as a completed cohort.
                        self._loaded_key = None
                    for lane, index in enumerate(cohort):
                        item_failure = failure
                        result = None
                        if item_failure is None:
                            try:
                                with torch.cuda.device(self.device), torch.cuda.stream(self._stream), self._allocation_scope(), torch.no_grad():
                                    result = self._export(lane, outputs.pop(lane), index)
                                    self._output_ready.record(self._stream)
                                torch.cuda.current_stream(self.device).wait_event(self._output_ready)
                                for value in memo_tensors(result._fields, result._monitors):
                                    if value.is_cuda:
                                        value.record_stream(torch.cuda.current_stream(value.device))
                                # CPU exports are ready on return; Result storage is
                                # independent of every future slot reset or close.
                            except Exception as exc:
                                item_failure = exc
                                if self.batch.execution.fail_fast:
                                    self._cancel.set()
                        kind = (FailureKind.CANCELLED if isinstance(item_failure, _BatchCancelled)
                                else FailureKind.CAPACITY if isinstance(item_failure, torch.cuda.OutOfMemoryError)
                                else FailureKind.RUNTIME)
                        if item_failure is not None:
                            # Traceback frames can otherwise retain an entire slot
                            # through a long-lived failure result after close().
                            item_failure = item_failure.with_traceback(None)
                            item_failure.__context__ = None
                        failed = None if item_failure is None else DistributedFailure(index=index,
                            device=self.device, kind=kind, exception=item_failure,
                            message=f"variant {self.batch.variant_ids[index]!r}: {type(item_failure).__name__}: {item_failure}")
                        record = ExecutionRecord(index=index, device=self.device,
                            status="completed" if failed is None else "cancelled" if kind == FailureKind.CANCELLED else "failed",
                            wall_time_s=time.perf_counter() - began, failure=failed)
                        records[index] = record
                        self._records = tuple(records[i] for i in sorted(records))
                        yield BatchItem(index=index, variant_id=self.batch.variant_ids[index],
                                        result=result, failure=failed, record=record)
                        del result
                    start += len(cohort)
        finally:
            self._finish_records(records)

    def _finish_records(self, records):
        for index in range(len(self.batch)):
            if index not in records:
                failure = DistributedFailure(index=index, device=self.device, kind=FailureKind.CANCELLED,
                                             message="Batch iterator closed before this variant completed.")
                records[index] = ExecutionRecord(index=index, device=self.device, status="cancelled", failure=failure)
        self._records = tuple(records[i] for i in sorted(records))

        if self._budget is not None:
            report = self.report.to_dict()
            report.update(memory=self._budget.stats(), capacity_retries=list(self._capacity_retries))
            self.report = BatchPlanReport.from_dict(report)

    def run(self, *, parameters=None):
        entries = [None] * len(self.batch)
        with self.iter_results(parameters=parameters) as iterator:
            for item in iterator:
                entries[item.index] = item.failure if item.failure is not None else item.result
        return BatchResult(entries, self._records, self.batch.variant_ids)

    @property
    def records(self):
        return self._records

    def request_cancel(self):
        self._cancel.set()

    @capture_participant
    def close(self):
        if self._closed:
            return
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("Close the active batch iterator before closing PreparedBatch.")
        try:
            self._stream.synchronize()
            self._slot = None
            self._models = ()
            self._fixed_material_cache.clear()
            self._local_material_cache = None
            self._loaded_key = None
            self._loaded_parameters = ()
            self._sources = None
            self._observers = None
            self._graphs.clear()
            self._budget = None
            self._closed = True
        finally:
            self._gate.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("PreparedBatch is closed.")
        return self

    def __exit__(self, *exc):
        self.close()


class BatchResultIterator:
    """Demand-driven result delivery; no unbounded completed-result queue."""

    def __init__(self, prepared, parameters, *, order):
        self.prepared = prepared
        self._generator = prepared._iterate(parameters, order)
        self._closed = False
        self._owns_prepared = False
        self._final_records = ()
        self._iteration_gate = threading.Lock()

    def __iter__(self):
        return self

    @capture_participant
    def __next__(self):
        if not self._iteration_gate.acquire(blocking=False):
            raise RuntimeError("BatchResultIterator.next() is already active.")
        try:
            if self._closed:
                raise StopIteration
            try:
                return next(self._generator)
            except BaseException:
                self._close_idle_generator()
                raise
        finally:
            self._iteration_gate.release()

    @property
    def records(self):
        return self._final_records if self._closed else self.prepared.records

    def close(self):
        if not self._iteration_gate.acquire(blocking=False):
            raise RuntimeError("Cannot close while next() is active; request_cancel(), then close after it returns.")
        try:
            self._close_idle_generator()
        finally:
            self._iteration_gate.release()

    def _close_idle_generator(self):
        if self._closed:
            return
        # Release ownership only after the generator has actually stopped.
        self._generator.close()
        self.prepared._finish_records({r.index: r for r in self.prepared.records})
        self._final_records = self.prepared.records
        self._closed = True
        self.prepared._gate.release()
        if self._owns_prepared:
            self.prepared.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ParallelPreparedBatch(PreparedBatch):
    """Persistent per-device slots with shared budgets and bounded output delivery."""

    @capture_participant
    def __init__(self, batch):
        from .pool import BatchMemoryPool
        self.batch = batch
        if batch.execution.tune:
            raise ValueError("Resolve measured tuning through BatchSimulation.prepare().")
        self._validate_inputs(batch.parameters)
        self.device = batch.execution.devices[0]
        self._gate = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False
        self._records = ()
        self._groups = batch.plan().groups
        self._children = []
        self._accounts = {}
        self._budget = None
        self._capacity_retries = []
        slots = batch.execution.slots_per_device
        if slots not in (1, 2):
            raise NotImplementedError("Automatic slot selection requires measured throughput tuning.")
        if batch.outputs.result_device not in ("local", "cpu", *batch.execution.devices):
            raise ValueError("A parallel Batch export device must belong to its execution devices.")
        try:
            for event in self._input_events(batch.parameters):
                event.synchronize()
            for device in batch.execution.devices:
                free, _ = torch.cuda.mem_get_info(device)
                requested = batch.execution.memory_limit_bytes
                if isinstance(requested, Mapping):
                    requested = requested[device]
                limit = max(0, int(free) - batch.execution.reserve_bytes)
                if requested is not None:
                    limit = min(limit, requested)
                self._accounts[device] = BatchMemoryPool(device, limit, batch.execution.reserve_bytes)
            for device in batch.execution.devices:
                for _ in range(slots):
                    child_batch = copy.copy(batch)
                    child_batch.execution = replace(batch.execution, devices=(device,), slots_per_device=1,
                        memory_limit_bytes=self._accounts[device].limit_bytes)
                    child_batch._memory_account = self._accounts[device]
                    child_batch._budget_slots = slots
                    child_batch._export_pools = {}
                    target = batch.outputs.result_device
                    if target not in ("local", "cpu", device):
                        account = self._accounts[target]
                        child_batch._export_pools[target] = BatchMemoryPool(target, account.limit_bytes,
                            batch.execution.reserve_bytes, account=account)
                    self._children.append(PreparedBatch(child_batch))
            self.capacity = sum(child.capacity for child in self._children)
            self._refresh_report()
        except BaseException:
            self.close()
            raise

    def _input_events(self, parameters):
        devices = {str(model.scene.device) for model in self.batch.simulations}
        devices.update(self.batch.execution.devices)
        if parameters is not None:
            devices.update(str(value.device) for kind in ("region_densities", "source_scales")
                           for value in getattr(parameters, kind).values() if value.is_cuda)
        events = []
        for device in sorted(devices):
            if torch.device(device).type == "cuda":
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    events.append(event)
        return events

    def _refresh_report(self):
        report = self.batch.plan().to_dict()
        report.update(execution_tier="native_parallel_pipeline", slots_per_device=self.batch.execution.slots_per_device,
                      total_capacity=sum(child.capacity for child in self._children))
        report["devices"] = [{"device": device, "slots": sum(c.device == device for c in self._children),
            "batch_sizes": [c.capacity for c in self._children if c.device == device],
            "memory": account.stats()} for device, account in self._accounts.items()]
        report["workers"] = [child.report.to_dict() for child in self._children]
        report.update(getattr(self, "_tuning_report", {}))
        self.report = BatchPlanReport.from_dict(report)

    @capture_participant
    def iter_results(self, *, parameters=None, order="completion"):
        if order not in ("completion", "input"):
            raise ValueError("order must be 'completion' or 'input'.")
        if self._closed:
            raise RuntimeError("PreparedBatch is closed.")
        parameters = self.batch.parameters if parameters is None else parameters
        self._validate_inputs(parameters)
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("PreparedBatch does not allow concurrent run or iteration.")
        try:
            if self._closed:
                raise RuntimeError("PreparedBatch is closed.")
            self._cancel.clear()
            self._records = ()
            events = self._input_events(parameters)
            # Source-scale materialization reads scalar values on the host.
            # Complete caller writes before a different worker thread reads them.
            for event in events:
                event.synchronize()
            for child in self._children:
                child._record_grad = torch.is_grad_enabled()
                child._cancel.clear()
                child._enter_stream()
                for event in events:
                    child._stream.wait_event(event)
            return ParallelBatchResultIterator(self, parameters, order=order)
        except BaseException:
            self._gate.release()
            raise

    def request_cancel(self):
        self._cancel.set()
        for child in self._children:
            child.request_cancel()

    def close(self):
        if self._closed:
            return
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("Close the active batch iterator before closing PreparedBatch.")
        try:
            for child in self._children:
                child.close()
            self._children.clear()
            self._accounts.clear()
            self._closed = True
        finally:
            self._gate.release()


class ParallelBatchResultIterator(BatchResultIterator):
    def __init__(self, prepared, parameters, *, order):
        self.prepared, self.parameters, self.order = prepared, parameters, order
        self._closed = self._owns_prepared = False
        self._iteration_gate = threading.Lock()
        self._condition = threading.Condition()
        self._queue = queue.Queue(maxsize=prepared.batch.outputs.max_pending_results)
        self._permits = threading.Semaphore(prepared.batch.outputs.max_pending_results)
        self._delivered = set()
        self._records = {}
        self._next_export = 0
        self._running = len(prepared._children)
        membership = {i: g["group_id"] for g in prepared._groups for i in g["variant_indices"]}
        groups = []
        sequence = (list(range(len(prepared.batch))) if order == "input" else
                    [i for group in prepared._groups for i in group["variant_indices"]])
        for index in sequence:
            if not groups or groups[-1][0] != membership[index]:
                groups.append((membership[index], deque()))
            groups[-1][1].append(index)
        self._tasks = deque(groups)
        self._executor = ThreadPoolExecutor(max_workers=self._running, thread_name_prefix="maxwell-batch")
        self._futures = None

    def _take_task(self, child, worker):
        with self._condition:
            if self.prepared._cancel.is_set() or not self._tasks:
                return None
            # Static scheduling advances the assignment cursor in worker order.
            if self.prepared.batch.execution.scheduling == "static":
                while getattr(self, "_assignment", 0) % len(self.prepared._children) != worker:
                    if self.prepared._cancel.is_set() or not self._tasks:
                        return None
                    self._condition.wait(.05)
                # The previous worker may have consumed the last task while
                # this worker slept, including when its turn now matches.
                if self.prepared._cancel.is_set() or not self._tasks:
                    return None
                self._assignment = getattr(self, "_assignment", 0) + 1
            group, pending = self._tasks[0]
            indices = [pending.popleft() for _ in range(min(child.capacity, len(pending)))]
            if not pending:
                self._tasks.popleft()
            self._condition.notify_all()
            return group, indices

    @contextlib.contextmanager
    def _phase(self, child):
        from ..fdtd.cuda.runtime.graph import coordinated_cuda_work
        with coordinated_cuda_work(), torch.cuda.device(child.device), torch.cuda.stream(child._stream), child._allocation_scope(), torch.no_grad():
            yield

    def _permission(self, index):
        with self._condition:
            while self.order == "input" and index != self._next_export:
                if self.prepared._cancel.is_set():
                    return False
                self._condition.wait(.05)
        while not self.prepared._cancel.is_set():
            if self._permits.acquire(timeout=.05):
                return True
        return False

    def _failure(self, child, index, error, began):
        kind = (FailureKind.CAPACITY if isinstance(error, torch.cuda.OutOfMemoryError) else
                FailureKind.CANCELLED if isinstance(error, _BatchCancelled) else FailureKind.RUNTIME)
        error = error.with_traceback(None)
        error.__context__ = None
        failure = DistributedFailure(index=index, device=child.device, kind=kind,
            exception=error, message=f"variant {self.prepared.batch.variant_ids[index]!r}: {error}")
        record = ExecutionRecord(index=index, device=child.device,
            status="cancelled" if kind == FailureKind.CANCELLED else "failed",
            wall_time_s=time.perf_counter() - began, failure=failure)
        return BatchItem(index=index, variant_id=self.prepared.batch.variant_ids[index], result=None, failure=failure, record=record)

    def _worker(self, child, worker):
        assigned = []
        began = time.perf_counter()
        try:
            while (task := self._take_task(child, worker)) is not None:
                group, assigned = task
                start = 0
                while start < len(assigned) and not self.prepared._cancel.is_set():
                    indices = assigned[start:start + child.capacity]
                    began = time.perf_counter()
                    lifecycle = None
                    error = None
                    try:
                        with self._phase(child):
                            child._load(indices, self.parameters, group)
                            lifecycle = child._cohort_steps()
                        while True:
                            with self._phase(child):
                                try:
                                    next(lifecycle)
                                except StopIteration as done:
                                    outputs = done.value
                                    break
                    except Exception as exc:
                        if (isinstance(exc, torch.cuda.OutOfMemoryError)
                                and (child.batch.execution.batch_size == "auto" or getattr(child.batch, "_auto_capacity", False))
                                and child.capacity > 1):
                            with self._phase(child):
                                if lifecycle is not None:
                                    lifecycle.close()
                                child._reduce_capacity()
                            continue
                        error = exc
                        child._loaded_key = None
                    finally:
                        if lifecycle is not None:
                            with self._phase(child):
                                lifecycle.close()
                    for lane, index in enumerate(indices):
                        if error is not None:
                            item = self._failure(child, index, error, began)
                            with self._condition:
                                self._records[index] = item.record
                            if self.prepared.batch.execution.fail_fast:
                                self.prepared.request_cancel()
                                break
                        if not self._permission(index):
                            break
                        events = []
                        try:
                            if error is None:
                                with self._phase(child):
                                    result = child._export(lane, outputs.pop(lane), index)
                                    devices = {str(t.device) for t in memo_tensors(result._fields, result._monitors) if t.is_cuda}
                                    for device in devices:
                                        event = torch.cuda.Event()
                                        event.record(torch.cuda.current_stream(device))
                                        events.append((device, event))
                                    record = ExecutionRecord(index=index, device=child.device, status="completed",
                                        wall_time_s=time.perf_counter() - began)
                                    item = BatchItem(index=index, variant_id=self.prepared.batch.variant_ids[index], result=result, failure=None, record=record)
                        except Exception as exc:
                            item = self._failure(child, index, exc, began)
                        with self._condition:
                            self._records[index] = item.record
                            self._queue.put((item, events))
                            if self.order == "input":
                                self._next_export += 1
                            self._condition.notify_all()
                        if item.failure is not None and self.prepared.batch.execution.fail_fast:
                            self.prepared.request_cancel()
                    with self._phase(child):
                        if error is None:
                            outputs.clear()
                    start += len(indices)
        except BaseException as exc:
            with self._condition:
                for index in assigned:
                    if index not in self._records:
                        self._records[index] = self._failure(child, index, exc, began).record
            self.prepared.request_cancel()
        finally:
            with self._condition:
                self._running -= 1
                self._condition.notify_all()

    def __next__(self):
        from ..fdtd.cuda.runtime.graph import coordinated_cuda_work
        if not self._iteration_gate.acquire(blocking=False):
            raise RuntimeError("BatchResultIterator.next() is already active.")
        try:
            if self._closed:
                raise StopIteration
            if self._futures is None:
                self._futures = [self._executor.submit(self._worker, child, worker)
                                 for worker, child in enumerate(self.prepared._children)]
            while True:
                try:
                    item, events = self._queue.get(timeout=.05)
                    self._permits.release()
                    with coordinated_cuda_work():
                        for device, event in events:
                            torch.cuda.current_stream(device).wait_event(event)
                        if item.result is not None:
                            for value in memo_tensors(item.result._fields, item.result._monitors):
                                if value.is_cuda:
                                    value.record_stream(torch.cuda.current_stream(value.device))
                    self._delivered.add(item.index)
                    return item
                except queue.Empty:
                    with self._condition:
                        # A producer can publish its final item and retire
                        # between get() timing out and this completion check.
                        if self._running or not self._queue.empty():
                            continue
                    for index in range(len(self.prepared.batch)):
                        if index not in self._delivered:
                            record = self._records.get(index)
                            if record is None:
                                failure = DistributedFailure(index=index, device=self.prepared.device,
                                    kind=FailureKind.CANCELLED, message="Batch cancelled before publication.")
                                record = ExecutionRecord(index=index, device=self.prepared.device, status="cancelled", failure=failure)
                                self._records[index] = record
                            self._delivered.add(index)
                            return BatchItem(index=index, variant_id=self.prepared.batch.variant_ids[index], result=None, failure=record.failure, record=record)
                    self._close_idle_generator()
                    raise StopIteration
        finally:
            self._iteration_gate.release()

    @property
    def records(self):
        return self._final_records if self._closed else tuple(self._records[i] for i in sorted(self._records))

    def _close_idle_generator(self):
        if self._closed:
            return
        self.prepared.request_cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        while not self._queue.empty():
            self._queue.get_nowait()
        self.prepared._finish_records(dict(self._records))
        self.prepared._refresh_report()
        self._final_records = self.prepared.records
        self._closed = True
        self.prepared._gate.release()
        if self._owns_prepared:
            self.prepared.close()


def memo_tensors(*values):
    """Visit output tensors for consumer-stream allocator lifetime tracking."""
    for value in values:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, Mapping):
            yield from memo_tensors(*value.values())
        elif isinstance(value, (list, tuple)):
            yield from memo_tensors(*value)
