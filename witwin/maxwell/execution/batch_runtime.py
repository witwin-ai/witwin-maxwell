"""Bounded cooperative execution of native FDTD field cohorts."""
from __future__ import annotations

import copy
import contextlib
import threading
import time
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
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.to(device=device, copy=True)
        return memo[id(value)]
    if isinstance(value, Mapping):
        return {key: _owned(item, device, memo) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_owned(item, device, memo) for item in value)
    if isinstance(value, list):
        return [_owned(item, device, memo) for item in value]
    return value


class PreparedBatch:
    """Reusable forward context with one persistent native CUDA field slot.

    The current execution tier has one device and one slot. Complete native
    timesteps support eager dispatch or coordinated CUDA graph replay.
    """

    @capture_participant
    def __init__(self, batch):
        self.batch = batch
        policy = batch.execution
        if len(policy.devices) != 1 or policy.slots_per_device != 1:
            raise NotImplementedError("PreparedBatch currently requires one GPU and one execution slot.")
        if policy.tune:
            raise NotImplementedError("Measured batch autotuning is not implemented yet.")
        if batch.outputs.materials or any(name.startswith("H") for name in batch.outputs.fields):
            raise NotImplementedError("Batch material export and magnetic full-field spectra are not implemented yet.")
        if batch.outputs.result_device not in ("local", "cpu", policy.devices[0]):
            raise NotImplementedError("Cross-device batch result export requires the pending multi-GPU scheduler.")
        self._validate_inputs(batch.parameters)
        if batch.variation is not None and batch.variation.structures:
            raise NotImplementedError(
                "Structure variation envelopes require compiled influence validation. "
                "Omit the envelope declaration to use full-scene cohort compilation."
            )
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
        if policy.memory_limit_bytes is not None or policy.batch_size == "auto":
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
                                   max(1, self._budget.limit_bytes // (2 * lane_peak)))
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
            "Fields, CPML, source patches and spectra use the native batch pipeline; scene compilation remains per variant.",
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
            if parameters.requires_grad:
                raise NotImplementedError("Batch adjoint execution is not implemented yet; trainable inputs are rejected.")
        from ..sources import PointDipole
        from ..monitors import PointMonitor, PlaneMonitor, FluxMonitor
        for model in batch.simulations:
            checked = copy.copy(model)
            checked._refresh_trainable_parameters()
            if checked.has_trainable_parameters:
                raise NotImplementedError("Batch adjoint execution is not implemented yet; trainable inputs are rejected.")
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
        inputs = () if parameters is None else tuple(
            value for kind in ("region_densities", "source_scales")
            for value in getattr(parameters, kind).values())
        key = (tuple(indices), None if parameters is None else tuple(
            (kind, name, id(value), object() if torch.is_inference(value) else value._version)
            for kind in ("region_densities", "source_scales")
            for name, value in getattr(parameters, kind).items()))
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
        for index in indices:
            free, _ = torch.cuda.mem_get_info(self.device)
            if free <= self.batch.execution.reserve_bytes:
                raise torch.cuda.OutOfMemoryError("Insufficient free device memory above BatchExecution.reserve_bytes.")
            model = copy.copy(self.batch.template or self.batch.simulations[index])
            model.config = copy.deepcopy(model.config)
            scene = materializer.scene(index).clone(device=self.device)
            if self.batch.outputs.monitors != "all":
                selected = set(self.batch.outputs.monitors)
                scene.monitors = [m for m in scene.monitors if m.name in selected]
            model.scene = scene
            model.scene_input = scene
            model.scene_module = None
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
            elif self.batch.template is not None and parameters.region_densities:
                from ..compiler.materials import LocalMaterialCache, PerturbationMedium
                if not any(isinstance(s.material, PerturbationMedium) for s in compiled_scene.structures):
                    if self._local_material_cache is None:
                        self._local_material_cache = LocalMaterialCache(compiled_scene, parameters.region_densities)
                        material_model = self._local_material_cache.base
                    else:
                        material_model = self._local_material_cache.compile(compiled_scene)
                    compiled_scene._material_model_override = material_model
            solver = model._build_fdtd_solver_for_scene(compiled_scene, initialize=True)
            solver.verbose = False
            validate_native_solver(solver)
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
        generators, pending, outputs = [], {}, {}
        try:
            for lane, (model, solver) in enumerate(zip(self._models, slot.solvers)):
                cfg = model.config
                steps = model._resolve_fdtd_time_steps(solver, model.scene)
                solver._cuda_graph_active = False
                iterator = solve_steps(solver, time_steps=steps, dft_frequency=model.frequencies,
                    dft_window=cfg.spectral_sampler.window, full_field_dft=bool(self.batch.outputs.fields),
                    normalize_source=cfg.spectral_sampler.normalize_source, shutoff=cfg.shutoff,
                    shutoff_check_interval=cfg.shutoff_check_interval, cooperative=True,
                    cooperative_observer=(observers, lane), cooperative_driver=True)
                generators.append(iterator)
                descriptor = next(iterator)
                if solver.dft_enabled and not descriptor["gpu_dft"]:
                    raise NotImplementedError("Batch full-field spectra require GPU-driven accumulation.")
                pending[lane] = (steps, descriptor["shutoff_min_step"])
            observers.bind()
            from ..fdtd.cuda.runtime.graph import CudaGraphRunner
            policy = self.batch.execution
            graph_enabled = policy.graph != "off"
            chunk_limit = 32 if policy.graph_steps == "auto" else policy.graph_steps
            # Ordinary full-field preparation owns fresh DFT buffers each run.
            # Monitor-only storage is persistent and can reuse captures.
            if self.batch.outputs.fields:
                self._graphs.clear()
            self._last_graph_replays = 0
            self._last_graph_error = None
            completed = 0
            while pending:
                if self._cancel.is_set():
                    raise _BatchCancelled("Batch cancellation requested.")
                deadline = min(value[0] for value in pending.values())
                for lane in pending:
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
                                                     memory_pool=self._budget).capture(advance_chunk)
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
                        continue
                    try:
                        generators[lane].send(completed)
                    except StopIteration as done:
                        outputs[lane] = done.value or {}
                        del pending[lane]
                        slot.active[lane] = 0
                    else:
                        raise RuntimeError("Batch lifecycle did not finalize after external stepping.")
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
        model = self._models[lane]
        solver = self._slot.solvers[lane]
        device = self.device if self.batch.outputs.result_device == "local" else self.batch.outputs.result_device
        memo = {}
        fields = {name: raw[name] for name in self.batch.outputs.fields if name in raw}
        if set(fields) != set(self.batch.outputs.fields):
            raise RuntimeError("Batch output plan requested unavailable field spectra.")
        result = Result(method="fdtd", scene=model.scene, frequencies=model.frequencies,
            fields=_owned(fields, device, memo), monitors=_owned(raw.get("observers", {}), device, memo),
            metadata={**model.metadata, "batch_index": index, "variant_id": self.batch.variant_ids[index]},
            solver_stats=model._build_fdtd_solver_stats(solver,
                time_steps=model._resolve_fdtd_time_steps(solver, model.scene),
                use_full_field_dft=bool(self.batch.outputs.fields), dft_cfg=model.config.spectral_sampler))
        result._solver_stats["batch_execution_tier"] = f"native_pipeline_{self._last_graph_mode}"
        result._solver_stats["batch_graph_replays"] = self._last_graph_replays
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
                                and self.batch.execution.batch_size == "auto" and self.capacity > 1):
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


def memo_tensors(*values):
    """Visit output tensors for consumer-stream allocator lifetime tracking."""
    for value in values:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, Mapping):
            yield from memo_tensors(*value.values())
        elif isinstance(value, (list, tuple)):
            yield from memo_tensors(*value)
