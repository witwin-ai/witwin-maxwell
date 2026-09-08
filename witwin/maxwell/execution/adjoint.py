"""Owned batch tapes feeding the native checkpoint adjoint."""
from __future__ import annotations

import contextlib
import copy

import torch


def validate_material_channels(scene):
    """Reject trainable channels absent from the native material pullback."""
    def trainable_geometry(geometry):
        return any(isinstance(value, torch.Tensor) and value.requires_grad
                   for value in vars(geometry).values())
    for region in scene.material_regions:
        if trainable_geometry(region.geometry):
            raise NotImplementedError("Batch adjoint does not register material-region geometry inputs; use density parameters.")
        if region.density.requires_grad and region.mu_bounds[0] != region.mu_bounds[1]:
            raise NotImplementedError("Batch adjoint has no material magnetic-permeability gradient channel.")
    if any(trainable_geometry(structure.geometry) for structure in scene.structures):
        for structure in scene.structures:
            material = structure.material
            if (float(getattr(material, "mu_r", 1.)) != 1.
                    or float(getattr(material, "sigma_e", 0.)) != 0.
                    or float(getattr(material, "sigma_m", 0.)) != 0.
                    or getattr(material, "mu_tensor", None) is not None
                    or getattr(material, "sigma_e_tensor", None) is not None
                    or getattr(material, "is_pec", False)):
                raise NotImplementedError(
                    "Batch geometry adjoint requires nonmagnetic, nonconductive dielectric structures; "
                    "magnetic, conductivity and PEC occupancy gradient channels are not available.")


def source_scale_pullback(bridge, inputs):
    """Pull accumulated source injection cotangents to the input frontier."""
    solver = bridge._last_solver
    values = bridge.simulation._batch_source_scales
    cotangents = {index: value for index, value in solver._batch_source_scale_gradients.items()}
    # Spectral normalization uses unit amplitude; physical source scales do
    # not change its denominator, including at zero scale.
    gradients = torch.autograd.grad(tuple(value[0] for value in values.values()), inputs,
        grad_outputs=tuple(cotangents[index] for index in values), allow_unused=True, retain_graph=True)
    return tuple(torch.zeros_like(value) if gradient is None else gradient
                 for value, gradient in zip(inputs, gradients))


def freeze_solver(solver):
    """Detach mutable runtime tensors from every persistent execution buffer."""
    memo = {id(solver.scene): solver.scene,
            id(solver._compiled_material_model): solver._compiled_material_model}
    def share_material(value):
        if isinstance(value, torch.Tensor):
            memo[id(value)] = value
        elif isinstance(value, dict):
            for item in value.values():
                share_material(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                share_material(item)
    share_material(solver._compiled_material_model)
    # Entries and primary DFT attributes are views of the frequency banks.
    # Clone each bank once and rebuild its views, rather than retaining a
    # second complete spectrum in the tape through per-frequency clones.
    for component, buffers in getattr(solver, "_dft_batched_fields", {}).items():
        for part, bank in buffers.items():
            if bank is None:
                continue
            frozen_bank = bank.detach().clone()
            memo[id(bank)] = frozen_bank
            for index, entry in enumerate(solver._dft_entries):
                memo[id(entry["fields"][component][part])] = frozen_bank[index]

    def freeze(value):
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, torch.Tensor):
            result = value.detach().clone()
        elif isinstance(value, dict):
            result = {}
            memo[id(value)] = result
            result.update((key, freeze(item)) for key, item in value.items())
        elif isinstance(value, list):
            result = [freeze(item) for item in value]
        elif isinstance(value, tuple):
            result = tuple(freeze(item) for item in value)
        elif hasattr(value, "__dict__") and type(value).__module__.startswith("witwin."):
            result = copy.copy(value)
            memo[id(value)] = result
            for name, item in vars(value).items():
                object.__setattr__(result, name, freeze(item))
        else:
            return value
        memo[id(value)] = result
        return result

    return freeze(solver)


class BatchAdjointTape:
    def __init__(self, bridge, solver, raw, checkpoints, steps, fields, pool):
        self.bridge = bridge
        self.pool = pool
        self.device = solver.device
        self.stats = None
        self.checkpoint_bytes = sum(value.numel()*value.element_size()
                                    for checkpoint in checkpoints for value in checkpoint.tensors.values())
        self.outputs = bridge.accept_forward(freeze_solver(solver), raw, tuple(checkpoints),
            time_steps=steps, use_full_field_dft=fields)
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(self.device))

    def attach(self):
        with torch.enable_grad():
            outputs = _BatchAdjointFunction.apply(self, *self.bridge.trainable_inputs)
            outputs = outputs if isinstance(outputs, tuple) else (outputs,)
            return self.bridge.rebuild_forward_outputs(outputs)[-1]


class _BatchAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tape, *inputs):
        ctx.tape = tape
        ctx.input_count = len(inputs)
        ctx.save_for_backward(*inputs, *getattr(tape.bridge.simulation, "_batch_input_guards", ()))
        ctx.set_materialize_grads(False)
        # Return distinct tensor objects so the tape's raw outputs never acquire
        # a grad_fn that points back to the tape itself.
        return tuple(value.view_as(value) for value in tape.outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        from ..fdtd.cuda.runtime.graph import coordinated_cuda_work
        tape = ctx.tape
        inputs = ctx.saved_tensors[:ctx.input_count]
        with coordinated_cuda_work(), torch.cuda.device(tape.device), (
                tape.pool.activate() if tape.pool is not None else contextlib.nullcontext()):
            torch.cuda.current_stream(tape.device).wait_event(tape.ready)
            try:
                gradients = tape.bridge.backward(inputs, grad_outputs)
            finally:
                if tape.stats is not None and tape.pool is not None:
                    tape.stats["batch_backward_memory"] = tape.pool.stats()
        return (None, *gradients)
