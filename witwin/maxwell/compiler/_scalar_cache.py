"""Compile-scoped reuse of read-only builtin scalar conversion, with no math change."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from functools import wraps
import struct
import threading

import torch


_SCALARS: ContextVar[tuple | None] = ContextVar('maxwell_compile_scalar_tensors', default=None)


def _task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def with_compile_scalar_cache(function):
    """Fresh ownership for each compile; nested/error/concurrent calls reset safely."""
    @wraps(function)
    def compiled(*args, **kwargs):
        tensors = {}
        token = _SCALARS.set((threading.get_ident(), _task(), tensors))
        try:
            return function(*args, **kwargs)
        finally:
            # An asynchronously inherited Context may outlive this synchronous
            # call. Its owner check prevents reuse; clearing also releases refs.
            tensors.clear()
            _SCALARS.reset(token)
    return compiled


def _builtin_key(value):
    kind = type(value)
    if kind is float:
        return kind, struct.pack('!d', value)
    if kind is complex:
        return kind, struct.pack('!dd', value.real, value.imag)
    if kind is int or kind is bool:
        return kind, value
    return None


def scalar_tensor(value, *, device, dtype):
    """Retain exact as_tensor behavior outside one owner/context or for nonbuiltins.

    Current material consumers read the scalar tensor without in-place mutation.
    Tensor/NumPy/sequence/custom-scalar values never enter this cache, preserving
    their original alias, dtype, conversion errors and autograd semantics.
    """
    context = _SCALARS.get()
    key = _builtin_key(value)
    if (context is None or key is None or device is None or dtype is None
            or context[0] != threading.get_ident()
            or context[1] is not _task()):
        return torch.as_tensor(value, device=device, dtype=dtype)
    actual_device = torch.device(device)
    if actual_device.type == 'cuda' and actual_device.index is None:
        actual_device = torch.device('cuda', torch.cuda.current_device())
    if actual_device.type not in ('cpu', 'cuda'):
        return torch.as_tensor(value, device=device, dtype=dtype)
    stream = (torch.cuda.current_stream(actual_device).cuda_stream
              if actual_device.type == 'cuda' else None)
    key = (actual_device, stream, dtype, torch.is_grad_enabled(), torch.is_inference_mode_enabled(), key)
    tensors = context[2]
    if key not in tensors:
        tensors[key] = torch.as_tensor(value, device=device, dtype=dtype)
    return tensors[key]
