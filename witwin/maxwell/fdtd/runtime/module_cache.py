from __future__ import annotations

import os

import torch

_VALID_FDTD_BACKENDS = {"cuda", "auto"}


def resolve_fdtd_backend_name(requested: str | None = None) -> str:
    backend = (requested or os.environ.get("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")).strip().lower()
    if backend not in _VALID_FDTD_BACKENDS:
        choices = ", ".join(sorted(_VALID_FDTD_BACKENDS))
        raise ValueError(f"WITWIN_MAXWELL_FDTD_BACKEND must be one of: {choices}.")
    if backend == "auto":
        return "cuda"
    return backend


def get_fdtd_module():
    from ..cuda.backend import get_native_fdtd_module

    return get_native_fdtd_module()


def require_cuda_scene(scene):
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(f"FDTD requires scene.device to be CUDA, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("FDTD requires CUDA, but torch.cuda.is_available() is False.")
