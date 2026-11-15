from __future__ import annotations

import torch


def get_fdtd_module():
    from ..cuda.backend import get_native_fdtd_module

    return get_native_fdtd_module()


def require_cuda_scene(scene):
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(f"FDTD requires scene.device to be CUDA, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FDTD requires CUDA, but torch.cuda.is_available() is False in this process. "
            "Check CUDA_VISIBLE_DEVICES and container or sandbox device access as well "
            "as the host driver."
        )
