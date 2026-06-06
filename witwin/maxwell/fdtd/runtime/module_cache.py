from __future__ import annotations

import os
import shutil
import sys

import torch

import slangtorch

if os.name != "nt":
    os.environ.setdefault("CC", "gcc-10")
    os.environ.setdefault("CXX", "g++-10")

_FDTD_MODULE_CACHE = {}
_VALID_FDTD_BACKENDS = {"slang", "cuda", "auto"}


def ensure_slang_build_tools_on_path():
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if not os.path.isdir(scripts_dir):
        return

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if scripts_dir not in path_entries:
        os.environ["PATH"] = scripts_dir + os.pathsep + current_path


def resolve_fdtd_backend_name(requested: str | None = None) -> str:
    backend = (requested or os.environ.get("WITWIN_MAXWELL_FDTD_BACKEND", "slang")).strip().lower()
    if backend not in _VALID_FDTD_BACKENDS:
        choices = ", ".join(sorted(_VALID_FDTD_BACKENDS))
        raise ValueError(f"WITWIN_MAXWELL_FDTD_BACKEND must be one of: {choices}.")
    if backend == "auto":
        try:
            from ..cuda.backend import is_available

            return "cuda" if is_available() else "slang"
        except Exception:
            return "slang"
    return backend


def get_fdtd_module(slang_path):
    if resolve_fdtd_backend_name() == "cuda":
        from ..cuda.backend import get_native_fdtd_module

        return get_native_fdtd_module()

    slang_path = os.path.abspath(slang_path)
    stat = os.stat(slang_path)
    stem, ext = os.path.splitext(os.path.basename(slang_path))
    shadow_name = f".{stem}_runtime_{stat.st_mtime_ns}_{stat.st_size}{ext}"
    shadow_path = os.path.join(os.path.dirname(slang_path), shadow_name)
    if not os.path.exists(shadow_path):
        shutil.copyfile(slang_path, shadow_path)
    slang_path = shadow_path
    ensure_slang_build_tools_on_path()
    module = _FDTD_MODULE_CACHE.get(slang_path)
    if module is None:
        module = slangtorch.loadModule(slang_path)
        _FDTD_MODULE_CACHE[slang_path] = module
    return module


def require_cuda_scene(scene):
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(f"FDTD requires scene.device to be CUDA, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("FDTD requires CUDA, but torch.cuda.is_available() is False.")
