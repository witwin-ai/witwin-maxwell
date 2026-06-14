from __future__ import annotations

import os
import shutil
import sys

import torch

_FDTD_MODULE_CACHE = {}
_VALID_FDTD_BACKENDS = {"slang", "cuda", "auto"}


def ensure_slang_build_tools_on_path():
    if os.name != "nt":
        os.environ.setdefault("CC", "gcc-10")
        os.environ.setdefault("CXX", "g++-10")

    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if not os.path.isdir(scripts_dir):
        return

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if scripts_dir not in path_entries:
        os.environ["PATH"] = scripts_dir + os.pathsep + current_path


def cuda_include_paths() -> list[str]:
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda_root:
        return []
    include_dir = os.path.join(cuda_root, "include")
    return [include_dir] if os.path.isdir(include_dir) else []


def current_env_library_paths() -> list[str]:
    env_root = os.path.dirname(sys.executable)
    candidates = [
        os.path.join(env_root, "Library", "lib"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


def _prepend_env_path(key: str, paths: list[str]) -> None:
    if not paths:
        return
    current = os.environ.get(key, "")
    entries = current.split(os.pathsep) if current else []
    prepend = [path for path in paths if path not in entries]
    if prepend:
        os.environ[key] = os.pathsep.join(prepend + entries)


def _cl_include_flag(path: str) -> str:
    return f'/I"{path}"' if " " in path else f"/I{path}"


def _link_libpath_flag(path: str) -> str:
    return f'/LIBPATH:"{path}"' if " " in path else f"/LIBPATH:{path}"


def ensure_cuda_build_env() -> list[str]:
    include_paths = cuda_include_paths()
    library_paths = current_env_library_paths()
    _prepend_env_path("INCLUDE", include_paths)
    _prepend_env_path("LIB", library_paths)

    cl_flags = [_cl_include_flag(path) for path in include_paths]
    current_cl = os.environ.get("CL", "")
    prepend = [flag for flag in cl_flags if flag not in current_cl]
    if prepend:
        os.environ["CL"] = " ".join(prepend + ([current_cl] if current_cl else []))

    link_flags = [_link_libpath_flag(path) for path in library_paths]
    current_link = os.environ.get("LINK", "")
    prepend_link = [flag for flag in link_flags if flag not in current_link]
    if prepend_link:
        os.environ["LINK"] = " ".join(prepend_link + ([current_link] if current_link else []))
    return include_paths


def _load_slangtorch():
    import slangtorch

    return slangtorch


def resolve_fdtd_backend_name(requested: str | None = None) -> str:
    backend = (requested or os.environ.get("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")).strip().lower()
    if backend not in _VALID_FDTD_BACKENDS:
        choices = ", ".join(sorted(_VALID_FDTD_BACKENDS))
        raise ValueError(f"WITWIN_MAXWELL_FDTD_BACKEND must be one of: {choices}.")
    if backend == "auto":
        return "cuda"
    return backend


def get_fdtd_module(slang_path):
    backend = resolve_fdtd_backend_name()
    if backend == "cuda":
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
        slangtorch = _load_slangtorch()
        module = slangtorch.loadModule(slang_path, includePaths=ensure_cuda_build_env())
        _FDTD_MODULE_CACHE[slang_path] = module
    return module


def require_cuda_scene(scene):
    device = torch.device(scene.device)
    if device.type != "cuda":
        raise ValueError(f"FDTD requires scene.device to be CUDA, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("FDTD requires CUDA, but torch.cuda.is_available() is False.")
