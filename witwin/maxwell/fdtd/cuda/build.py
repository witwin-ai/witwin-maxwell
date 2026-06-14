from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load


def _candidate_vcvars64_paths() -> list[Path]:
    paths: list[Path] = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            try:
                install_root = subprocess.check_output(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    text=True,
                    encoding="mbcs",
                    errors="replace",
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                install_root = ""
            if install_root:
                paths.append(Path(install_root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    for env_name in ("VSINSTALLDIR",):
        root = os.environ.get(env_name)
        if root:
            paths.append(Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
        paths.append(
            Path(program_files)
            / "Microsoft Visual Studio"
            / "2022"
            / edition
            / "VC"
            / "Auxiliary"
            / "Build"
            / "vcvars64.bat"
        )

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _load_vcvars64_environment() -> bool:
    for vcvars in _candidate_vcvars64_paths():
        if not vcvars.exists():
            continue
        fd, probe_name = tempfile.mkstemp(prefix="witwin_maxwell_vcvars_probe_", suffix=".cmd")
        os.close(fd)
        probe = Path(probe_name)
        probe.write_text(f'@echo off\ncall "{vcvars}" >nul\nset\n', encoding="utf-8")
        try:
            output = subprocess.check_output(
                ["cmd.exe", "/d", "/c", str(probe)],
                text=True,
                encoding="mbcs",
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
        updates: dict[str, str] = {}
        for line in output.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            updates[key] = value
        for key, value in updates.items():
            os.environ[key] = value
        path_value = updates.get("PATH", updates.get("Path"))
        if path_value is not None:
            os.environ["PATH"] = path_value
            os.environ["Path"] = path_value
        return True
    return False


def _ensure_windows_build_tools_on_path() -> None:
    if os.name != "nt":
        return
    if shutil.which("cl") is None:
        _load_vcvars64_environment()
    if shutil.which("cl") is None:
        return

    current_path = os.environ.get("PATH") or os.environ.get("Path") or ""
    prefixes: list[str] = []
    vc_tools = os.environ.get("VCToolsInstallDir")
    if vc_tools:
        prefixes.append(str(Path(vc_tools) / "bin" / "Hostx64" / "x64"))
    vs_install = os.environ.get("VSINSTALLDIR")
    if vs_install:
        prefixes.append(
            str(Path(vs_install) / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja")
        )
    if not prefixes:
        return
    merged_path = os.pathsep.join([*prefixes, current_path])
    os.environ["PATH"] = merged_path
    os.environ["Path"] = merged_path


def _ensure_cuda_home_from_nvcc() -> None:
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return
    cuda_home = Path(nvcc).resolve().parents[1]
    os.environ["CUDA_HOME"] = str(cuda_home)
    if os.name == "nt":
        os.environ["CUDA_PATH"] = str(cuda_home)


def source_root() -> Path:
    return Path(__file__).resolve().parent


def prebuilt_root() -> Path:
    return source_root() / "prebuilt"


def extension_suffix() -> str:
    return ".pyd" if os.name == "nt" else ".so"


def prebuilt_extension_path() -> Path:
    return prebuilt_root() / f"witwin_maxwell_fdtd_cuda{extension_suffix()}"


def extension_sources() -> list[Path]:
    root = source_root()
    return [
        root / "extension.cpp",
        root / "kernels" / "common.cu",
        root / "kernels" / "magnetic.cu",
        root / "kernels" / "electric.cu",
        root / "kernels" / "boundary.cu",
        root / "kernels" / "projection.cu",
        root / "kernels" / "dispersive.cu",
        root / "kernels" / "sources.cu",
        root / "kernels" / "spectral.cu",
        root / "kernels" / "observers.cu",
        root / "kernels" / "adjoint.cu",
    ]


def _conda_torch_ldflags() -> list[str]:
    # Conda-distributed torch keeps c10.lib/torch_cuda.lib in <env>\Library\lib
    # instead of site-packages\torch\lib, which cpp_extension does not search.
    if os.name != "nt":
        return []
    library_lib = Path(sys.prefix) / "Library" / "lib"
    if (library_lib / "c10.lib").exists():
        return [f"/LIBPATH:{library_lib}"]
    return []


def _load_extension_file(module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("witwin_maxwell_fdtd_cuda", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_packaged_prebuilt_extension():
    module_path = prebuilt_extension_path()
    if not module_path.exists():
        return None
    return _load_extension_file(module_path)


def _load_prebuilt_extension(build_directory: Path):
    # Profilers such as Nsight Systems break torch's MSVC detection in child
    # processes, which rewrites build.ninja and forces a rebuild that then
    # fails under the profiler. Loading the already-built module directly
    # avoids invoking the build toolchain entirely.
    module_path = build_directory / f"witwin_maxwell_fdtd_cuda{extension_suffix()}"
    if not module_path.exists():
        raise FileNotFoundError(
            f"WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1 but {module_path} does not exist; "
            "run a normal build first."
        )
    return _load_extension_file(module_path)


def build_extension(*, verbose: bool = False):
    root = source_root()
    default_build_directory = Path(tempfile.gettempdir()) / "witwin_maxwell_fdtd_cuda"
    build_directory = Path(os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", default_build_directory))
    if os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT") != "1":
        module = _load_packaged_prebuilt_extension()
        if module is not None:
            return module
    if os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT") == "1":
        return _load_prebuilt_extension(build_directory)
    _ensure_windows_build_tools_on_path()
    _ensure_cuda_home_from_nvcc()
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="witwin_maxwell_fdtd_cuda",
        sources=[str(path) for path in extension_sources()],
        build_directory=str(build_directory),
        extra_include_paths=[str(root), str(root / "kernels")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=_conda_torch_ldflags(),
        verbose=verbose,
    )
