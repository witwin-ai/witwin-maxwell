from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.cpp_extension import load

STABLE_ABI_VERSION = "stable_abi_v17"
_REQUIRED_STABLE_OPS = (
    "trilinear_border_forward",
    "trilinear_border_entries",
    "trilinear_border_backward",
    "trilinear_border_grid_backward",
    "accumulate_point_observers_indexed",
    "seed_inject_point",
    "accumulate_tfsf_sample_adjoint",
    "sample_wire_emf",
    "update_wire_state",
    "deposit_wire_current",
    "plane_flux_reduce",
    "add_table_time_shifted_source_patch",
    "advance_copper_electric",
    "begin_copper_step",
    "correct_copper_magnetic",
    "accumulate_copper_spectra",
    "thin_sheet_gather",
    "thin_sheet_scatter",
    "thin_sheet_copper_rhs",
    "thin_sheet_chord_forcing",
    "thin_sheet_copper_state",
    "thin_sheet_chord_residual",
    "thin_sheet_chord_initialize",
    "thin_sheet_chord_matvec",
    "thin_sheet_chord_direction",
    "gyromagnetic_snapshot",
    "gyromagnetic_step",
)



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
    os.environ.setdefault("VSLANG", "1033")
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
    from torch.utils import cpp_extension

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        if nvcc is not None:
            cuda_home = str(Path(nvcc).resolve().parents[1])
        else:
            cuda_home = cpp_extension.CUDA_HOME
    if not cuda_home:
        raise RuntimeError(
            "FDTD native extension compilation needs a CUDA toolkit. Set CUDA_HOME "
            "to its directory before starting Python, or put nvcc on PATH. "
            "Toolkit discovery is separate from CUDA device access."
        )
    os.environ["CUDA_HOME"] = str(cuda_home)
    # The extension module caches this path at import, before late environment setup.
    cpp_extension.CUDA_HOME = str(cuda_home)
    if os.name == "nt":
        os.environ["CUDA_PATH"] = str(cuda_home)


def source_root() -> Path:
    return Path(__file__).resolve().parent


def prebuilt_root() -> Path:
    return source_root() / "prebuilt"


def extension_suffix() -> str:
    return ".pyd" if os.name == "nt" else ".so"


def extension_sources() -> list[Path]:
    root = source_root()
    return [
        root / "extension.cpp",
        root / "kernels" / "common.cu",
        root / "kernels" / "electric.cu",
        root / "kernels" / "boundary.cu",
        root / "kernels" / "aniso.cu",
        root / "kernels" / "gyromagnetic.cu",
        root / "kernels" / "dispersive.cu",
        root / "kernels" / "sources.cu",
        root / "kernels" / "spectral.cu",
        root / "kernels" / "observers.cu",
        root / "kernels" / "wire.cu",
        root / "kernels" / "copper.cu",
        root / "kernels" / "thin_sheet.cu",
        root / "kernels" / "adjoint.cu",
        root / "kernels" / "trilinear.cu",
    ]


class NativeLibrary(NamedTuple):
    """One packaged native library: file stem and torch.ops namespace, ABI marker, sources."""

    name: str
    abi: str
    root: Path
    sources: tuple[str, ...]
    required_ops: tuple[str, ...]
    subdirectory: str = ""
    cuda_flags: tuple[str, ...] = ()


MAIN = NativeLibrary(
    "witwin_maxwell_fdtd_cuda", STABLE_ABI_VERSION, source_root(),
    tuple(str(path.relative_to(source_root())) for path in extension_sources()),
    _REQUIRED_STABLE_OPS,
)
CORE = NativeLibrary(
    "witwin_maxwell_fdtd_core", "core_abi_v15", source_root() / "core",
    ("binding.cpp", "yee.cu", "step.cu", "observers.cu"),
    ("core_launch", "core_step_launch", "core_observer_gather", "launch", "compile_delayed", "release"),
    subdirectory="core",
)
SETUP = NativeLibrary(
    "witwin_maxwell_setup", "setup_abi_v7", source_root().parents[1] / "compiler" / "accel" / "cuda",
    ("binding.cpp", "setup.cu", "poles.cu", "subpixel.cu", "coefficients.cu", "tables.cu", "mesh.cu", "bvh.cu"),
    ("sample", "poles", "distance", "polygon_distance", "coverage", "conductor_edges",
     "bulk", "window", "reconstruct", "coefficients", "hash", "gather", "remap",
     "mesh_keys", "mesh_topology", "mesh_bounds", "mesh_distance"),
    subdirectory="setup", cuda_flags=("--fmad=false",),
)
LIBRARIES = (MAIN, CORE, SETUP)


def prebuilt_extension_path(library=MAIN) -> Path:
    return prebuilt_root() / f"{library.name}{extension_suffix()}"


def prebuilt_abi_marker_path(library_path: Path | None = None, library=MAIN) -> Path:
    selected = prebuilt_extension_path(library) if library_path is None else Path(library_path)
    return selected.with_suffix(f"{selected.suffix}.abi")


def _cuda_gencode_flags() -> list[str]:
    """Translate the release architecture list directly into nvcc flags."""
    value = os.environ.get("WITWIN_CUDA_GENCODE_ARCHES")
    if not value:
        return []
    flags: list[str] = []
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        include_ptx = entry.endswith("+PTX")
        number = entry.removesuffix("+PTX").replace(".", "")
        if not number.isdigit():
            raise ValueError(f"Invalid CUDA architecture {entry!r} in WITWIN_CUDA_GENCODE_ARCHES.")
        flags.append(f"-gencode=arch=compute_{number},code=sm_{number}")
        if include_ptx:
            flags.append(f"-gencode=arch=compute_{number},code=compute_{number}")
    return flags


def _conda_torch_ldflags() -> list[str]:
    # Conda-distributed torch keeps c10.lib/torch_cuda.lib in <env>\Library\lib
    # instead of site-packages\torch\lib, which cpp_extension does not search.
    if os.name != "nt":
        return []
    library_lib = Path(sys.prefix) / "Library" / "lib"
    if (library_lib / "c10.lib").exists():
        return [f"/LIBPATH:{library_lib}"]
    return []


class _StableOpsModule:
    """Attribute-compatible view of the dispatcher operators."""

    def __init__(self, library_path: Path, namespace: str) -> None:
        # Only strings live on the instance, so copy/deepcopy of solvers that hold
        # this view behave as plain-attribute copies.
        self.__file__ = str(library_path)
        self._namespace = namespace

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def __getattr__(self, name: str):
        # copy.copy builds the instance without __init__ and probes dunders before
        # restoring __dict__; resolving them would recurse through _namespace.
        if name.startswith("__") or "_namespace" not in self.__dict__:
            raise AttributeError(name)
        return getattr(getattr(torch.ops, self._namespace), name)


def _load_extension_file(library_path: Path, library=MAIN) -> _StableOpsModule:
    torch.ops.load_library(str(library_path))
    namespace = getattr(torch.ops, library.name)
    if any(not hasattr(namespace, name) for name in library.required_ops):
        raise ImportError(f"{library_path} does not register the Stable ABI Maxwell operators.")
    return _StableOpsModule(library_path, library.name)


def _load_packaged_prebuilt_extension(library=MAIN):
    module_path = prebuilt_extension_path(library)
    if not module_path.exists():
        return None
    marker_path = prebuilt_abi_marker_path(module_path, library)
    if not marker_path.exists():
        return None
    if marker_path.read_text(encoding="utf-8").strip() != library.abi:
        return None
    return _load_extension_file(module_path, library)


def _load_prebuilt_extension(build_directory: Path, library=MAIN):
    # Profilers such as Nsight Systems break torch's MSVC detection in child
    # processes, which rewrites build.ninja and forces a rebuild that then
    # fails under the profiler. Loading the already-built module directly
    # avoids invoking the build toolchain entirely.
    module_path = build_directory / f"{library.name}{extension_suffix()}"
    if not module_path.exists():
        raise FileNotFoundError(
            f"WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1 but {module_path} does not exist; "
            "run a normal build first."
        )
    # A binary from another ABI registers operators with mismatched schemas;
    # loading it would silently pair stale kernels with current Python bindings.
    marker_path = prebuilt_abi_marker_path(module_path, library)
    found = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else None
    if found != library.abi:
        raise ImportError(
            f"WITWIN_MAXWELL_FDTD_CUDA_PREBUILT=1 but {module_path} has ABI marker "
            f"{found!r} at {marker_path}, expected {library.abi!r}; rebuild it with "
            "a normal (non-prebuilt) run of this source tree."
        )
    return _load_extension_file(module_path, library)


def build_extension(*, verbose: bool = False, library=MAIN):
    root = source_root()
    default_build_directory = Path(tempfile.gettempdir()) / "witwin_maxwell_fdtd_cuda" / library.abi
    build_directory = Path(os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR", default_build_directory)) / library.subdirectory
    if os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_SKIP_PREBUILT") != "1":
        # Missing or stale markers are rejected before loading the library so a
        # JIT rebuild remains safe. A matching marker is an ABI promise: if
        # loading or probing that binary fails, propagate the error instead of
        # trying to register the same dispatcher namespace again in-process.
        module = _load_packaged_prebuilt_extension(library)
        if module is not None:
            return module
    if os.environ.get("WITWIN_MAXWELL_FDTD_CUDA_PREBUILT") == "1":
        return _load_prebuilt_extension(build_directory, library)
    _ensure_windows_build_tools_on_path()
    _ensure_cuda_home_from_nvcc()
    build_directory.mkdir(parents=True, exist_ok=True)
    library_path = load(
        name=library.name,
        sources=[str(library.root / name) for name in library.sources],
        build_directory=str(build_directory),
        extra_include_paths=[str(library.root), str(root), str(root / "kernels")],
        extra_cflags=(
            ["/O2", "/DTORCH_TARGET_VERSION=0x020a000000000000", "/DUSE_CUDA"]
            if os.name == "nt"
            else ["-O3", "-DTORCH_TARGET_VERSION=0x020a000000000000", "-DUSE_CUDA"]
        ),
        extra_cuda_cflags=[
            "-O3",
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            "-DUSE_CUDA",
            *_cuda_gencode_flags(),
            *library.cuda_flags,
        ],
        extra_ldflags=_conda_torch_ldflags(),
        is_python_module=False,
        verbose=verbose,
    )
    # Mark every JIT build directory, including an overridden one, so a later
    # prebuilt load can verify which ABI produced the library.
    prebuilt_abi_marker_path(Path(library_path), library).write_text(f"{library.abi}\n", encoding="utf-8")
    return _StableOpsModule(Path(library_path), library.name)
