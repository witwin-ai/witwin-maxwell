"""Immutable file bindings and local process evidence; no numerical execution."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time

SCHEMA = "source-bound-pytest-shards/1"
CODE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h", ".so", ".toml", ".ini", ".cfg"}
TEXT_INPUT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv", ".xml"}
IGNORE_PARTS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
RUNTIME_ENV = {"PCB_SHARD_REQUEST", "PCB_SHARD_EVENTS"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_events(path):
    if not Path(path).exists():
        return []
    snapshot = Path(path).read_bytes()
    # Parse only newline-terminated records from this single snapshot. A child
    # can finish its write between reads, or die before its last newline. Neither
    # case makes the previous valid records corrupt or the trailing bytes final.
    complete = snapshot[:snapshot.rfind(b"\n") + 1]
    return [json.loads(line) for line in complete.splitlines()]


def command(argv, cwd):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return {"argv": list(map(str, argv)), "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def source_files(source, external):
    source = Path(source)
    tracked = command(["git", "ls-files", "-z"], source)
    tracked_names = set(tracked["stdout"].split("\x00")) if tracked["returncode"] == 0 else set()
    files = set()
    # Contract manifests and historical baseline receipts are actual test inputs,
    # even though they are outside the numerical package and test directories.
    for name in tracked_names:
        path = source / name
        if path.is_file() and path.suffix in TEXT_INPUT_SUFFIXES:
            files.add(path.resolve())
    for directory in ("witwin/maxwell", "tests", "scripts", "benchmark", "docs/reference"):
        root = source / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (path.is_file() and not (set(path.parts) & IGNORE_PARTS)
                    and (path.suffix in CODE_SUFFIXES or str(path.relative_to(source)) in tracked_names
                         or directory == "docs/reference" and path.suffix in TEXT_INPUT_SUFFIXES)):
                files.add(path.resolve())
    manifest = source / "docs/reference/fdtd-capability-guards.json"
    if manifest.is_file():
        files.add(manifest.resolve())
        baseline = json.loads(manifest.read_text())["baseline"]
        for key in ("path", "detailed_path"):
            path = source / baseline[key]
            if not path.is_file():
                raise RuntimeError(f"Missing guard-census baseline input: {path}")
            files.add(path.resolve())
    for name in ("pyproject.toml", "pytest.ini", "setup.cfg", "conftest.py", "hatch_build.py"):
        path = source / name
        if path.is_file():
            files.add(path.resolve())
    for root in map(Path, external):
        if root.is_file():
            files.add(root.resolve())
        else:
            for path in root.rglob("*"):
                if path.is_file() and not (set(path.parts) & IGNORE_PARTS):
                    files.add(path.resolve())
    return files


def environment_binding(environment):
    # Shell bookkeeping is normalized; numerical/test/toolchain variables are not.
    values = {key: value for key, value in environment.items()
              if key not in RUNTIME_ENV | {"_", "SHLVL", "OLDPWD"}}
    public_names = {"PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PYTEST_ADDOPTS",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MPLCONFIGDIR", "PWD"}
    external = {}
    for name, value in values.items():
        if name.startswith("GSP_") and not name.endswith("SHA256") and Path(value).is_file():
            external[str(Path(value).resolve())] = digest(value)
    return {"sha256": object_sha(values), "key_sha256": {key: object_sha(value) for key, value in sorted(values.items())},
            "public_values": {key: values[key] for key in sorted(public_names & values.keys())},
            "GSP_input_files": external}


def file_binding(source, control, external, core_root):
    source, control, core_root = map(lambda p: Path(p).resolve(), (source, control, core_root))
    files = source_files(source, external)
    for root in (control, core_root):
        files.update(path.resolve() for path in root.rglob("*")
                     if path.is_file() and not (set(path.parts) & IGNORE_PARTS)
                     and (root == control or path.suffix in CODE_SUFFIXES))
    files.add(Path(sys.executable).resolve())
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and (Path(cuda_home) / "bin/nvcc").is_file():
        files.add((Path(cuda_home) / "bin/nvcc").resolve())
    hashes = {str(path): digest(path) for path in sorted(files)}
    head = command(["git", "rev-parse", "HEAD"], source)
    return {"source_root": str(source), "control_root": str(control), "core_root": str(core_root),
            "source_head": head["stdout"] if head["returncode"] == 0 else None,
            "files": hashes, "files_sha256": object_sha(hashes)}


def runtime_identity(request):
    # Only child processes import the numerical stack; this function does not
    # allocate or probe CUDA. Actual tests retain their normal conftest behavior.
    import pytest
    import torch
    import witwin.core as core
    import witwin.maxwell as maxwell
    source = Path(request["source_root"]).resolve()
    control = Path(request["control_root"]).resolve()
    assert Path(maxwell.__file__).resolve() == source / "witwin/maxwell/__init__.py", maxwell.__file__
    assert Path(__file__).resolve().parent == control, __file__
    core_root = Path(core.__file__).resolve().parent
    binding = file_binding(source, control, request["external_paths"], core_root)
    versions = {"python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda, "pytest": pytest.__version__,
                "core_distribution": importlib.metadata.version("witwin"),
                "maxwell_distribution": importlib.metadata.version("witwin-maxwell")}
    modules, native = {}, {}
    for name, module in sorted(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if name.startswith(("witwin.core", "witwin.maxwell", "shard_common", "shard_progress", "module_environment")):
            modules[name] = str(path)
            if path.suffix == ".py":
                expected = source / "witwin/maxwell" if name.startswith("witwin.maxwell") else (
                    core_root if name.startswith("witwin.core") else control)
                if not path.is_relative_to(expected):
                    raise RuntimeError(f"Unexpected actual import {name}: {path}")
        if path.suffix == ".so" and (name.startswith(("witwin", "torch")) or "torch_extensions" in str(path)):
            native[str(path)] = digest(path)
    # Dispatcher-only Stable ABI libraries are not Python modules.
    for location in sorted(torch.ops.loaded_libraries):
        path = Path(location).resolve()
        native[str(path)] = digest(path)
    return {"gsp_environment": {key: value for key, value in os.environ.items() if key.startswith("GSP_")},
            "binding": binding, "binding_sha256": object_sha(binding), "versions": versions,
            "actual_module_files": modules, "loaded_native_binaries": native,
            "helper_file": str(Path(__file__).resolve()), "helper_sha256": digest(__file__)}


def process_fingerprint(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": stat[0], "ppid": int(stat[1]), "pgid": int(stat[2]),
                "sid": int(stat[3]), "start_ticks": int(stat[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "argv": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def observed_live(record):
    actual = process_fingerprint(record["pid"])
    return bool(actual and actual["boot_id"] == record["boot_id"]
                and actual["start_ticks"] == record["start_ticks"] and actual["state"] not in ("Z", "X"))
