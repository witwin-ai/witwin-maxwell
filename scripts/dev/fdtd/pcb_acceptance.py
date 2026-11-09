"""Bind PCB acceptance evidence to actual dependencies, source and CUDA runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return {"command": list(map(str, args)), "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def package_identity(module, distribution):
    root = Path(module.__file__).resolve().parent
    sources = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
               if p.is_file() and p.suffix in {".py", ".cpp", ".cu", ".h", ".cuh"}}
    return {"distribution": distribution, "version": importlib.metadata.version(distribution),
            "module_file": str(module.__file__), "sources": sources,
            "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}


def identity():
    import witwin.core
    import witwin.maxwell
    root = Path(__file__).resolve().parents[3]
    if not torch.cuda.is_available():
        raise RuntimeError("PCB CUDA acceptance requires accessible GPU devices.")
    # Allocation and a synchronized operation exercise CUDA, beyond NVML visibility.
    probe = torch.arange(32, device="cuda", dtype=torch.float32).square().sum()
    torch.cuda.synchronize()
    if probe.item() != 10416:
        raise RuntimeError("CUDA arithmetic preflight failed.")
    toolkit = Path(os.environ.get("CUDA_HOME", "/nonexistent")) / "bin/nvcc"
    repos = {}
    for path in (root, root.parent / "witwin-core", root.parent / "maxwell-bench"):
        repos[path.name] = {"head": command(["git", "rev-parse", "HEAD"], path),
                            "working_changes": command(["git", "status", "--short"], path)}
    return {
        "schema": "pcb-acceptance-identity/1", "python": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_home": os.environ.get("CUDA_HOME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_preflight_sum": probe.item(),
        "devices": [{"name": torch.cuda.get_device_name(i),
                     "capability": torch.cuda.get_device_capability(i)}
                    for i in range(torch.cuda.device_count())],
        "toolkit": command([str(toolkit), "--version"]) if toolkit.exists() else None,
        "core": package_identity(witwin.core, "witwin"),
        "maxwell": package_identity(witwin.maxwell, "witwin-maxwell"),
        "repositories": repos,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "torch": report["torch"],
                      "core": report["core"]["module_file"],
                      "maxwell_source_sha256": report["maxwell"]["source_sha256"]}))


if __name__ == "__main__":
    main()
