"""Prepare real FDTD solvers on the host for admission and validation checks.

The CUDA requirement, the kernel module lookup and the core-path activation are
stubbed, so ``Simulation.prepare()`` builds the real solver on CPU tensors without
initializing CUDA. Nothing may step such a solver. There is no CPU FDTD path: the
public preparation of a CPU scene raises.
"""
from __future__ import annotations


def stub_cuda_preparation(monkeypatch):
    for name in ("witwin.maxwell.simulation._require_cuda_device",
                 "witwin.maxwell.fdtd.solver._require_cuda_scene",
                 "witwin.maxwell.fdtd.runtime.initialization.require_cuda_scene"):
        monkeypatch.setattr(name, lambda *args, **kwargs: None)
    for name in ("witwin.maxwell.fdtd.solver._get_fdtd_module",
                 "witwin.maxwell.fdtd.runtime.initialization.get_fdtd_module"):
        monkeypatch.setattr(name, lambda: object())
    # Core activation binds CUDA launches to the prepared fields; the host solver stays unbound.
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.core_path.activate_core_path", lambda solver: None)
