"""Torch copper step for ordinary CopperFoil sheets (test oracle).

Imaginary, paired, phase and material-ADE foils always run the Torch copper recurrence in
``witwin.maxwell.fdtd.copper_foil.advance_copper_electric``; ordinary foils get the fused
CUDA electric step prepared by ``fdtd.cuda.backend.prepare_copper_electric``. Historical
copper pins were recorded with every foil on the Torch recurrence (formerly selected by
``WITWIN_MAXWELL_COPPER_KERNEL=torch``). Skipping the fused preparation reproduces that
arithmetic exactly, on the working tree and on frozen baseline trees alike, because both
dispatch on the same ``fused_electric`` entry key.
"""
from __future__ import annotations

from witwin.maxwell.fdtd.cuda import backend


def _keep_torch_step(solver, entry):
    return None


def install(monkeypatch):
    """Keep ordinary foils on the Torch copper step for the rest of this test."""
    monkeypatch.setattr(backend, "prepare_copper_electric", _keep_torch_step)


def apply():
    """Process-wide variant for fresh-interpreter probes (no pytest monkeypatch)."""
    backend.prepare_copper_electric = _keep_torch_step
