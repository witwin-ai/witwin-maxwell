"""Admission for the explicit CPU reference FDTD profile.

``Simulation.fdtd(..., backend="cpu_reference")`` runs the ordinary FDTD
pipeline with the Torch kernels in :mod:`.cpu_reference` on a CPU scene. It is
a validation profile for small problems (notably the conformal thin-sheet
runtime) and is never selected implicitly: a CPU scene without the explicit
backend still fails with the original CUDA requirement. Execution modes whose
state contracts are not implemented on this profile fail here, before any
solver allocation.
"""
from __future__ import annotations

import torch


def admit_cpu_reference_scene(simulation, prepared_scene):
    device = torch.device(prepared_scene.device)
    if device.type != "cpu":
        raise ValueError("FDTD backend='cpu_reference' requires scene.device='cpu'.")
    config = simulation.config
    if config.parallel is not None:
        raise NotImplementedError("FDTD backend='cpu_reference' supports single-device execution only.")
    if simulation.has_trainable_parameters:
        raise NotImplementedError("FDTD backend='cpu_reference' is forward-only; trainable scene parameters are not supported.")
    if simulation.material_cache is not None:
        raise NotImplementedError("FDTD backend='cpu_reference' does not support FDTDMaterialCache.")
    if simulation.port_sweep is not None or simulation._waveport_excitation() is not None:
        raise NotImplementedError("FDTD backend='cpu_reference' does not support port sweeps or wave-port excitations.")
    if config.initial_condition is not None:
        raise NotImplementedError("FDTD backend='cpu_reference' does not support initial conditions.")
    from .boundary.runtime import resolve_absorber
    absorber = str(config.absorber)
    if absorber == "auto":
        absorber, _ = resolve_absorber(
            prepared_scene, absorber, complex_fields=prepared_scene.boundary.uses_kind("bloch"),
        )
    if absorber not in ("cpml", "none"):
        raise NotImplementedError("FDTD backend='cpu_reference' supports the CPML absorber only.")
    boundary = prepared_scene.boundary
    for axis in "xyz":
        if boundary.axis_kind(axis) in ("periodic", "bloch"):
            raise NotImplementedError("FDTD backend='cpu_reference' supports non-periodic Cartesian boundaries only.")
    if getattr(prepared_scene, "coordinates", None) is not None:
        raise NotImplementedError("FDTD backend='cpu_reference' supports Cartesian coordinates only.")
    prepared_scene._fdtd_cpu_reference = True
