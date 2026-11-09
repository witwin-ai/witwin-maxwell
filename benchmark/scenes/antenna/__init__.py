"""FDTD-driven antenna benchmark scenes.

Each builder returns a standard ``Scene`` carrying geometry, a lumped feed port,
and a ``ClosedSurfaceMonitor`` NF2FF box. The validation runner or test drives the
scene through the public ``Scene -> Simulation -> Result`` path and consumes the
far field through :meth:`Result.antenna` (no monkeypatch), comparing against
analytic antenna references.

* ``half_wave_dipole`` -- center-fed thin-wire half-wave dipole; wave-level PASS
  (sin^2 pattern, ~2.15 dBi directivity, radiation-resistance class, radiated vs
  accepted power closure). The input reactance carries a documented delta-gap
  feed offset.
* ``patch`` -- probe-fed rectangular microstrip patch on a finite grounded
  dielectric slab, with matched-broadside and power-closure gates near 3.2 GHz.
* ``printed_patch`` -- fixed physical PEC sheet patch, square galvanic probe and
  finite source aperture, with independently controlled lateral/substrate meshes.
  Convergence qualification is recorded separately from nominal matching.
"""

from __future__ import annotations

from benchmark.scenes.antenna.half_wave_dipole import half_wave_dipole_scene
from benchmark.scenes.antenna.patch import patch_antenna_scene
from benchmark.scenes.antenna.printed_patch import printed_patch_scene

__all__ = [
    "half_wave_dipole_scene",
    "patch_antenna_scene",
    "printed_patch_scene",
]
