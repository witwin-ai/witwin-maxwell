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
  dielectric slab. This exercises the real ``Result.antenna`` NF2FF pipeline over
  a grounded-dielectric structure (the finite substrate leaves a homogeneous air
  Huygens surface). The pipeline runs end to end and returns valid ``AntennaData``
  (six faces, finite gains, positive radiated power, radiated-vs-accepted power
  closure). The matched-broadside ``TM010`` resonance and the ``D >= 5 dBi`` gate
  are a DOCUMENTED GAP: the probe on this thick finite-ground slab is
  reactance-dominated (``|Gamma| ~ 1``) and the pattern is off-broadside. Feed/
  ground redesign and the external-reference cross-check remain open items,
  deferred past stage E2c to a future stage.
"""

from __future__ import annotations

from benchmark.scenes.antenna.half_wave_dipole import half_wave_dipole_scene
from benchmark.scenes.antenna.patch import patch_antenna_scene

__all__ = [
    "half_wave_dipole_scene",
    "patch_antenna_scene",
]
