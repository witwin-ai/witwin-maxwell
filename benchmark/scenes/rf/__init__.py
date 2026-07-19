"""RF port validation scenes.

Every builder returns a standard ``Scene``. The validation runner
(``benchmark/rf_validation.py``) drives each scene through the public
``Scene -> Simulation -> Result`` path and compares against analytic
transmission-line / waveguide references (first-line per the S-reference-solver
policy).

Honest status (audit S1, 2026-07-18) -- these are NOT six passing wave-level
scenes:

* ``rectangular_waveguide`` -- terminated guide (walls through the PML, PML held
  fixed in metres across tiers); reaches a wave-level FDTD S-matrix; beta(omega)
  is de-embedded from the fields (NRW) with the a_passive/a_driven precondition,
  passivity and reciprocity recorded.
* ``coax_thru`` -- terminated bench (conductors through the PML); the TEM launch
  is clean (arg(S21)/L ~ k0), but the S=b/a premise is still violated because the
  passive port is re-illuminated by the far-end reflection (a_passive/a_driven
  ~ 1): the ~0.3 m TEM wavelength is large versus the PML that fits the compact
  transverse cross-section under the uniform-num_layers boundary API. Recorded as
  a wave-level gap with the measured residual; NOT a port/line impedance mismatch.
* ``microstrip_two_port`` / ``differential_pair`` -- BLOCKED: a contour-snap error
  fires first, and underneath it WaveModeSpec('tem') is categorically inapplicable
  to their inhomogeneous (substrate + air) cross-sections (a hybrid mode solve is
  required).
* ``series_parallel_rlc`` -- lumped resonance is parasitic-dominated; open gap.
* ``lumped_open_short_match`` -- broken bench (feed decoupled from load); FAIL.

The builders construct geometry and ports only; excitation, run length, and
spectral sampling are chosen by the runner so a scene can be reused across grid
tiers for the convergence report.
"""

from __future__ import annotations

from benchmark.scenes.rf.coax_thru import coax_thru_scene
from benchmark.scenes.rf.differential_pair import differential_pair_scene
from benchmark.scenes.rf.lumped_open_short_match import lumped_one_port_scene
from benchmark.scenes.rf.microstrip_two_port import microstrip_two_port_scene
from benchmark.scenes.rf.rectangular_waveguide import rectangular_waveguide_scene
from benchmark.scenes.rf.series_parallel_rlc import series_rlc_scene

__all__ = [
    "coax_thru_scene",
    "differential_pair_scene",
    "lumped_one_port_scene",
    "microstrip_two_port_scene",
    "rectangular_waveguide_scene",
    "series_rlc_scene",
]
