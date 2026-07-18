"""RF port wave-level validation scenes.

Every builder here returns a standard ``Scene`` that is driven by a *real* FDTD
run through the public ``Scene -> Simulation -> Result`` path. The scenes exist
so that scattering/impedance quantities extracted from the FDTD fields can be
compared against analytic transmission-line / waveguide references (the
first-line reference per the S-reference-solver policy). No scene here relies on
an algebraic identity, a symmetric fixture, or a post-processing shortcut for its
exit gate; those classifications live with the wave-level gate tests under
``tests/rf/wave_validation/``.

The builders are intentionally light: they construct geometry and ports only.
Excitation, run length, and spectral sampling are chosen by the validation
runner (``benchmark/rf_validation.py``) so that a single scene can be reused at
several grid/dt settings for the three-tier convergence report.
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
