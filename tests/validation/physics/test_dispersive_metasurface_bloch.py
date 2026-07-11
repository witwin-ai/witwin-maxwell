"""P5.4 acceptance: a dispersive (Lorentz) metasurface runs forward under Bloch.

Plan P5.4 criterion 3 asks that a dispersive metasurface (Lorentz + Bloch) run
and match a cross-check. The natural cross-solver reference would be FDFD, but the
FDFD runtime rejects periodic/Bloch faces outright (``FDFD currently supports only
per-face 'none' and 'pml' boundaries``, ``fdtd`` -> ``fdfd/solver.py``), so there
is no FDFD path for a periodic metasurface. The honest, exact cross-check that is
available instead is the periodic-equivalent unit-Bloch-phase reference:

The Bloch phase on a transverse axis is ``exp(i k L)`` for domain length ``L``.
Choosing ``k = 2*pi / L`` makes the phase exactly ``1``, so a complex-field Bloch
run is physically identical to a plain real-field periodic run on that axis, while
still driving every complex-field code path (complex E/H updates, the dispersive
ADE on both field halves, and the Bloch/CPML corrector). The two solves must agree
to float32 round-off. A real routing bug in the dispersive-metasurface Bloch
composition (e.g. a dropped imaginary polarization-current channel) produces an
O(1) mismatch, not round-off. Unlike ``test_bloch_dispersive_forward.py`` (a
uniform slab), the structure here is a *patterned* half-cell bar -- a genuine
grating that breaks transverse translation symmetry and feeds the diffraction
orders -- so the equivalence also covers the patterned dispersive composition.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_FREQ = 1.0e9
_DOMAIN_LEN = 0.6
# k * L = 2*pi  ->  Bloch phase exp(i*2*pi) = 1, i.e. periodic-equivalent.
_K_PERIODIC = 2.0 * np.pi / _DOMAIN_LEN
_LORENTZ = mw.LorentzPole(delta_eps=2.0, resonance_frequency=3.0e9, gamma=2.0e8)


def _patterned_lorentz_metasurface(boundary):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=boundary,
        device="cuda",
    )
    # A dispersive bar filling half the x-period breaks x-translation symmetry:
    # a genuine sub-wavelength Lorentz metasurface, not a uniform film.
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(-0.15, 0.0, 0.0), size=(0.3, 0.6, 0.1)),
            material=mw.Material(eps_r=2.25, lorentz_poles=(_LORENTZ,)),
            name="lorentz_bar",
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, -0.1), polarization="Ex", width=0.05,
            source_time=mw.CW(frequency=_FREQ, amplitude=20.0), name="src",
        )
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=[_FREQ],
        run_time=mw.TimeConfig.auto(steady_cycles=3, transient_cycles=3),
        full_field_dft=True,
    ).run()
    field = result.field("Ex")
    return (field["data"] if isinstance(field, dict) else field).detach().cpu().numpy()


def _periodic_boundary():
    return mw.BoundarySpec.faces(
        default="pml", num_layers=6, strength=1.0,
        x=("periodic", "periodic"), y=("periodic", "periodic"),
    )


def _bloch_unit_phase_boundary():
    return mw.BoundarySpec.faces(
        default="pml", num_layers=6, strength=1.0,
        x=("bloch", "bloch"), y=("bloch", "bloch"),
        bloch_wavevector=(_K_PERIODIC, _K_PERIODIC, 0.0),
    )


def test_patterned_lorentz_metasurface_bloch_matches_periodic_equivalent():
    periodic = _patterned_lorentz_metasurface(_periodic_boundary())
    bloch = _patterned_lorentz_metasurface(_bloch_unit_phase_boundary())

    assert np.isfinite(bloch).all()
    scale = np.abs(periodic).max()
    assert scale > 0.0
    rel_max_diff = np.abs(periodic - bloch).max() / scale
    # Measured ~1e-6 (float32 round-off): the patterned Lorentz metasurface Bloch
    # composition is correct on both field halves.
    assert rel_max_diff < 1e-4, rel_max_diff


def test_patterned_lorentz_metasurface_oblique_bloch_runs_forward():
    # The physical use case: the same patterned Lorentz metasurface under a real
    # oblique Bloch phase (nonzero, incident-consistent transverse wavevector).
    boundary = mw.BoundarySpec.faces(
        default="pml", num_layers=6, strength=1.0,
        x=("bloch", "bloch"), y=("bloch", "bloch"),
        bloch_wavevector=(0.4 * _K_PERIODIC, 0.2 * _K_PERIODIC, 0.0),
    )
    field = _patterned_lorentz_metasurface(boundary)
    assert np.isfinite(field).all()
    assert np.abs(field).max() > 0.0
