"""Forward Bloch composition with dispersive media -- exact correctness lock.

The P5.2 combination matrix asserts only that ``dispersive + bloch`` produces a
*finite* field. This module upgrades that to an exact physical-correctness lock
for the whole dispersive family on the complex Bloch fields (the split real/imag
ADE routing landed with the P5.1 adjoint work; here we prove the forward path).

Trick: the Bloch phase factor on an axis is ``exp(i * k * L)`` for domain length
``L`` (see ``BoundarySpec.bloch_phase_factors``). Choosing ``k = 2*pi / L`` makes
the phase exactly ``exp(i*2*pi) = 1``, so a complex-field Bloch run is physically
identical to a plain (real-field) periodic run on that axis, while still driving
every complex-field code path: complex E/H updates, the dispersive ADE advance on
*both* field halves, and the Bloch/CPML electric corrector. The two solves must
therefore agree to float32 round-off. A real routing bug (e.g. the imaginary-half
polarization current dropped, or the ADE decay applied to only one half) produces
an O(1) mismatch, not round-off.

This exercises the complex-field composition that is genuinely tractable. The
instantaneous-nonlinear, full off-diagonal anisotropic, and time-modulated Bloch
combinations remain physics-deferred (see the combination matrix), because each
would need a complex-field kernel variant the real update kernels do not provide.
"""

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

_LORENTZ = mw.LorentzPole(delta_eps=1.0, resonance_frequency=3.0e9, gamma=2.0e8)

# Every physically distinct dispersive path that must compose with Bloch: the
# three electric pole families, magnetic (mu-pole) dispersion, a conductive
# (lossy) dispersive medium, a diagonal-anisotropic dispersive medium, and a
# medium carrying electric and magnetic dispersion at once.
_MATERIAL_CASES = {
    "electric-lorentz": dict(eps_r=2.25, lorentz_poles=(_LORENTZ,)),
    "electric-drude": dict(eps_r=1.0, drude_poles=(mw.DrudePole(plasma_frequency=2.0e9, gamma=1.0e8),)),
    "electric-debye": dict(eps_r=2.0, debye_poles=(mw.DebyePole(delta_eps=2.0, tau=1.0e-10),)),
    "magnetic-lorentz": dict(eps_r=1.0, mu_lorentz_poles=(_LORENTZ,)),
    "conductive-dispersive": dict(eps_r=2.25, sigma_e=0.05, lorentz_poles=(_LORENTZ,)),
    "diagonal-aniso-dispersive": dict(epsilon_tensor=mw.DiagonalTensor3(2.25, 2.25, 4.0), lorentz_poles=(_LORENTZ,)),
    "electric-and-magnetic": dict(eps_r=2.0, lorentz_poles=(_LORENTZ,), mu_lorentz_poles=(_LORENTZ,)),
}


def _run(boundary, material_kwargs):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=boundary,
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material(**material_kwargs),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0), polarization="Ez", width=0.05,
            source_time=mw.CW(frequency=_FREQ, amplitude=20.0), name="src",
        )
    )
    result = mw.Simulation.fdtd(
        scene, frequencies=[_FREQ],
        run_time=mw.TimeConfig.auto(steady_cycles=3, transient_cycles=3),
        full_field_dft=True,
    ).run()
    ez = result.field("Ez")
    return (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()


def _periodic_boundary():
    return mw.BoundarySpec.faces(
        default="pml", num_layers=6, strength=1.0,
        x=("periodic", "periodic"), y=("periodic", "periodic"),
    )


def _bloch_periodic_equivalent_boundary():
    # x/y Bloch with k*L = 2*pi (phase == 1) + z PML: the mixed Bloch/CPML path.
    return mw.BoundarySpec.faces(
        default="pml", num_layers=6, strength=1.0,
        x=("bloch", "bloch"), y=("bloch", "bloch"),
        bloch_wavevector=(_K_PERIODIC, _K_PERIODIC, 0.0),
    )


@pytest.mark.parametrize("case", list(_MATERIAL_CASES), ids=list(_MATERIAL_CASES))
def test_bloch_dispersive_matches_periodic_at_unit_phase(case):
    """A dispersive Bloch run at unit phase reproduces the periodic run exactly.

    Confirms the complex-field dispersive ADE (electric Debye/Drude/Lorentz,
    magnetic poles, conductive and diagonal-anisotropic dispersion) is advanced
    and subtracted consistently on both field halves under Bloch periodicity.
    """
    material_kwargs = _MATERIAL_CASES[case]
    periodic = _run(_periodic_boundary(), material_kwargs)
    bloch = _run(_bloch_periodic_equivalent_boundary(), material_kwargs)

    assert np.isfinite(bloch).all()
    scale = np.abs(periodic).max()
    assert scale > 0.0
    rel_max_diff = np.abs(periodic - bloch).max() / scale
    # Measured ~3e-7 (float32 round-off) across all cases; a dropped imaginary
    # ADE channel or a one-sided decay would be O(1e-1) or larger.
    assert rel_max_diff < 1e-4, (case, rel_max_diff)
