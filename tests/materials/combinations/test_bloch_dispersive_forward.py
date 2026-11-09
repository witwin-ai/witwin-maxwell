"""Independent real-quadrature oracles for complex Bloch dispersive updates.

A unit spatial Bloch phase makes each real field bank periodic. Pure CW complex
runs carry an analytic cos(omega*t)-i*sin(omega*t) drive, so two ordinary periodic
runs (source phase zero and pi/2) independently determine the two physical time
banks and their finite-window spectrum. This remains valid before transients
decay, including electric/magnetic ADE and static conductivity. Instantaneous
nonlinearity, off-diagonal tensors and modulation retain their explicit guards.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

# Equivalence against real periodic or supercell runs needs the same absorber on both
# sides: complex fixed-k runs default to CFS-CPML, so these runs pin alpha_max = 0.
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


def _run(boundary, material_kwargs, *, phase=0.):
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
            source_time=mw.CW(frequency=_FREQ, amplitude=20.0, phase=phase), name="src",
        )
    )
    result = mw.Simulation.fdtd(
        scene, absorber="cpml", frequencies=[_FREQ],
        run_time=mw.TimeConfig.auto(steady_cycles=3, transient_cycles=3),
        full_field_dft=True, cpml_config={"alpha_max": 0.0}).run()
    ez = result.field("Ez")
    peak = (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()
    real = result.solver.Ez.detach().cpu().numpy().copy()
    imaginary = (result.solver.Ez_imag.detach().cpu().numpy().copy()
                 if getattr(result.solver, "complex_fields_enabled", False) else None)
    return peak, real, imaginary


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
    """Both analytic CW banks reproduce independent real periodic quadratures.

    Confirms the complex-field dispersive ADE (electric Debye/Drude/Lorentz,
    magnetic poles, conductive and diagonal-anisotropic dispersion) is advanced
    and subtracted consistently on both field halves under Bloch periodicity.
    """
    material_kwargs = _MATERIAL_CASES[case]
    periodic, real, _ = _run(_periodic_boundary(), material_kwargs)
    quadrature, imaginary, _ = _run(_periodic_boundary(), material_kwargs, phase=np.pi / 2)
    bloch, bloch_real, bloch_imaginary = _run(_bloch_periodic_equivalent_boundary(), material_kwargs)

    # An analytic carrier is cos(omega*t)-i*sin(omega*t). In a finite window,
    # startup transients need not be one-sided in frequency, so a single cosine
    # run is not its complex spectral reference. Two independent real solves
    # test both ADE banks, source quadratures and their Fourier normalization.
    periodic = .5 * (periodic + 1j * quadrature)

    assert np.isfinite(bloch).all()
    scale = np.abs(periodic).max()
    assert scale > 0.0
    rel_max_diff = np.abs(periodic - bloch).max() / scale
    # Measured ~3e-7 (float32 round-off) across all cases; a dropped imaginary
    # ADE channel or a one-sided decay would be O(1e-1) or larger.
    assert rel_max_diff < 1e-4, (case, rel_max_diff)
    for expected, actual in ((real, bloch_real), (imaginary, bloch_imaginary)):
        scale = np.abs(expected).max()
        assert scale > 0
        assert np.abs(expected - actual).max() / scale < 1e-4
