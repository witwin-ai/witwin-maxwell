"""Space-time modulation composed with dispersion and instantaneous nonlinearity.

This is the electro-optic-modulator edge of the P5.2 material combination matrix:
a single Material (or a Scene) that is simultaneously time-modulated and either
dispersive (electric/magnetic poles) or nonlinear (Kerr / chi2 / TPA).

Physics of the composition (see witwin/maxwell/fdtd/runtime/materials.py and the
apply_polarization_modulated_kernel in cuda/kernels/dispersive.cu):

  D(x, t) = eps_inf(x) * m(x, t) * E + P_dispersive + P_nonlinear

with m(x, t) = 1 + amp*cos(phase)*cos(Omega t) - amp*sin(phase)*sin(Omega t). The
modulation scales the eps_inf background, so the conservative Ampere update divides
BOTH curl(H) and the ADE polarization current by the same eps_inf * m_next, and the
field-dependent nonlinear coefficients enter the same modulated E update. The tests
below prove:

  * the combination compiles/runs (guards lifted for the meaningful edges);
  * it reduces exactly to the pure-dispersive solve as the modulation depth -> 0
    (the dispersive machinery is untouched by the modulated code path);
  * it generates the omega +/- Omega sidebands of a time-modulated medium on top of
    a dispersive slab (the two features co-exist);
  * the modulation acts on eps_inf only, so a dispersive medium converts a smaller
    fraction of carrier to sidebands than a non-dispersive medium of the same static
    index (the poles are not modulated);
  * the modulated + Kerr path reduces to the pure-modulation solve as chi3 -> 0.

Physically meaningless / not-yet-composed edges (anisotropic tensor, static
conductivity) still raise a physics-worded error.
"""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene

_C0 = 299_792_458.0

_CARRIER = 1.0e9
_FMOD = 2.5e8
_FRES = 3.0e9
_DELTA_EPS = 1.0
_GAMMA = 5.0e7
_EPS_INF = 4.0


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _lorentz_poles():
    return (mw.LorentzPole(delta_eps=_DELTA_EPS, resonance_frequency=_FRES, gamma=_GAMMA),)


def _lorentz_eps_at(frequency):
    """Analytic Re{eps_Lorentz(f)} of the single test pole (below-resonance value)."""
    w = 2.0 * np.pi * frequency
    w0 = 2.0 * np.pi * _FRES
    g = 2.0 * np.pi * _GAMMA
    denom = (w0 * w0 - w * w) ** 2 + (g * w) ** 2
    return _DELTA_EPS * w0 * w0 * (w0 * w0 - w * w) / denom


# ---------------------------------------------------------------------------
# Compile-layer guard convergence (CPU)
# ---------------------------------------------------------------------------


def test_single_material_modulation_composes_dispersion_and_nonlinearity():
    modulation = mw.ModulationSpec(frequency=_FMOD, amplitude=0.25)

    electric = mw.Material(eps_r=_EPS_INF, modulation=modulation, lorentz_poles=_lorentz_poles())
    assert electric.is_modulated and electric.is_electric_dispersive

    debye = mw.Material(
        eps_r=_EPS_INF, modulation=modulation, debye_poles=(mw.DebyePole(delta_eps=1.0, tau=2.0e-10),)
    )
    assert debye.is_modulated and debye.is_electric_dispersive

    magnetic = mw.Material(
        eps_r=_EPS_INF,
        modulation=modulation,
        mu_lorentz_poles=(mw.LorentzPole(delta_eps=0.5, resonance_frequency=_FRES, gamma=_GAMMA),),
    )
    assert magnetic.is_modulated and magnetic.is_magnetic_dispersive

    kerr = mw.Material(eps_r=_EPS_INF, modulation=modulation, kerr_chi3=1.0e-18)
    assert kerr.is_modulated and kerr.is_nonlinear

    chi2 = mw.Material(
        eps_r=_EPS_INF, modulation=modulation, nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-9)
    )
    assert chi2.is_modulated and chi2.is_nonlinear


def test_single_material_modulation_still_rejects_anisotropy_and_conductivity():
    modulation = mw.ModulationSpec(frequency=_FMOD, amplitude=0.25)

    with pytest.raises(NotImplementedError, match="anisotropic"):
        mw.Material(eps_r=_EPS_INF, modulation=modulation, epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0))
    with pytest.raises(NotImplementedError, match="conductivity"):
        mw.Material(eps_r=_EPS_INF, modulation=modulation, sigma_e=1.0)


def _prepare(scene):
    return mw.Simulation.fdtd(
        scene,
        frequencies=[_CARRIER],
        run_time=mw.TimeConfig(time_steps=8),
        full_field_dft=False,
    ).prepare().solver


def test_scene_modulation_plus_dispersive_material_prepares():
    """The cross-material runtime guard is lifted: a modulated slab may share a Scene
    with a (disjoint) dispersive material and both feature flags stay enabled."""
    modulation = mw.ModulationSpec(frequency=_FMOD, amplitude=0.25)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.16, 0.16), (-0.16, 0.16))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=_CARRIER, amplitude=80.0),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(-0.1, 0.0, 0.0), size=(0.2, 0.64, 0.64)),
                material=mw.Material(eps_r=_EPS_INF, modulation=modulation),
            ),
            mw.Structure(
                geometry=Box(position=(0.2, 0.0, 0.0), size=(0.16, 0.64, 0.64)),
                material=mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=2.0e-10),
            ),
        ],
    )
    solver = _prepare(scene)
    assert solver.modulation_enabled
    assert solver.electric_dispersive_enabled


# ---------------------------------------------------------------------------
# FDTD runtime (GPU)
# ---------------------------------------------------------------------------


def _slab_scene(*, material, amplitude=80.0, time_steps):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.16, 0.16), (-0.16, 0.16))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=_CARRIER, amplitude=amplitude),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.64, 0.64)),
                material=material,
            )
        ],
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("probe", components=("Ez",), position=(0.42, 0.0, 0.0), interval=1)
    )
    return scene


def _run(material, *, amplitude=80.0, time_steps):
    scene = _slab_scene(material=material, amplitude=amplitude, time_steps=time_steps)
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_CARRIER],
        run_time=mw.TimeConfig(time_steps=time_steps),
        full_field_dft=False,
    ).run()
    trace = _to_numpy(result.monitor("probe")["field"]).astype(np.float64)
    dt = float(result.solver.dt)
    del result
    torch.cuda.empty_cache()
    return trace, dt


def _sideband_amps(trace, dt):
    """Windowed-FFT magnitudes at the carrier and the two first-order sidebands."""
    tail = trace[len(trace) // 4 :]
    window = np.hanning(tail.size)
    spectrum = np.abs(np.fft.rfft(tail * window))
    freqs = np.fft.rfftfreq(tail.size, d=dt)

    def peak(target):
        index = int(np.argmin(np.abs(freqs - target)))
        low = max(index - 2, 0)
        high = min(index + 3, spectrum.size)
        return float(spectrum[low:high].max())

    return peak(_CARRIER), peak(_CARRIER - _FMOD), peak(_CARRIER + _FMOD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulated_dispersive_reduces_to_pure_dispersive_at_zero_depth():
    """As the modulation depth -> 0 the modulated dispersive solve must reproduce the
    pure-dispersive solve. The modulated E update and the modulated ADE current
    subtraction perturb the field by O(depth), so at depth = 1e-3 the two traces
    agree to a few x 1e-3. This proves the dispersive machinery is untouched by the
    modulated code path and its 1/m_next current fold reduces correctly."""
    time_steps = 1500
    depth = 1.0e-3
    modulated, dt_m = _run(
        mw.Material(
            eps_r=_EPS_INF,
            modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=depth),
            lorentz_poles=_lorentz_poles(),
        ),
        time_steps=time_steps,
    )
    reference, dt_r = _run(
        mw.Material(eps_r=_EPS_INF, lorentz_poles=_lorentz_poles()),
        time_steps=time_steps,
    )
    # Same material apart from the modulation -> identical Courant dt, so a direct
    # time-trace comparison is well posed.
    assert dt_m == dt_r
    denom = float(np.max(np.abs(reference)))
    assert denom > 0.0
    rel_diff = float(np.max(np.abs(modulated - reference))) / denom
    # Expected O(depth); assert an order of magnitude of headroom above depth.
    assert rel_diff < 20.0 * depth, rel_diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulated_dispersive_slab_generates_sidebands():
    """A time-modulated Lorentz slab converts carrier power into omega +/- Omega
    sidebands, exactly as a non-dispersive modulated slab does, while the pole is
    present. The unmodulated Lorentz slab shows no such peaks."""
    time_steps = 4096
    modulated, dt = _run(
        mw.Material(
            eps_r=_EPS_INF,
            modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=0.25),
            lorentz_poles=_lorentz_poles(),
        ),
        time_steps=time_steps,
    )
    reference, dt_ref = _run(
        mw.Material(eps_r=_EPS_INF, lorentz_poles=_lorentz_poles()),
        time_steps=time_steps,
    )
    assert dt == dt_ref

    carrier, lower, upper = _sideband_amps(modulated, dt)
    ref_carrier, ref_lower, ref_upper = _sideband_amps(reference, dt_ref)

    assert carrier > 0.0 and ref_carrier > 0.0
    # Sidebands are a substantial fraction of the carrier (amplitude spectrum).
    assert lower > 3.0e-2 * carrier
    assert upper > 3.0e-2 * carrier
    # The unmodulated dispersive reference has no sideband content at those bins.
    assert lower > 10.0 * ref_lower
    assert upper > 10.0 * ref_upper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulation_acts_on_eps_inf_only_not_the_dispersive_poles():
    """The modulation scales eps_inf; the dispersive polarization is unmodulated (it
    enters the update as an additive current divided by the SAME eps_inf * m_next).
    A Lorentz medium therefore modulates only the eps_inf fraction of its index, so
    it converts a smaller fraction of carrier to sidebands than a non-dispersive
    medium of the SAME static index n0 = sqrt(eps_inf + eps_pole(f0)).

    First-order phase-modulation theory gives the sideband/carrier ratio proportional
    to the modulated index amplitude:
        Lorentz:        dn_L propto eps_inf * delta / (2 n0)
        non-dispersive: dn_N propto (eps_inf + eps_pole) * delta / (2 n0)
    so ratio_L / ratio_N ~ eps_inf / (eps_inf + eps_pole) < 1. The exact FDTD value
    departs from this quasi-static estimate because the pole is driven by the
    modulated field, so this asserts the robust one-sided inequality (poles present
    but not modulated) rather than the estimate itself."""
    time_steps = 4096
    eps_pole = _lorentz_eps_at(_CARRIER)
    predicted_ratio = _EPS_INF / (_EPS_INF + eps_pole)
    assert 0.0 < predicted_ratio < 1.0

    lorentz, dt_l = _run(
        mw.Material(
            eps_r=_EPS_INF,
            modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=0.25),
            lorentz_poles=_lorentz_poles(),
        ),
        time_steps=time_steps,
    )
    # Non-dispersive medium matched to the Lorentz static index at the carrier.
    nondispersive, dt_n = _run(
        mw.Material(
            eps_r=_EPS_INF + eps_pole,
            modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=0.25),
        ),
        time_steps=time_steps,
    )

    cL, lL, uL = _sideband_amps(lorentz, dt_l)
    cN, lN, uN = _sideband_amps(nondispersive, dt_n)
    ratio_L = 0.5 * (lL + uL) / cL
    ratio_N = 0.5 * (lN + uN) / cN

    assert ratio_L > 0.0 and ratio_N > 0.0
    # The dispersive medium modulates only eps_inf, so it converts strictly less than
    # the matched-index non-dispersive medium, and stays within a factor of the
    # eps_inf fraction (poles present, not modulated).
    assert ratio_L < ratio_N
    assert predicted_ratio * 0.4 < ratio_L / ratio_N < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulated_kerr_reduces_to_pure_modulation_at_zero_chi3():
    """The modulated + Kerr path routes the field-dependent dynamic coefficients
    through the modulated E update. As chi3 -> 0 the dynamic curl collapses to the
    static curl, so a modulated Kerr material must reproduce the plain modulated
    material. A moderate drive keeps the field far from the parametric-buildup
    regime of a resonant modulated cavity."""
    time_steps = 1500
    amplitude = 50.0
    kerr, dt_k = _run(
        mw.Material(
            eps_r=_EPS_INF,
            modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=0.25),
            kerr_chi3=1.0e-14,
        ),
        amplitude=amplitude,
        time_steps=time_steps,
    )
    linear, dt_l = _run(
        mw.Material(eps_r=_EPS_INF, modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=0.25)),
        amplitude=amplitude,
        time_steps=time_steps,
    )
    assert dt_k == dt_l
    denom = float(np.max(np.abs(linear)))
    assert denom > 0.0
    assert np.all(np.isfinite(kerr))
    rel_diff = float(np.max(np.abs(kerr - linear))) / denom
    assert rel_diff < 5.0e-3, rel_diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_modulated_kerr_scene_enables_both_channels_and_runs_finite():
    """A single modulated Kerr Material prepares with both feature flags set and runs
    to a finite field, exercising the composed modulated + nonlinear kernel path."""
    scene = _slab_scene(
        material=mw.Material(
            eps_r=_EPS_INF,
            modulation=mw.ModulationSpec(frequency=_FMOD, amplitude=0.2),
            kerr_chi3=1.0e-16,
        ),
        amplitude=50.0,
        time_steps=64,
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[_CARRIER],
        run_time=mw.TimeConfig(time_steps=64),
        full_field_dft=False,
    ).prepare().solver
    assert solver.modulation_enabled
    assert solver.kerr_enabled
    assert solver.nonlinear_enabled

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[_CARRIER],
        run_time=mw.TimeConfig(time_steps=256),
        full_field_dft=False,
    ).run()
    trace = _to_numpy(result.monitor("probe")["field"])
    assert np.all(np.isfinite(trace))
    assert float(np.abs(trace).max()) > 0.0
