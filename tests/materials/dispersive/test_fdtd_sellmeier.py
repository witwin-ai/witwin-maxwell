import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

_C0 = 299_792_458.0

# Schott BK7 borosilicate crown glass Sellmeier coefficients.
# B_i are dimensionless; C_i are quoted in micron^2 and converted to meters^2.
_BK7_B = (1.03961212, 0.231792344, 1.01046945)
_BK7_C_UM2 = (0.00600069867, 0.0200179144, 103.560653)
_BK7_C_M2 = tuple(c * 1.0e-12 for c in _BK7_C_UM2)


def _bk7_index(lambda_um: float) -> float:
    """Analytic BK7 refractive index from the raw Sellmeier equation."""
    lam2 = lambda_um * lambda_um
    n2 = 1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(_BK7_B, _BK7_C_UM2))
    return float(np.sqrt(n2))


def test_sellmeier_lowers_to_zero_damping_lorentz_poles():
    material = mw.Material.sellmeier(b_coefficients=_BK7_B, c_coefficients=_BK7_C_M2, name="BK7")

    assert material.is_dispersive
    assert not material.is_magnetic_dispersive
    assert float(material.eps_r) == 1.0
    assert len(material.lorentz_poles) == len(_BK7_B)
    assert not material.debye_poles and not material.drude_poles

    for pole, b, c_m2 in zip(material.lorentz_poles, _BK7_B, _BK7_C_M2):
        assert pole.gamma == 0.0
        assert pole.delta_eps == pytest.approx(b)
        # resonance omega0 = 2*pi*c / sqrt(C) -> resonance_frequency = c / sqrt(C).
        assert pole.resonance_frequency == pytest.approx(_C0 / np.sqrt(c_m2), rel=1e-12)


def test_sellmeier_material_matches_analytic_bk7_index():
    material = mw.Material.sellmeier(b_coefficients=_BK7_B, c_coefficients=_BK7_C_M2, name="BK7")

    for lambda_um in np.linspace(0.5, 1.5, 41):
        frequency = _C0 / (lambda_um * 1.0e-6)
        epsilon = material.relative_permittivity(frequency)
        # A lossless Sellmeier medium must have no imaginary permittivity.
        assert abs(epsilon.imag) < 1e-12
        n_material = float(np.sqrt(epsilon.real))
        n_reference = _bk7_index(float(lambda_um))
        assert n_material == pytest.approx(n_reference, rel=1e-9, abs=1e-9)


def test_sellmeier_eps_inf_offsets_baseline():
    eps_inf = 2.25
    material = mw.Material.sellmeier(
        b_coefficients=_BK7_B, c_coefficients=_BK7_C_M2, eps_inf=eps_inf
    )
    lambda_um = 1.0
    frequency = _C0 / (lambda_um * 1.0e-6)
    epsilon = material.relative_permittivity(frequency)
    # eps_inf shifts the baseline: n^2 = eps_inf + sum term = (bk7 n^2 - 1) + eps_inf.
    expected_eps = (_bk7_index(lambda_um) ** 2 - 1.0) + eps_inf
    assert epsilon.real == pytest.approx(expected_eps, rel=1e-9)


def test_sellmeier_validation_errors():
    with pytest.raises(ValueError, match="equal length"):
        mw.Material.sellmeier(b_coefficients=(1.0, 2.0), c_coefficients=(1e-12,))
    with pytest.raises(ValueError, match="at least one"):
        mw.Material.sellmeier(b_coefficients=(), c_coefficients=())
    with pytest.raises(ValueError, match="must be > 0"):
        mw.Material.sellmeier(b_coefficients=(1.0,), c_coefficients=(0.0,))
    with pytest.raises(ValueError, match="must be > 0"):
        mw.Material.sellmeier(b_coefficients=(1.0,), c_coefficients=(-1e-12,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_sellmeier_slab_imparts_index_phase_delay():
    micron = 1.0e-6
    lambda_um = 1.0
    frequency = _C0 / (lambda_um * micron)
    slab_thickness = 0.4 * micron

    material = mw.Material.sellmeier(b_coefficients=_BK7_B, c_coefficients=_BK7_C_M2, name="BK7")
    n_slab = float(np.sqrt(material.relative_permittivity(frequency).real))

    def build_scene(with_slab: bool):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-3.0 * micron, 3.0 * micron),
                                     (-1.0 * micron, 1.0 * micron),
                                     (-1.0 * micron, 1.0 * micron))),
            grid=mw.GridSpec.uniform(0.04 * micron),
            boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
            device="cuda",
            sources=[
                mw.PlaneWave(
                    direction=(1.0, 0.0, 0.0),
                    polarization=(0.0, 0.0, 1.0),
                    source_time=mw.CW(frequency=frequency, amplitude=1.0),
                    name="pw",
                )
            ],
        )
        if with_slab:
            scene.add_structure(
                mw.Structure(
                    geometry=Box(position=(0.0, 0.0, 0.0),
                                 size=(slab_thickness, 2.0 * micron, 2.0 * micron)),
                    material=material,
                )
            )
        scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=1.5 * micron, fields=("Ez",)))
        return scene

    def run(with_slab: bool):
        return mw.Simulation.fdtd(
            build_scene(with_slab),
            frequencies=[frequency],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()

    def complex_mean(result):
        data = result.monitor("post")["data"]
        assert torch.is_tensor(data)
        field = data.detach().cpu().numpy()[3:-3, 3:-3]
        return complex(field.mean())

    vacuum = complex_mean(run(False))
    slab = complex_mean(run(True))

    # The Sellmeier slab must remain lossless and stable (finite, non-collapsing field).
    assert np.isfinite(slab.real) and np.isfinite(slab.imag)
    assert abs(slab) > 0.1 * abs(vacuum)

    measured_phase = float(np.angle(slab / vacuum))
    expected_phase = (n_slab - 1.0) * (2.0 * np.pi / lambda_um) * (slab_thickness / micron)
    # A ~1.5-index, 0.4 um slab delays the transmitted phase by ~1.3 rad. Fabry-Perot
    # ripple and grid dispersion loosen the match, so assert a robust band around it.
    assert 0.8 < measured_phase < 2.2
    assert abs(measured_phase - expected_phase) < 0.6
