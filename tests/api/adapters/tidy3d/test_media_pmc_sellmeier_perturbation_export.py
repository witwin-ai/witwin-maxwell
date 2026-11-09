"""Reference-adapter export of PMC, custom Sellmeier and pole-perturbation media.

Compared against the genuine installed tidy3d classes (PMCMedium, CustomSellmeier,
PerturbationPoleResidue).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.adapters.tidy3d import _convert_material

td = pytest.importorskip("tidy3d")

_C0 = 299_792_458.0
_FREQUENCIES = (0.6e9, 1.0e9, 1.7e9)
_SHAPE = (3, 4, 5)


def _uniform(value):
    coords = dict(x=[-1.0, 1.0], y=[-1.0, 1.0], z=[-1.0, 1.0])
    return td.SpatialDataArray(np.full((2, 2, 2), float(value)), coords=coords)


def _scalar(value):
    return complex(np.asarray(value).ravel()[0])


def test_pmc_exports_to_reference_pmc_medium():
    exported = _convert_material(mw.Material.pmc(), td)
    assert isinstance(exported, td.PMCMedium)
    assert not isinstance(_convert_material(mw.Material.pec(), td), td.PMCMedium)


def test_uniform_custom_sellmeier_export_matches_reference_custom_sellmeier():
    c1, c2 = (_C0 / 1.2e9) ** 2, (_C0 / 2.5e9) ** 2  # meters^2
    material = mw.Material.custom_sellmeier(
        b_coefficients=(torch.full(_SHAPE, 1.2, dtype=torch.float64), 0.5),
        c_coefficients=(c1, torch.full(_SHAPE, c2, dtype=torch.float64)),
        eps_inf=1.0,
    )
    exported = _convert_material(material, td)
    scalar = mw.Material.sellmeier(b_coefficients=(1.2, 0.5), c_coefficients=(c1, c2))
    # The reference expresses C in micrometre^2.
    reference = td.CustomSellmeier(coeffs=((_uniform(1.2), _uniform(c1 * 1e12)), (_uniform(0.5), _uniform(c2 * 1e12))))
    for frequency in _FREQUENCIES:
        expected = _scalar(reference.eps_model(frequency))
        assert _scalar(exported.eps_model(frequency)) == pytest.approx(expected, rel=1e-9)
        assert complex(scalar.relative_permittivity(frequency)) == pytest.approx(expected, rel=1e-9)


def test_spatially_varying_custom_sellmeier_export_is_rejected():
    b = torch.full(_SHAPE, 1.2, dtype=torch.float64)
    b[0] = 0.4
    material = mw.Material.custom_sellmeier(b_coefficients=(b,), c_coefficients=((_C0 / 1.2e9) ** 2,))
    with pytest.raises(NotImplementedError, match="spatially-varying custom dispersive pole"):
        _convert_material(material, td)


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_uniform_pole_perturbation_export_matches_reference_perturbation_pole_residue(value):
    delta_eps, f0, gamma = 1.0, 2.0e9, 2.0e8
    eps_sensitivity, lorentz_sensitivity = 0.4, 0.25
    base = mw.Material.lorentz(eps_inf=2.0, delta_eps=delta_eps, resonance_frequency=f0, gamma=gamma)
    material = mw.PerturbationMedium(
        base,
        perturbation=torch.full(_SHAPE, value),
        eps_sensitivity=eps_sensitivity,
        lorentz_sensitivities=(lorentz_sensitivity,),
    )
    exported = _convert_material(material, td)

    # Reference: a pole-residue medium whose eps_inf and residue are linear in
    # temperature T = 300 K + value. The Lorentz residue is proportional to
    # delta_eps, so d(c)/dT = c * lorentz_sensitivity / delta_eps.
    unperturbed = td.Lorentz(eps_inf=2.0, coeffs=((delta_eps, f0, 0.5 * gamma),)).pole_residue
    (pole, residue), = unperturbed.poles
    reference = td.PerturbationPoleResidue(
        eps_inf=unperturbed.eps_inf,
        poles=unperturbed.poles,
        eps_inf_perturbation=td.ParameterPerturbation(
            heat=td.LinearHeatPerturbation(coeff=eps_sensitivity, temperature_ref=300.0)
        ),
        poles_perturbation=(
            (None, td.ParameterPerturbation(
                heat=td.LinearHeatPerturbation(coeff=residue * lorentz_sensitivity / delta_eps,
                                               temperature_ref=300.0)
            )),
        ),
    ).perturbed_copy(temperature=_uniform(300.0 + value))
    for frequency in _FREQUENCIES:
        assert _scalar(exported.eps_model(frequency)) == pytest.approx(
            _scalar(reference.eps_model(frequency)), rel=1e-9
        )
    if value == 0.0:
        for frequency in _FREQUENCIES:
            assert _scalar(exported.eps_model(frequency)) == pytest.approx(
                complex(base.relative_permittivity(frequency)), rel=1e-12
            )


def test_spatially_varying_pole_perturbation_export_is_rejected():
    perturbation = torch.zeros(_SHAPE)
    perturbation[0] = 1.0
    material = mw.PerturbationMedium(
        mw.Material.lorentz(eps_inf=2.0, delta_eps=1.0, resonance_frequency=2.0e9, gamma=2.0e8),
        perturbation=perturbation,
        lorentz_sensitivities=(0.25,),
    )
    with pytest.raises(NotImplementedError, match="spatially-varying PerturbationMedium"):
        _convert_material(material, td)
