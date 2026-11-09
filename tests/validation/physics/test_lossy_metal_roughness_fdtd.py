"""Forward FDTD check that conductor roughness raises surface absorption as predicted.

Normal-incidence reflection from a good-conductor half-space (the same effectively 1D
scene as ``test_lossy_metal_sibc.py``) is read from the vacuum standing-wave ratio. A
rough conductor must absorb more than the smooth one. For the narrowband order-0
surface the runtime realizes ``Z_s = K_loss * R`` (pure resistance, ``K_loss`` the
roughness loss factor at the source frequency); for the broadband rational surface ADE
the fitted rough impedance ``K(omega) * Z_smooth(omega)`` is stepped directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")

_C = 299792458.0
_ETA0 = 376.730313668
_SIGMA = 50.0
_FREQUENCY = 2.0e9
_BAND = (0.5e9, 5.0e9)
_ROUGHNESS = mw.HammerstadRoughness(rms_height=1.5e-3)  # Rq ~ delta(2 GHz) = 1.59 mm


def _scene(material, dx):
    trans = 4.0 * dx
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-trans / 2.0, trans / 2.0), (-trans / 2.0, trans / 2.0))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.faces(default="periodic", num_layers=10, strength=1.0, x=("pml", "pml")),
        device="cuda",
        sources=[mw.PointDipole(position=(-0.3, 0.0, 0.0), polarization=(0.0, 0.0, 1.0), width=2.0 * dx,
                                source_time=mw.CW(frequency=_FREQUENCY, amplitude=40.0), name="pw")],
        structures=[mw.Structure(geometry=Box(position=(0.3, 0.0, 0.0), size=(0.4, 1.0, 1.0)),
                                 material=material, name="metal")],
    )


def _simulation(material, cells_per_wavelength=40.0):
    dx = (_C / _FREQUENCY) / cells_per_wavelength
    return mw.Simulation.fdtd(
        _scene(material, dx), absorber="cpml", frequencies=[_FREQUENCY],
        run_time=mw.TimeConfig.auto(steady_cycles=16, transient_cycles=24), full_field_dft=True,
    )


def _absorption(material, cells_per_wavelength=40.0):
    result = _simulation(material, cells_per_wavelength).run()
    ez = result.field("Ez")
    ez = (ez["data"] if isinstance(ez, dict) else ez).detach().cpu().numpy()
    assert np.isfinite(ez).all()
    line = np.abs(ez).reshape(ez.shape[0], -1).mean(axis=1)
    xs = result.solver.scene.x_nodes64
    v = line[(xs >= -0.28) & (xs <= 0.08)]
    gamma = float((v.max() - v.min()) / (v.max() + v.min()))
    return 1.0 - gamma * gamma


def _analytic_absorption(impedance):
    gamma = abs((impedance - _ETA0) / (impedance + _ETA0))
    return 1.0 - gamma * gamma


def test_narrowband_surface_resistance_is_scaled_by_the_loss_factor():
    smooth = _simulation(mw.LossyMetalMedium(conductivity=_SIGMA)).prepare().solver
    rough_metal = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=_ROUGHNESS)
    rough = _simulation(rough_metal).prepare().solver
    smooth_r = [write["surface_r"] for write in smooth._surface_impedance["writes"]]
    rough_r = [write["surface_r"] for write in rough._surface_impedance["writes"]]
    factor = rough_metal.roughness_loss_factor(_FREQUENCY)
    assert factor > 1.5
    for a, b in zip(smooth_r, rough_r):
        assert b / a == pytest.approx(factor, rel=1e-12)


@pytest.mark.parametrize(
    "roughness", [_ROUGHNESS, mw.HurayRoughness.cannonball(0.3e-3)], ids=["hammerstad", "huray"]
)
def test_narrowband_rough_absorption_increases_as_predicted(roughness):
    smooth_metal = mw.LossyMetalMedium(conductivity=_SIGMA)
    rough_metal = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=roughness)
    resistance = smooth_metal.surface_impedance_at_freq(_FREQUENCY).real
    expected_smooth = _analytic_absorption(resistance)
    expected_rough = _analytic_absorption(resistance * rough_metal.roughness_loss_factor(_FREQUENCY))
    smooth = _absorption(smooth_metal)
    rough = _absorption(rough_metal)
    assert rough > smooth
    assert rough == pytest.approx(expected_rough, rel=0.02)
    assert rough / smooth == pytest.approx(expected_rough / expected_smooth, rel=0.02)


def test_broadband_rough_rational_surface_absorbs_more_and_stays_stable():
    """The fitted rough impedance reaches the broadband surface ADE.

    Only the ordering and stability are gated here. The generic Z-form surface ADE
    itself over-predicts absorption once ``|Z_s|`` grows (measured for a *smooth*
    15 S/m conductor: +12% at 45 and 60 cells/wavelength), so a quantitative rough
    gate would test that runtime limit rather than the roughness model.
    """
    smooth_metal = mw.LossyMetalMedium(conductivity=_SIGMA)
    rough_metal = mw.LossyMetalMedium(conductivity=_SIGMA, roughness=_ROUGHNESS)
    smooth = _absorption(mw.SurfaceImpedanceMedium(impedance=smooth_metal.rational_surface_impedance(_BAND)), 45.0)
    rough = _absorption(mw.SurfaceImpedanceMedium(impedance=rough_metal.rational_surface_impedance(_BAND)), 45.0)
    expected_ratio = (
        _analytic_absorption(rough_metal.surface_impedance_at_freq(_FREQUENCY))
        / _analytic_absorption(smooth_metal.surface_impedance_at_freq(_FREQUENCY))
    )
    assert expected_ratio > 1.4
    assert math.isfinite(rough) and 0.0 < rough < 1.0
    assert rough / smooth > 1.3
