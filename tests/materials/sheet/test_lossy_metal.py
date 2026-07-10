import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene

_MU0 = 4.0e-7 * np.pi


def test_lossy_metal_construction_and_surface_impedance():
    sigma = 5.8e7  # copper [S/m]
    metal = mw.LossyMetalMedium(conductivity=sigma, name="copper")
    assert metal.is_lossy_metal
    assert metal.conductivity == pytest.approx(sigma)

    frequency = 10.0e9
    omega = 2.0 * np.pi * frequency
    delta = np.sqrt(2.0 / (omega * _MU0 * sigma))
    assert metal.skin_depth(frequency) == pytest.approx(delta, rel=1e-12)

    z_s = metal.surface_impedance_at_freq(frequency)
    magnitude = np.sqrt(omega * _MU0 / (2.0 * sigma))
    assert z_s.real == pytest.approx(magnitude, rel=1e-12)
    assert z_s.imag == pytest.approx(-magnitude, rel=1e-12)
    # Leontovich relation |Z_s| = sqrt(2) / (sigma * delta).
    assert abs(z_s) == pytest.approx(np.sqrt(2.0) / (sigma * delta), rel=1e-12)

    with pytest.raises(ValueError):
        mw.LossyMetalMedium(conductivity=0.0)
    with pytest.raises(ValueError):
        metal.surface_impedance(0.0)


def test_lossy_metal_compile_raises_not_implemented():
    scene = prepare_scene(
        mw.Scene(
            domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
            grid=mw.GridSpec.uniform(0.05),
            boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
            device="cpu",
            structures=[
                mw.Structure(
                    geometry=Box(position=(0.1, 0.0, 0.0), size=(0.2, 0.3, 0.3)),
                    material=mw.LossyMetalMedium(conductivity=5.8e7),
                )
            ],
        )
    )
    with pytest.raises(NotImplementedError, match="LossyMetalMedium"):
        scene.compile_materials()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_lossy_metal_simulation_prepare_raises_not_implemented():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.1, 0.0, 0.0), size=(0.2, 0.3, 0.3)),
                material=mw.LossyMetalMedium(conductivity=5.8e7),
            )
        ],
    )
    with pytest.raises(NotImplementedError, match="LossyMetalMedium"):
        mw.Simulation.fdtd(scene, frequencies=[1.0e9]).prepare()
