"""Spatially varying (custom) Sellmeier dielectrics."""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene

_FREQUENCY = 1.0e9
_C = 299_792_458.0
# Squared resonance wavelengths [m^2]: resonances near 1.2 GHz and 2.5 GHz.
_C1 = (_C / 1.2e9) ** 2
_C2 = (_C / 2.5e9) ** 2

# Same slab convention as the custom-pole tests: faces mid-cell on the 0.05
# grid, covered node window 7 x 15 x 15.
_SLAB_BOX = Box(position=(0.0, 0.0, 0.0), size=(0.35, 0.75, 0.75))
_GRID_SHAPE = (7, 15, 15)


_NODE_SAMPLING = mw.SubpixelSpec(samples=1, averaging="arithmetic")


def _build_scene(*structures, device="cpu", subpixel_samples=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device=device,
        **({} if subpixel_samples is None else {"subpixel_samples": subpixel_samples}),
    )
    for structure in structures:
        scene.add_structure(structure)
    return scene


def _full(value):
    return torch.full(_GRID_SHAPE, float(value), dtype=torch.float64)


def _split(left, right):
    grid = torch.empty(_GRID_SHAPE, dtype=torch.float64)
    grid[:4] = float(left)
    grid[4:] = float(right)
    return grid


def _piecewise_boxes():
    left = Box(position=(-0.075, 0.0, 0.0), size=(0.2, 0.75, 0.75))
    right = Box(position=(0.1, 0.0, 0.0), size=(0.15, 0.75, 0.75))
    return left, right


def _compile(material, geometry=_SLAB_BOX):
    return prepare_scene(_build_scene(mw.Structure(geometry=geometry, material=material))).compile_materials()


def _relative_eps(*structures, frequency=_FREQUENCY, subpixel_samples=None):
    eps, _ = prepare_scene(_build_scene(*structures, subpixel_samples=subpixel_samples)).compile_relative_materials(
        frequency=frequency)
    return eps


def test_custom_sellmeier_validation():
    grid = _full(1.0)
    with pytest.raises(ValueError, match="equal length"):
        mw.Material.custom_sellmeier(b_coefficients=(grid,), c_coefficients=(_C1, _C2))
    with pytest.raises(ValueError, match="at least one"):
        mw.Material.custom_sellmeier(b_coefficients=(), c_coefficients=())
    with pytest.raises(ValueError, match="Material.sellmeier"):
        mw.Material.custom_sellmeier(b_coefficients=(1.0,), c_coefficients=(_C1,))
    with pytest.raises(ValueError, match="3D tensor"):
        mw.Material.custom_sellmeier(b_coefficients=(torch.ones(4, 4),), c_coefficients=(_C1,))
    with pytest.raises(ValueError, match="share a shape"):
        mw.Material.custom_sellmeier(b_coefficients=(grid,), c_coefficients=(torch.full((2, 2, 2), _C1),))
    with pytest.raises(ValueError, match="> 0"):
        mw.Material.custom_sellmeier(b_coefficients=(grid,), c_coefficients=(-1.0,))
    with pytest.raises(ValueError, match="> 0"):
        mw.Material.custom_sellmeier(b_coefficients=(grid,), c_coefficients=(_split(_C1, 0.0),))
    with pytest.raises(ValueError, match=">= 0"):
        mw.Material.custom_sellmeier(b_coefficients=(_split(1.0, -0.5),), c_coefficients=(_C1,))
    with pytest.raises(NotImplementedError, match="requires grad"):
        mw.Material.custom_sellmeier(b_coefficients=(grid.clone().requires_grad_(True),), c_coefficients=(_C1,))
    smooth_c = torch.linspace(_C2, _C1, int(np.prod(_GRID_SHAPE)), dtype=torch.float64).reshape(_GRID_SHAPE)
    with pytest.raises(NotImplementedError, match="distinct values"):
        mw.Material.custom_sellmeier(b_coefficients=(grid,), c_coefficients=(smooth_c,))


def test_uniform_custom_sellmeier_compiles_exactly_to_sellmeier():
    reference = mw.Material.sellmeier(b_coefficients=(1.2, 0.5), c_coefficients=(_C1, _C2), eps_inf=1.5)
    custom = mw.Material.custom_sellmeier(
        b_coefficients=(_full(1.2), 0.5), c_coefficients=(_C1, _full(_C2)), eps_inf=1.5
    )
    # A tensor B yields a custom pole; a scalar B over a uniform C stays a scalar pole.
    assert isinstance(custom.lorentz_poles[0], mw.CustomLorentzPole)
    assert type(custom.lorentz_poles[1]) is mw.LorentzPole
    expected = _compile(reference)
    compiled = _compile(custom)
    assert len(compiled["lorentz_poles"]) == len(expected["lorentz_poles"]) == 2
    for got, want in zip(compiled["lorentz_poles"], expected["lorentz_poles"]):
        assert got["pole"] == want["pole"]
        assert torch.equal(got["weight"], want["weight"])
    for key in ("eps_r", "mu_r", "sigma_e"):
        assert torch.equal(compiled[key], expected[key])
    assert torch.equal(
        _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=custom)),
        _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=reference)),
    )


def test_graded_b_matches_piecewise_sellmeier_reference():
    custom = mw.Material.custom_sellmeier(b_coefficients=(_split(1.5, 0.4),), c_coefficients=(_C1,), eps_inf=2.0)
    # Pins the legacy node-sampling policy; graded-vs-piecewise identity is a node-grid property, not a subpixel-quality test.
    left_box, right_box = _piecewise_boxes()
    graded = _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=custom), subpixel_samples=_NODE_SAMPLING)
    reference = _relative_eps(
        mw.Structure(geometry=left_box,
                     material=mw.Material.sellmeier(b_coefficients=(1.5,), c_coefficients=(_C1,), eps_inf=2.0)),
        mw.Structure(geometry=right_box,
                     material=mw.Material.sellmeier(b_coefficients=(0.4,), c_coefficients=(_C1,), eps_inf=2.0)),
        subpixel_samples=_NODE_SAMPLING,
    )
    assert torch.allclose(graded, reference, rtol=1e-5, atol=1e-6)


def test_piecewise_constant_c_splits_into_one_pole_per_level():
    custom = mw.Material.custom_sellmeier(
        b_coefficients=(_split(1.5, 0.4),), c_coefficients=(_split(_C1, _C2),), eps_inf=2.0
    )
    resonances = sorted(pole.resonance_frequency for pole in custom.lorentz_poles)
    assert resonances == pytest.approx(sorted((_C / np.sqrt(_C1), _C / np.sqrt(_C2))), rel=1e-12)
    # Pins the legacy node-sampling policy; graded-vs-piecewise identity is a node-grid property, not a subpixel-quality test.
    left_box, right_box = _piecewise_boxes()
    graded = _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=custom), subpixel_samples=_NODE_SAMPLING)
    reference = _relative_eps(
        mw.Structure(geometry=left_box,
                     material=mw.Material.sellmeier(b_coefficients=(1.5,), c_coefficients=(_C1,), eps_inf=2.0)),
        mw.Structure(geometry=right_box,
                     material=mw.Material.sellmeier(b_coefficients=(0.4,), c_coefficients=(_C2,), eps_inf=2.0)),
        subpixel_samples=_NODE_SAMPLING,
    )
    assert torch.allclose(graded, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_uniform_custom_sellmeier_matches_sellmeier_run():
    def run(material):
        scene = _build_scene(mw.Structure(geometry=_SLAB_BOX, material=material), device="cuda")
        scene.add_source(mw.PlaneWave(direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0),
                                      source_time=mw.CW(frequency=_FREQUENCY, amplitude=80.0), name="pw"))
        scene.add_monitor(mw.PointMonitor("probe", (0.35, 0.0, 0.0), fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene, frequencies=[_FREQUENCY],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False,
        ).run()
        return result.monitor("probe")["data"]

    reference = mw.Material.sellmeier(b_coefficients=(1.2,), c_coefficients=(_C1,), eps_inf=1.5)
    custom = mw.Material.custom_sellmeier(b_coefficients=(_full(1.2),), c_coefficients=(_C1,), eps_inf=1.5)
    custom_probe = run(custom)
    assert torch.is_tensor(custom_probe)
    assert torch.equal(custom_probe, run(reference))
