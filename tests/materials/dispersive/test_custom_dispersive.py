import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box, Sphere
from witwin.maxwell.scene import prepare_scene

_FREQUENCY = 1.0e9


def _build_scene(*structures, device="cpu", sources=()):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device=device,
        sources=list(sources),
    )
    for structure in structures:
        scene.add_structure(structure)
    return scene


def _plane_wave(amplitude=80.0):
    return mw.PlaneWave(
        direction=(1.0, 0.0, 0.0),
        polarization=(0.0, 0.0, 1.0),
        source_time=mw.CW(frequency=_FREQUENCY, amplitude=amplitude),
        name="pw",
    )


# The slab spans x in [-0.175, 0.175] so every face sits mid-cell on the 0.05
# grid; the covered node window is the 7 nodes at x = -0.15 .. 0.15 and the
# custom parameter grids below use shape (7, 15, 15).
_SLAB_BOX = Box(position=(0.0, 0.0, 0.0), size=(0.35, 0.75, 0.75))
_GRID_SHAPE = (7, 15, 15)


def _split_grid(left_value, right_value):
    grid = torch.empty(_GRID_SHAPE)
    grid[:4] = float(left_value)
    grid[4:] = float(right_value)
    return grid


def _piecewise_reference_boxes():
    # Left part covers x in [-0.175, 0.025] (nodes -0.15 .. 0.0), right part
    # covers x in [0.025, 0.175] (nodes 0.05 .. 0.15); the internal interface
    # at x = 0.025 is mid-cell so both scenes rasterize identical node fields.
    left = Box(position=(-0.075, 0.0, 0.0), size=(0.2, 0.75, 0.75))
    right = Box(position=(0.1, 0.0, 0.0), size=(0.15, 0.75, 0.75))
    return left, right


def test_custom_pole_parameter_grid_validation():
    with pytest.raises(TypeError, match="torch.Tensor"):
        mw.CustomDebyePole(delta_eps=3.0, tau=1.0e-10)
    with pytest.raises(ValueError, match="3D"):
        mw.CustomDebyePole(delta_eps=torch.ones(4, 4), tau=1.0e-10)
    with pytest.raises(ValueError, match=">= 0"):
        mw.CustomLorentzPole(
            delta_eps=torch.full((2, 2, 2), -1.0),
            resonance_frequency=1.0e9,
            gamma=1.0e8,
        )
    with pytest.raises(ValueError, match="positive value"):
        mw.CustomDrudePole(plasma_frequency=torch.zeros(2, 2, 2), gamma=1.0e8)
    with pytest.raises(ValueError, match="tau"):
        mw.CustomDebyePole(delta_eps=torch.ones(2, 2, 2), tau=0.0)
    with pytest.raises(ValueError, match="finite"):
        mw.CustomDebyePole(delta_eps=torch.full((2, 2, 2), float("nan")), tau=1.0e-10)


def test_custom_pole_susceptibility_matches_scalar_pole():
    omega = 2.0 * np.pi * _FREQUENCY
    debye = mw.CustomDebyePole(delta_eps=torch.full((2, 2, 2), 3.0), tau=1.0e-10)
    assert torch.allclose(
        debye.susceptibility(omega),
        torch.full((2, 2, 2), mw.DebyePole(delta_eps=3.0, tau=1.0e-10).susceptibility(omega)),
    )
    drude = mw.CustomDrudePole(plasma_frequency=torch.full((2, 2, 2), 2.0e9), gamma=1.0e8)
    assert torch.allclose(
        drude.susceptibility(omega),
        torch.full((2, 2, 2), mw.DrudePole(plasma_frequency=2.0e9, gamma=1.0e8).susceptibility(omega)),
    )
    lorentz = mw.CustomLorentzPole(
        delta_eps=torch.full((2, 2, 2), 2.0), resonance_frequency=2.0e9, gamma=2.0e8
    )
    assert torch.allclose(
        lorentz.susceptibility(omega),
        torch.full(
            (2, 2, 2),
            mw.LorentzPole(delta_eps=2.0, resonance_frequency=2.0e9, gamma=2.0e8).susceptibility(omega),
        ),
    )


def test_material_rejects_non_pole_entries():
    with pytest.raises(TypeError, match="DebyePole or CustomDebyePole"):
        mw.Material(debye_poles=(object(),))


def test_custom_pole_material_blocks_scalar_frequency_evaluation():
    material = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.CustomDebyePole(delta_eps=torch.ones(2, 2, 2), tau=1.0e-10),),
    )
    assert material.is_dispersive
    assert material.has_custom_poles
    with pytest.raises(NotImplementedError, match="spatially-varying"):
        material.relative_permittivity(_FREQUENCY)

    magnetic = mw.Material(
        mu_debye_poles=(mw.CustomDebyePole(delta_eps=torch.ones(2, 2, 2), tau=1.0e-10),),
    )
    with pytest.raises(NotImplementedError, match="spatially-varying"):
        magnetic.relative_permeability(_FREQUENCY)


def test_custom_pole_requires_box_geometry():
    material = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.CustomDebyePole(delta_eps=torch.ones(2, 2, 2), tau=1.0e-10),),
    )
    scene = _build_scene(
        mw.Structure(geometry=Sphere(position=(0.0, 0.0, 0.0), radius=0.2), material=material)
    )
    with pytest.raises(ValueError, match="Box structure geometry only"):
        prepare_scene(scene).compile_materials()


def test_uniform_custom_debye_matches_scalar_debye_compiled_model():
    custom = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.CustomDebyePole(delta_eps=torch.full(_GRID_SHAPE, 3.0), tau=1.0e-10),),
    )
    scalar = mw.Material.debye(eps_inf=2.0, delta_eps=3.0, tau=1.0e-10)

    eps_custom, _ = prepare_scene(
        _build_scene(mw.Structure(geometry=_SLAB_BOX, material=custom))
    ).compile_relative_materials(frequency=_FREQUENCY)
    eps_scalar, _ = prepare_scene(
        _build_scene(mw.Structure(geometry=_SLAB_BOX, material=scalar))
    ).compile_relative_materials(frequency=_FREQUENCY)

    assert torch.allclose(eps_custom, eps_scalar, rtol=1e-6, atol=1e-6)


def test_custom_parameter_grid_is_resampled_to_node_coverage():
    # A uniform grid at twice the covered node count trilinearly resamples to
    # the same uniform per-node values.
    coarse = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.CustomDebyePole(delta_eps=torch.full((14, 30, 30), 3.0), tau=1.0e-10),),
    )
    exact = mw.Material(
        eps_r=2.0,
        debye_poles=(mw.CustomDebyePole(delta_eps=torch.full(_GRID_SHAPE, 3.0), tau=1.0e-10),),
    )
    eps_coarse, _ = prepare_scene(
        _build_scene(mw.Structure(geometry=_SLAB_BOX, material=coarse))
    ).compile_relative_materials(frequency=_FREQUENCY)
    eps_exact, _ = prepare_scene(
        _build_scene(mw.Structure(geometry=_SLAB_BOX, material=exact))
    ).compile_relative_materials(frequency=_FREQUENCY)
    assert torch.allclose(eps_coarse, eps_exact, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "custom_material, left_material, right_material",
    [
        (
            mw.Material(
                eps_r=2.0,
                lorentz_poles=(
                    mw.CustomLorentzPole(
                        delta_eps=_split_grid(2.0, 0.5),
                        resonance_frequency=2.0e9,
                        gamma=2.0e8,
                    ),
                ),
            ),
            mw.Material.lorentz(eps_inf=2.0, delta_eps=2.0, resonance_frequency=2.0e9, gamma=2.0e8),
            mw.Material.lorentz(eps_inf=2.0, delta_eps=0.5, resonance_frequency=2.0e9, gamma=2.0e8),
        ),
        (
            mw.Material(
                eps_r=1.0,
                drude_poles=(
                    mw.CustomDrudePole(plasma_frequency=_split_grid(2.0e9, 1.0e9), gamma=1.0e8),
                ),
            ),
            mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8),
            mw.Material.drude(eps_inf=1.0, plasma_frequency=1.0e9, gamma=1.0e8),
        ),
    ],
    ids=["lorentz", "drude"],
)
def test_graded_custom_pole_matches_piecewise_uniform_reference(
    custom_material, left_material, right_material
):
    left_box, right_box = _piecewise_reference_boxes()
    graded_scene = _build_scene(mw.Structure(geometry=_SLAB_BOX, material=custom_material))
    reference_scene = _build_scene(
        mw.Structure(geometry=left_box, material=left_material),
        mw.Structure(geometry=right_box, material=right_material),
    )

    eps_graded, _ = prepare_scene(graded_scene).compile_relative_materials(frequency=_FREQUENCY)
    eps_reference, _ = prepare_scene(reference_scene).compile_relative_materials(frequency=_FREQUENCY)

    assert torch.allclose(eps_graded, eps_reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_uniform_custom_drude_matches_scalar_drude_run():
    def run(material):
        scene = _build_scene(
            mw.Structure(geometry=_SLAB_BOX, material=material),
            device="cuda",
            sources=(_plane_wave(),),
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.5, 0.0, 0.0), fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[_FREQUENCY],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()
        return result.monitor("probe")["data"]

    custom = mw.Material(
        eps_r=1.0,
        drude_poles=(mw.CustomDrudePole(plasma_frequency=torch.full(_GRID_SHAPE, 2.0e9), gamma=1.0e8),),
    )
    scalar = mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8)

    custom_probe = run(custom)
    scalar_probe = run(scalar)
    assert torch.is_tensor(custom_probe)
    assert torch.allclose(custom_probe, scalar_probe, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_graded_custom_lorentz_matches_piecewise_uniform_run():
    def run(*structures):
        scene = _build_scene(*structures, device="cuda", sources=(_plane_wave(),))
        scene.add_monitor(mw.PointMonitor("probe", (0.5, 0.0, 0.0), fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[_FREQUENCY],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()
        return result.monitor("probe")["data"]

    graded = mw.Material(
        eps_r=2.0,
        lorentz_poles=(
            mw.CustomLorentzPole(
                delta_eps=_split_grid(2.0, 0.5), resonance_frequency=2.0e9, gamma=2.0e8
            ),
        ),
    )
    left_box, right_box = _piecewise_reference_boxes()
    left = mw.Material.lorentz(eps_inf=2.0, delta_eps=2.0, resonance_frequency=2.0e9, gamma=2.0e8)
    right = mw.Material.lorentz(eps_inf=2.0, delta_eps=0.5, resonance_frequency=2.0e9, gamma=2.0e8)

    graded_probe = run(mw.Structure(geometry=_SLAB_BOX, material=graded))
    reference_probe = run(
        mw.Structure(geometry=left_box, material=left),
        mw.Structure(geometry=right_box, material=right),
    )

    # The compiled node fields are identical; the only numerical difference is
    # that the reference scene integrates two ADE slots with the same recursion
    # constants where the graded scene integrates one, so the runs agree to
    # floating-point accumulation error.
    assert torch.is_tensor(graded_probe)
    reference_scale = reference_probe.abs().max()
    assert torch.allclose(graded_probe, reference_probe, rtol=1e-3, atol=1e-4 * float(reference_scale))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_custom_drude_auto_dt_uses_peak_plasma_frequency():
    custom_scene = _build_scene(
        mw.Structure(
            geometry=_SLAB_BOX,
            material=mw.Material(
                eps_r=1.0,
                drude_poles=(
                    mw.CustomDrudePole(plasma_frequency=_split_grid(2.0e9, 1.0e9), gamma=1.0e8),
                ),
            ),
        ),
        device="cuda",
        sources=(_plane_wave(),),
    )
    scalar_scene = _build_scene(
        mw.Structure(
            geometry=_SLAB_BOX,
            material=mw.Material.drude(eps_inf=1.0, plasma_frequency=2.0e9, gamma=1.0e8),
        ),
        device="cuda",
        sources=(_plane_wave(),),
    )

    custom_dt = mw.Simulation.fdtd(custom_scene, frequencies=[_FREQUENCY]).prepare().solver.dt
    scalar_dt = mw.Simulation.fdtd(scalar_scene, frequencies=[_FREQUENCY]).prepare().solver.dt
    assert custom_dt == scalar_dt
