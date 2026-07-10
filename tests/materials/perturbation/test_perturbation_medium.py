import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box, Sphere
from witwin.maxwell.scene import prepare_scene

_FREQUENCY = 1.0e9

# All box faces sit mid-cell on the 0.05 grid so the perturbation node window
# and the soft geometry occupancy agree exactly between the perturbed scene and
# a directly-constructed reference scene.
_SLAB_BOX = Box(position=(0.0, 0.0, 0.0), size=(0.35, 0.35, 0.35))
_GRID_SHAPE = (7, 7, 7)


def _build_scene(material, *, geometry=_SLAB_BOX, device="cpu"):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device=device,
    )
    scene.add_structure(mw.Structure(geometry=geometry, material=material))
    return scene


def _compiled_eps(material, *, frequency=None, geometry=_SLAB_BOX):
    prepared = prepare_scene(_build_scene(material, geometry=geometry))
    eps, _ = prepared.compile_relative_materials(frequency=frequency)
    return eps


def test_perturbation_medium_validation():
    base = mw.Material(eps_r=2.25)
    with pytest.raises(TypeError, match="base must be a Material"):
        mw.PerturbationMedium(2.25, perturbation=torch.zeros(2, 2, 2))
    with pytest.raises(TypeError, match="torch.Tensor"):
        mw.PerturbationMedium(base, perturbation=1.0)
    with pytest.raises(ValueError, match="3D"):
        mw.PerturbationMedium(base, perturbation=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="finite"):
        mw.PerturbationMedium(base, perturbation=torch.full((2, 2, 2), float("inf")))
    with pytest.raises(ValueError, match="PEC"):
        mw.PerturbationMedium(mw.Material.pec(), perturbation=torch.zeros(2, 2, 2))
    with pytest.raises(ValueError, match="non-positive permittivity"):
        mw.PerturbationMedium(
            base, perturbation=torch.full((2, 2, 2), -3.0), eps_sensitivity=1.0
        )


def test_perturbation_medium_blocks_scalar_frequency_evaluation():
    material = mw.PerturbationMedium(
        mw.Material(eps_r=2.25), perturbation=torch.ones(2, 2, 2), eps_sensitivity=0.5
    )
    with pytest.raises(NotImplementedError, match="spatially varying"):
        material.evaluate_at_frequency(_FREQUENCY)
    with pytest.raises(NotImplementedError, match="PerturbationMedium"):
        material.relative_permittivity(_FREQUENCY)
    # The static sample stays the base sample; the compiler applies the delta.
    assert float(material.evaluate_static().eps_r) == 2.25


def test_perturbation_medium_requires_box_geometry():
    material = mw.PerturbationMedium(
        mw.Material(eps_r=2.25), perturbation=torch.ones(2, 2, 2), eps_sensitivity=0.5
    )
    scene = _build_scene(material, geometry=Sphere(position=(0.0, 0.0, 0.0), radius=0.2))
    with pytest.raises(ValueError, match="Box structure geometry only"):
        prepare_scene(scene).compile_materials()


def test_zero_perturbation_reproduces_base_material_exactly():
    base = mw.Material(eps_r=2.25)
    perturbed = mw.PerturbationMedium(
        base, perturbation=torch.zeros(_GRID_SHAPE), eps_sensitivity=0.7
    )
    assert torch.equal(_compiled_eps(perturbed), _compiled_eps(base))


def test_uniform_perturbation_matches_directly_constructed_material():
    delta, sensitivity = 1.5, 0.8
    perturbed = mw.PerturbationMedium(
        mw.Material(eps_r=2.25),
        perturbation=torch.full(_GRID_SHAPE, delta),
        eps_sensitivity=sensitivity,
    )
    direct = mw.Material(eps_r=2.25 + sensitivity * delta)
    assert torch.allclose(_compiled_eps(perturbed), _compiled_eps(direct), rtol=1e-6, atol=1e-6)


def test_perturbation_of_dispersive_base_shifts_eps_inf():
    base = mw.Material.lorentz(
        eps_inf=2.0, delta_eps=1.0, resonance_frequency=2.0e9, gamma=2.0e8
    )
    perturbed = mw.PerturbationMedium(
        base, perturbation=torch.full(_GRID_SHAPE, 0.5), eps_sensitivity=1.0
    )
    direct = mw.Material.lorentz(
        eps_inf=2.5, delta_eps=1.0, resonance_frequency=2.0e9, gamma=2.0e8
    )
    assert torch.allclose(
        _compiled_eps(perturbed, frequency=_FREQUENCY),
        _compiled_eps(direct, frequency=_FREQUENCY),
        rtol=1e-6,
        atol=1e-6,
    )


def test_nonuniform_perturbation_is_differentiable_at_compile_time():
    perturbation = (0.5 * torch.rand(_GRID_SHAPE)).requires_grad_(True)
    material = mw.PerturbationMedium(
        mw.Material(eps_r=2.25), perturbation=perturbation, eps_sensitivity=0.5
    )
    prepared = prepare_scene(_build_scene(material))
    eps, _ = prepared.compile_material_tensors()
    eps.sum().backward()
    assert perturbation.grad is not None
    assert torch.isfinite(perturbation.grad).all()
    assert perturbation.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_zero_perturbation_run_matches_base_material_run():
    def run(material):
        scene = _build_scene(material, device="cuda")
        scene.add_source(
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=_FREQUENCY, amplitude=80.0),
                name="pw",
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.4, 0.0, 0.0), fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[_FREQUENCY],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()
        return result.monitor("probe")["data"]

    base = mw.Material(eps_r=2.25)
    perturbed = mw.PerturbationMedium(
        base, perturbation=torch.zeros(_GRID_SHAPE), eps_sensitivity=0.7
    )
    base_probe = run(base)
    perturbed_probe = run(perturbed)
    assert torch.is_tensor(perturbed_probe)
    assert torch.allclose(perturbed_probe, base_probe, rtol=1e-6, atol=1e-8)


def _abs2(z):
    if z.is_complex():
        return (z * z.conj()).real
    return z * z


def _perturbation_sim(perturbation):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
            material=mw.PerturbationMedium(
                mw.Material(eps_r=2.0), perturbation=perturbation, eps_sensitivity=2.0
            ),
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, -0.18),
            polarization="Ez",
            width=0.04,
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0),
        )
    )
    scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.18), fields=("Ez",)))
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=24),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_perturbation_gradient_matches_finite_difference():
    def loss_of(perturbation):
        result = _perturbation_sim(perturbation).run()
        return _abs2(result.monitor("probe")["data"]).sum()

    torch.manual_seed(0)
    base = 0.4 * torch.rand(2, 2, 2, device="cuda")
    trainable = base.clone().requires_grad_(True)
    loss = loss_of(trainable)
    loss.backward()
    backward_grad = trainable.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()
    assert backward_grad.abs().sum() > 0

    delta = 1.0e-2
    flat = base.flatten()
    fd_grad = torch.zeros_like(flat)
    for index in range(flat.numel()):
        with torch.no_grad():
            plus = flat.clone()
            plus[index] += delta
            minus = flat.clone()
            minus[index] -= delta
            loss_plus = loss_of(plus.reshape(base.shape))
            loss_minus = loss_of(minus.reshape(base.shape))
        fd_grad[index] = (loss_plus - loss_minus) / (2.0 * delta)
    fd_grad = fd_grad.reshape(base.shape)

    scale = float(fd_grad.abs().max())
    assert torch.allclose(backward_grad, fd_grad, rtol=2e-3, atol=scale * 2e-3)
