"""PerturbationMedium perturbing dispersive pole strengths (pole-residue perturbation)."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene

_FREQUENCY = 1.0e9
# Faces mid-cell on the 0.05 grid; the covered node window is 7 x 7 x 7.
_SLAB_BOX = Box(position=(0.0, 0.0, 0.0), size=(0.35, 0.35, 0.35))
_GRID_SHAPE = (7, 7, 7)


def _build_scene(*structures, device="cpu", subpixel_samples=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device=device,
        **({} if subpixel_samples is None else {"subpixel_samples": subpixel_samples}),
    )
    for structure in structures:
        scene.add_structure(structure)
    return scene


def _compile(material):
    return prepare_scene(_build_scene(mw.Structure(geometry=_SLAB_BOX, material=material))).compile_materials()


def _relative_eps(*structures, frequency=_FREQUENCY, subpixel_samples=None):
    eps, _ = prepare_scene(_build_scene(*structures, subpixel_samples=subpixel_samples)).compile_relative_materials(
        frequency=frequency)
    return eps


def _mixed_base():
    return mw.Material(
        eps_r=2.0,
        debye_poles=(mw.DebyePole(delta_eps=3.0, tau=1.0e-10),),
        drude_poles=(mw.DrudePole(plasma_frequency=2.0e9, gamma=1.0e8),),
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=2.0e9, gamma=2.0e8),),
    )


_SENSITIVITIES = dict(debye_sensitivities=(0.5,), drude_sensitivities=(2.5e8,), lorentz_sensitivities=(0.25,))


def test_pole_perturbation_validation():
    base = _mixed_base()
    with pytest.raises(ValueError, match="lorentz_sensitivities has 2 entries"):
        mw.PerturbationMedium(base, perturbation=torch.zeros(_GRID_SHAPE), lorentz_sensitivities=(1.0, 1.0))
    with pytest.raises(NotImplementedError, match="fixed perturbation"):
        mw.PerturbationMedium(
            base, perturbation=torch.zeros(_GRID_SHAPE, requires_grad=True), lorentz_sensitivities=(1.0,)
        )
    # A trainable perturbation with only the eps channel stays differentiable.
    trainable = mw.PerturbationMedium(
        base, perturbation=torch.zeros(_GRID_SHAPE, requires_grad=True), lorentz_sensitivities=(0.0,)
    )
    assert not trainable.perturbs_poles
    assert trainable.lorentz_poles == base.lorentz_poles
    with pytest.raises(ValueError, match=">= 0"):
        mw.PerturbationMedium(base, perturbation=torch.full(_GRID_SHAPE, -1.0), lorentz_sensitivities=(2.0,))
    gain = mw.Material(
        lorentz_poles=(mw.LorentzPole(delta_eps=0.1, resonance_frequency=2.0e9, gamma=2.0e8, allow_gain=True),),
    )
    with pytest.raises(NotImplementedError, match="gain"):
        mw.PerturbationMedium(gain, perturbation=torch.zeros(_GRID_SHAPE), lorentz_sensitivities=(0.1,))
    custom = mw.Material(
        lorentz_poles=(mw.CustomLorentzPole(delta_eps=torch.ones(3, 3, 3), resonance_frequency=2.0e9, gamma=2.0e8),),
    )
    with pytest.raises(ValueError, match="grid shape"):
        mw.PerturbationMedium(custom, perturbation=torch.zeros(_GRID_SHAPE), lorentz_sensitivities=(0.1,))


def test_zero_perturbation_with_pole_sensitivities_reproduces_base_exactly():
    base = _mixed_base()
    perturbed = mw.PerturbationMedium(
        base, perturbation=torch.zeros(_GRID_SHAPE), eps_sensitivity=0.7, **_SENSITIVITIES
    )
    assert perturbed.perturbs_poles
    assert isinstance(perturbed.debye_poles[0], mw.CustomDebyePole)
    assert isinstance(perturbed.drude_poles[0], mw.CustomDrudePole)
    assert isinstance(perturbed.lorentz_poles[0], mw.CustomLorentzPole)
    expected = _compile(base)
    compiled = _compile(perturbed)
    for kind in ("debye_poles", "drude_poles", "lorentz_poles"):
        assert len(compiled[kind]) == len(expected[kind]) == 1
        assert compiled[kind][0]["pole"] == expected[kind][0]["pole"]
        assert torch.equal(compiled[kind][0]["weight"], expected[kind][0]["weight"])
    assert torch.equal(compiled["eps_r"], expected["eps_r"])
    assert torch.equal(
        _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=perturbed)),
        _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=base)),
    )


def test_uniform_pole_perturbation_matches_directly_constructed_material():
    value = 0.5
    perturbed = mw.PerturbationMedium(
        _mixed_base(), perturbation=torch.full(_GRID_SHAPE, value), eps_sensitivity=0.4, **_SENSITIVITIES
    )
    direct = mw.Material(
        eps_r=2.0 + 0.4 * value,
        debye_poles=(mw.DebyePole(delta_eps=3.0 + 0.5 * value, tau=1.0e-10),),
        drude_poles=(mw.DrudePole(plasma_frequency=2.0e9 + 2.5e8 * value, gamma=1.0e8),),
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0 + 0.25 * value, resonance_frequency=2.0e9, gamma=2.0e8),),
    )
    for frequency in (0.5e9, _FREQUENCY, 1.9e9):
        assert torch.allclose(
            _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=perturbed), frequency=frequency),
            _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=direct), frequency=frequency),
            rtol=1e-5, atol=1e-5,
        )


def test_graded_lorentz_strength_perturbation_matches_piecewise_reference():
    # Pins the legacy node-sampling policy; graded-vs-piecewise identity is a node-grid property, not a subpixel-quality test.
    # The cell-centred perturbation ramps 0->2 over x in [0, 0.05]; the reference steps at x=0.025. A window centred on
    # x=0 or 0.05 sees a mean perturbation 0.25 away from the step, i.e. 0.75*0.25*|chi(1 GHz)/delta_eps| ~= 0.25 in eps,
    # consistent with the measured default-quadrature maximum 0.176 (also on master 986b81e8).
    node_sampling = mw.SubpixelSpec(samples=1, averaging="arithmetic")
    base = mw.Material.lorentz(eps_inf=2.0, delta_eps=1.0, resonance_frequency=2.0e9, gamma=2.0e8)
    perturbation = torch.empty(_GRID_SHAPE)
    perturbation[:4] = 0.0
    perturbation[4:] = 2.0
    perturbed = mw.PerturbationMedium(
        base, perturbation=perturbation, eps_sensitivity=0.0, lorentz_sensitivities=(0.75,)
    )
    left = Box(position=(-0.075, 0.0, 0.0), size=(0.2, 0.35, 0.35))
    right = Box(position=(0.1, 0.0, 0.0), size=(0.15, 0.35, 0.35))
    reference = _relative_eps(
        mw.Structure(geometry=left, material=base),
        mw.Structure(geometry=right, material=mw.Material.lorentz(
            eps_inf=2.0, delta_eps=2.5, resonance_frequency=2.0e9, gamma=2.0e8)),
        subpixel_samples=node_sampling,
    )
    graded = _relative_eps(mw.Structure(geometry=_SLAB_BOX, material=perturbed), subpixel_samples=node_sampling)
    assert torch.allclose(graded, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_zero_pole_perturbation_run_matches_base_run():
    def run(material):
        scene = _build_scene(mw.Structure(geometry=_SLAB_BOX, material=material), device="cuda")
        scene.add_source(mw.PlaneWave(direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0),
                                      source_time=mw.CW(frequency=_FREQUENCY, amplitude=80.0), name="pw"))
        scene.add_monitor(mw.PointMonitor("probe", (0.4, 0.0, 0.0), fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene, frequencies=[_FREQUENCY],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"), full_field_dft=False,
        ).run()
        return result.monitor("probe")["data"]

    base = mw.Material.lorentz(eps_inf=2.0, delta_eps=1.0, resonance_frequency=2.0e9, gamma=2.0e8)
    perturbed = mw.PerturbationMedium(
        base, perturbation=torch.zeros(_GRID_SHAPE), eps_sensitivity=0.7, lorentz_sensitivities=(0.25,)
    )
    perturbed_probe = run(perturbed)
    assert torch.is_tensor(perturbed_probe)
    assert torch.equal(perturbed_probe, run(base))
