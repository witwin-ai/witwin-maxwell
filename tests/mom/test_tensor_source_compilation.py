"""Physical mode declarations and native phase-origin compilation."""
from dataclasses import replace
import pytest
import torch
from witwin.maxwell.sources import MaxwellPlaneModes
from witwin.maxwell.mom.tensor_carrier import compile_tensor_carrier
from test_tensor_carrier_persistence import carrier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('reciprocal', [True, False])
def test_physical_modes_preserve_translated_superposition_and_native_gradients(reciprocal):
    original = carrier(reciprocal)
    amplitude = original.electric.new_tensor([.7+.1j, -.2+.3j], requires_grad=True)
    origins = original.origin.new_tensor([[.04, -.03, .02], [-.02, .01, -.04]], requires_grad=True)
    declarations = [MaxwellPlaneModes(original.medium.frequency, original.wavevectors,
        amplitude[index]*original.electric, amplitude[index]*original.magnetic,
        origins[index]) for index in range(2)]
    compiled = compile_tensor_carrier(declarations, original.medium)
    assert compiled.medium is original.medium
    assert compiled.wavevectors.shape == (4, 3)
    points = original.origin.new_tensor([[.02, .03, .01], [-.04, .01, -.02]])
    actual = compiled.evaluate(points, original.medium)
    expected = [original.electric.new_zeros(2, 3), original.magnetic.new_zeros(2, 3)]
    for index in range(2):
        phase = torch.exp(1j*((points-origins[index]).to(original.electric.dtype)
                             @original.wavevectors.T))
        for field, values in enumerate((original.electric, original.magnetic)):
            expected[field] = expected[field]+amplitude[index]*(phase@values)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=1e-12, atol=1e-13)
    losses = [sum(value.abs().square().sum() for value in fields) for fields in (actual, expected)]
    gradients = [torch.autograd.grad(loss, (amplitude, origins), retain_graph=True) for loss in losses]
    for value, target in zip(*gradients):
        torch.testing.assert_close(value, target, rtol=1e-11, atol=1e-12)
    identity = compiled.identity
    declarations[0].electric.mul_(.9)
    assert compiled.identity == identity
    with pytest.raises(ValueError, match='Maxwell curl'):
        compile_tensor_carrier(declarations, original.medium)


def test_mode_compilation_rejects_wrong_frequency_and_forged_fields():
    original = carrier()
    declaration = MaxwellPlaneModes(original.medium.frequency, original.wavevectors,
                                    original.electric, original.magnetic, original.origin)
    with pytest.raises(ValueError, match='frequencies must match'):
        compile_tensor_carrier([replace(declaration, frequency=2e8)], original.medium)
    with pytest.raises(ValueError, match='Maxwell curl'):
        compile_tensor_carrier([replace(declaration, magnetic=.9*original.magnetic)], original.medium)
    for sources in ([], [object()]):
        with pytest.raises(TypeError, match='MaxwellPlaneModes'):
            compile_tensor_carrier(sources, original.medium)
    for changes in ({'frequency': True}, {'electric': original.electric.real},
                    {'wavevectors': original.wavevectors[:0]},
                    {'origin': original.origin[:2]}, {'kind': 'plane_wave'}):
        with pytest.raises(ValueError):
            replace(declaration, **changes)


def test_scene_background_sampling_clone_and_old_backend_rejection():
    import witwin.maxwell as mw
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.mom.physics import validate_scene_system
    original = carrier(False)
    background = mw.Material(
        epsilon_tensor=mw.ComplexTensor3x3(original.medium.epsilon_r.detach().cpu().tolist()),
        mu_tensor=mw.ComplexTensor3x3(original.medium.mu_r.detach().cpu().tolist(), reciprocal=False))
    declaration = MaxwellPlaneModes(original.medium.frequency, original.wavevectors,
                                    original.electric, original.magnetic, original.origin)
    scene = mw.Scene(background=background, sources=[declaration],
        structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2.))])
    clone = scene.clone()
    assert clone.background is background
    exterior, interiors = validate_scene_system(clone, original.medium.frequency, tensor_exterior=True)
    assert not exterior.reciprocal
    torch.testing.assert_close(exterior.epsilon_r, original.medium.epsilon_r, rtol=0, atol=0)
    torch.testing.assert_close(exterior.mu_r, original.medium.mu_r, rtol=0, atol=0)
    assert len(interiors) == 1
    compiled = compile_tensor_carrier(clone.sources, exterior)
    assert compiled.medium is exterior
    with pytest.raises(NotImplementedError, match='dyadic exterior runtime'):
        validate_scene_system(clone, original.medium.frequency)
    with pytest.raises(NotImplementedError, match='Grid preparation'):
        prepare_scene(clone)


@pytest.mark.parametrize('dimension', [1, 2, 3])
def test_dyadic_configuration_restores_budgets_and_binds_actual_kernel(dimension):
    import json
    from witwin.maxwell.mom.periodic_config import MoMDyadicPeriodicConfig
    from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
    from witwin.maxwell.mom.periodic_dyadic import QuasiperiodicDyadicRegularKernel
    original = carrier()
    config = MoMDyadicPeriodicConfig(
        vectors_m=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))[:dimension],
        shells=4, max_shells=8, height_m=.1, max_halvings=3, absolute_tolerance=.02,
        normal=(0., 0., 1.) if dimension == 1 else None, cache_max_bytes=0)
    restored = MoMDyadicPeriodicConfig.restore(json.loads(json.dumps(config.payload())))
    assert restored == config
    kernel = restored.compile(medium=original.medium, integration=DyadicIntegrationConfig())
    assert isinstance(kernel, QuasiperiodicDyadicRegularKernel)
    assert kernel.medium is original.medium
    assert kernel.lattice.dimension == dimension
    assert kernel.kernel.cache_max_bytes == 0
    assert kernel.integration.absolute_tolerance == config.absolute_tolerance/8
    assert kernel.integration.relative_tolerance == 0
    for changes in ({'shells': True}, {'height_m': float('nan')}, {'max_halvings': 2},
                    {'cache_max_bytes': -1}, {'unexpected': 1}, {'version': 'mom_bravais_ewald_config_v1'}):
        with pytest.raises(ValueError):
            MoMDyadicPeriodicConfig.restore(dict(config.payload(), **changes))


@pytest.mark.parametrize('trainable', [False, True])
def test_background_perturbation_cannot_be_silently_sampled_as_base_material(trainable):
    import witwin.maxwell as mw
    from witwin.maxwell.simulation import _scene_trainable_material_parameters
    from witwin.maxwell.mom.physics import validate_scene_system
    perturbation = torch.ones(1, 1, 1, device='cuda', requires_grad=trainable)
    background = mw.PerturbationMedium(mw.Material(eps_r=2.), perturbation=perturbation)
    scene = mw.Scene(background=background)
    tracked = _scene_trainable_material_parameters(scene)
    assert len(tracked) == int(trainable)
    if trainable:
        assert tracked[0] is perturbation
        assert mw.Simulation.mom(scene, frequency=3e8).has_trainable_parameters
    with pytest.raises(NotImplementedError, match='perturbation fields'):
        validate_scene_system(scene, 3e8, tensor_exterior=True)
