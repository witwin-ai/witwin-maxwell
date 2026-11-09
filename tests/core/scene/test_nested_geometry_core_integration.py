"""CPU Scene compiler contracts for nested tensor-authored core geometry."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.adjoint_inputs import scene_trainable_material_tensors, ancestry_safe_trainable_tensors
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('slot', ['position', 'size', 'rotation'])
@pytest.mark.parametrize('shared_epsilon', [False, True])
def test_scene_preserves_nested_geometry_ancestry_and_stack_material_compilation(slot, shared_epsilon):
    theta = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
    if slot == 'size':
        values = (.55+theta, .45+theta*.2, .35+theta*.3)
    elif slot == 'position':
        values = (theta, .05+theta*.2, theta*.1)
    else:
        values = (theta, theta*.7, theta*.3)

    def build(stacked):
        geometry = mw.Box(**{slot: torch.stack(values) if stacked else values})
        material = mw.Material(eps_r=2.+theta if shared_epsilon else 2.1)
        return mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"), domain=mw.Domain(bounds=((-1.,1.),)*3),
            grid=mw.GridSpec.uniform(.2), boundary=mw.BoundarySpec.none(),
            structures=(mw.Structure(geometry, material),),device='cpu')

    nested, stacked = build(False), build(True)
    actual, expected = getattr(nested.structures[0].geometry,slot), getattr(stacked.structures[0].geometry,slot)
    assert actual.dtype == expected.dtype == torch.float64 and actual.requires_grad
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    candidates=scene_trainable_material_tensors(nested)
    assert any(tensor is actual for tensor in candidates)
    safe=ancestry_safe_trainable_tensors(candidates)
    assert safe
    tensor=prepare_scene(nested).permittivity
    reference=prepare_scene(stacked).permittivity
    torch.testing.assert_close(tensor,reference,rtol=0,atol=0)
    weights=torch.linspace(.1,1.7,tensor.numel(),dtype=tensor.dtype).reshape(tensor.shape)
    gx=torch.autograd.grad((tensor*weights).sum(),theta,retain_graph=True)[0]
    gy=torch.autograd.grad((reference*weights).sum(),theta)[0]
    torch.testing.assert_close(gx,gy,rtol=0,atol=0)
    assert gx.abs()>1e-4


def test_shared_zero_derivative_geometry_still_has_structural_dependency():
    theta=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    geometry=mw.Box(position=(theta*0.,0.,0.))
    scene=mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=2, averaging="polarized"), domain=mw.Domain(bounds=((-1.,1.),)*3),grid=mw.GridSpec.uniform(.5),
        structures=(mw.Structure(geometry,mw.Material(eps_r=theta)),),device='cpu')
    actual=scene.structures[0].geometry.position
    assert actual.requires_grad and actual.grad_fn is not None
    dependency,=torch.autograd.grad(actual.sum(),theta,allow_unused=True)
    assert dependency is not None and dependency==0
    assert any(tensor is actual for tensor in scene_trainable_material_tensors(scene))
