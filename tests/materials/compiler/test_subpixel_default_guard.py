"""The forward default requires fixed geometry and region tensors; gradient modes are explicit."""
import re

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene


MESSAGE = (
    "kottke_edge (the default subpixel averaging) is not differentiable in geometry; "
    "set SubpixelSpec(averaging='kottke_smooth') or 'polarized' for shape gradients."
)


class GeometryScene(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.size = torch.nn.Parameter(torch.tensor(.37))

    def to_scene(self):
        return scene(geometry=mw.Box(size=(self.size, .41, .43)))


def scene(*, geometry=None, material=None, spec=None, region=False, density=None):
    kwargs = {} if spec is None else {"subpixel_samples": spec}
    geometry = mw.Box(size=(.37, .41, .43)) if geometry is None else geometry
    material = mw.Material(eps_r=4.) if material is None else material
    result = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5),) * 3),
        grid=mw.GridSpec.uniform(.125), boundary=mw.BoundarySpec.none(),
        device="cpu", **kwargs,
    )
    if region:
        result.add_material_region(mw.MaterialRegion(
            name="design", geometry=geometry,
            density=torch.full((2, 2, 2), .5) if density is None else density,
            eps_bounds=(1., 4.),
        ))
    else:
        result.add_structure(mw.Structure(geometry, material))
    return result


@pytest.mark.parametrize("kind", ["tensor", "nested", "region", "module"])
@pytest.mark.parametrize("explicit", [False, True])
def test_trainable_geometry_fails_closed(kind, explicit):
    parameter = torch.tensor(.37, requires_grad=True)
    geometry = mw.Box(size=(parameter, .41, .43))
    if kind == "nested":
        geometry = mw.Difference(geometry, mw.Sphere(radius=.1))
    value = GeometryScene().to_scene() if kind == "module" else scene(
        geometry=geometry, region=kind == "region",
    )
    if explicit:
        value.subpixel = mw.SubpixelSpec(samples=4, averaging="kottke_edge")
    with pytest.raises(ValueError, match=re.escape(MESSAGE)):
        prepare_scene(value).compile_materials()
    assert value.subpixel.averaging == "kottke_edge"


def test_default_allows_material_only_gradients():
    epsilon = torch.tensor(4., requires_grad=True)
    value = scene(material=mw.Material(eps_r=epsilon))
    model = prepare_scene(value).compile_materials()
    gradient, = torch.autograd.grad(sum(x.sum() for x in model["edge_components"]["eps"].values()), epsilon)
    assert torch.isfinite(gradient) and gradient > 0
    assert value.subpixel == mw.SubpixelSpec()


def test_trainable_density_requires_explicit_averaging():
    density = torch.full((2, 2, 2), .5, requires_grad=True)
    value = scene(region=True, density=density)
    message = (
        "kottke_edge (the default subpixel averaging) does not support gradients "
        "through MaterialRegion design densities; set SubpixelSpec(samples=1, averaging='arithmetic') "
        "for density/topology optimization."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        prepare_scene(value).compile_materials()

    value.subpixel = mw.SubpixelSpec(averaging="polarized")
    prepare_scene(value).compile_materials()


@pytest.mark.parametrize("mode", ["kottke_smooth", "polarized"])
def test_explicit_differentiable_geometry_modes_are_allowed(mode):
    size = torch.tensor(.37, requires_grad=True)
    value = scene(geometry=mw.Box(size=(size, .41, .43)),
                  spec=mw.SubpixelSpec(samples=4, averaging=mode))
    model = prepare_scene(value).compile_materials()
    gradient, = torch.autograd.grad(sum(x.sum() for x in model["edge_components"]["eps"].values()), size)
    assert torch.isfinite(gradient) and gradient.abs() > 0
    assert value.subpixel.averaging == mode
