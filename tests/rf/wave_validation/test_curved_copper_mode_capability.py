"""An unsupported curved aperture cannot silently become a dielectric mode."""
from types import SimpleNamespace

import pytest

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.modes.conductors import copper_intersects_mode,prepare_good_conductor_context


def _scene(vertices):
    geometry=mw.SurfaceMesh(vertices,((0,1,2),),device='cpu')
    return SimpleNamespace(structures=(mw.Structure(geometry,mw.CopperFoil(thickness=35e-6,frequency_range=(.5e9,2e9)),name='curved'),))


@pytest.mark.parametrize('model',(None,'good_conductor'))
@pytest.mark.parametrize('coplanar',(False,True))
def test_curved_copper_aperture_rejects_both_default_and_planar_expansion(model,coplanar):
    vertices=((0.,-.01,-.01),(0.,.01,-.01),(0.,0.,.01)) if coplanar else ((-.01,-.01,0.),(.01,-.01,0.),(0.,.01,0.))
    scene=_scene(vertices)
    source=dict(position=(0.,0.,0.),size=(0.,.01,.01),normal_axis='x',wave_family='tem')
    with pytest.raises(NotImplementedError,match='curved finite-Et mode solve'):
        if model is None:
            copper_intersects_mode(scene,source)
        else:
            prepare_good_conductor_context(SimpleNamespace(scene=scene),source)


@pytest.mark.parametrize('vertices',(
    ((.01,-.01,0.),(.02,-.01,0.),(.015,.01,0.)),
    ((0.,-.01,.021),(0.,.021,-.01),(0.,.021,.021)),
))
def test_remote_or_triangle_corner_miss_remains_available(vertices):
    # The second triangle's bounding box overlaps the aperture, but the physical
    # triangle does not: y+z >= .011 outside y/z in [-.005,.005].
    source=dict(position=(0.,0.,0.),size=(0.,.01,.01),normal_axis='x')
    assert copper_intersects_mode(_scene(vertices),source) is False
