"""Physical foil endpoints survive center/size ULPs without closing real gaps."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.scene import prepare_scene


def _faces(normal,shift=0.):
    transverse=[axis for axis in range(3) if axis!=normal]
    position=np.zeros(3);size=np.full(3,.006);size[normal]=0.
    low,high=.001+shift,.023
    position[transverse[0]]=(low+high)/2;size[transverse[0]]=high-low
    foil=mw.Structure(mw.Box(position=torch.tensor(position,dtype=torch.float64),
        size=torch.tensor(size,dtype=torch.float64)),
        mw.CopperFoil(thickness=35e-6,frequency_range=(.5e9,2e9)),name="foil")
    scene=mw.Scene(domain=mw.Domain(bounds=((- .030,.030),)*3),grid=mw.GridSpec.uniform(.001),
        boundary=mw.BoundarySpec.none(),structures=(foil,),device="cpu")
    prepared=prepare_scene(scene)
    return prepared,compile_copper_faces(prepared,device="cpu")


@pytest.mark.parametrize("normal",range(3))
def test_float64_reconstructed_aligned_foil_bounds_keep_all_whole_edges(normal):
    scene,faces=_faces(normal)
    tangent=next(axis for axis in range(3) if axis!=normal)
    face=next(face for face in faces if face.component==tangent)
    values=getattr(scene,"xyz"[tangent]+"_nodes64")
    unique=face.indices[:,tangent].unique().numpy()
    assert len(unique)==22
    assert values[unique[0]]==pytest.approx(.001,abs=1e-15)
    assert values[unique[-1]+1]==pytest.approx(.023,abs=1e-15)


@pytest.mark.parametrize("shift",(1e-12,1e-9,.00025))
def test_a_real_foil_endpoint_offset_does_not_acquire_a_whole_grid_edge(shift):
    scene,faces=_faces(2,shift)
    face=next(face for face in faces if face.component==0)
    nodes=scene.x_nodes64
    assert nodes[int(face.indices[:,0].min())]==pytest.approx(.002,abs=1e-15)
