"""Independent Ampere/source-work sign controls for terminal gap RF currents."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import compile_port_geometry, _compile_current
from witwin.maxwell.fdtd.coords import component_coords
from witwin.maxwell.scene import prepare_scene


@pytest.mark.parametrize('axis','xyz')
@pytest.mark.parametrize('reverse',[False,True])
def test_gap_current_opposes_ampere_source_and_delivers_positive_passive_power(axis,reverse):
    scene=prepare_scene(mw.Scene(domain=mw.Domain(bounds=((0.,1.),)*3),grid=mw.GridSpec.uniform(.1),
        boundary=mw.BoundarySpec.none(),device='cpu'))
    normal='xyz'.index(axis)
    negative=[.5]*3
    positive=[.5]*3
    negative[normal],positive[normal]=.2,.8
    if reverse:
        negative,positive=positive,negative
    center=[.5]*3
    center[normal]=.45
    size=[.5]*3
    size[normal]=0.
    port=mw.LumpedPort('gap',negative=negative,positive=positive,voltage_path=mw.AxisPath(axis),
        current_surface=mw.Box(position=torch.tensor(center,dtype=torch.float64),size=torch.tensor(size,dtype=torch.float64)))
    geometry=compile_port_geometry(scene,port)
    voltage=3.+.4j
    current=.03+.005j
    density=-current/(.5*.5)
    tangents={'x':('y','z'),'y':('z','x'),'z':('x','y')}[axis]
    fields={}
    for field in ('Ex','Ey','Ez','Hx','Hy','Hz'):
        coordinates=tuple(torch.tensor(value,dtype=torch.float64) for value in component_coords(scene,field))
        shape=tuple(len(value) for value in coordinates)
        values=torch.zeros(shape,dtype=torch.complex128)
        if field=='E'+axis:
            values.fill_(voltage/.6)
        if field=='H'+tangents[0]:
            index='xyz'.index(tangents[1]);broadcast=[1]*3;broadcast[index]=shape[index]
            values[:]=-.5*density*(coordinates[index]-.5).reshape(broadcast)
        if field=='H'+tangents[1]:
            index='xyz'.index(tangents[0]);broadcast=[1]*3;broadcast[index]=shape[index]
            values[:]=.5*density*(coordinates[index]-.5).reshape(broadcast)
        fields[field]=values
    direction=-1 if reverse else 1
    torch.testing.assert_close(geometry.integrate_voltage(fields),torch.tensor(direction*voltage,dtype=torch.complex128))
    torch.testing.assert_close(geometry.integrate_current(fields),torch.tensor(direction*current,dtype=torch.complex128))
    # The independently specified impressed J supplies -Re(E J*)/2 watts.
    supplied=-.5*((voltage/.6)*density.conjugate()*(.5*.5*.6)).real
    assert supplied>0
    assert float(geometry.average_power(fields))==pytest.approx(supplied,rel=1e-13)
    # The generic propagation-contour helper keeps physical Ampere orientation;
    # TEM WavePorts use it with a transverse E voltage arrow and must retain it.
    components,indices,weights=_compile_current(port,scene,direction,device=torch.device('cpu'))
    raw=sum(torch.sum(fields[name][tuple(index.T)]*weight) for name,index,weight in zip(components,indices,weights))
    torch.testing.assert_close(raw,torch.tensor(-direction*current,dtype=torch.complex128))
    assert ((direction*voltage)/complex(raw)).real<0
