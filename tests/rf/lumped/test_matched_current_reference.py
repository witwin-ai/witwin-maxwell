"""Independent physical source-current quadrature and midpoint phase audits."""
import math
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.compiler.ports import compile_port_geometry
from scripts.dev.fdtd.matched_current_reference import compile_known_current_edges,known_current_spectrum


@pytest.mark.parametrize('axis',(0,1,2))
@pytest.mark.parametrize('direction',(-1,1))
def test_reference_current_edges_match_native_weighted_voltage_volume_and_work(axis,direction):
    start=np.zeros(3);stop=np.zeros(3);start[axis]=-.005;stop[axis]=.01
    if direction<0:start,stop=stop,start
    center=np.zeros(3);center[axis]=.0025
    size=np.full(3,.01);size[axis]=.015
    contour=np.zeros(3);contour[axis]=-.0025
    contour_size=np.full(3,.015);contour_size[axis]=0.
    box=lambda p,s:mw.Box(position=torch.tensor(p,dtype=torch.float64),size=torch.tensor(s,dtype=torch.float64))
    port=mw.LumpedPort('feed',positive=stop,negative=start,voltage_path=mw.AxisPath('xyz'[axis]),
        voltage_aperture=box(center,size),current_surface=box(contour,contour_size))
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,.02),)*3),grid=mw.GridSpec.uniform(.005),
        ports=(port,),structures=(mw.Structure(geometry=mw.Box(size=(.1,.1,.1)),
        material=mw.Material(eps_r=2.4,sigma_e=.02)),),device='cpu')
    prepared=prepare_scene(scene);native=compile_port_geometry(prepared,port,device='cpu')
    axes=tuple(getattr(prepared,name+'_nodes64') for name in 'xyz')
    epsilon=2.4*mw.constants.EPSILON_0;dt=1e-12
    edges=compile_known_current_edges(axes,axis=axis,path=(start[axis],stop[axis]),
        aperture_bounds={i:(-.005,.005) for i in range(3) if i!=axis},dt=dt,epsilon=epsilon,conductivity=.02)
    actual={edge.index:edge for edge in edges}
    assert set(actual)==set(map(tuple,native.voltage_indices.tolist()))
    for index,weight in zip(native.voltage_indices.tolist(),native.voltage_weights.tolist()):
        edge=actual[tuple(index)]
        assert edge.voltage_weight==pytest.approx(weight,rel=3e-14)
        volume=np.prod([getattr(prepared,'d'+'xyz'[i]+('_primal64' if i==axis else '_dual64'))[index[i]] for i in range(3)])
        assert edge.edge_length*edge.dual_area==pytest.approx(volume,rel=2e-14)
        assert edge.soft_e_per_amp==pytest.approx(dt/(epsilon+.02*dt/2)*weight/volume,rel=3e-14)
    generator=np.random.default_rng(1267);electric=generator.normal(size=len(edges));current=.23
    field_work=sum(electric[i]*edge.current_density_per_amp*current*edge.edge_length*edge.dual_area for i,edge in enumerate(edges))
    measured_voltage=sum(electric[i]*edge.voltage_weight for i,edge in enumerate(edges))
    assert field_work==pytest.approx(measured_voltage*current,rel=3e-14,abs=3e-18)
    for normal in set(edge.index[axis] for edge in edges):
        assert sum(edge.current_density_per_amp*edge.dual_area for edge in edges if edge.index[axis]==normal)==pytest.approx(direction,rel=3e-14)


def test_radial_cylindrical_source_conserves_known_flux_at_every_radius():
    axes=(np.linspace(.2,.4,9),np.linspace(-.5,.5,11),np.linspace(-.1,.1,11))
    dt=1e-12;eps=2.4*mw.constants.EPSILON_0
    edges=compile_known_current_edges(axes,axis=0,path=(.35,.25),aperture_bounds={1:(-.2,.2),2:(-.04,.04)},
        dt=dt,epsilon=eps,cylindrical_radial=True)
    for normal in set(edge.index[0] for edge in edges):
        assert sum(edge.current_density_per_amp*edge.dual_area for edge in edges if edge.index[0]==normal)==pytest.approx(-1.,rel=2e-14)
    # Fixed physical radial flux needs 1/r current density. No transformed
    # chart epsilon is substituted into the independent physical coefficient.
    central=[edge for edge in edges if edge.index[1:]==(5,5)]
    np.testing.assert_allclose([edge.current_density_per_amp*edge.position[0] for edge in central],
        central[0].current_density_per_amp*central[0].position[0],rtol=2e-14)


def test_known_source_spectrum_uses_authoritative_engine_midpoint_clock():
    dt=1e-12;frequency=20e9;n=np.arange(120)
    values=np.exp(-.5*((n-50)/15)**2)*np.sin(2*np.pi*frequency*n*dt)
    # Source text files deliberately truncate timestamp precision.
    time=np.array([float(f'{value:.6g}') for value in n*dt])
    exact=2*dt*np.sum(values*np.exp(2j*np.pi*frequency*(n+.5)*dt))
    actual=known_current_spectrum(time,values,[frequency],dt=dt)[0]
    assert actual==pytest.approx(exact,rel=3e-14)
    wrong_clock=2*dt*np.sum(values*np.exp(2j*np.pi*frequency*n*dt))
    assert np.angle(actual/wrong_clock)==pytest.approx(math.pi*frequency*dt,rel=3e-14)
