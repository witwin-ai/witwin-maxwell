"""Independent film work, actual tile intersections and immutable diagnostics."""
import math
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_response
from witwin.maxwell.fdtd.copper_foil import copper_spectral_power
from witwin.maxwell.postprocess.fdtd_loss import conductor_region_power


def regional_record(frequency,dt,thickness):
    foil=mw.CopperFoil(thickness=thickness,frequency_range=(.8e9,1.2e9))
    response=compile_copper_response(foil)
    phasor=torch.tensor([[[2.+.3j,2.+.3j],[7.-.2j,7.-.2j]]],dtype=torch.complex128)
    area=torch.tensor([2.,4.],dtype=torch.float64)
    total,edge=copper_spectral_power(phasor,frequency,dt,response.state_space.B,response.rates,
        response.state_space.D,area,return_edge_power=True)
    angle=torch.pi*frequency*dt;s=-2j*torch.tan(angle)/dt
    admittance=response.state_space.D.to(torch.complex128)[None]+torch.einsum('ni,nj,fn->fij',
        response.state_space.B.to(torch.complex128),response.state_space.B.to(torch.complex128),1/(s[:,None]+response.rates[None]))
    original=.5*torch.einsum('fim,fij,fjm->fm',phasor.conj(),admittance,phasor).real
    original=(original*area).sum(-1)*torch.cos(angle).square()
    assert torch.equal(original,total)
    torch.testing.assert_close(edge.sum(-1),total,rtol=3e-15,atol=0.)
    return dict(coordinate_frame='cartesian',surface_power=total,surface_power_traces=[dict(power=edge,area=area,
        component='Ex',indices=torch.tensor([[0,0,1],[1,0,1]]),
        lower_bounds=torch.tensor([[0.,0.,3.],[1.,0.,3.]],dtype=torch.float64),
        upper_bounds=torch.tensor([[1.,2.,3.],[3.,2.,3.]],dtype=torch.float64),
        normal_axis='z',normal_coordinate=3.)]),phasor


@pytest.mark.parametrize('ratio',[.1,1.,10.])
def test_regional_heat_matches_independent_finite_film_admittance_and_legacy_total(ratio):
    mu,sigma,eps=1.25663706212e-6,5.8e7,8.8541878128e-12
    frequency=torch.tensor([1e9],dtype=torch.float64);dt=1e-12
    delta=math.sqrt(2/(2*math.pi*1e9*mu*sigma));thickness=ratio*delta
    record,phasor=regional_record(frequency,dt,thickness)
    # Full conductive-layer ABCD-to-Y conversion, including displacement current.
    omega=2*math.tan(math.pi*1e9*dt)/dt;s=-1j*omega
    gamma=np.sqrt(s*mu*(sigma+s*eps));z=s*mu/gamma
    diagonal=np.cosh(gamma*thickness)/(z*np.sinh(gamma*thickness))
    cross=-1/(z*np.sinh(gamma*thickness));y=np.array([[diagonal,cross],[cross,diagonal]])
    e=phasor[0,:,0].numpy();density=.5*np.real(e.conj()@y@e)*math.cos(math.pi*1e9*dt)**2
    actual=conductor_region_power(record,((.25,2.),(.5,1.5),(2.,4.)))
    assert actual.item()==pytest.approx(density*1.75,rel=3e-4)
    a=conductor_region_power(record,((0.,1.37),(0.,2.),(2.,4.)))
    b=conductor_region_power(record,((1.37,3.),(0.,2.),(2.,4.)))
    torch.testing.assert_close(a+b,record['surface_power'],rtol=3e-15,atol=0.)
    # The collapsed plane is assigned to [lo,hi); no finite thickness is guessed.
    torch.testing.assert_close(conductor_region_power(record,((0.,3.),(0.,2.),(3.,4.))),record['surface_power'],rtol=3e-15,atol=0.)
    assert conductor_region_power(record,((0.,3.),(0.,2.),(2.,3.))).item()==0.


def test_missing_roi_geometry_and_periodic_fragments_are_rejected():
    with pytest.raises(ValueError,match='per-edge work'):
        conductor_region_power(dict(surface_power=torch.ones(1)),((0.,1.),)*3)
    record,_=regional_record(torch.tensor([1e9],dtype=torch.float64),1e-12,35e-6)
    record['periodic_axes']=('x',)
    with pytest.raises(NotImplementedError,match='wrapped surface-tile'):
        conductor_region_power(record,((0.,1.),)*3)
    record['periodic_axes']=()
    record['coordinate_frame']='mapped_chart'
    with pytest.raises(NotImplementedError,match='mapped partial-tile metrics'):
        conductor_region_power(record,((0.,1.),)*3)


def test_diagnostic_ownership_clones_nested_tensors_without_detaching_existing_live_graphs():
    parameter=torch.tensor(3.,requires_grad=True)
    record={'surface_power_traces':[{'power':parameter.square().reshape(1)}]}
    result=mw.Result(method='fdtd',frequency=1e9,
        scene=mw.Scene(domain=mw.Domain(bounds=((0.,1.),)*3),grid=mw.GridSpec.uniform(.5),device='cpu'),
        metadata={'conductor_losses':{'foil':record},'conductor_loss_columns':{'feed':{'foil':record}}})
    a=result.conductor_losses['foil']['surface_power_traces'][0]['power']
    b=result.conductor_loss_columns['feed']['foil']['surface_power_traces'][0]['power']
    derivative,=torch.autograd.grad(a.sum()+b.sum(),parameter)
    assert derivative.item()==12.
    a.zero_()
    assert result.conductor_losses['foil']['surface_power_traces'][0]['power'].item()==9.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_actual_spectral_diagnostic_geometry_and_half_volume_survive_result_roundtrip(tmp_path):
    from tests.materials.surface_impedance.test_copper_foil_power_loss import _simulation
    result=_simulation(graph=True,steps=128).run()
    before=result.conductor_losses
    bounds=((-1.,1.),)*3
    for record in before.values():
        total=conductor_region_power(record,bounds)
        torch.testing.assert_close(total,record['surface_power'],rtol=4e-15,atol=0.)
        for trace in record['bulk_electric_traces']:
            assert trace['normal_axis']=='x'
            assert trace['normal_coordinate']==0.
            torch.testing.assert_close(trace['normal_half_widths'],torch.full_like(trace['normal_half_widths'],.0025))
    path=tmp_path/'regional.pt';result.save(path)
    loaded=mw.Result.load(path,scene=result.scene,map_location='cpu')
    for name,record in loaded.conductor_losses.items():
        torch.testing.assert_close(conductor_region_power(record,bounds),before[name]['surface_power'],rtol=4e-15,atol=0.)
        assert all(not trace['power'].requires_grad for trace in record['surface_power_traces'])
    first=next(iter(result.conductor_losses.values()))
    first['surface_power_traces'][0]['power'].zero_()
    for name,record in result.conductor_losses.items():
        torch.testing.assert_close(conductor_region_power(record,bounds),before[name]['surface_power'],rtol=4e-15,atol=0.)
