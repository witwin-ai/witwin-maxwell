"""Rayleigh reconstruction of actual periodic equivalent-current fields."""
import runpy
import math
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SystemDecodeHandle
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_fields import decode_periodic
from witwin.maxwell.mom.periodic_diffraction import rayleigh_orders
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('side',[-1,1])
def test_rayleigh_scattered_fields_reconstruct_periodic_decoder(side):
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=original.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    sample=body.mesh.vertices
    handle=SystemDecodeHandle(system,original.exterior,[mw.PlaneWave()],mw.MoMConfig(integration=replace(original.integration,order=6,singular_order=6)),{})
    lattice=LatticeSpec(sample.new_tensor([[.7,0.,0.],[0.,.8,0.]]),sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,3.5,real_shells=3,reciprocal_shells=6)
    torch.manual_seed(8)
    currents=.01*torch.randn(2*system.total_edges,device=sample.device,dtype=torch.complex128)
    axis=torch.arange(-6,7,device=sample.device)
    orders=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    plane=side*.5
    result=rayleigh_orders(handle,currents,lattice=lattice,orders=orders,side=side,plane_offset=plane,quadrature_order=6)
    point=sample.new_tensor([[.13,.17,plane]])
    actual=decode_periodic(handle,currents,point,0,'scattered',kernel=kernel,query_config=FieldQueryConfig(adaptive=False))
    phase=torch.exp(1j*(point@result['tangential_wavevectors'].T))
    for name,expected in zip(('electric','magnetic'),actual):
        reconstructed=phase@result[name]
        torch.testing.assert_close(reconstructed,expected,rtol=1e-7,atol=1e-9)
        assert (result['wavevectors']*result[name]).sum(-1).abs().max()<1e-11
    grid=torch.arange(17,device=sample.device,dtype=sample.dtype)/17-.5
    uv=torch.stack(torch.meshgrid(grid,grid,indexing='ij'),-1).reshape(-1,2)
    plane_points=uv@lattice.vectors+plane*result['normal']
    plane_phase=torch.exp(1j*(plane_points@result['tangential_wavevectors'].T))
    e,h=plane_phase@result['electric'],plane_phase@result['magnetic']
    direct=.5*lattice.cell_measure*(torch.linalg.cross(e,h.conj()).real*(side*result['normal'])).sum(-1).mean()
    torch.testing.assert_close(result['outward_power_W'].sum(),direct,rtol=1e-12,atol=1e-14)
    grazing=LatticeSpec(sample.new_tensor([[2*math.pi/handle.exterior.k.real,0.,0.],
        [0.,2*math.pi/handle.exterior.k.real,0.]]),sample.new_zeros(3))
    with pytest.raises(ValueError,match='grazing'):
        rayleigh_orders(handle,currents,lattice=grazing,orders=orders.new_tensor([[1,0]]),
            side=side,plane_offset=plane)
    assert result['propagating_lossless'].sum()==1
    assert not result['physical_qualification'] and result['field_kind']=='scattered'
    with pytest.raises(ValueError,match='strictly outside'):
        rayleigh_orders(handle,currents,lattice=lattice,orders=orders,side=side,plane_offset=.03)


@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('skew',[False,True])
def test_coherent_total_rayleigh_plane_flux(side,skew):
    from witwin.maxwell.mom.periodic_diffraction import rayleigh_total
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=original.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    sample=body.mesh.vertices
    direction=(.6,0.,.8) if skew else (0.,0.,1.)
    polarization=(0.,1.,0.) if skew else (1.,0.,0.)
    sources=[mw.PlaneWave(direction=direction,polarization=polarization),
        mw.PlaneWave(direction=direction,polarization=polarization,source_time=mw.CW(
        frequency=original.exterior.frequency,amplitude=.7,phase=.6)),
        mw.PlaneWave(direction=(direction[0],0.,-direction[2]),polarization=polarization,source_time=mw.CW(
            frequency=original.exterior.frequency,amplitude=.3,phase=-.4))]
    handle=SystemDecodeHandle(system,original.exterior,sources,
        mw.MoMConfig(integration=replace(original.integration,order=6,singular_order=6)),{})
    lattice=LatticeSpec(sample.new_tensor([[.7,0.,0.],[0.,.8,0.]]),sample.new_zeros(3))
    if skew:
        lattice=LatticeSpec(sample.new_tensor([[1.5,.2,0.],[.1,1.4,0.]]),sample.new_zeros(3))
        tangential=sample.new_tensor([handle.exterior.k.real*.6,0.,0.])
        lattice=LatticeSpec(lattice.vectors,tangential-lattice.reciprocal[0])
    kernel=EwaldHelmholtzKernel(lattice,3.5,real_shells=3,reciprocal_shells=12 if skew else 6,
        quadrature_order=128 if skew else 64)
    torch.manual_seed(19)
    currents=.01*torch.randn(2*system.total_edges,device=sample.device,dtype=torch.complex128)
    shells=12 if skew else 6
    axis=torch.arange(-shells,shells+1,device=sample.device)
    orders=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
    plane=side*.5
    result=rayleigh_total(handle,currents,lattice=lattice,orders=orders,side=side,plane_offset=plane)
    point=sample.new_tensor([[.13,.17,plane]])
    expected=decode_periodic(handle,currents,point,0,'total',kernel=kernel,
        query_config=FieldQueryConfig(adaptive=False))
    phase=torch.exp(1j*(point@result['tangential_wavevectors'].T))
    for name,value in zip(('electric','magnetic'),expected):
        torch.testing.assert_close(phase@result[name],value,rtol=1e-7,atol=1e-9)
    grid_size=4*shells+1
    grid=torch.arange(grid_size,device=sample.device,dtype=sample.dtype)/grid_size-.5
    uv=torch.stack(torch.meshgrid(grid,grid,indexing='ij'),-1).reshape(-1,2)
    points=uv@lattice.vectors+plane*result['normal']
    phase=torch.exp(1j*(points@result['tangential_wavevectors'].T))
    e,h=phase@result['electric'],phase@result['magnetic']
    direct=.5*lattice.cell_measure*(torch.linalg.cross(e,h.conj()).real*(side*result['normal'])).sum(-1).mean()
    torch.testing.assert_close(result['outward_power_W'].sum(),direct,rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(result['outward_power_W'],result['incident_outward_power_W']+
        result['scattered']['outward_power_W']+result['interference_outward_power_W'],rtol=1e-12,atol=1e-14)
    expected_incident=side*direction[2]*.5*lattice.cell_measure/handle.exterior.eta.real*(
        abs(1+.7*complex(math.cos(.6),math.sin(.6)))**2-.3**2)
    torch.testing.assert_close(result['incident_outward_power_W'].sum(),expected_incident,rtol=1e-12,atol=1e-14)
    assert 'wavevectors' not in result and not result['physical_qualification']
    incident_index=orders.new_tensor([1,0] if skew else [0,0])
    with pytest.raises(ValueError,match='omit an incident'):
        rayleigh_total(handle,currents,lattice=lattice,orders=orders[(orders!=incident_index).any(-1)],
            side=side,plane_offset=plane)


def test_rayleigh_independent_estimates_and_unresolved_evanescent_tail():
    from witwin.maxwell.mom.periodic_diffraction import estimate_rayleigh_total,RayleighConvergenceError
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=original.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    sample=body.mesh.vertices
    handle=SystemDecodeHandle(system,original.exterior,[mw.PlaneWave()],mw.MoMConfig(),{})
    lattice=LatticeSpec(sample.new_tensor([[.7,0.,0.],[0.,.8,0.]]),sample.new_zeros(3))
    torch.manual_seed(8)
    currents=.01*torch.randn(2*system.total_edges,device=sample.device,dtype=torch.complex128)
    result=estimate_rayleigh_total(handle,currents,lattice=lattice,shells=6,side=1,plane_offset=.5)
    assert result['quadrature_converged'] and result['order_truncation_converged']
    assert not result['physical_qualification'] and not result['convergence']['certified_error_bound']
    # A near plane has non-negligible evanescent amplitudes even though all
    # retained propagating orders and cell powers can already agree.
    with pytest.raises(RayleighConvergenceError) as error:
        estimate_rayleigh_total(handle,currents,lattice=lattice,shells=1,side=1,plane_offset=.08)
    checks=error.value.receipt['checks']
    assert [check['axis'] for check in checks]==['surface_quadrature','diffraction_orders']
    tail=checks[1]['quantities']
    assert not tail[0]['passed'] and tail[1]['passed']
    assert not error.value.receipt['physical_qualification']
    print(error.value.receipt)
