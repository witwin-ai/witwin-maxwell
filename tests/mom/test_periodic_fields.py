"""Periodic physical-field routing and current differentiation controls."""
import runpy
import math
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SystemDecodeHandle
import pytest
import torch
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_fields import decode_periodic
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.lifting_fields import decode_system

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_periodic_fields_shift_native_gradient_and_free_interior(monkeypatch):
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    wave=mw.PlaneWave(direction=(.2,0.,math.sqrt(.96)),polarization=(0.,1.,0.))
    handle=SystemDecodeHandle(original.system,original.exterior,[wave],mw.MoMConfig(integration=original.integration),{})
    sample=handle.system.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_tensor([.2*handle.exterior.k.real,0.,0.]))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=6,quadrature_order=64)
    torch.manual_seed(42)
    coefficients=torch.randn(2*handle.system.total_edges,device=sample.device,dtype=torch.complex128,requires_grad=True)
    point=sample.new_tensor([[.2,.12,.1]])
    fixed=FieldQueryConfig(adaptive=False)
    actual=decode_periodic(handle,coefficients,point,0,'total',kernel=kernel,query_config=fixed)
    shifted=decode_periodic(handle,coefficients,point+lattice.vectors[0],0,'total',kernel=kernel,query_config=fixed)
    for a,b in zip(actual,shifted):
        torch.testing.assert_close(b,a*torch.exp(1j*(lattice.vectors[0]@lattice.bloch)),rtol=1e-12,atol=1e-12)
    gradient,=torch.autograd.grad(actual[0].real.sum(),coefficients)
    delta=torch.randn_like(coefficients)
    step=1e-6
    plus=decode_periodic(handle,coefficients.detach()+step*delta,point,0,'total',kernel=kernel,query_config=fixed)
    minus=decode_periodic(handle,coefficients.detach()-step*delta,point,0,'total',kernel=kernel,query_config=fixed)
    numerical=(plus[0]-minus[0]).real.sum()/(2*step)
    torch.testing.assert_close((gradient.conj()*delta).real.sum(),numerical,rtol=1e-7,atol=1e-8)
    from witwin.maxwell.mom import periodic_fields
    monkeypatch.setattr(periodic_fields,'regular_surface_fields',lambda *a,**k:pytest.fail('Interior used periodic images'))
    inside=sample.mean(0,keepdim=True)
    interior=decode_periodic(handle,coefficients,inside,1,'total',kernel=kernel,query_config=fixed)
    free=decode_system(handle,coefficients,inside,1,'total',query_config=fixed)
    for a,b in zip(interior,free):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not handle.last_periodic_field_log['physical_qualification']


def test_periodic_fields_match_direct_images_and_complete_field_refinement():
    import copy
    from witwin.maxwell.mom.periodic import LossyImageReference
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.fields import _potentials
    from witwin.maxwell.mom.lifting_fields import system_local_coefficients
    from witwin.maxwell.mom.periodic_fields import PeriodicFieldConvergenceError
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    exterior=compile_medium(mw.Material(sigma_e=.02),3e8)
    handle=SystemDecodeHandle(original.system,exterior,[mw.PlaneWave()],mw.MoMConfig(integration=original.integration),{})
    sample=handle.system.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=8,quadrature_order=64)
    torch.manual_seed(27)
    coefficients=torch.randn(2*handle.system.total_edges,device=sample.device,dtype=torch.complex128)
    points=sample.new_tensor([[.7,.6,.5]])
    actual=decode_periodic(handle,coefficients,points,0,'scattered',kernel=kernel,
        query_config=FieldQueryConfig(adaptive=False))
    reference=[torch.zeros_like(points,dtype=coefficients.dtype) for _ in range(2)]
    for index,local in enumerate(handle.body_handles):
        oracle=copy.copy(local)
        oracle.kernel=LossyImageReference(lattice,12)
        fields=_potentials(oracle,system_local_coefficients(handle,coefficients,index),points,exterior,
            order=3,singular_order=3)
        reference=[a+b for a,b in zip(reference,fields)]
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    converged=decode_periodic(handle,coefficients,points,0,'scattered',kernel=kernel,
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-9,max_order=12))
    assert handle.last_periodic_field_log['converged']
    assert not handle.last_periodic_field_log['ewald_truncation_verified']
    assert all(torch.isfinite(value).all() for value in converged)
    with pytest.raises(PeriodicFieldConvergenceError) as failure:
        decode_periodic(handle,coefficients,points,0,'scattered',kernel=kernel,
            query_config=FieldQueryConfig(relative_tolerance=1e-14,absolute_tolerance=1e-16,max_order=4))
    assert failure.value.receipt['failure_reason']=='surface_quadrature_budget_exhausted'
    assert not failure.value.receipt['checks'][-1]['passed']


def test_analytic_incident_carrier_has_no_regular_image_field():
    from witwin.maxwell.mom.lifting import incident_trace
    from witwin.maxwell.mom.quadrature import triangle_rule
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    wave=mw.PlaneWave(direction=(.2,0.,math.sqrt(.96)),polarization=(0.,1.,0.))
    handle=SystemDecodeHandle(original.system,original.exterior,[wave],mw.MoMConfig(integration=original.integration),{})
    sample=handle.system.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_tensor([.2*handle.exterior.k.real,0.,0.]))
    kernel=EwaldHelmholtzKernel(lattice,2.,real_shells=2,reciprocal_shells=8,quadrature_order=96)
    queries=sample.new_tensor([[.2,.12,.1],[.017,.016,.015]])
    medium=handle.exterior
    errors=[]
    for order in (2,6):
        e=torch.zeros_like(queries,dtype=torch.complex128)
        h=torch.zeros_like(e)
        for local in handle.body_handles:
            mesh=local.basis.mesh
            source,w=triangle_rule(mesh.triangles,order)
            j,m,dj,dm=incident_trace(local,source,torch.arange(len(mesh.faces),device=sample.device))
            g,gradient=kernel.regular_part(queries[:,None,None]-source,medium.k)
            def potential(current,divergence):
                return ((g[...,None]*current+gradient*divergence[...,None]/medium.k**2)*w[...,None]).sum((1,2))
            def curl(current):
                return (torch.linalg.cross(gradient,current[None])*w[...,None]).sum((1,2))
            e=e+1j*medium.omega*medium.mu*potential(j,dj)-curl(m)
            h=h+curl(j)+1j*medium.omega*medium.epsilon*potential(m,dm)
        errors.append(float((e.abs().square().sum(-1)+abs(medium.eta)**2*h.abs().square().sum(-1)).sqrt().max()))
    assert errors[1]<errors[0]*1e-3
    assert errors[1]<1e-10
    print({'analytic_carrier_regular_field_V_per_m':errors})


def test_periodic_rhs_reuses_exact_local_lifting_and_rejects_wrong_sector(monkeypatch):
    from witwin.maxwell.mom.periodic_operator import build_periodic_rhs
    from witwin.maxwell.mom import lifting
    handle,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    sample=handle.system.bodies[0].mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[2.,0.,0.]]),sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,2.,reciprocal_shells=6)
    expected=handle.build_rhs()
    actual,receipt=build_periodic_rhs(handle,kernel=kernel)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-15)
    assert len(receipt['bodies'])==2 and not receipt['physical_qualification']
    wrong=EwaldHelmholtzKernel(LatticeSpec(lattice.vectors,sample.new_tensor([.2,0.,0.])),2.,reciprocal_shells=6)
    monkeypatch.setattr(lifting,'build_lifted_rhs',lambda *a,**k:pytest.fail('Wrong sector reached integration'))
    with pytest.raises(ValueError,match='Bloch vector'):
        build_periodic_rhs(handle,kernel=wrong)


def test_periodic_field_ewald_check_detects_skew_heat_error_and_preserves_gradient():
    from dataclasses import replace
    from witwin.maxwell.mom.mesh import SurfaceSystem
    from witwin.maxwell.mom.periodic_fields import PeriodicFieldConvergenceError
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=original.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    wave=mw.PlaneWave(direction=(.6,0.,.8),polarization=(0.,1.,0.))
    handle=SystemDecodeHandle(system,original.exterior,[wave],
        mw.MoMConfig(integration=replace(original.integration,order=6,singular_order=6)),{})
    sample=body.mesh.vertices
    lattice=LatticeSpec(sample.new_tensor([[1.5,.2,0.],[.1,1.4,0.]]),sample.new_zeros(3))
    lattice=LatticeSpec(lattice.vectors,sample.new_tensor([handle.exterior.k.real*.6,0.,0.])-lattice.reciprocal[0])
    kernel=EwaldHelmholtzKernel(lattice,3.5,real_shells=3,reciprocal_shells=12,quadrature_order=64)
    torch.manual_seed(19)
    coefficients=.01*torch.randn(2*system.total_edges,device=sample.device,dtype=torch.complex128)
    coefficients.requires_grad_()
    points=sample.new_tensor([[.13,.17,.5]])
    config=FieldQueryConfig(adaptive=False,relative_tolerance=1e-7,absolute_tolerance=1e-10)
    with pytest.raises(PeriodicFieldConvergenceError) as failure:
        decode_periodic(handle,coefficients,points,0,'total',kernel=kernel,query_config=config,ewald_check=True)
    receipt=failure.value.receipt
    assert receipt['failure_reason']=='independent_ewald_estimate_failed'
    assert [c['passed'] for c in receipt['ewald_checks']]==[True,True,False]
    print({'failed_ewald_checks':receipt['ewald_checks']})
    kernel=replace(kernel,quadrature_order=128)
    fields=decode_periodic(handle,coefficients,points,0,'total',kernel=kernel,query_config=config,ewald_check=True)
    receipt=handle.last_periodic_field_log
    assert receipt['ewald_truncation_verified'] and not receipt['converged']
    assert receipt['failure_reason']=='fixed_order_error_unestimated'
    assert not receipt['physical_qualification']
    gradient,=torch.autograd.grad(fields[0].real.sum(),coefficients)
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    print({'passed_ewald_checks':receipt['ewald_checks']})
    adaptive=replace(config,adaptive=True,max_order=12)
    converged=decode_periodic(handle,coefficients,points,0,'total',kernel=kernel,
        query_config=adaptive,ewald_check=True)
    receipt=handle.last_periodic_field_log
    assert receipt['converged'] and receipt['ewald_truncation_verified']
    assert receipt['final_order']==12 and receipt['failure_reason'] is None
    for actual,expected in zip(converged,fields):
        torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-10)
