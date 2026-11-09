"""Recovered tensor source queries, native gradients and owning kernel guards."""
import copy
from dataclasses import replace
import pytest
import torch
from test_tensor_volume_source import setup
from witwin.maxwell.mom.surface_volume import VolumeFieldResponse,CoupledFieldConvergenceError
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.lifting_fields import decode_system
from witwin.maxwell.mom.tensor_volume import cell_tensor_current_fields
from witwin.maxwell.mom.dyadic_green import DyadicGreenKernel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_owned_tensor_volume_query_composition_gradient_and_mutation():
    op,binding=setup();kernel=op.body_handles[0].tensor_kernel
    current=torch.tensor([[.2+.1j,-.3j,.4]],device='cuda',dtype=torch.complex128,requires_grad=True)
    coefficients=torch.linspace(.001,.002,op.shape[0],device='cuda',dtype=torch.float64).to(torch.complex128).requires_grad_()
    points=binding.volume.vertices.new_tensor([[.013,.017,.011]])
    fixed=FieldQueryConfig(adaptive=False)
    view=VolumeFieldResponse(op,(binding,),(current,),coefficients,order=3,query_config=fixed)
    actual=view.fields(points,1)
    surface=decode_system(op,coefficients,points,1,'total',query_config=fixed)
    volume,receipt=cell_tensor_current_fields(binding.volume,current,points,kernel,order=3,radial_order=3)
    expected=tuple(a+b for a,b in zip(surface,volume))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-10)
    ga=torch.autograd.grad(sum(x.abs().square().sum() for x in actual),(current,coefficients))
    gb=torch.autograd.grad(sum(x.abs().square().sum() for x in expected),(current,coefficients))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=2e-11,atol=1e-8)
    assert receipt['static_diagonal_included'] and not receipt['volume_quadrature_converged']
    assert view.last_field_receipt['volume']['representation']=='tensor_volume_electric_current_fields'
    outside=points.new_tensor([[.3,.2,.4]])
    external=view.fields(outside,0)
    reference=decode_system(op,coefficients,outside,0,'total',query_config=fixed)
    for a,b in zip(external,reference):torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-10)
    assert view.last_field_receipt['volume'] is None
    op.body_handles[0].tensor_kernel=DyadicGreenKernel(kernel.medium,kernel.integration)
    with pytest.raises(RuntimeError,match='mutated'):view.fields(points,1)
    op.body_handles[0].tensor_kernel=None
    with pytest.raises(ValueError,match='owning regional'):view.fields(points,1)
    op.body_handles[0].tensor_kernel=kernel
    with pytest.raises(ValueError,match='native cell'):
        cell_tensor_current_fields(binding.volume,current.real,points,kernel)
    with torch.no_grad():view.currents[0].add_(.1)
    with pytest.raises(RuntimeError,match='mutated'):view.fields(points,1)


def test_cloned_tensor_query_view_and_complete_error_exhaustion():
    original,binding=setup();op=copy.copy(original)
    op.body_handles=tuple(copy.copy(h) for h in original.body_handles)
    for h in (op,*op.body_handles):
        h.incident_carrier=replace(h.incident_carrier,waves=())
    current=torch.ones(1,3,device='cuda',dtype=torch.complex128)
    coefficients=torch.zeros(op.shape[0],device='cuda',dtype=torch.complex128)
    config=FieldQueryConfig(relative_tolerance=1e-14,absolute_tolerance=1e-20,max_order=16)
    view=VolumeFieldResponse(op,(binding,),(current,),coefficients,order=2,query_config=config)
    points=binding.volume.vertices.new_tensor([[.013,.017,.011]])
    with pytest.raises(CoupledFieldConvergenceError) as info:view.fields(points,1)
    r=info.value.receipt
    assert r['failure_reason']=='coupled_field_order_budget_exhausted'
    assert r['field_error_metric']=='separate_E_and_vacuum_impedance_H'
    assert r['volume']['query_count']==1 and not r['physical_qualification']
    assert not r['converged'] and r['checks']


def test_actual_anisotropic_volume_query_uses_owned_current_map():
    op,binding=setup(anisotropic=True)
    current=torch.tensor([[.2,-.1,.3j]],device='cuda',dtype=torch.complex128)
    coefficients=torch.zeros(op.shape[0],device='cuda',dtype=torch.complex128)
    points=binding.volume.vertices.new_tensor([[.013,.017,.011]])
    # Zero surface currents isolate direct volume composition. The real
    # tensor kernel still runs; this does not certify a recovered solution.
    view=VolumeFieldResponse(op,(binding,),(current,),coefficients,order=2,
        query_config=FieldQueryConfig(adaptive=False))
    actual=view.fields(points,1)
    expected,_=cell_tensor_current_fields(binding.volume,current,points,op.body_handles[0].tensor_kernel,
        order=2,radial_order=2)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-11,atol=1e-10)
    assert all(x['converged'] for x in view.last_field_receipt['volume']['spectral_evaluations'])


def test_nonfinite_current_product_and_stable_complete_field_metric(monkeypatch):
    from witwin.maxwell.mom import lifting_fields
    from witwin.maxwell.mom.operator import IntegrationConfig
    op,binding=setup();points=binding.volume.vertices.new_tensor([[.013,.017,.011]])
    kernel=op.body_handles[0].tensor_kernel
    huge=torch.full((1,3),1e308,device='cuda',dtype=torch.complex128)
    with pytest.raises(ValueError,match='fields are not finite'):
        cell_tensor_current_fields(binding.volume,huge,points,kernel,order=2,radial_order=2)
    op.integration=IntegrationConfig(order=2,singular_order=2)
    op.body_handles[0].integration=op.integration
    def finite_large_surface(handle,coefficients,points,*args,**kwargs):
        order=handle.body_handles[0].integration.order
        field=torch.full_like(points,1e200 if order==2 else 1.1e200,dtype=torch.complex128)
        handle.last_field_query_log={'manufactured_finite_large_surface':True}
        return field,torch.zeros_like(field)
    monkeypatch.setattr(lifting_fields,'decode_system',finite_large_surface)
    zeros=torch.zeros(1,3,device='cuda',dtype=torch.complex128)
    coefficients=torch.zeros(op.shape[0],device='cuda',dtype=torch.complex128)
    config=FieldQueryConfig(relative_tolerance=.01,absolute_tolerance=0,max_order=4)
    view=VolumeFieldResponse(op,(binding,),(zeros,),coefficients,order=2,query_config=config)
    with pytest.raises(CoupledFieldConvergenceError) as info:view.fields(points,1)
    r=info.value.receipt
    assert r['failure_reason']=='coupled_field_order_budget_exhausted'
    assert not r['checks'][0]['E']['passed']
    assert r['checks'][0]['E']['maximum_relative_difference']==pytest.approx(1/11,rel=1e-12)
    with pytest.raises(CoupledFieldConvergenceError) as info:
        view.fields(points,1,query_config=replace(config,relative_tolerance=1e200))
    assert info.value.receipt['failure_reason']=='nonfinite_coupled_error_metric'
    assert not info.value.receipt['converged']
