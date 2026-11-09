"""Actual periodic scalar matrix/RHS/field integration, not physical certification."""
import runpy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.runtime import SystemDecodeHandle
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import DenseOperator,CompactRWGRiesz,StoppingRule,solve_reference
from witwin.maxwell.mom.periodic import LatticeSpec,EwaldHelmholtzKernel
from witwin.maxwell.mom.periodic_operator import assemble_periodic_system,build_periodic_rhs
from witwin.maxwell.mom.periodic_fields import decode_periodic
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dimension',[1,2])
def test_actual_periodic_scalar_solve_fields_and_zero_contrast(dimension,tmp_path):
    original,_=runpy.run_path('tests/mom/test_surface_volume.py')['problem']()
    body=original.system.bodies[0]
    system=SurfaceSystem.create([(body.structure_index,body.name,body.basis,body.interior)])
    sample=body.mesh.vertices
    vectors=sample.new_tensor([[.7,0.,0.],[0.,.8,0.]])[:dimension]
    lattice=LatticeSpec(vectors,sample.new_zeros(3))
    kernel=EwaldHelmholtzKernel(lattice,3.5,real_shells=3,reciprocal_shells=4,quadrature_order=64)
    config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3))
    decoder=SystemDecodeHandle(system,original.exterior,[mw.PlaneWave()],config,{})
    matrix,assembly=assemble_periodic_system(system,original.exterior,config.integration,kernel=kernel)
    rhs,rhs_receipt=build_periodic_rhs(decoder,kernel=kernel)
    metric=CompactRWGRiesz(body.basis.mass_matrix(),abs(original.exterior.eta)**2)
    operator=DenseOperator(matrix,metric,{})
    coefficients,receipt=solve_reference(operator,rhs,StoppingRule(method='direct',rtol=1e-10,atol=1e-14))
    assert receipt['converged'] and coefficients.norm()>0
    true_residual=(matrix@coefficients-rhs).norm()/rhs.norm()
    assert true_residual<1e-10
    fixed=FieldQueryConfig(adaptive=False)
    outside=sample.new_tensor([[.17,.14,.13]])
    fields=decode_periodic(decoder,coefficients,outside,0,'total',kernel=kernel,query_config=fixed)
    translated=decode_periodic(decoder,coefficients,outside+vectors.sum(0),0,'total',kernel=kernel,query_config=fixed)
    for a,b in zip(fields,translated):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    inside=decode_periodic(decoder,coefficients,sample.mean(0,keepdim=True),1,'total',kernel=kernel,query_config=fixed)
    assert all(torch.isfinite(value).all() for value in (*fields,*inside))
    # Transparent material must produce no scattered forcing, not merely a small residual.
    matched=SurfaceSystem.create([(body.structure_index,body.name,body.basis,original.exterior)])
    matched_decoder=SystemDecodeHandle(matched,original.exterior,[mw.PlaneWave()],config,{})
    zero,_=build_periodic_rhs(matched_decoder,kernel=kernel)
    assert torch.count_nonzero(zero)==0
    incident=matched_decoder.incident_carrier.evaluate(outside,original.exterior)
    transparent=decode_periodic(matched_decoder,torch.zeros_like(coefficients),outside,0,'total',kernel=kernel,query_config=fixed)
    for a,b in zip(transparent,incident):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    evidence={'dimension':dimension,'true_relative_residual':float(true_residual),
        'coefficient_norm':float(coefficients.norm()),'physical_qualification':False,
        'assembly':assembly,'rhs':rhs_receipt,'solver':receipt}
    if dimension==2:
        from witwin.maxwell.mom.periodic_diffraction import rayleigh_power_balance
        axis=torch.arange(-3,4,device=sample.device)
        orders=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        ledger=rayleigh_power_balance(decoder,coefficients,lattice=lattice,orders=orders,
            lower_plane=-.5,upper_plane=.5,quadrature_order=6)
        refined_axis=torch.arange(-6,7,device=sample.device)
        refined_orders=torch.stack(torch.meshgrid(refined_axis,refined_axis,indexing='ij'),-1).reshape(-1,2)
        deeper=rayleigh_power_balance(decoder,coefficients,lattice=lattice,orders=refined_orders,
            lower_plane=-.5,upper_plane=.5,quadrature_order=6)
        quadrature=rayleigh_power_balance(decoder,coefficients,lattice=lattice,orders=orders,
            lower_plane=-.5,upper_plane=.5,quadrature_order=12)
        for key in ('scattered_power_W','extinction_power_W','flux_inferred_absorption_W'):
            torch.testing.assert_close(ledger[key],deeper[key],rtol=1e-8,atol=1e-15)
            torch.testing.assert_close(ledger[key],quadrature[key],rtol=1e-8,atol=1e-15)
        torch.testing.assert_close(ledger['net_total_outward_power_W'],
            -ledger['flux_inferred_absorption_W'],rtol=1e-8,atol=1e-15)
        assert ledger['net_incident_outward_power_W'].abs()<1e-15
        transparent_ledger=rayleigh_power_balance(matched_decoder,torch.zeros_like(coefficients),
            lattice=lattice,orders=orders,lower_plane=-.5,upper_plane=.5)
        assert transparent_ledger['lossless_balance_relative_defect']==0
        assert transparent_ledger['scattered_power_W']==0
        assert transparent_ledger['flux_inferred_absorption_W']==0
        evidence['power_balance']=ledger
        print({key:float(ledger[key]) for key in ('scattered_power_W','extinction_power_W',
            'flux_inferred_absorption_W','lossless_balance_relative_defect')})
        assert not ledger['physical_qualification']
    torch.save({'coefficients':coefficients,'outside_fields':fields,'inside_fields':inside,
        'evidence':evidence},tmp_path/'periodic-solve.pt')
    print({k:evidence[k] for k in ('dimension','true_relative_residual','coefficient_norm','physical_qualification')})
