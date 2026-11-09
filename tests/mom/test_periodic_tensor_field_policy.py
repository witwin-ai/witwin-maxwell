"""Independent field gates cannot be masked by the other tensor field."""
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0,MU_0
from witwin.maxwell.mom.runtime import SystemDecodeHandle
from witwin.maxwell.mom.mesh import SurfaceSystem
from witwin.maxwell.mom.tensor_medium import TensorMedium
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.periodic_fields import decode_periodic,PeriodicFieldConvergenceError
from test_generalized_system_blocks import basis_at

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('varying_field',(0,1),ids=('electric','magnetic'))
@pytest.mark.parametrize('trace_side',(None,'interior'))
def test_tensor_periodic_refinement_separates_fields_and_wraps_trace(varying_field,trace_side,monkeypatch):
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    medium=TensorMedium(3e8,torch.diag(eye.new_tensor([2.,3.,4.])),eye)
    basis=basis_at(0.)
    system=SurfaceSystem.create([(0,'tensor',basis,medium)])
    config=mw.MoMConfig(integration=IntegrationConfig(order=1,singular_order=1),
        dyadic_integration=mw.DyadicIntegrationConfig(),
        periodic=mw.MoMPeriodicConfig(((2.,0.,0.),),bloch_rad_per_m=(.2,0.,0.)))
    handle=SystemDecodeHandle(system,compile_medium(mw.Material(),3e8),[],config,{})
    kernel=config.periodic.compile(device='cuda',dtype=torch.complex128)
    point=(basis.mesh.vertices.mean(0,keepdim=True) if trace_side is None else
           basis.mesh.triangles[:1].mean(1))
    shifted=point+kernel.lattice.vectors[0]
    eta0=math.sqrt(MU_0/EPSILON_0)
    calls=[]
    # Manufactured order dependence isolates the error gate and routing. It is
    # not a tensor field oracle or an accuracy test of surface quadrature.
    def fields(local,coefficients,wrapped,region,kind,side,*,query_config):
        torch.testing.assert_close(wrapped,point,rtol=0,atol=1e-15)
        assert region==1 and kind=='total' and side==trace_side
        calls.append(local.integration.order)
        values=[eye.new_zeros((1,3)),eye.new_zeros((1,3))]
        values[varying_field][0,0]=0. if local.integration.order==1 else 2e-6
        values[1-varying_field][0,0]=1e8
        values[1]/=eta0
        return tuple(values)
    from witwin.maxwell.mom import periodic_fields
    monkeypatch.setattr(periodic_fields,'decode_system',fields)
    def forbidden(*args,**kwargs):
        raise AssertionError('A tensor interior must not receive periodic image potentials')
    monkeypatch.setattr(periodic_fields,'regular_surface_fields',forbidden)
    coefficients=eye.new_zeros(2*basis.n_edges)
    policy=mw.FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-9,max_order=2)
    with pytest.raises(PeriodicFieldConvergenceError) as failure:
        decode_periodic(handle,coefficients,shifted,1,'total',trace_side,
                        kernel=kernel,query_config=policy)
    check=failure.value.receipt['checks'][-1]
    assert not check['passed']
    assert check['absolute_difference'][0][varying_field]>check['threshold'][0][varying_field]
    assert calls==[1,2]
    from dataclasses import replace
    calls.clear()
    result=decode_periodic(handle,coefficients,shifted,1,'total',trace_side,kernel=kernel,
                           query_config=replace(policy,max_order=4))
    assert calls==[1,2,4] and handle.last_periodic_field_log['converged']
    assert handle.last_periodic_field_log['field_error_metric']=='separate_E_and_vacuum_impedance_H'
    scale=1. if varying_field==0 else eta0
    torch.testing.assert_close(result[varying_field][0,0]*scale,
        eye.new_tensor(2e-6)*torch.exp(eye.new_tensor(.4j)),rtol=1e-13,atol=1e-15)
