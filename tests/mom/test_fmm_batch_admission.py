"""Shared-space CUDA admission, ordered failures and actual M2L ownership."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
from witwin.maxwell.mom.cluster import build_support_octree
from witwin.maxwell.mom.fmm_moments import NestedRWGMultipoles
from witwin.maxwell.mom.fmm_translation import ElectromagneticM2L
from witwin.maxwell.mom.fmm_error import multipole_truncation_bound, MultipoleTruncationError
from witwin.maxwell.mom.interpolation_error import KernelErrorBudget, KernelBoundUnavailable
from witwin.maxwell.mom.physics import compile_medium

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def projection(center,dtype):
    basis=RWGBasis(sphere_mesh(radius=.01,center=(center,0.,0.),subdivisions=0,device='cuda',dtype=dtype))
    triangles=basis.mesh.vertices[basis.mesh.faces]
    support=torch.cat((triangles[basis.plus_faces],triangles[basis.minus_faces]),dim=1)
    return NestedRWGMultipoles(basis,build_support_octree(support.amin(1),support.amax(1),leaf_size=30),order=4)


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
@pytest.mark.parametrize('storage',('dense','compact'))
def test_batched_support_admission_actual_action_gradient_and_bounded_calls(dtype,storage,monkeypatch):
    import witwin.maxwell.mom.fmm_translation as module
    test,source=projection(-.5,dtype),projection(.5,dtype)
    medium=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.001,sigma_m=.002),3e8)
    tol=6e-5 if dtype==torch.float32 else 3e-13
    rules=('bivariate_cauchy_v1','real_support_min_v1','positive_tail_min_v1','first_wave_min_v1',
           'direct_positive_tail_min_v1','direct_first_wave_min_v1')
    for rule in rules:
        receipt=multipole_truncation_bound(test.centers[0]-source.centers[0],test.radii[0],source.radii[0],
            medium.k,test.order,kernel_bound_rule=rule)
        budget=KernelErrorBudget(2*receipt['green_absolute'],2*receipt['gradient_component_absolute'])
        reference=ElectromagneticM2L(test,source,0,0,medium,table_storage=storage,error_budget=budget,kernel_bound_rule=rule)
        calls=[];shared=[]
        native=module.multipole_truncation_bounds
        test_check=test.assert_current;source_check=source.assert_current
        with monkeypatch.context() as patch:
            def observed(d,*args,**kwargs):
                calls.append(len(d));return native(d,*args,**kwargs)
            patch.setattr(module,'multipole_truncation_bounds',observed)
            patch.setattr(test,'assert_current',lambda:(shared.append('test'),test_check())[1])
            patch.setattr(source,'assert_current',lambda:(shared.append('source'),source_check())[1])
            objects=ElectromagneticM2L.create_many(test,source,[(0,0)]*65,medium,
                table_storage=storage,error_budget=budget,kernel_bound_rule=rule)
        assert calls==[32,32,1] and shared==['test','source']
        x=torch.randn((2,2,test.count,4),device='cuda',dtype=test.dtype,requires_grad=True);y=torch.randn_like(x)
        expected=reference.action(x)
        for index in (0,31,32,64):
            obj=objects[index];torch.testing.assert_close(obj.action(x),expected,rtol=tol,atol=tol)
            derivative,=torch.autograd.grad(obj.action(x),x,y,retain_graph=True)
            torch.testing.assert_close(derivative,obj.action(y,adjoint=True),rtol=tol,atol=tol)
            assert obj.error_budget==budget and obj.kernel_bound_rule==rule
        if storage=='compact':
            assert len({obj.tensors()[0].untyped_storage().data_ptr() for obj in objects})==65
        objects[0].tensors()[0].add_(1)
        with pytest.raises(ValueError,match='changed'):objects[0].assert_current()
        if storage=='compact':
            objects[32].assert_current()
        else:
            # Dense batch views intentionally share a tensor version counter.
            with pytest.raises(ValueError,match='changed'):objects[32].assert_current()


def test_batched_support_admission_rejects_before_coefficients_and_preserves_first_failure(monkeypatch):
    import witwin.maxwell.mom.fmm_translation as module
    test,source=projection(-.5,torch.float64),projection(.5,torch.float64)
    medium=compile_medium(mw.Material(),3e8)
    def forbidden(*args,**kwargs):raise AssertionError('Rejected admission allocated coefficients')
    monkeypatch.setattr(module,'multipole_to_local',forbidden)
    monkeypatch.setattr(module.CompactElectromagneticM2L,'create_many',forbidden)
    budget=KernelErrorBudget(1e-100,1e-100)
    for storage in ('dense','compact'):
        with pytest.raises(MultipoleTruncationError) as actual:
            ElectromagneticM2L.create_many(test,source,[(0,0),(99,0)],medium,
                table_storage=storage,error_budget=budget)
        scalar=multipole_truncation_bound(test.centers[0]-source.centers[0],test.radii[0],source.radii[0],medium.k,4)
        assert actual.value.receipt==scalar
        with pytest.raises(ValueError,match='separated complete-support'):
            ElectromagneticM2L.create_many(test,test,[(0,0)],medium,table_storage=storage,error_budget=budget)
        with pytest.raises(TypeError,match='KernelErrorBudget'):
            ElectromagneticM2L.create_many(test,source,[(0,0)],medium,table_storage=storage,error_budget=object())
        with pytest.raises(MemoryError,match='before coefficients'):
            ElectromagneticM2L.create_many(test,source,[(0,0)],medium,table_storage=storage,table_budget_bytes=1)
        assert ElectromagneticM2L.create_many(test,source,[],medium,table_storage=storage)==()
    # A valid separated pair can lack the requested Cauchy contour.
    nearby=projection(-.44,torch.float64)
    with pytest.raises(KernelBoundUnavailable):
        ElectromagneticM2L.create_many(test,nearby,[(0,0)],medium,separation_ratio=.9,error_budget=budget)
    test.centers.add_(1)
    with pytest.raises(ValueError,match='changed'):
        ElectromagneticM2L.create_many(test,source,[(0,0)],medium,error_budget=budget)
