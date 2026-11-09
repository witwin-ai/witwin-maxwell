"""Actual tensor-source HB equations and exact decode-only bundle replay."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_volume_source import setup
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.mesh import SurfaceSystem,RWGBasis
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material,compile_tensor_harmonic_material
from witwin.maxwell.mom.tensor_harmonic_preparation import prepare_mixed_tensor_harmonic_balance
from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
from witwin.maxwell.mom.tensor_harmonic_bundle import snapshot_mixed_tensor_harmonic_bundle,restore_mixed_tensor_harmonic_bundle

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_tensor_source_harmonic_bundle(monkeypatch,tmp_path):
    _,binding=setup();volume=binding.volume
    nonlinear=mw.NonlinearSusceptibility(chi2=.002,chi3=.001)
    # Sample one identical declaration through the scalar and tensor compilers
    # to isolate regional tensor dispatch in this scalar-limit control. Public
    # tensor-source runtime admission remains independently guarded.
    tensor_material=compile_tensor_harmonic_material(mw.Material(
        eps_r=2.,mu_r=1.3,nonlinearity=nonlinear),3e7,2,device='cuda')
    scalar_material=compile_harmonic_material(mw.Material(eps_r=2.,mu_r=1.3,nonlinearity=nonlinear),3e7,2)
    exterior=compile_harmonic_material(mw.Material(),3e7,2)
    query=FieldQueryConfig(relative_tolerance=1e-4,absolute_tolerance=1e-10,max_order=48)
    states=[]
    for material,tensor in ((scalar_material,False),(tensor_material,True)):
        config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),
            dyadic_integration=DyadicIntegrationConfig() if tensor else None,riesz_storage='compact_rwg_v1')
        operators=tuple(SystemOperator(SurfaceSystem.create([
            (0,'nonlinear',RWGBasis(volume.boundary),material.linear_samples[n])]),
            exterior.linear_samples[n],[mw.PlaneWave()] if n==0 else [],config) for n in range(2))
        model,prepared=prepare_mixed_tensor_harmonic_balance(operators,(volume,),(material,),exterior,
            query_config=query,max_order=64,tensor_face_rule='projected_duffy',tensor_kernel_batch=8192)
        solution=model.solve(pump_scale=.5,relative_tolerance=1e-9,absolute_tolerance=1e-10)
        assert solution.receipt['converged']
        state=recover_harmonic_state(model,prepared,solution)
        states.append(state)
        assert state.cell_electric[0].norm()>0 and state.cell_electric[2].norm()>0
        assert state.total_volume_current[1].norm()>0
    # This exercises tensor dispatch in its independently comparable scalar
    # limit; it is not a general anisotropic nonlinear physical reference.
    errors=[float((states[1].cell_electric[n]-states[0].cell_electric[n]).norm()/states[0].cell_electric[n].norm())
            for n in range(3)]
    print({'tensor_source_harmonic_scalar_limit_errors':errors,'physical_qualification':False},flush=True)
    assert max(errors)<.005
    assert all(p['receipt']['source_matrix']['tensor_kernel_batch']==8192 for p in prepared['harmonics'])
    bundle,owner=snapshot_mixed_tensor_harmonic_bundle(state,prepared)
    path=tmp_path/'tensor-source-harmonic.pt';torch.save({'bundle':bundle,'owner':owner},path)
    saved=torch.load(path,weights_only=True)
    decoder=HarmonicFieldDecoder(prepared,state)
    points=prepared['static']['points']
    ac=tuple(decoder.query_fields(points,1,harmonic=n,query_config=query) for n in (1,2))
    dc=decoder.query_dc(points,1)
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
    def forbidden(*args,**kwargs):raise AssertionError('Bundle replay must not assemble or solve')
    with monkeypatch.context() as patch:
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(DenseHarmonicBalance,'solve',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored_model,new_prepared,new_state=restore_mixed_tensor_harmonic_bundle(
            saved['bundle'],expected_binding=saved['owner'],device='cuda')
        replay=HarmonicFieldDecoder(new_prepared,new_state)
        for name in ('cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(new_state,name),getattr(state,name),rtol=0,atol=0)
        for n,expected in zip((1,2),ac):
            for a,b in zip(replay.query_fields(points,1,harmonic=n,query_config=query),expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(replay.query_dc(points,1),dc):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(restored_model.residual(new_state.cell_electric,pump_scale=.5),
                                   model.residual(state.cell_electric,pump_scale=.5),rtol=0,atol=0)
    broken=copy.deepcopy(saved['bundle']);broken['state']['state']['total_volume_current'][1,0,0]+=1
    with pytest.raises(ValueError,match='owning binding'):
        restore_mixed_tensor_harmonic_bundle(broken,expected_binding=owner,device='cuda')
    mutated_kernel=new_prepared['operators'][0].body_handles[0].tensor_kernel
    object.__setattr__(mutated_kernel,'cache_max_bytes',mutated_kernel.cache_max_bytes+1)
    with pytest.raises((ValueError,RuntimeError)):
        replay.query_fields(points,1,harmonic=1,query_config=query)


def test_internal_tensor_ac_admission_and_public_guard():
    from witwin.maxwell.mom.tensor_volume import tensor_volume_static_kernels
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    from witwin.maxwell.mom.tensor_harmonic_runtime import _admit
    from witwin.maxwell.mom.harmonic_config import HarmonicBalanceConfig
    op,binding=setup()
    tensor=compile_tensor_harmonic_material(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.,2.)),3e7,1,device='cuda')
    exterior=compile_harmonic_material(mw.Material(),3e7,1)
    kernels=tensor_volume_static_kernels(tensor.linear_samples[0])
    assert len(kernels)==2
    eye=torch.eye(3,device='cuda',dtype=torch.complex128)
    bad=TensorMedium(3e7,torch.diag(eye.new_tensor([1.,-1.,1.])),eye)
    with pytest.raises(NotImplementedError,match='definite real or strictly lossy'):
        tensor_volume_static_kernels(bad)
    changed=replace(tensor,linear_samples=(bad,))
    # Geometry-only operator: rejection precedes refresh or matrix assembly.
    with pytest.raises(NotImplementedError,match='definite real or strictly lossy'):
        prepare_mixed_tensor_harmonic_balance((SimpleNamespace(system=op.system),),
            (binding.volume,),(changed,),exterior)
    config=mw.MoMConfig(dyadic_integration=DyadicIntegrationConfig(),
        harmonic_balance=HarmonicBalanceConfig(volumes=(binding.volume,),harmonics=1,branch_id="tensor-volume-admission"))
    with pytest.raises(NotImplementedError,match='scalar regional AC kernels'):
        _admit(config,(tensor,),exterior)
