"""CPU contracts for optional spectral precision; no field solve or CUDA."""
from types import SimpleNamespace
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.observers import BatchObserverEngine,prepare_observers,_sampled_plane_gathers
from witwin.maxwell.fdtd.adjoint.seeds import _seed_batch_contribution,_apply_point_seeds_native,_apply_plane_seeds_native
from witwin.maxwell.fdtd.adjoint.accumulation import build_point_cell_plan
from tests.fdtd.cuda.torch_reference import _seed_inject_point


def test_default_and_opt_in_sampler_precision_are_explicit():
    assert mw.SpectralSampler().observer_dtype==torch.float32
    assert mw.SpectralSampler(observer_dtype=torch.float64).observer_dtype==torch.float64


@pytest.mark.parametrize('dtype',(torch.float16,torch.complex128,'float64',None))
def test_unregistered_observer_dtype_rejects(dtype):
    with pytest.raises(ValueError,match='observer_dtype'):
        mw.SpectralSampler(observer_dtype=dtype)


def test_source_normalization_precision_has_an_explicit_boundary():
    with pytest.raises(ValueError,match='normalize_source=False'):
        mw.SpectralSampler(observer_dtype=torch.float64,normalize_source=True)
    assert mw.SpectralSampler(observer_dtype=torch.float32,normalize_source=True).normalize_source


def test_complex_field_observation_rejects_before_allocation():
    solver=SimpleNamespace(_observer_dtype=torch.float64,complex_fields_enabled=True)
    with pytest.raises(ValueError,match='real time-domain fields'):
        prepare_observers(solver,(3e9,),'none',96)


def test_mixed_cohort_precision_rejects_before_sampling():
    solvers=(SimpleNamespace(_observer_dtype=torch.float32),SimpleNamespace(_observer_dtype=torch.float64))
    engine=BatchObserverEngine.__new__(BatchObserverEngine)
    engine.slot=SimpleNamespace(solvers=solvers)
    with pytest.raises(ValueError,match='one observer_dtype'):
        engine.bind()


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
def test_sample_interpolation_coefficients_use_observation_precision(dtype):
    brackets=[([0,1],[1,2],[.2,.6]),([0,1],[1,2],[.3,.7])]
    # Preparation accepts the same (lower,upper,upper_weight) triples as the
    # primary midpoint sampler; no field values are sampled on CPU here.
    import numpy as np
    brackets=[tuple(np.asarray(value)for value in triple)for triple in brackets]
    _,weights=_sampled_plane_gathers('x',1,brackets,device='cpu',dtype=dtype)
    assert weights.dtype==dtype
    torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights[...,0]),rtol=0,atol=3*torch.finfo(dtype).eps)


@pytest.mark.parametrize('kind',('point','plane'))
def test_fp64_transpose_reduces_before_cast_and_matches_direct_autograd(kind):
    field=torch.arange(24,dtype=torch.float32).reshape(2,3,4).requires_grad_()
    real=torch.tensor([[.2,.3],[.4,.5]],dtype=torch.float64)
    imag=torch.tensor([[.7,.9],[.1,.4]],dtype=torch.float64)
    cos=torch.tensor([[.3,.1],[.7,.2]],dtype=torch.float64)
    sin=torch.tensor([[.6,.2],[.9,.1]],dtype=torch.float64)
    entries=torch.tensor([0,1]);step=1
    if kind=='point':
        i=torch.tensor([0,1]);j=torch.tensor([1,2]);k=torch.tensor([2,3])
        batch=SimpleNamespace(grad_real=real,grad_imag=imag,cos_pack=cos,sin_pack=sin,
            state_field_name='Ex')
        batch.cells,batch.cell_offsets,batch.point_order=build_point_cell_plan(i,j,k,field.shape)
        selected=field[i,j,k]
        apply=_apply_point_seeds_native
    else:
        real=real[:,None,:].expand(2,3,2).contiguous();imag=imag[:,None,:].expand(2,3,2).contiguous()
        field=torch.arange(12,dtype=torch.float32).reshape(2,3,2).requires_grad_()
        batch=SimpleNamespace(grad_real=real,grad_imag=imag,cos_pack=cos,sin_pack=sin,
            axis_code=0,plane_index=1,state_field_name='Ex')
        selected=field[1]
        apply=_apply_plane_seeds_native
    expected=torch.autograd.grad((real*selected.double()*cos[:,step].reshape((2,)+(1,)*(real.ndim-1))+
        imag*selected.double()*sin[:,step].reshape((2,)+(1,)*(imag.ndim-1))).sum(),field)[0]
    target=torch.zeros_like(field)
    runtime=SimpleNamespace(point_batches=(batch,),plane_batches=(batch,))
    # Point seeds now dispatch both observer precisions to the native gather.
    # Supply the independent CPU oracle for this host-only dispatch contract.
    apply(SimpleNamespace(_seed_inject_point=_seed_inject_point),{'Ex':target},runtime,step)
    torch.testing.assert_close(target,expected,rtol=0,atol=0)


def test_batch_declaration_distinguishes_observer_precisions():
    from witwin.maxwell.execution.plan import _batch_declaration
    unresolved=[]
    legacy=_batch_declaration(mw.SpectralSampler(), 'sampler', unresolved)
    high=_batch_declaration(mw.SpectralSampler(observer_dtype=torch.float64), 'sampler', unresolved)
    assert legacy!=high and not unresolved
