"""Material compiler scalar reuse: exact constructors, context safety and bytes.

The undecorated compile body is the production uncached constructor path;
no historical checkout or assessment driver is imported.
"""
import asyncio
import cProfile
import contextvars
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import re
import struct
import threading

import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler import _scalar_cache as cache
from witwin.maxwell.compiler import materials as candidate
from witwin.maxwell.scene import prepare_scene


def convert(value,dtype=torch.float64):return cache.scalar_tensor(value,device='cpu',dtype=dtype)

@pytest.mark.parametrize('value',[True,False,0,1,-1,2**60+3,0.,-0.,1.25,float('inf'),complex(1.25,-.0)])
def test_exact_builtin_conversion_and_scoped_reuse(value):
    dtype=torch.complex128 if type(value)is complex else torch.float64
    @cache.with_compile_scalar_cache
    def run():
        a,b=convert(value,dtype),convert(value,dtype)
        assert a is b
        torch.testing.assert_close(a,torch.as_tensor(value,dtype=dtype),rtol=0,atol=0,equal_nan=True)
        return a
    a,b=run(),run();assert a is not b and cache._SCALARS.get()is None

def test_float_bits_nan_payload_negative_zero_and_exact_integer_keys():
    nan1=struct.unpack('!d',bytes.fromhex('7ff8000000000011'))[0]
    nan2=struct.unpack('!d',bytes.fromhex('7ff8000000000022'))[0]
    @cache.with_compile_scalar_cache
    def run():
        p,n=convert(0.),convert(-0.);assert p is not n
        assert not torch.signbit(p) and torch.signbit(n)
        assert convert(nan1)is convert(nan1) and convert(nan1)is not convert(nan2)
        a,b=convert(2**60+1,torch.int64),convert(2**60+2,torch.int64)
        assert a.item()==2**60+1 and b.item()==2**60+2 and a is not b
        assert convert(True)is not convert(1) and convert(1)is not convert(1.)
    run()

def test_tensor_alias_autograd_numpy_and_custom_scalar_use_original_path():
    tensor=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    array=np.array([2.,3.]);classed=type('FloatSubclass',(float,),{})(2.)
    @cache.with_compile_scalar_cache
    def run():
        assert convert(tensor)is tensor
        y=3*convert(tensor)**2
        assert torch.autograd.grad(y,tensor)[0].item()==12.
        a,b=convert(array),convert(array)
        assert a is not b and a.data_ptr()==b.data_ptr()
        array[0]=7.;assert a[0].item()==7.
        assert convert(classed)is not convert(classed)
        assert len(cache._SCALARS.get()[2])==0
    run()

def test_dtype_grad_and_inference_contexts_are_isolated():
    @cache.with_compile_scalar_cache
    def run():
        original=convert(2.)
        assert convert(2.,torch.float32)is not original
        with torch.no_grad():nograd=convert(2.);assert nograd is not original
        assert convert(2.)is original
        with torch.inference_mode():
            inference=convert(2.);assert inference.is_inference() and inference is not original
        assert not original.is_inference() and convert(2.)is original
    run()

def test_exception_and_nested_compile_release_only_their_own_cache():
    observed=[]
    @cache.with_compile_scalar_cache
    def inner():observed.append(cache._SCALARS.get());raise ValueError('intentional')
    @cache.with_compile_scalar_cache
    def outer():
        a=convert(2.);own=cache._SCALARS.get()
        with pytest.raises(ValueError,match='intentional'):inner()
        assert cache._SCALARS.get()is own and convert(2.)is a
        assert observed[0][2]=={}
        return own
    own=outer();assert own[2]=={} and cache._SCALARS.get()is None

def test_concurrent_threads_do_not_share_or_retain_scalar_storage():
    barrier=threading.Barrier(2)
    @cache.with_compile_scalar_cache
    def run():
        a=convert(2.);barrier.wait();assert convert(2.)is a
        return a,cache._SCALARS.get()
    with ThreadPoolExecutor(2)as pool:a,b=list(pool.map(lambda _:run(),range(2)))
    assert a[0]is not b[0] and a[1][2]==b[1][2]=={} and cache._SCALARS.get()is None

def test_explicit_inherited_thread_context_cannot_access_owner_cache():
    @cache.with_compile_scalar_cache
    def parent():
        own=convert(2.);context=contextvars.copy_context()
        with ThreadPoolExecutor(1)as pool:other=pool.submit(context.run,convert,2.).result()
        assert own is not other and convert(2.)is own and len(cache._SCALARS.get()[2])==1
    parent()

def test_actual_profiler_conversion_count_changes_without_skipping_requests():
    def counted(fn):
        profiler=cProfile.Profile();profiler.enable();fn();profiler.disable()
        return sum(x.callcount for x in profiler.getstats()if isinstance(x.code,str)and'as_tensor'in x.code)
    def old():
        return [torch.as_tensor(2.,device='cpu',dtype=torch.float64)for _ in range(5)]
    @cache.with_compile_scalar_cache
    def new():
        return [candidate._scalar_tensor(2.,device='cpu',dtype=torch.float64)for _ in range(5)]
    assert counted(old)==5 and counted(new)==1

def test_inherited_async_context_cannot_reuse_parent_cache_and_drops_references():
    async def check():
        async def child():return convert(2.),cache._SCALARS.get()
        @cache.with_compile_scalar_cache
        def parent():return convert(2.),asyncio.create_task(child())
        original,task=parent();other,context=await task
        assert original is not other and context[2]=={}
    asyncio.run(check())

def test_read_only_consumers_preserve_cache_versions_and_field_vjp():
    occupancy=torch.tensor([.1,.5,.9],dtype=torch.float64,requires_grad=True)
    @cache.with_compile_scalar_cache
    def run():
        value=convert(3.);version=value._version
        field=torch.tensor([1.,2.,4.],dtype=torch.float64)
        out=candidate._blend_material(field,occupancy,value=3.)
        frequency=candidate._assign_modulation_omega(field,occupancy,value=3.)
        assert value._version==version and torch.equal(value,torch.tensor(3.,dtype=torch.float64))
        grad=torch.autograd.grad(out.sum(),occupancy)[0]
        torch.testing.assert_close(grad,3.-field,rtol=0,atol=0)
        torch.testing.assert_close(frequency,torch.full_like(field,3.),rtol=0,atol=0)
    run()

@pytest.mark.parametrize('value,dtype',[(2**80,torch.int64),(1+2j,torch.float64)])
def test_original_conversion_errors_are_not_cached_or_changed(value,dtype):
    with pytest.raises(Exception)as original:torch.as_tensor(value,dtype=dtype)
    @cache.with_compile_scalar_cache
    def run():
        with pytest.raises(type(original.value),match=re.escape(str(original.value))):convert(value,dtype)
        assert cache._SCALARS.get()[2]=={}
    run()

def test_field_vjp_survives_cache_release_and_original_default_conversion():
    occupancy=torch.tensor([.2,.5],dtype=torch.float64,requires_grad=True)
    @cache.with_compile_scalar_cache
    def build():
        assert cache.scalar_tensor(2.,device=None,dtype=torch.float64).item()==2.
        return candidate._blend_material(torch.ones_like(occupancy),occupancy,value=3.).square().sum()
    loss=build();assert cache._SCALARS.get()is None
    torch.testing.assert_close(torch.autograd.grad(loss,occupancy)[0],4*(1+2*occupancy),rtol=0,atol=0)

def _tensors(value,path=''):
    if isinstance(value,torch.Tensor):return {path:value}
    if isinstance(value,dict):return {p:t for k,v in value.items() for p,t in _tensors(v,path+'/'+str(k)).items()}
    if isinstance(value,(tuple,list)):return {p:t for k,v in enumerate(value) for p,t in _tensors(v,path+'/'+str(k)).items()}
    if dataclasses.is_dataclass(value):return {p:t for f in dataclasses.fields(value) for p,t in _tensors(getattr(value,f.name),path+'/'+f.name).items()}
    return {}

@pytest.mark.parametrize('samples,averaging',[(1,'arithmetic'),((1,1,3),'polarized'),((1,1,512),'polarized')])
def test_real_CPU_compiled_all_material_and_PEC_tensor_coefficients_bitwise(samples,averaging):
    before=torch.cuda.is_initialized();nodes=np.linspace(-1,1,5)
    scene=mw.Scene(domain=mw.Domain(bounds=((-1.,1.),)*3),grid=mw.GridSpec.custom(nodes,nodes,nodes),device='cpu',
        boundary=mw.BoundarySpec.pml(num_layers=1),subpixel_samples=mw.SubpixelSpec(samples=samples,averaging=averaging),structures=[
            mw.Structure(mw.Box(position=(-.1,0.,.2),size=(1.1,1.7,1.5)),mw.Material(eps_r=4.,mu_r=1.,sigma_e=.01),name='one'),
            mw.Structure(mw.Box(position=(.2,-.1,0.),size=(1.7,1.1,1.3)),mw.Material(eps_r=2.5,mu_r=1.4,sigma_e=.02),name='two'),
            mw.Structure(mw.PlanarPEC.rectangle(((-1.,1.),(-1.,1.)),coordinate=0.),mw.Material.pec(),name='PEC')])
    prepared=prepare_scene(scene)
    assert cache._SCALARS.get()is None
    original=candidate.compile_material_model.__wrapped__(prepared,subpixel=scene.subpixel)
    changed=candidate.compile_material_model(prepared,subpixel=scene.subpixel)
    a,b=_tensors(original),_tensors(changed)
    assert a.keys()==b.keys() and len(a)>30
    for key in a:
        assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape,key
        assert torch.equal(a[key],b[key]),key
        assert a[key].detach().contiguous().numpy().tobytes()==b[key].detach().contiguous().numpy().tobytes(),key
    assert torch.cuda.is_initialized()==before and cache._SCALARS.get()is None
