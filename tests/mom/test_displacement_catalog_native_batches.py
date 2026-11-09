"""Bounded native IEEE catalog batching on new ordered displacement inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest
import torch
from witwin.maxwell.mom.fmm_storage import shared_displacement_catalog

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def _original():
    path=os.environ.get('GSP_ORIGINAL_CATALOG_MODULE')
    digest=os.environ.get('GSP_ORIGINAL_CATALOG_MODULE_SHA256')
    if path is None and digest is None:
        pytest.skip('Requires frozen original native32 catalog module')
    assert path and digest and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location('original_native_catalog',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.shared_displacement_catalog,digest


def test_original_ordered_ieee_owners_and_bindings_across_native_chunks(tmp_path,monkeypatch):
    reference,digest=_original()
    # Nonbinary positions, nextafter neighbors and signed zero exercise exact
    # native keys; only topology and IEEE bit records cross to host metadata.
    centers=torch.arange(2053*3,device='cuda',dtype=torch.float64).reshape(-1,3)*.013
    centers[:,1]=torch.remainder(centers[:,1],.19)
    centers[0]=0.;centers[1]=centers[0];centers[1,0]=1.
    centers[2]=centers[1];centers[2,0]=torch.nextafter(centers[2,0],centers.new_tensor(torch.inf))
    centers[3]=torch.tensor((0.,-0.,1.),device='cuda',dtype=torch.float64)
    pairs=[(0,1),(1,0),(0,2),(2,0),(0,3),(3,0)]
    for index in range(12001):
        a=(17*index+11)%2053;b=(43*index+7)%2053
        if a==b:b=(b+1)%2053
        pairs.append((a,b))
        if index%7==0:pairs.append((b,a))
        if index%11==0:pairs.append((0,1))
    assert len(pairs)>10*1024
    torch.cuda.synchronize();started=time.perf_counter();expected=reference(centers,pairs)
    torch.cuda.synchronize();original_seconds=time.perf_counter()-started
    tensor=torch.tensor;chunks=[]
    def tracked(data,*args,**kwargs):
        if isinstance(data,tuple) and data and isinstance(data[0],tuple):
            chunks.append(len(data));assert len(data)<=1024
        return tensor(data,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch,'tensor',tracked)
        started=time.perf_counter();actual=shared_displacement_catalog(centers,pairs)
        torch.cuda.synchronize();batched_seconds=time.perf_counter()-started
    assert actual==expected
    assert chunks==[min(1024,len(pairs)-start) for start in range(0,len(pairs),1024)]
    owners,bindings=actual
    assert bindings[0][0]==bindings[1][0] and bindings[0][1]!=bindings[1][1]
    assert bindings[2][0]==bindings[3][0] and bindings[0][0]!=bindings[2][0]
    assert tuple(pairs[index] for index in owners)==tuple(pairs[index] for index in expected[0])
    assert shared_displacement_catalog(centers,())==((),())
    record=dict(version='native_displacement_catalog_batches_control_v1',original_module_sha256=digest,
        logical_pairs=len(pairs),unique_owners=len(owners),owner_indices_and_all_bindings_exact=True,
        maximum_native_pair_chunk=max(chunks),native_pair_chunks=len(chunks),original_pair_chunks=(len(pairs)+31)//32,
        original_seconds=original_seconds,batched_seconds=batched_seconds,timing_scope='one new ordered catalog input, shared GPU descriptive only',
        no_physical_partition_or_coefficient_or_action=True,production_qualification=False)
    (tmp_path/'catalog.json').write_text(json.dumps(record,indent=2)+'\n')


@pytest.mark.parametrize('case',['zero','nonfinite','float32','trainable','out_of_range','boolean_index'])
def test_native_catalog_refusal_contract_is_retained(case):
    centers=torch.tensor([[0.,0.,0.],[1.,0.,0.]],device='cuda',dtype=torch.float64)
    pairs=[(0,1)]
    if case=='zero':pairs=[(0,0)]
    elif case=='nonfinite':centers[1,0]=torch.inf
    elif case=='float32':centers=centers.float()
    elif case=='trainable':centers.requires_grad_(True)
    elif case=='out_of_range':pairs=[(0,2)]
    elif case=='boolean_index':pairs=[(False,1)]
    with pytest.raises(ValueError):shared_displacement_catalog(centers,pairs)
