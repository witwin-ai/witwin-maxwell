"""Actual scalar FFT forward/current-backward resource and graph lifetime."""
import json
import pytest
import torch
from witwin.maxwell.mom.fmm_fft import CartesianFFTTranslation
from witwin.maxwell.mom.fmm_expansion import multipole_moments

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_fft_p12_chunked_rhs_actual_peaks_and_no_saved_current_grids(tmp_path,monkeypatch):
    d=torch.tensor([-.016618783759270284,.0018717383556100717,-.009306902746826528],device='cuda',dtype=torch.float64)
    op=CartesianFFTTranslation(d,71.13533891222738,12,rhs_batch=4)
    generator=torch.Generator(device='cuda').manual_seed(1729)
    points=torch.randn((7,3),generator=generator,device='cuda',dtype=torch.float64)
    points=points/points.norm(dim=-1,keepdim=True)*.0017
    charges=torch.randn((33,7),generator=generator,device='cuda',dtype=torch.complex128)
    x=multipole_moments(points,charges,12).detach().requires_grad_(True)
    original=op._fourier_fields;field_calls=[]
    def observed():field_calls.append(1);return original()
    monkeypatch.setattr(op,'_fourier_fields',observed)
    saved=[]
    def pack(value):saved.append(value);return value
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):output=op.apply(x)
    torch.cuda.synchronize()
    forward=dict(baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),end_allocated_bytes=torch.cuda.memory_allocated())
    assert saved==[] and field_calls==[1]
    degrees=op.indices.sum(-1)
    dual=torch.randn(output.shape,generator=generator,device='cuda',dtype=op.dtype)*(.0029**degrees)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    gradient,=torch.autograd.grad(output,x,dual)
    torch.cuda.synchronize()
    backward=dict(baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),end_allocated_bytes=torch.cuda.memory_allocated())
    assert field_calls==[1,1] and saved==[]
    assert bool(torch.isfinite(output).all() & torch.isfinite(gradient).all())
    assert not any(isinstance(value,torch.Tensor) and value.numel()>=op.side**3 for value in vars(op).values())
    # A measured single-process p12 case, not a guarantee across input sizes,
    # library plans, allocator state or the eventual composed EM/FMM runtime.
    record=dict(rhs_count=33,rhs_batch=4,forward=forward,backward=backward,retained_bytes=op.retained_bytes,
        planned_visible_tensor_workspace_bytes=op.planned_tensor_workspace_bytes,no_current_grids_saved=True,
        limitations=['Single scalar translation at one geometry','Absolute process peaks include input/output retained state',
            'FFT library and allocator requirements vary','No EM system/setup/solve/scaling qualification'],
        physical_qualification=False,complexity_qualification=False,whole_fmm_peak_qualified=False)
    (tmp_path/'resource.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
