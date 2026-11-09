"""R3b CUDA timing evidence; invoke only through the assigned GPU-1 lock runner."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile

from witwin.maxwell.mom.layered.assembly import assemble_potentials
from witwin.maxwell.mom.layered.build import build_extension
from witwin.maxwell.mom.layered.lowfreq import scale_potentials
from witwin.maxwell.mom.layered.network import recover_scaled_chunk, solve_scaled_chunk
from tests.mom.layered.test_layered_vias import vertical_loop, underpass


def timed(function):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record()
    result=function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end)/1000,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    ext=build_extension()
    anchor=torch.empty(0,device="cuda",dtype=torch.float64)
    peaks={}
    for dtype,threads,iterations in ((32,131072,4096),(64,131072,4096)):
        times=[]
        for _ in range(3):
            peak_anchor=anchor.to(torch.float32 if dtype==32 else torch.float64)
            seconds,_=timed(lambda:torch.ops.witwin_maxwell_mom_layered.fma_peak(peak_anchor,threads,iterations))
            times.append(seconds)
        operations=threads*iterations*16
        peaks[f"fp{dtype}_GFLOPs"]=operations/min(times)/1e9
    source=torch.empty(32*1024**2,device="cuda",dtype=torch.float64)
    target=torch.empty_like(source)
    seconds,_=timed(lambda:target.copy_(source))
    peaks["copy_GBs"]=2*source.numel()*source.element_size()/seconds/1e9
    del source,target
    out=dict(peaks=peaks,cases=[])
    for name,prepare in (("vertical_loop",lambda:vertical_loop()[0]),("rfic_underpass",underpass)):
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        prepared=prepare()
        torch.cuda.synchronize()
        setup=time.perf_counter()-started
        seconds,(vector,scalar)=timed(lambda:assemble_potentials(
            prepared.layout,prepared.table,frequency_index=-1,far_dtype=torch.float32,_network_accuracy=True))
        transform,(matrix,rhs,recovery)=timed(lambda:scale_potentials(prepared,vector,scalar))
        solve,(solution,info)=timed(lambda:solve_scaled_chunk(matrix,rhs))
        recover,_=timed(lambda:recover_scaled_chunk(prepared,solution,rhs,recovery))
        count=len(prepared.table.frequencies)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
            assemble_potentials(prepared.layout,prepared.table,frequency_index=0,
                                far_dtype=torch.float32,_network_accuracy=True)
            torch.cuda.synchronize()
        kernel_ms={}
        for event in trace.events():
            if event.device_type==torch.autograd.DeviceType.CUDA:
                kernel_ms[event.name]=kernel_ms.get(event.name,0.)+event.device_time_total/1000
        row=dict(case=name,unknowns=prepared.layout.basis.n_edges,faces=len(prepared.layout.mesh.faces),
                 setup_seconds=setup,assembly_seconds_per_frequency=seconds/count,
                 transform_seconds_per_frequency=transform/count,solve_seconds_per_frequency=solve/count,
                 recovery_seconds_per_frequency=recover/count,peak_bytes=torch.cuda.max_memory_allocated(),
                 kernel_ms=kernel_ms,factors=info[0],backward_error=info[2])
        print(json.dumps({key:value for key,value in row.items() if key!="kernel_ms"}),flush=True)
        out["cases"].append(row)
        del prepared,vector,scalar,matrix,rhs,recovery,solution
    args.output.write_text(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
