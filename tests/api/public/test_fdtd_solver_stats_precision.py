"""Metadata precision contracts; no field solve or global field gather."""
from types import SimpleNamespace
import pytest
import torch
import witwin.maxwell as mw


def model(*, parallel=None):
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.02,0.02),)*3),
        grid=mw.GridSpec.uniform(.01),boundary=mw.BoundarySpec.pml(num_layers=1),device='cpu')
    return mw.Simulation.fdtd(scene,frequency=1e9,parallel=parallel)


def stats(simulation, runtime):
    # Bind the reference frequency exactly as Simulation._configure_fdtd_solver_time does.
    runtime._spectral_reference_frequency=None
    return simulation._build_fdtd_solver_stats(runtime,time_steps=16,
        use_full_field_dft=False,dft_cfg=simulation.config.spectral_sampler)


def test_partitioned_stats_do_not_require_or_gather_global_fields():
    class PartitionedRuntime:
        dt=1e-12
        parallel_stats={'transport':'cuda_p2p','shards':2}
        def gather_fields(self):
            raise AssertionError('Statistics must not gather global field tensors')
    simulation=model(parallel=mw.FDTDParallelConfig(devices=('cuda:0','cuda:1'),transport='cuda_p2p'))
    initialized=torch.cuda.is_initialized()
    result=stats(simulation,PartitionedRuntime())
    assert result['parallel_stats']==PartitionedRuntime.parallel_stats
    assert result['dt']==1e-12 and result['steps_run']==16
    assert 'field_dtype' not in result and 'observer_dtype' not in result
    assert torch.cuda.is_initialized()==initialized


@pytest.mark.parametrize('field_dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('observer_dtype',[torch.float32,torch.float64])
def test_single_device_precision_reports_actual_runtime_dtypes(field_dtype,observer_dtype):
    runtime=SimpleNamespace(dt=1e-12,Ex=torch.empty(0,dtype=field_dtype),_observer_dtype=observer_dtype)
    result=stats(model(),runtime)
    if torch.float64 in (field_dtype,observer_dtype):
        assert result['field_dtype']==str(field_dtype)
        assert result['observer_dtype']==str(observer_dtype)
    else:
        assert 'field_dtype' not in result and 'observer_dtype' not in result
