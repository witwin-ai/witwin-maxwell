"""Independent clipped aperture/current duality through physical refinement."""
import pytest
from scripts.dev.fdtd.bend_pcb_refinement_preflight import compile_source_duality


@pytest.mark.parametrize('spacing,edges',((.01,18),(.005,100),(.0025,648)))
def test_refined_native_voltage_and_independent_current_share_physical_aperture(spacing,edges):
    report=compile_source_duality(spacing)
    assert report['passed']
    assert report['cpu_only']
    assert report['independent_source_edges']==report['native_source_edges']==edges
    assert report['voltage_weight_sum']==pytest.approx(.02,rel=2e-7)
    assert report['maximum_relative_voltage_weight_error']<2e-7
    assert all(value==pytest.approx(-1.,abs=2e-15)for value in report['radial_current_flux_per_amp'])


def test_cpu_source_preflight_can_follow_other_gpu_tests_without_using_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    report=compile_source_duality(.01)
    assert report['passed']
    assert report['cuda_context_preexisting']
    assert not report['cuda_context_initialized_by_preflight']
