"""Host-only observer-path admission metadata; no GPU execution is modelled."""
from types import SimpleNamespace
import pytest
import torch
from witwin.maxwell.fdtd.observers import make_single_observer_engine
from witwin.maxwell.fdtd.runtime.stepping import _make_tail_runner


def metadata():
    return SimpleNamespace(observers_enabled=True,complex_fields_enabled=False,
        Ex=torch.zeros((2,2,2),dtype=torch.float32),_plane_observer_groups={},
        _point_observer_groups={'Ez':object()},_observer_spectral_entries=[{'frequency':1e9}],
        _cuda_graph_active=True,dft_enabled=False)


@pytest.mark.parametrize('enabled',[False,True])
def test_one_native_point_does_not_require_a_dedicated_grouped_observer_graph(enabled):
    initialized=torch.cuda.is_initialized();solver=metadata()
    assert make_single_observer_engine(solver,enabled=enabled) is None
    assert solver._single_observer_engine is None
    assert solver._observer_cuda_graph_active is False
    assert torch.cuda.is_initialized()==initialized


def test_full_field_dft_disabled_has_no_table_tail_capture_requirement():
    initialized=torch.cuda.is_initialized();solver=metadata()
    assert _make_tail_runner(solver,use_gpu_dft=False) is None
    assert not hasattr(solver,'_dft_step')
    assert torch.cuda.is_initialized()==initialized


def test_point_observer_policy_does_not_turn_off_independent_field_graph_flag():
    solver=metadata()
    assert make_single_observer_engine(solver,enabled=True) is None
    assert solver._cuda_graph_active is True
    # This flag is only host admission metadata here; actual field capture is
    # qualified separately by the genuine CUDA matrix.
    assert solver._observer_cuda_graph_active is False
