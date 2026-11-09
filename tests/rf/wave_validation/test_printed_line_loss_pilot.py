"""Physical pilot length changes preserve the material and observation domain."""
import pytest
import torch

from scripts.dev.fdtd.printed_line_loss_pilot import prepare_pilot
from tests._fresh_interpreter import run_without_cuda


def test_two_length_combined_pilot_preserves_grid_roi_and_drive_basis():
    run_without_cuda(__file__, '_check_two_length_combined_pilot')


def _check_two_length_combined_pilot():
    short = prepare_pilot('stripline', separation=.12)
    long = prepare_pilot('stripline', separation=.20)
    assert short['native_grid_sha256'] == long['native_grid_sha256']
    assert short['loss_roi_bounds_m'] == long['loss_roi_bounds_m']
    assert short['source_voltage_dofs'] == long['source_voltage_dofs']
    assert short['coherent_drive_groups'] == long['coherent_drive_groups']
    assert short['regional_schema'] == long['regional_schema']
    assert short['per_basis_drive_count'] == 2
    assert short['planned_duration_s'] == 64e-9 and short['checkpoint_target_s'] == 32e-9
    assert short['native_plane_complex128_output_bytes'] < 1024 ** 3
    assert short['single_frequency_dense_complex128_bytes'] < 128 * 1024 ** 2
    assert short['cuda_initialized'] == torch.cuda.is_initialized()


def test_pilot_rejects_undeclared_length_before_scene_or_field_execution():
    with pytest.raises(ValueError, match='source separation'):
        prepare_pilot('stripline', separation=.16)


@pytest.fixture(autouse=True)
def _cpu_preparation_keeps_CUDA_initialization_state():
    before = torch.cuda.is_initialized()
    yield
    assert torch.cuda.is_initialized() == before
