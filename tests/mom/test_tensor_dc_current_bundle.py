"""Full discrete DC tensor current bundle control with scalar AC maps."""
import runpy
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_discrete_tensor_dc_current_bundle(tmp_path,monkeypatch):
    runpy.run_path('tests/mom/test_tensor_harmonic_current_preparation.py')[
        'test_actual_ac_current_hb_preparation_and_ownership'](monkeypatch,tmp_path,False,tensor_current=True)
