"""Run the existing F1/F4 criteria on the default native sheet copy path."""
import pytest
import torch

from . import test_thin_sheet_feeds as gates

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def cuda_feed_factory(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_THIN_SHEET_KERNEL", raising=False)
    simulation = gates._simulation

    def native_simulation(*args, **kwargs):
        kwargs["device"] = "cuda"
        return simulation(*args, **kwargs)

    monkeypatch.setattr(gates, "_simulation", native_simulation)


@pytest.mark.parametrize("material", ["pec", "copper"])
@pytest.mark.parametrize("width_cells", [pytest.param(1, id="narrow"), 2, 4])
def test_f1_axis_aligned_feed_equivalence_on_fused_cuda(material, width_cells, monkeypatch, cuda_feed_factory):
    gates.test_f1_axis_aligned_feed_equivalence(material, width_cells, monkeypatch)


def test_f4_loaded_unexcited_feed_on_fused_cuda(cuda_feed_factory):
    gates.test_f4_loaded_unexcited_feed_for_twenty_thousand_steps()
