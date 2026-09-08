"""Compiled equality proofs share coefficients and promote on later variation."""
import pytest
import torch
from tests.fdtd.cuda.test_batch_runtime import simulation, run_cohort
from witwin.maxwell.fdtd.runtime.batch import NativeFieldSlot, COEFFICIENTS, FIELDS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def test_shared_coefficients_promote_without_reallocating_fields():
    slot = NativeFieldSlot(3, "cuda:0")
    slot.bind([simulation(0, steps=20).prepare().solver for _ in range(3)])
    assert all(slot.buffers[name].ndim == 3 for name in COEFFICIENTS)
    pointers = {name: slot.buffers[name].data_ptr() for name in FIELDS}
    sims = [simulation(i, steps=20) for i in range(3)]
    slot.bind([sim.prepare().solver for sim in sims])
    assert slot.buffers["cex_curl"].ndim == 4
    assert slot.buffers["chx_curl"].ndim == 3
    assert {name: slot.buffers[name].data_ptr() for name in FIELDS} == pointers
    run_cohort(slot, steps=20)
    for lane, sim in enumerate(sims):
        reference = sim.prepare()
        reference.run()
        for name in FIELDS:
            torch.testing.assert_close(slot.buffers[name][lane], getattr(reference.solver, name), rtol=0, atol=0)
    coefficient_pointer = slot.buffers["cex_curl"].data_ptr()
    slot.bind([simulation(0, steps=20).prepare().solver for _ in range(3)])
    assert slot.buffers["cex_curl"].ndim == 4
    assert slot.buffers["cex_curl"].data_ptr() == coefficient_pointer
