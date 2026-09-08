"""Stable DFT storage, schedule invalidation and independent retained tapes."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation
from tests.gradients.test_batch_adjoint import density_batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def test_repeated_full_spectra_preserve_all_captured_addresses(monkeypatch):
    batch = mw.Simulation.batch([simulation(steps=31, frequencies=(.8e9, 1e9))],
        outputs=mw.BatchOutput(fields=("Ez",)), execution=mw.BatchExecution(
            devices=("cuda:0",), batch_size=1, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run()[0]
        solver = prepared._slot.solvers[0]
        banks = [v for b in solver._dft_batched_fields.values() for v in b.values() if v is not None]
        names = ("_dft_weighted_cos_table", "_dft_weighted_sin_table", "_dft_step",
                 "_dft_weighted_cos", "_dft_weighted_sin")
        pointers = [v.data_ptr() for v in banks] + [getattr(solver, n).data_ptr() for n in names]
        graphs = tuple(prepared._graphs.values())
        # A cache hit must not repeat the O(T*F) host schedule calculation.
        import witwin.maxwell.fdtd.runtime.spectral as spectral
        def forbidden(*args):
            raise AssertionError("The fixed DFT schedule was rebuilt")
        monkeypatch.setattr(spectral, "_dft_window_weight", forbidden)
        second = prepared.run()[0]
        banks = [v for b in solver._dft_batched_fields.values() for v in b.values() if v is not None]
        assert pointers == [v.data_ptr() for v in banks] + [getattr(solver, n).data_ptr() for n in names]
        assert tuple(prepared._graphs.values()) == graphs
        torch.testing.assert_close(second.tensor("Ez"), first.tensor("Ez"), rtol=0, atol=0)


def test_schedule_cache_rebuilds_changed_frequency_window_and_horizon():
    solver = simulation().prepare().solver
    for frequencies, window, steps in [((.8e9, 1e9), "none", 27),
                                      ((1e9, 1.2e9), "none", 27),
                                      ((1e9, 1.2e9), "hanning", 31)]:
        solver.enable_dft(frequencies, window_type=window, end_step=steps, components=("Ez",), reuse_buffers=True)
        assert solver.build_dft_step_tables(steps)
        reference = simulation().prepare().solver
        reference.enable_dft(frequencies, window_type=window, end_step=steps, components=("Ez",))
        assert reference.build_dft_step_tables(steps)
        torch.testing.assert_close(solver._dft_weighted_cos_table, reference._dft_weighted_cos_table, rtol=0, atol=0)
        torch.testing.assert_close(solver._dft_weighted_sin_table, reference._dft_weighted_sin_table, rtol=0, atol=0)


def test_full_spectrum_tapes_share_only_owned_frequency_banks(monkeypatch):
    from witwin.maxwell.execution import adjoint
    from witwin.maxwell.execution.capacity import tensor_memory_manifest
    frozen = []
    freeze = adjoint.freeze_solver
    def record(solver):
        owned = freeze(solver)
        frozen.append(owned)
        for component, buffers in owned._dft_batched_fields.items():
            for part, bank in buffers.items():
                if bank is None:
                    continue
                assert bank.data_ptr() != solver._dft_batched_fields[component][part].data_ptr()
                for entry in owned._dft_entries:
                    assert entry["fields"][component][part].untyped_storage().data_ptr() == bank.untyped_storage().data_ptr()
        return owned
    monkeypatch.setattr(adjoint, "freeze_solver", record)
    density = torch.full((2, 2, 2, 2), .4, device="cuda:0", requires_grad=True)
    batch = density_batch(density)
    batch.outputs = mw.BatchOutput(fields=("Ez",))
    with batch.prepare() as prepared:
        first, second = prepared.run(), prepared.run()
    def objective(results):
        return sum(r.tensor("Ez").abs().square().sum() for r in results)
    for a, b in zip(first, second):
        torch.testing.assert_close(a.tensor("Ez"), b.tensor("Ez"), rtol=0, atol=0)
    first_gradient, = torch.autograd.grad(objective(first), density)
    second_gradient, = torch.autograd.grad(objective(second), density)
    # Material interpolation's GPU pullback uses floating-point reductions;
    # forward spectra are bitwise stable, while VJPs use a numerical tolerance.
    torch.testing.assert_close(first_gradient, second_gradient, rtol=2e-6, atol=1e-6)
    assert first_gradient.abs().max() > 0
    for solver in frozen:
        manifest = tensor_memory_manifest({"banks": solver._dft_batched_fields, "entries": solver._dft_entries})
        expected = sum(v.numel() * v.element_size() for b in solver._dft_batched_fields.values()
                       for v in b.values() if v is not None)
        assert manifest["storage_bytes_by_device"]["cuda:0"] == expected
