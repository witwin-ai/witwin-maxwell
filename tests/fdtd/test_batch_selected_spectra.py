"""Requested spectra allocate and differentiate only selected Yee components."""
import pytest
import torch
import witwin.maxwell as mw
from tests.fdtd.cuda.test_batch_runtime import simulation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("frequencies,shape", [(4, (80, 80, 80)), (3, (80, 80, 80)), (4, (128, 128, 64))])
def test_omitted_components_cover_coarsened_and_split_native_launches(frequencies, shape):
    from witwin.maxwell.fdtd.cuda.backend import _accumulate_dft_batched, get_compiled_extension
    get_compiled_extension()
    field = torch.randn(shape, device="cuda:0")
    empty = torch.empty(0, device="cuda:0")
    empty_accum = empty.view(1, 0).expand(frequencies, 0)
    real = torch.zeros((frequencies, *shape), device="cuda:0")
    imag = torch.zeros_like(real)
    cos = torch.linspace(-.5, .7, frequencies, device="cuda:0")
    sin = torch.linspace(.3, -.2, frequencies, device="cuda:0")
    _accumulate_dft_batched(Ex=empty, Ey=empty, Ez=field,
        ExRealAccum=empty_accum, ExImagAccum=empty_accum,
        EyRealAccum=empty_accum, EyImagAccum=empty_accum,
        EzRealAccum=real, EzImagAccum=imag, weightedCos=cos, weightedSin=sin)
    torch.testing.assert_close(real, field[None] * cos[:, None, None, None], rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(imag, field[None] * sin[:, None, None, None], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("components", [("Ez",), ("Ex", "Ey"), ("Ex", "Ey", "Ez")])
@pytest.mark.parametrize("frequency_count", [1, 3, 31])
def test_selected_spectrum_storage_and_values_match_full_dft(components, frequency_count):
    frequencies = tuple(.9e9 + i * .01e9 for i in range(frequency_count))
    sims = [simulation(i, steps=19, frequencies=frequencies) for i in range(2)]
    batch = mw.Simulation.batch(sims, variation=mw.BatchVariation(sources=("drive",)),
        outputs=mw.BatchOutput(fields=components), execution=mw.BatchExecution(
            devices=("cuda:0",), batch_size=2, graph="required", graph_steps=8))
    with batch.prepare() as prepared:
        results = prepared.run()
        results.results()
        for solver in prepared._slot.solvers:
            assert set(solver._dft_batched_fields) == set(components)
            actual = sum(v.numel() * v.element_size() for fields in solver._dft_batched_fields.values()
                         for v in fields.values() if v is not None)
            expected = frequency_count * 8 * sum(getattr(solver, c).numel() for c in components)
            assert actual == expected
            assert solver._dft_empty_field.untyped_storage().nbytes() == 0
        repeated = prepared.run()
    for i, sim in enumerate(sims):
        sim.config.full_field_dft = True
        reference = sim.run()
        assert set(results[i]._fields) == set(components)
        for component in components:
            torch.testing.assert_close(results[i]._fields[component], reference._fields[component.upper()], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(repeated[i]._fields[component], results[i]._fields[component], rtol=0, atol=0)


def test_selected_spectrum_gradient_matches_source_scale_finite_difference():
    scales = torch.tensor([.7, -1.2], device="cuda:0", requires_grad=True)
    batch = mw.Simulation.batch(simulation(steps=27, frequencies=(.9e9, 1e9)),
        parameters=mw.BatchParameters(source_scales={"drive": scales}),
        outputs=mw.BatchOutput(fields=("Ez",)), execution=mw.BatchExecution(
            devices=("cuda:0",), batch_size=2, graph="required", memory_limit_bytes=128 * 1024**2))
    with batch.prepare() as prepared:
        results = prepared.run()
        loss = sum(r._fields["Ez"].abs().square().sum() for r in results)
        loss.backward()
        actual = scales.grad.clone()
        def objective(value):
            with torch.no_grad():
                output = prepared.run(parameters=mw.BatchParameters(source_scales={"drive": value}))
                return sum(r._fields["Ez"].abs().square().sum() for r in output)
        for i in range(2):
            delta = torch.zeros_like(scales)
            delta[i] = .002
            expected = (objective(scales.detach() + delta) - objective(scales.detach() - delta)) / .004
            torch.testing.assert_close(actual[i], expected, rtol=3e-3, atol=1e-8)
