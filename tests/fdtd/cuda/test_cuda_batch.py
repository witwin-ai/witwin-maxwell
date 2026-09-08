"""Native batch addressing reuses the independently exercised field formulas."""
import pytest
import torch

from witwin.maxwell.fdtd.cuda.backend import get_compiled_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA batch kernels require a GPU")


@pytest.fixture(scope="module")
def native():
    return get_compiled_extension()


def inputs(batch, component, electric, dense=False):
    shape = (5, 6, 7)
    first = list(shape)
    second = list(shape)
    first[1 if component == 2 else 2] += -1 if electric else 1
    second[1 if component == 0 else 0] += -1 if electric else 1
    device = "cuda:0"
    field = torch.randn(batch, *shape, device=device)
    first = torch.randn(batch, *first, device=device)
    second = torch.randn(batch, *second, device=device)
    coefficient_shape = (batch, *shape) if dense else shape
    decay = torch.rand(coefficient_shape, device=device) * 0.1 + 0.9
    curl = torch.rand(coefficient_shape, device=device) * 0.1
    inv_a = torch.rand(shape[1 if component == 0 else 0], device=device) + 0.1
    inv_b = torch.rand(shape[1 if component == 2 else 2], device=device) + 0.1
    active = torch.ones(batch, dtype=torch.int32, device=device)
    return [field, first, second, decay, curl, inv_a, inv_b, active]


def independent(native, values, component, electric, modes):
    field, first, second, decay, curl, inv_a, inv_b, active = values
    expected = field.clone()
    name = f"update_{'electric_e' if electric else 'magnetic_h'}{'xyz'[component]}_standard"
    operation = getattr(native, name)
    for lane in range(field.shape[0]):
        if active[lane] == 0:
            continue
        args = [expected[lane], first[lane], second[lane],
                decay[lane] if decay.ndim == 4 else decay,
                curl[lane] if curl.ndim == 4 else curl, inv_a, inv_b]
        if electric:
            args.extend(modes)
        operation(*args, None, None)
    return expected


def batched(native, values, component, electric, modes):
    operation = native.update_electric_batch if electric else native.update_magnetic_batch
    operation(*values, component, *modes if electric else ())


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8, 10])
@pytest.mark.parametrize("component,electric", [(c, e) for e in (False, True) for c in range(3)])
@pytest.mark.parametrize("dense", [False, True])
def test_batch_matches_independent_components(native, batch, component, electric, dense):
    values = inputs(batch, component, electric, dense)
    modes = (2, 2, 4, 5)
    expected = independent(native, values, component, electric, modes)
    batched(native, values, component, electric, modes)
    torch.testing.assert_close(values[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("modes", [(0, 0, 0, 0), (2, 2, 2, 2), (4, 4, 4, 4), (5, 5, 5, 5)])
def test_electric_boundary_access_stays_inside_each_lane(native, component, modes):
    values = inputs(3, component, True)
    values[1][1].mul_(1000)
    expected = independent(native, values, component, True, modes)
    batched(native, values, component, True, modes)
    torch.testing.assert_close(values[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("electric", [False, True])
def test_poisoned_inactive_lane_does_not_change_active_results(native, electric):
    values = inputs(4, 0, electric, dense=True)
    values[-1][-1] = 0
    for tensor in values[:5]:
        tensor[-1].fill_(float("nan"))
    expected = independent(native, values, 0, electric, (2, 2, 2, 2))
    batched(native, values, 0, electric, (2, 2, 2, 2))
    torch.testing.assert_close(values[0], expected, rtol=0, atol=0, equal_nan=True)
    assert torch.isfinite(values[0][:-1]).all()


def test_batch_uses_nondefault_stream_and_graph_replay_with_updated_mask(native):
    values = inputs(3, 0, True)
    initial = values[0].clone()
    expected = independent(native, values, 0, True, (2, 2, 2, 2))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        batched(native, values, 0, True, (2, 2, 2, 2))
        values[0].copy_(initial)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        batched(native, values, 0, True, (2, 2, 2, 2))
    torch.cuda.current_stream().wait_stream(stream)
    values[0].copy_(initial)
    values[-1][1] = 0
    expected[1].copy_(initial[1])
    graph.replay()
    torch.testing.assert_close(values[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mutation,match", [
    (lambda values: values.__setitem__(7, values[7].float()), "active"),
    (lambda values: values.__setitem__(7, values[7][:-1]), "active"),
    (lambda values: values.__setitem__(1, values[1][:, :-1].contiguous()), "Yee"),
    (lambda values: values.__setitem__(3, values[0]), "overlap"),
    (lambda values: values.__setitem__(6, values[6][:-1]), "inv_b"),
    (lambda values: values.__setitem__(7, values[0].view(torch.int32).flatten()[:3]), "overlap"),
])
def test_invalid_layouts_are_rejected_before_launch(native, mutation, match):
    values = inputs(3, 0, True)
    mutation(values)
    with pytest.raises(RuntimeError, match=match):
        batched(native, values, 0, True, (2, 2, 2, 2))


def test_unsupported_bloch_kernel_does_not_silently_use_real_periodicity(native):
    with pytest.raises(RuntimeError, match="supports only"):
        batched(native, inputs(3, 0, True), 0, True, (3, 3, 3, 3))
