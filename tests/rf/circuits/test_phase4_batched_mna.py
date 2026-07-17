import pytest
import torch

from witwin.maxwell.compiler import compile_batched_mna_factors


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Batched MNA execution requires CUDA.",
)


@pytest.mark.parametrize("unknown_count", (8, 32))
def test_batched_mna_cached_solve_matches_independent_dense_reference(unknown_count):
    generator = torch.Generator(device="cuda").manual_seed(20260715 + unknown_count)
    raw = torch.randn(
        (5, unknown_count, unknown_count),
        generator=generator,
        device="cuda",
        dtype=torch.float64,
    )
    matrices = raw @ raw.transpose(-1, -2)
    matrices = matrices + unknown_count * torch.eye(
        unknown_count,
        device="cuda",
        dtype=torch.float64,
    )
    rhs = torch.randn(
        (5, unknown_count),
        generator=generator,
        device="cuda",
        dtype=torch.float64,
    )
    compiled = compile_batched_mna_factors(matrices)
    output = torch.empty_like(rhs)

    actual = compiled.solve(rhs, out=output)
    expected = torch.linalg.solve(matrices, rhs.unsqueeze(-1)).squeeze(-1)

    assert actual.data_ptr() == output.data_ptr()
    assert compiled.batch_size == 5
    assert compiled.unknown_count == unknown_count
    assert compiled.condition.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-13)


def test_batched_mna_solve_is_cuda_graph_replay_safe():
    matrices = torch.eye(16, device="cuda", dtype=torch.float32).expand(4, -1, -1).clone()
    compiled = compile_batched_mna_factors(matrices)
    rhs = torch.ones((4, 16), device="cuda", dtype=torch.float32)
    output = torch.empty_like(rhs)
    compiled.solve(rhs, out=output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled.solve(rhs, out=output)

    rhs.fill_(3.0)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, rhs, rtol=0.0, atol=0.0)


def test_batched_mna_rejects_cpu_and_singular_batches():
    with pytest.raises(ValueError, match="requires CUDA"):
        compile_batched_mna_factors(torch.eye(2).reshape(1, 2, 2))
    singular = torch.zeros((2, 4, 4), device="cuda")
    with pytest.raises(ValueError, match="singular"):
        compile_batched_mna_factors(singular)
