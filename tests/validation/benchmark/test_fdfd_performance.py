import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
def test_fdfd_performance_case_reports_metrics():
    from benchmark import fdfd_performance as perf

    case = perf.run_case(size=24, solver_type="gmres", max_iter=60, tol=1e-6, restart=20,
                         preconditioner="ssor")

    assert case.status == "ok"
    assert case.preconditioner == "ssor"
    assert case.precond_setup_s >= 0.0
    assert case.unknowns > 0
    assert case.nnz > 0
    assert case.assembly_s > 0.0
    assert case.solve_s > 0.0
    assert case.matvecs > 0
    assert case.residual == case.residual  # not NaN
    assert case.peak_gpu_gb > 0.0

    markdown = perf.render_markdown([case], timestamp="test")
    assert "| 24^3 |" in markdown
    assert "Peak GPU" in markdown
