"""The fused core Yee update against the frozen Torch Yee oracle.

The oracle (``tests/oracles/yee_cpu_reference.py``) is the kernel module of the retired
CPU reference profile; its step driver applies the per-component updates to a prepared
solver's own tensors. Since W7 row 16e deleted the native per-component kernels, the
core launches are compared with it directly, from the same random state, at the
step-level bound of the numerical-equivalence standard.
"""
import pytest
import torch

from benchmark.core_path.gates import make_simulation
from benchmark.core_path.scenes import build_core_case
from tests.oracles import yee_cpu_reference
from tests.support.numerical_equivalence import assert_numerically_equivalent

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.mark.parametrize("name,cpml", [("boundary_pec", False), ("pmc_cavity", False), ("boundary_none", False), ("cpml_dense", True)])
def test_core_yee_step_matches_the_torch_oracle(name, cpml):
    solver = make_simulation(build_core_case(name), steps=8, full_field_dft=True).prepare().solver
    assert solver._core_runtime is not None and solver.uses_cpml is cpml
    state = {key: value for key, value in vars(solver).items()
             if (key in FIELDS or key.startswith("psi_")) and isinstance(value, torch.Tensor)}
    generator = torch.Generator(device=solver.device).manual_seed(3)
    for tensor in state.values():
        tensor.copy_(torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype))
    start = {key: tensor.clone() for key, tensor in state.items()}
    solver._core_runtime.magnetic()
    solver._core_runtime.electric()
    core = {key: tensor.clone() for key, tensor in state.items()}
    for key, tensor in state.items():
        tensor.copy_(start[key])
    yee_cpu_reference.magnetic_step(solver)
    yee_cpu_reference.electric_step(solver)
    assert_numerically_equivalent(state, core, level="step")
