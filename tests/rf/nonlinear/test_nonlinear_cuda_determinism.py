"""Bitwise repeatability of standalone nonlinear CUDA transients and stamps."""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.nonlinear_devices import (
    NonlinearMNASystem,
    _scatter_currents,
    _stamp_conductance,
    compile_nonlinear_devices,
    run_nonlinear_transient,
)

import _rectifier_fixtures as fx


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _build_shared_node_system(*, device):
    circuit = mw.Circuit("three_phase_rectifier")
    sources = [circuit.node(f"src{i}") for i in range(3)]
    out = circuit.node("out")
    node_index = {node.name: index for index, node in enumerate([*sources, out])}
    diodes = [
        mw.Diode(
            f"d{i}", node, out,
            saturation_current=fx.SATURATION_CURRENT, ideality=fx.IDEALITY,
        )
        for i, node in enumerate(sources)
    ]
    cap = mw.VoltageDependentCapacitor(
        "cs", out, circuit.ground,
        q_coefficients=torch.tensor([0.0, fx.SMOOTHING_CAPACITANCE], dtype=fx.DTYPE),
    )
    compiled = compile_nonlinear_devices(
        [*diodes, cap], node_index, dtype=fx.DTYPE, device=device,
    )
    gs = 1.0 / fx.SOURCE_RESISTANCE
    gl = 1.0 / fx.LOAD_RESISTANCE
    conductance = torch.diag(torch.tensor([gs, gs, gs, gl], dtype=fx.DTYPE, device=device))
    injection = torch.zeros(4, dtype=fx.DTYPE, device=device)
    system = NonlinearMNASystem(4, conductance, injection, compiled)

    def source_injection(t):
        phase = 2.0 * math.pi * fx.FREQUENCY * t
        currents = [
            fx.AMPLITUDE * gs * math.sin(phase + i * 2.0 * math.pi / 3.0)
            for i in range(3)
        ]
        return torch.tensor([*currents, 0.0], dtype=fx.DTYPE, device=device)

    return system, source_injection, 3


@pytest.mark.parametrize(
    "builder", [fx.build_full_wave_system, _build_shared_node_system],
    ids=["full_wave", "shared_node"],
)
def test_cuda_transient_is_bitwise_repeatable(builder):
    # The standalone Newton path is host-synchronous and launch-bound on CUDA
    # (~1.8 s per step for these 3-4 node systems, vs ~10 ms on CPU), so the
    # repeatability check uses the first 12 steps. Every step assembles the
    # shared-node duplicate stamps, which is what this test pins.
    times = fx.coarse_times()[:13]
    outputs = []
    for _ in range(3):
        system, source_injection, output_index = builder(device="cuda")
        result = run_nonlinear_transient(
            system, times, source_injection, mw.NonlinearSolveConfig(),
            integration="trapezoidal",
        )
        assert result.converged
        assert result.node_voltages.is_cuda
        assert torch.isfinite(result.node_voltages).all()
        assert result.node_voltages[:, output_index].abs().max() > 0
        outputs.append(result.node_voltages)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_duplicate_stamps_are_bitwise_repeatable(dtype):
    count = 43
    positive = torch.zeros(count, dtype=torch.long, device="cuda")
    negative = torch.ones(count, dtype=torch.long, device="cuda")
    generator = torch.Generator().manual_seed(211)
    # Mixed signs and magnitudes expose changes in floating-point addition order.
    values = (
        torch.randn(count, generator=generator, dtype=dtype)
        * torch.logspace(-6, 6, count, dtype=dtype)
    ).cuda()
    conductance = values.abs()
    matrix_base = torch.tensor([[0.125, -0.25], [-0.375, 0.5]], dtype=dtype, device="cuda")
    current_base = torch.tensor([0.125, -0.25], dtype=dtype, device="cuda")
    for repeat in range(200):
        matrix = matrix_base.clone()
        currents = current_base.clone()
        _stamp_conductance(matrix, positive, negative, conductance)
        _scatter_currents(currents, positive, negative, values, sign=-1.0)
        if repeat == 0:
            expected_matrix = matrix.clone()
            expected_currents = currents.clone()
            assert not torch.equal(matrix, matrix_base)
            assert not torch.equal(currents, current_base)
        assert torch.equal(matrix, expected_matrix)
        assert torch.equal(currents, expected_currents)
