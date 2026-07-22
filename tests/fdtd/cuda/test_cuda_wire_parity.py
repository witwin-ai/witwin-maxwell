from __future__ import annotations

import os

import pytest
import torch


_CUDA_BUILD = pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA wire kernels.",
)


def _extension():
    from witwin.maxwell.fdtd.cuda import backend

    return backend.get_compiled_extension()


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    return (2e-6, 2e-6) if dtype == torch.float32 else (1e-12, 1e-12)


def test_wire_kernels_are_registered_in_the_native_backend():
    from witwin.maxwell.fdtd.cuda import backend

    assert backend._KERNELS["sampleWireEmf3D"] is backend._sample_wire_emf
    assert backend._KERNELS["updateWireState1D"] is backend._update_wire_state
    assert backend._KERNELS["depositWireCurrent3D"] is backend._deposit_wire_current


@_CUDA_BUILD
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sample_wire_emf_matches_deterministic_torch_reference(dtype):
    device = torch.device("cuda")
    ex = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 4) * 0.25 - 1.0
    ey = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 4) * -0.125 + 0.75
    ez = torch.arange(24, device=device, dtype=dtype).reshape(2, 3, 4) * 0.0625 + 0.5
    segment_offsets = torch.tensor([0, 3, 7, 9], device=device, dtype=torch.int64)
    edge_components = torch.tensor(
        [0, 1, 2, 0, 2, 1, 0, 2, 1], device=device, dtype=torch.int32
    )
    edge_offsets = torch.tensor(
        [0, 1, 2, 7, 8, 9, 10, 20, 3], device=device, dtype=torch.int64
    )
    weights = torch.tensor(
        [0.5, -0.25, 0.75, -0.125, 0.375, 0.625, -0.5, 0.2, 0.8],
        device=device,
        dtype=dtype,
    )
    emf = torch.empty(3, device=device, dtype=dtype)

    gathered = torch.stack(
        (
            ex.flatten()[edge_offsets],
            ey.flatten()[edge_offsets],
            ez.flatten()[edge_offsets],
        )
    )
    selected = gathered.gather(0, edge_components.to(torch.int64).unsqueeze(0)).squeeze(0)
    weighted = selected * weights
    expected = torch.stack((weighted[0:3].sum(), weighted[3:7].sum(), weighted[7:9].sum()))

    extension = _extension()
    extension.sample_wire_emf(
        ex,
        ey,
        ez,
        segment_offsets,
        edge_components,
        edge_offsets,
        weights,
        emf,
    )
    first = emf.clone()
    emf.zero_()
    extension.sample_wire_emf(
        ex,
        ey,
        ez,
        segment_offsets,
        edge_components,
        edge_offsets,
        weights,
        emf,
    )

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(emf, expected, rtol=rtol, atol=atol)
    assert torch.equal(emf, first)


def _wire_state_inputs(dtype: torch.dtype):
    device = torch.device("cuda")
    return {
        "emf": torch.tensor([0.8, -0.35, 0.2], device=device, dtype=dtype),
        "tail": torch.tensor([0, 1, 1], device=device, dtype=torch.int64),
        "head": torch.tensor([1, 2, 3], device=device, dtype=torch.int64),
        "inductance": torch.tensor([1.2, 0.75, 1.5], device=device, dtype=dtype),
        "node_capacitance": torch.tensor([0.5, 1.25, 0.8, 1.1], device=device, dtype=dtype),
        "grounded": torch.tensor([True, False, False, True], device=device, dtype=torch.bool),
        "node_offsets": torch.tensor([0, 1, 4, 5, 6], device=device, dtype=torch.int64),
        "node_segments": torch.tensor([0, 0, 1, 2, 1, 2], device=device, dtype=torch.int64),
        "node_signs": torch.tensor([1, -1, 1, 1, -1, -1], device=device, dtype=torch.int32),
        "current": torch.tensor([0.1, -0.2, 0.05], device=device, dtype=dtype),
        "charge": torch.tensor([4.0, 0.3, -0.16, -2.0], device=device, dtype=dtype),
    }


def _reference_wire_state(inputs: dict[str, torch.Tensor], dt: float):
    voltage = torch.where(
        inputs["grounded"],
        torch.zeros_like(inputs["charge"]),
        inputs["charge"] / inputs["node_capacitance"],
    )
    current = inputs["current"] + dt * (
        inputs["emf"]
        + voltage[inputs["tail"]]
        - voltage[inputs["head"]]
    ) / inputs["inductance"]
    charge = inputs["charge"].clone()
    offsets = (0, 1, 4, 5, 6)
    for node, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        incidence = (
            inputs["node_signs"][begin:end].to(charge.dtype)
            * current[inputs["node_segments"][begin:end]]
        ).sum()
        charge[node] -= dt * incidence
    charge = torch.where(inputs["grounded"], torch.zeros_like(charge), charge)
    return current, charge


@_CUDA_BUILD
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_update_wire_state_matches_incidence_reference_and_is_deterministic(dtype):
    dt = 0.04
    inputs = _wire_state_inputs(dtype)
    expected_current, expected_charge = _reference_wire_state(inputs, dt)
    current_a = inputs["current"].clone()
    charge_a = inputs["charge"].clone()
    current_b = inputs["current"].clone()
    charge_b = inputs["charge"].clone()
    extension = _extension()

    args = (
        inputs["emf"],
        inputs["tail"],
        inputs["head"],
        inputs["inductance"],
        inputs["node_capacitance"],
        inputs["grounded"],
        inputs["node_offsets"],
        inputs["node_segments"],
        inputs["node_signs"],
        dt,
    )
    extension.update_wire_state(*args, current_a, charge_a)
    extension.update_wire_state(*args, current_b, charge_b)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(current_a, expected_current, rtol=rtol, atol=atol)
    torch.testing.assert_close(charge_a, expected_charge, rtol=rtol, atol=atol)
    assert torch.equal(current_a, current_b)
    assert torch.equal(charge_a, charge_b)
    assert charge_a[0] == 0
    assert charge_a[3] == 0


def _deposit_inputs(dtype: torch.dtype):
    device = torch.device("cuda")
    return {
        "ex": torch.arange(12, device=device, dtype=dtype).reshape(2, 2, 3) * 0.1,
        "ey": torch.arange(12, device=device, dtype=dtype).reshape(2, 2, 3) * -0.2,
        "ez": torch.arange(12, device=device, dtype=dtype).reshape(2, 2, 3) * 0.3,
        "edge_group_offsets": torch.tensor([0, 2, 5, 6, 8, 10], device=device, dtype=torch.int64),
        "target_components": torch.tensor([0, 1, 2, 0, 2], device=device, dtype=torch.int32),
        "target_offsets": torch.tensor([1, 4, 8, 10, 2], device=device, dtype=torch.int64),
        "contribution_segments": torch.tensor(
            [0, 2, 1, 0, 2, 1, 2, 0, 1, 2], device=device, dtype=torch.int64
        ),
        "contribution_scales": torch.tensor(
            [0.5, -0.25, 0.3, 0.2, -0.1, 0.75, -0.4, 0.6, -0.2, 0.5],
            device=device,
            dtype=dtype,
        ),
        "current": torch.tensor([0.4, -0.2, 0.9], device=device, dtype=dtype),
    }


def _reference_deposit(inputs: dict[str, torch.Tensor]):
    fields = [inputs["ex"].clone(), inputs["ey"].clone(), inputs["ez"].clone()]
    offsets = (0, 2, 5, 6, 8, 10)
    components = (0, 1, 2, 0, 2)
    targets = (1, 4, 8, 10, 2)
    for target, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        contribution = (
            inputs["contribution_scales"][begin:end]
            * inputs["current"][inputs["contribution_segments"][begin:end]]
        ).sum()
        fields[components[target]].view(-1)[targets[target]] -= contribution
    return tuple(fields)


@_CUDA_BUILD
def test_native_module_launches_the_wire_keyword_contract_end_to_end():
    from witwin.maxwell.fdtd.cuda import backend

    dtype = torch.float32
    state = _wire_state_inputs(dtype)
    deposit = _deposit_inputs(dtype)
    emf = torch.empty(3, device="cuda", dtype=dtype)
    segment_offsets = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
    edge_components = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    edge_offsets = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
    weights = torch.ones(3, device="cuda", dtype=dtype)
    _extension()
    module = backend.get_native_fdtd_module()

    module.sampleWireEmf3D(
        Ex=deposit["ex"],
        Ey=deposit["ey"],
        Ez=deposit["ez"],
        segmentOffsets=segment_offsets,
        edgeComponents=edge_components,
        edgeOffsets=edge_offsets,
        weights=weights,
        emf=emf,
    ).launchRaw()
    module.updateWireState1D(
        emf=emf,
        tail=state["tail"],
        head=state["head"],
        inductance=state["inductance"],
        nodeCapacitance=state["node_capacitance"],
        grounded=state["grounded"],
        nodeOffsets=state["node_offsets"],
        nodeSegments=state["node_segments"],
        nodeSigns=state["node_signs"],
        dt=0.04,
        current=state["current"],
        charge=state["charge"],
    ).launchRaw()
    module.depositWireCurrent3D(
        Ex=deposit["ex"],
        Ey=deposit["ey"],
        Ez=deposit["ez"],
        edgeGroupOffsets=deposit["edge_group_offsets"],
        targetComponents=deposit["target_components"],
        targetOffsets=deposit["target_offsets"],
        contributionSegments=deposit["contribution_segments"],
        contributionScales=deposit["contribution_scales"],
        current=state["current"],
    ).launchRaw()
    torch.cuda.synchronize()

    assert torch.all(torch.isfinite(emf))
    assert torch.all(torch.isfinite(state["current"]))
    assert torch.all(torch.isfinite(state["charge"]))
    assert all(torch.all(torch.isfinite(deposit[name])) for name in ("ex", "ey", "ez"))


@_CUDA_BUILD
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deposit_wire_current_matches_grouped_reference_without_atomics(dtype):
    inputs = _deposit_inputs(dtype)
    expected = _reference_deposit(inputs)
    fields_a = [inputs[name].clone() for name in ("ex", "ey", "ez")]
    fields_b = [inputs[name].clone() for name in ("ex", "ey", "ez")]
    extension = _extension()
    args = (
        inputs["edge_group_offsets"],
        inputs["target_components"],
        inputs["target_offsets"],
        inputs["contribution_segments"],
        inputs["contribution_scales"],
        inputs["current"],
    )
    extension.deposit_wire_current(*fields_a, *args)
    extension.deposit_wire_current(*fields_b, *args)

    rtol, atol = _tolerance(dtype)
    for actual, repeated, reference in zip(fields_a, fields_b, expected):
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        assert torch.equal(actual, repeated)


@_CUDA_BUILD
def test_wire_kernel_sequence_is_cuda_graph_safe_and_replay_deterministic():
    dtype = torch.float32
    sample = _deposit_inputs(dtype)
    state = _wire_state_inputs(dtype)
    segment_offsets = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
    edge_components = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    edge_offsets = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
    weights = torch.tensor([0.5, -0.25, 0.75], device="cuda", dtype=dtype)
    emf = torch.empty(3, device="cuda", dtype=dtype)
    current = state["current"].clone()
    charge = state["charge"].clone()
    ex = sample["ex"].clone()
    ey = sample["ey"].clone()
    ez = sample["ez"].clone()
    current_initial = current.clone()
    charge_initial = charge.clone()
    ex_initial = ex.clone()
    ey_initial = ey.clone()
    ez_initial = ez.clone()
    extension = _extension()

    def run_sequence():
        current.copy_(current_initial)
        charge.copy_(charge_initial)
        ex.copy_(ex_initial)
        ey.copy_(ey_initial)
        ez.copy_(ez_initial)
        extension.sample_wire_emf(
            ex,
            ey,
            ez,
            segment_offsets,
            edge_components,
            edge_offsets,
            weights,
            emf,
        )
        extension.update_wire_state(
            emf,
            state["tail"],
            state["head"],
            state["inductance"],
            state["node_capacitance"],
            state["grounded"],
            state["node_offsets"],
            state["node_segments"],
            state["node_signs"],
            0.04,
            current,
            charge,
        )
        extension.deposit_wire_current(
            ex,
            ey,
            ez,
            sample["edge_group_offsets"],
            sample["target_components"],
            sample["target_offsets"],
            sample["contribution_segments"],
            sample["contribution_scales"],
            current,
        )

    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        run_sequence()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_sequence()
    graph.replay()
    torch.cuda.synchronize()
    first = tuple(value.clone() for value in (emf, current, charge, ex, ey, ez))
    graph.replay()
    torch.cuda.synchronize()

    for value, reference in zip((emf, current, charge, ex, ey, ez), first):
        assert torch.equal(value, reference)
