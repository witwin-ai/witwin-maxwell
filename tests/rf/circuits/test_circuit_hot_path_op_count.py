"""Op-count contract for the circuit-bound port per-step hot path.

Deterministic host/device op tallies only (no timing is asserted). The
circuit-bound (MNA companion) path is the heaviest remaining per-step port
path; these ceilings lock in the eager per-step budget and the fixed-schedule
CUDA-graph reduction so a regression (reintroduced per-step allocation, a lost
graph capture, or a new host<->device sync) turns the suite red.

Falsification (recorded 2026-07-17): forcing ``prepare_circuit_graph_runners``
to leave the graph inactive drops the run back onto the eager schedule
(133 launches/step); ``test_circuit_bound_graph_step``'s
``_circuit_graph_active is True`` guard then fires red. Tightening the eager
``launches`` ceiling below the measured 133/step (e.g. to 100/step) turns
``test_circuit_bound_eager_step`` red, confirming that ceiling is not vacuous.
Both were verified red under a scratch monkeypatch and restored to green.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.circuits import (
    prepare_circuit_graph_runners,
    prepare_circuit_time_series,
)
from witwin.maxwell.fdtd.ports import (
    accumulate_port_observers,
    apply_port_runtimes,
    prepare_port_spectral_accumulators,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Circuit hot-path op-count contract requires CUDA.",
)


def _port():
    return mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
        reference_impedance=50.0,
    )


def _rlc_ladder_circuit():
    circuit = mw.Circuit("rlc_ladder")
    a = circuit.node("a")
    b = circuit.node("b")
    circuit.add(mw.Resistor("R1", a, b, 60.0))
    circuit.add(mw.Inductor("L1", b, circuit.ground, 1.2e-9))
    circuit.add(mw.Capacitor("C1", a, circuit.ground, 0.8e-12))
    circuit.add(mw.Capacitor("C2", b, circuit.ground, 0.5e-12))
    circuit.bind_port("feed", positive=a, negative=circuit.ground)
    return circuit


def _circuit_solver(*, cuda_graph):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(_port(),),
        circuits=(_rlc_ladder_circuit(),),
        device="cuda",
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=96),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    ).prepare().solver
    prepare_port_spectral_accumulators(solver, 96, "none")
    prepare_circuit_time_series(solver, 96)
    if cuda_graph:
        prepare_circuit_graph_runners(solver, True)
    return solver


def _hot_path_op_inventory(solver, *, steps: int = 16) -> dict[str, int]:
    for _ in range(8):
        apply_port_runtimes(solver)
        accumulate_port_observers(solver)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        acc_events=True,
    ) as profile:
        for _ in range(steps):
            apply_port_runtimes(solver)
            accumulate_port_observers(solver)
    torch.cuda.synchronize()

    tally = {
        "launches": 0,
        "allocs": 0,
        "memcpy_dtod": 0,
        "memcpy_hostside": 0,
        "scalar_sync": 0,
        "device_mem": 0,
    }
    for event in profile.key_averages():
        key = event.key
        if "cudaLaunchKernel" in key:
            tally["launches"] += event.count
        if key in ("aten::empty", "aten::empty_strided", "aten::empty_like"):
            tally["allocs"] += event.count
        if "Memcpy DtoD" in key:
            tally["memcpy_dtod"] += event.count
        if "Memcpy HtoD" in key or "Memcpy DtoH" in key:
            tally["memcpy_hostside"] += event.count
        if key in ("aten::item", "aten::_local_scalar_dense"):
            tally["scalar_sync"] += event.count
        tally["device_mem"] += max(0, getattr(event, "self_device_memory_usage", 0))
    return tally


def test_circuit_bound_eager_step_stays_within_op_count_ceiling():
    # Default (cuda_graph off) production path: generic transient MNA assembly
    # and per-device power/current readout. Measured base schedule: 133 launches,
    # 18 allocs, 31 DtoD per step, 0 scalar syncs, 0 host<->device transfers.
    # Ceilings carry headroom so a genuine reduction passes while a regression
    # (reintroduced diagnostics, a per-step host sync) turns this red.
    solver = _circuit_solver(cuda_graph=False)
    steps = 16
    tally = _hot_path_op_inventory(solver, steps=steps)

    assert tally["scalar_sync"] == 0
    assert tally["memcpy_hostside"] == 0
    assert tally["launches"] <= steps * 160
    assert tally["allocs"] <= steps * 24
    assert tally["memcpy_dtod"] <= steps * 40


def test_circuit_bound_graph_step_stays_within_op_count_ceiling():
    # Fixed-schedule CUDA-graph path: the RLC ladder is graph-eligible, so the
    # captured step collapses the eager 133-launch/18-alloc schedule to 8
    # launches and 0 allocations per step. Measured base schedule: 8 launches,
    # 0 allocs, 25 DtoD per step. If graph capture silently regresses to the
    # eager fallback the launch tally jumps to 133 and this ceiling fails.
    solver = _circuit_solver(cuda_graph=True)
    assert solver._circuit_graph_active is True, getattr(
        solver, "_circuit_graph_error", None
    )

    steps = 16
    tally = _hot_path_op_inventory(solver, steps=steps)

    assert tally["scalar_sync"] == 0
    assert tally["memcpy_hostside"] == 0
    assert tally["allocs"] == 0
    assert tally["launches"] <= steps * 12
    assert tally["memcpy_dtod"] <= steps * 32
