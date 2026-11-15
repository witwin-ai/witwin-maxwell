"""The passivity guard on the core path's K-step chunks.

The core path ends a chunk at every multiple of SAMPLE_INTERVAL while a guard state
exists and samples after the replay, so its samples, windows and trip step are those
of the per-step loop at every chunk size. A trip is the guard's verdict, not a core
failure: it propagates without rebuilding the existing path.
"""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import passivity

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="core path requires CUDA")]

GUARD_KEYS = ("history", "peak", "window_max")


def _prepared(*, steps, graph=True, chunk=32, shutoff=0., slab=True):
    # shutoff > 0 adds chunk boundaries at its 100-step check interval, off the 32-step samples.
    scene = mw.Scene(domain=mw.Domain(bounds=((-.1, .1),) * 3), grid=mw.GridSpec.uniform(.025), boundary=mw.BoundarySpec.faces(x="pec", y="periodic" if slab else "pec", z="pml", num_layers=4))
    if slab:
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(geometry=mw.Box(position=(0., 0., 0.), size=(.2, .2, .05)),
                         material=mw.Material(eps_r=3.)),
        ))
    scene = scene.updated(sources=(
        *scene.sources,
        mw.PointDipole(position=(.025, .025, .0125), polarization="Ez",
                       source_time=mw.GaussianPulse(frequency=2e9, fwidth=1.5e9)),
    ), monitors=(*scene.monitors, mw.PointMonitor("point", position=(0., 0., 0.), fields=("Ez",))))
    prepared = mw.Simulation.fdtd(scene, frequencies=2e9, config=mw.FDTDConfig(
        absorber="cpml", cuda_graph=graph, shutoff=shutoff, full_field_dft=False,
        run_time=mw.TimeConfig(time_steps=steps))).prepare()
    solver = prepared.solver
    assert solver._core_runtime is not None and hasattr(solver, "_core_runtime")
    solver._core_runtime.chunk_size = chunk
    return prepared


def _run(prepared, *, trips):
    if trips:
        with pytest.raises(mw.fdtd.FDTDPassivityError, match="real-field closed-transverse PML") as error:
            prepared.run()
        return str(error.value)
    prepared.run()
    return None


def _assert_sampling_steps(solver, last_step):
    """Every multiple of SAMPLE_INTERVAL up to last_step is a chunk end."""
    ends = set(torch.tensor(solver._core_step_graph_chunks).cumsum(0).tolist())
    assert set(range(passivity.SAMPLE_INTERVAL, last_step + 1, passivity.SAMPLE_INTERVAL)) <= ends


def _guard(solver):
    state = solver._passivity_state
    return {key: state[key].clone() for key in GUARD_KEYS} | {
        key: state[key] for key in ("sample_count", "window_count", "start_step")}


def _assert_same_guard(actual, expected):
    for key in GUARD_KEYS:
        assert torch.equal(actual[key], expected[key]), key
    for key in ("sample_count", "window_count", "start_step"):
        assert actual[key] == expected[key], key


@pytest.fixture
def short_windows(monkeypatch):
    # Real SAMPLE_INTERVAL (the chunk alignment under test); fewer, shorter windows.
    monkeypatch.setattr(passivity, "WINDOW_SAMPLES", 4)
    monkeypatch.setattr(passivity, "FIT_WINDOWS", 4)


@pytest.fixture
def growing_energy(monkeypatch, short_windows):
    def synthetic(solver):
        state = solver._passivity_state
        sample = state["window_count"] * passivity.WINDOW_SAMPLES + state["sample_count"] + 1
        return torch.tensor(math.exp(1e-5 * sample * passivity.SAMPLE_INTERVAL), dtype=torch.float64)

    monkeypatch.setattr(passivity, "total_energy", synthetic)


@pytest.mark.parametrize("chunk", (32, 16, 1))
def test_core_chunks_trip_at_the_per_step_check_point(growing_energy, chunk):
    steps = 4096
    reference = _prepared(steps=steps, graph=False, shutoff=1e-30)
    message = _run(reference, trips=True)
    assert not reference.solver._core_step_graph_active
    expected = _guard(reference.solver)
    trip = int(expected["history"][-1, 0])
    assert trip % passivity.SAMPLE_INTERVAL == 0 and trip < steps

    prepared = _prepared(steps=steps, chunk=chunk, shutoff=1e-30)
    solver = prepared.solver
    assert _run(prepared, trips=True) == message
    # No rebuild: the core solver raised at the same check point.
    assert prepared.solver is solver and solver._core_runtime is not None
    assert solver._core_step_graph_active and solver._core_step_graph_k == chunk
    assert sum(solver._core_step_graph_chunks) == trip
    _assert_sampling_steps(solver, trip)
    _assert_same_guard(_guard(solver), expected)


def test_core_chunks_nonfinite_energy_fails_closed(monkeypatch, short_windows):
    def nonfinite(solver):
        return torch.tensor(math.inf, dtype=torch.float64)

    monkeypatch.setattr(passivity, "total_energy", nonfinite)
    prepared = _prepared(steps=4096)
    with pytest.raises(mw.fdtd.FDTDPassivityError, match="non-finite field energy"):
        prepared.run()
    assert prepared.solver._core_runtime is not None
    assert prepared.solver._core_step_graph_active


@pytest.mark.parametrize("chunk", (32, 16, 1))
@pytest.mark.parametrize("slab,trips", ((True, True), (False, False)))
def test_core_chunk_real_energy_verdict_equals_the_per_step_loop(short_windows, chunk, slab, trips):
    # The real energy functional with short windows. The thin 4-layer z-CPML under the periodic
    # eps3 slab grows (must trip); the empty PEC guide drains through it (must not trip).
    # Every sample, the verdict and the fields at the stop step are bitwise those of the per-step loop.
    steps = 1600
    reference = _prepared(steps=steps, graph=False, shutoff=1e-30, slab=slab)
    message = _run(reference, trips=trips)
    expected = _guard(reference.solver)
    assert expected["window_count"] >= passivity.FIT_WINDOWS
    prepared = _prepared(steps=steps, chunk=chunk, shutoff=1e-30, slab=slab)
    assert _run(prepared, trips=trips) == message
    solver = prepared.solver
    assert solver._core_runtime is not None and solver._core_step_graph_active
    last = int(expected["history"][-1, 0]) if trips else steps
    assert sum(solver._core_step_graph_chunks) == last
    _assert_sampling_steps(solver, last)
    _assert_same_guard(_guard(solver), expected)
    if not trips:
        assert solver.passivity_guard_status == "active"
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(solver, name), getattr(reference.solver, name)), name


def test_skipped_guard_does_not_split_chunks(monkeypatch):
    # Uncounted storage skips the guard with the unchanged warning and status; chunks keep K.
    monkeypatch.setattr(passivity, "uncounted_storage", lambda solver: ["dispersive/ADE media"])
    prepared = _prepared(steps=200)
    with pytest.warns(RuntimeWarning, match="passivity guard skipped .*dispersive/ADE media"):
        result = prepared.run()
    assert result.metadata["passivity_guard"] == "skipped: energy storage not counted (dispersive/ADE media)"
    assert prepared.solver._passivity_state is None
    assert prepared.solver._core_step_graph_chunks == [32] * 6 + [8]
