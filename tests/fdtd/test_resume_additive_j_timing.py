"""Resume identity pins the additive electric-current sampling convention."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources
from witwin.maxwell.fdtd import resume
from witwin.maxwell.fdtd.distributed import resume as distributed_resume


MESSAGE = "checkpoint predates the additive-J n+1/2 source timing; re-run from the start"


def _source(kind):
    waveform = mw.CW(frequency=1e9)
    if kind == "point":
        return mw.PointDipole(position=(0., 0., 0.), profile="ideal", source_time=waveform)
    if kind == "uniform":
        return mw.UniformCurrentSource(size=(.02,) * 3, source_time=waveform)
    if kind in ("electric", "magnetic", "mixed"):
        keys = {"electric": ("Jz",), "magnetic": ("Mz",), "mixed": ("Jz", "Mz")}[kind]
        return mw.CustomCurrentSource(mw.CurrentDataset(
            ((-.02, .02),) * 3, {key: np.ones((2, 2, 2)) for key in keys}), source_time=waveform)
    if kind == "plane_wave":
        return mw.PlaneWave(source_time=waveform)
    return None


def _scene(kind="none", device="cpu"):
    source = _source(kind)
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.05, 0.05),) * 3), grid=mw.GridSpec.uniform(0.01),
        boundary=mw.BoundarySpec.pec(), device=device,
        sources=[] if source is None else [source],
    )


def _execution(kind="none", device="cpu", graph=False):
    return mw.Simulation.fdtd(
        _scene(kind, device), frequency=1e9, run_time=mw.TimeConfig(time_steps=8),
        backend="cpu_reference" if device == "cpu" else "native", cuda_graph=graph, full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    ).prepare()


@pytest.mark.parametrize("kind,additive_j", [
    ("none", False), ("point", True), ("uniform", True), ("electric", True),
    ("magnetic", False), ("mixed", True), ("plane_wave", False),
])
def test_only_additive_electric_sources_carry_timing_metadata(kind, additive_j):
    sources = compile_fdtd_sources(_scene(kind), default_frequency=1e9)
    solver = SimpleNamespace(_compiled_sources=sources)
    assert resume._additive_j_signature(solver) == (
        {"additive_j_injection": "n+1/2"} if additive_j else {})


def _distributed_metadata(solver):
    # Only signature/preflight metadata is exercised here, without shard execution.
    return SimpleNamespace(
        shard_layouts=(), logical_scene=solver.scene,
        shards=(SimpleNamespace(rank=0, solver=solver),), _nccl=False, dt=solver.dt,
        _time_plan=solver._time_plan,
    )


def test_non_j_fingerprint_matches_f52aa33f():
    solver = _execution().solver
    # Computed with f52aa33f's fingerprint function on this prepared CPU scene.
    assert resume._configuration_fingerprint(solver) == (
        "9893bfe97ee88e81f7f7f0b148f688c0ef0d8c797ddb4e17871471b7f55f9471")
    assert "additive_j_injection" not in resume._solver_signature(solver)
    distributed_signature = distributed_resume._signature(_distributed_metadata(solver))
    assert "additive_j_injection" not in distributed_signature
    assert distributed_signature["logical_fingerprint"] == (
        "7818cb5d85c17ce8c896e96dfec9694f2ee659dd0b09d6c5e3ea32c2d9255685")


@pytest.mark.parametrize("convention", [None, "n"])
@pytest.mark.parametrize("distributed", [False, True])
def test_old_additive_j_checkpoint_is_refused_before_fingerprint_comparison(tmp_path, convention, distributed):
    execution = _execution("point")
    checkpoint = execution.run_until(3)
    solver = _distributed_metadata(execution.solver) if distributed else execution.solver
    payload = checkpoint._snapshot()
    if distributed:
        payload["signature"] = distributed_resume._signature(solver)
    payload["signature"].pop("additive_j_injection")
    if convention is not None:
        payload["signature"]["additive_j_injection"] = convention
    # Old checkpoints also lack the timing hash; the timing diagnostic must win.
    payload["signature"]["logical_fingerprint" if distributed else "configuration_fingerprint"] = "old fingerprint"
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    restored = mw.FDTDResumeCheckpoint.load(path)
    before = execution.solver.Ez.clone()
    with pytest.raises(NotImplementedError, match=MESSAGE.replace("+", r"\+")):
        resume.preflight_resume_checkpoint(solver, restored, total_steps=8)
    assert torch.equal(execution.solver.Ez, before)


def test_non_j_disk_checkpoint_keeps_old_signature_and_resumes(tmp_path):
    checkpoint = _execution().run_until(3)
    assert "additive_j_injection" not in checkpoint.signature
    checkpoint.save(tmp_path / "non-j.pt")
    target = _execution()
    target.run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path / "non-j.pt"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["point", "uniform", "electric"])
@pytest.mark.parametrize("graph", [False, True])
def test_new_additive_j_disk_checkpoint_round_trips(tmp_path, device, kind, graph):
    if device == "cpu" and (kind != "point" or graph):
        pytest.skip("CPU reference profile supports eager point-current execution")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    expected = _execution(kind, device, graph).run()
    checkpoint = _execution(kind, device, graph).run_until(3)
    assert checkpoint.signature["additive_j_injection"] == "n+1/2"
    checkpoint.save(tmp_path / "new.pt")
    restored = mw.FDTDResumeCheckpoint.load(tmp_path / "new.pt")
    actual = _execution(kind, device, graph).run(resume_from=restored)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(actual.solver, name), getattr(expected.solver, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["point", "uniform", "electric"])
def test_native_resume_refuses_old_additive_j_checkpoint(tmp_path, kind):
    checkpoint = _execution(kind, "cuda").run_until(3)
    payload = checkpoint._snapshot()
    payload["signature"].pop("additive_j_injection")
    torch.save(payload, tmp_path / "old.pt")
    target = _execution(kind, "cuda")
    before = target.solver.Ez.clone()
    with pytest.raises(NotImplementedError, match=MESSAGE.replace("+", r"\+")):
        target.run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path / "old.pt"))
    assert torch.equal(target.solver.Ez, before)
