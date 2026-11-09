"""Bitwise release gates for native source patches."""

import numpy as np
import pytest
import torch

import witwin.maxwell as mw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)
_STEPS = 512


def _oblique_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.96, 0.96),) * 3),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4), device="cuda:0",
    )
    scene.add_source(mw.PlaneWave(
        direction=(0.6, 0.0, 0.8), polarization="Ey",
        source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.35e9, amplitude=80.0),
        injection=mw.TFSF(bounds=((-0.32, 0.32),) * 3),
    ))
    scene.add_monitor(mw.PointMonitor("probe", (0.08, 0.08, 0.08), fields=_FIELDS))
    return scene


def _seam_scene(bloch=False):
    boundary = mw.BoundarySpec.faces(
        default="pml", num_layers=3 if bloch else 40, x="bloch" if bloch else "periodic",
        y="bloch" if bloch else "periodic", z="pml",
        bloch_wavevector=(2.3, 1.1, 0.0) if bloch else (0.0, 0.0, 0.0),
    )
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
                     grid=mw.GridSpec.uniform(0.05), boundary=boundary, device="cuda:0")
    # Unequal corner currents make the four endpoint writes order-sensitive.
    coords = (np.linspace(-0.4, 0.4, 3), np.linspace(-0.4, 0.4, 3), (-0.05, 0.05))
    current = np.random.default_rng(17).uniform(0.1, 2.0, (3, 3, 2))
    waveform = mw.CW(frequency=1e9, phase=0.37) if bloch else mw.GaussianPulse(frequency=1e9, fwidth=1e9)
    scene.add_source(mw.CustomCurrentSource(mw.CurrentDataset(coords, {"Jz": current}), source_time=waveform))
    scene.add_monitor(mw.PointMonitor("probe", (0.35, 0.35, 0.1), fields=_FIELDS))
    return scene


def _prepare(scene, *, core_path=False):
    torch.manual_seed(17)
    return mw.Simulation.fdtd(
        scene, frequencies=_FREQUENCIES, backend="native", cuda_graph=False, fdtd_core_path=core_path,
        run_time=mw.TimeConfig(time_steps=_STEPS), shutoff=0,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
    ).prepare()


def _snapshot(solver, result):
    names = _FIELDS + tuple(name + "_imag" for name in _FIELDS) if solver.complex_fields_enabled else _FIELDS
    values = {name: getattr(solver, name).detach().cpu().clone() for name in names}
    spectra = result.raw_monitor("probe")["components"]
    if set(spectra) != set(_FIELDS):
        pytest.fail("The probe must export all six field spectra")
    values.update({"spectrum/" + name: torch.as_tensor(value).detach().cpu().clone()
                   for name, value in spectra.items()})
    if not all(torch.isfinite(value).all() for value in values.values()):
        pytest.fail("Nonfinite fields cannot qualify determinism")
    if not any(torch.count_nonzero(values[name]) for name in _FIELDS):
        pytest.fail("The source did not excite the scene")
    return values


def _assert_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


def _repeat(run):
    expected = run()
    # Finish all three independent executions before comparing their snapshots.
    actual = [run(), run()]
    for values in actual:
        _assert_equal(expected, values)


def _oblique_run(*, fallback=False):
    execution = _prepare(_oblique_scene())
    state = execution.solver._tfsf_state
    if state["provider"] != "plane_wave_aux" or any(state.get(key) is None for key in ("electric_batch", "magnetic_batch")):
        pytest.fail("Expected packed oblique auxiliary patches")
    if fallback:
        state["electric_batch"] = state["magnetic_batch"] = None
    return _snapshot(execution.solver, execution.run())


def test_oblique_pulsed_tfsf_bitwise():
    # plane_wave_aux + packed faces -> addBatchedInterpolatedSourcePatches3D, including cross-warp edge collisions.
    _repeat(_oblique_run)


def test_interpolated_per_term_fallback_bitwise():
    # plane_wave_aux without packed faces -> addInterpolatedSourcePatch3D; unique offsets per launch, serial terms: no xfail.
    _repeat(lambda: _oblique_run(fallback=True))


def _seam_run(bloch):
    execution = _prepare(_seam_scene(bloch))
    solver = execution.solver
    term, = solver._source_terms
    if (term["field_name"] != "Ez" or term["delay_patch"] is not None or term["cw_cos_patch"] is not None
            or term.get("periodic_compiled", False) or solver.complex_fields_enabled != bloch):
        pytest.fail("Expected an undelayed seam patch with the requested field representation")
    if term["offsets"][:2] != (0, 0) or term["patch"].shape[:2] != solver.Ez.shape[:2]:
        pytest.fail("The patch must include both endpoints of both tangential axes")
    return _snapshot(solver, execution.run())


def test_periodic_seam_patch_bitwise():
    # apply_compiled_source_terms: undelayed, non-Bloch, periodic, not periodic_compiled -> addSourcePatchEzPeriodic3D.
    _repeat(lambda: _seam_run(False))


def test_bloch_seam_patch_bitwise():
    # apply_compiled_source_terms: Bloch + undelayed current -> addSourcePatchBloch3D with x/y phase-rotated copies.
    _repeat(lambda: _seam_run(True))


def _batch_run():
    torch.manual_seed(17)
    simulations = []
    for lane in range(3):
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(0.025),
                         boundary=mw.BoundarySpec.pml(num_layers=3), device="cuda:0")
        for name, amplitude in (("drive", 1.0 + lane), ("overlap", 0.3)):
            scene.add_source(mw.PointDipole(
                name=name, position=(-0.05, 0.0, 0.0), polarization="Ez",
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9, amplitude=amplitude),
            ))
        scene.add_monitor(mw.PointMonitor("probe", (0.1, 0.0, 0.0), fields=_FIELDS))
        simulations.append(mw.Simulation.fdtd(
            scene, frequencies=_FREQUENCIES, backend="native", cuda_graph=False, shutoff=0,
            run_time=mw.TimeConfig(time_steps=120, dt=2e-11),
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
        ))
    batch = mw.Simulation.batch(
        simulations, variation=mw.BatchVariation(sources=("drive",)),
        execution=mw.BatchExecution(devices=("cuda:0",), batch_size=3, graph="off",
                                    memory_limit_bytes=4 * 1024**3),
    )
    with batch.prepare() as execution:
        results = execution.run()
        with torch.cuda.stream(execution._stream):
            return {f"lane{lane}/{name}": value for lane, solver in enumerate(execution._slot.solvers)
                    for name, value in _snapshot(solver, results[lane]).items()}


def test_batched_patch_bitwise():
    # BatchSourceEngine.advance -> batch_source_patch; PML means wrap=0, unique offsets per lane/ordinal: no xfail.
    _repeat(_batch_run)


def test_oblique_checkpoint_resume_bitwise(tmp_path):
    # Direct and resumed plane_wave_aux runs both retain packed faces -> addBatchedInterpolatedSourcePatches3D.
    expected = _oblique_run()
    resumed = []
    for replay in range(3):
        execution = _prepare(_oblique_scene())
        checkpoint = execution.run_until(211)
        path = tmp_path / f"oblique-{replay}.pt"
        checkpoint.save(path)
        del checkpoint, execution
        target = _prepare(_oblique_scene())
        result = target.run(resume_from=mw.FDTDResumeCheckpoint.load(path, map_location="cpu"))
        resumed.append(_snapshot(target.solver, result))
    for values in resumed:
        _assert_equal(expected, values)


def test_core_path_periodic_seam_patch_bitwise():
    # fdtd_core_path=True fails closed when unsupported; its source patch with s.wrap set atomicAdds every
    # periodic boundary image (fdtd/cuda/core/step.cu periodic wrap branch).
    _repeat(lambda: (lambda execution: _snapshot(execution.solver, execution.run()))(
        _prepare(_seam_scene(False), core_path=True)))
