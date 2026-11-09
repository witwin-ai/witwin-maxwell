"""Actual prepared providers and native finite-record origin identities."""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.representation import normalization_sample_time, normalization_signal
from witwin.maxwell.fdtd.runtime.spectral import source_normalization_spectrum


CASES = [
    ("axis_pulse", "plane_wave_axis_aligned", 1.),
    ("axis_cw", "plane_wave_axis_aligned", 1.),
    ("special_pulse", "plane_wave_ref_x_ez", 1.),
    ("special_cw", "plane_wave_ref_x_ez", 1.),
    ("oblique_pulse", "plane_wave_aux", 1.),
    # Unshifted CW face carriers use physical Yee instants, hence beta=0.
    ("oblique_cw", "plane_wave_discrete_cw", 0.),
    ("beam_cw", "analytic_profile", 0.),
    ("beam_pulse", "analytic_profile", 0.),
    ("astigmatic_cw", "analytic_profile", 0.),
    ("slab_pulse", "plane_wave_axis_aligned", 1.),
    ("slab_cw", "plane_wave_axis_aligned", 1.),
    ("grating_cw", "plane_wave_grating_slab_cw", 0.),
    # Pulsed fixed-k slabs use the fixed-k incident line; its drive keeps n*dt, hence beta=0.
    ("grating_pulse", "plane_wave_fixed_k", 0.),
]


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def execution(case, *, device="cpu", normalized=True, steps=512, dt=4.7619e-12,
              field_dtype=torch.float32, observer_dtype=torch.float32, full_field_dft=True):
    f = 2e9
    waveform = mw.CW(f) if case.endswith("cw") else mw.GaussianPulse(f, 1e9)
    direction = ((1., 0., 0.) if case.startswith("special") else (.6, 0., .8)
                 if case.startswith("oblique") else (.2, 0., math.sqrt(.96))
                 if case.startswith("grating") else (0., 0., 1.))
    electric = "Ez" if case.startswith("special") else "Ey" if direction[0] else "Ex"
    magnetic = "Hx" if electric == "Ey" else "Hy"
    boundary = mw.BoundarySpec.pml(num_layers=4)
    injection = mw.TFSF(bounds=((-0.1, .1),) * 3)
    if case.startswith("slab"):
        injection = mw.TFSF.slab(axis="z", bounds=(-.1, .1))
    if case.startswith("grating"):
        # Explicit k selects the fixed-k incident line for the pulsed row.
        boundary = mw.BoundarySpec.faces(default="pml", num_layers=4, strength=1., x="bloch", y="bloch", z="pml",
                                   bloch_wavevector=(2 * math.pi * f / 299792458. * direction[0], 0., 0.))
        injection = mw.TFSF.slab(axis="z", bounds=(-.1, .1))
    source = (mw.GaussianBeam(direction=direction, polarization=electric, source_time=waveform,
                injection=injection, beam_waist=.3, focus=(0., 0., 0.)) if case.startswith("beam")
              else mw.PlaneWave(direction=direction, polarization=electric, source_time=waveform, injection=injection))
    if case.startswith("astigmatic"):
        source = mw.AstigmaticGaussianBeam(direction=direction, polarization=electric,
            source_time=waveform, injection=injection, beam_waist=(.3, .4), focus=(0., 0., 0.))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, .3),) * 3), grid=mw.GridSpec.uniform(.025),
        boundary=boundary, device=device, sources=[source], monitors=[
            mw.PointMonitor("point", (0., 0., 0.), fields=(electric, magnetic)),
            mw.FieldTimeMonitor("trace", components=(electric, magnetic), position=(0., 0., 0.))])
    return mw.Simulation.fdtd(scene, frequency=f, backend="cpu_reference" if device == "cpu" else "native",
        cuda_graph=False, full_field_dft=full_field_dft, field_dtype=field_dtype,
        run_time=mw.TimeConfig(time_steps=steps, dt=dt, transient_cycles=0),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=normalized,
                                             observer_dtype=observer_dtype)).prepare(), electric, magnetic


@pytest.mark.parametrize("case,provider,beta", CASES)
def test_actual_prepared_provider_metadata(case, provider, beta):
    device = "cuda" if case.startswith("grating") else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual Bloch provider preparation requires CUDA")
    prepared, _, _ = execution(case, steps=1, device=device)
    solver = prepared.solver
    source = solver._compiled_sources[0]
    assert source["reference_provider_kind"] == provider == solver._normalization_reference_provider
    assert source["reference_waveform_kind"] == source["source_time"]["kind"]
    assert solver._normalization_label_offset == beta
    if provider in {"plane_wave_axis_aligned", "plane_wave_ref_x_ez", "plane_wave_aux"} and not case.endswith("cw"):
        assert source["reference_incident_kind"] == "physical_upstream_yee_entry_plane"
        expected = sum(min(lo * d, hi * d) for (lo, hi), d in
                       zip(source["reference_incident_bounds"], source["reference_incident_direction"]))
        assert source["reference_incident_projection"] == pytest.approx(expected, abs=1e-15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native paired provider gates require CUDA")
@pytest.mark.parametrize("case,provider,beta", CASES)
def test_native_provider_origin_matches_unchanged_sample_record(case, provider, beta):
    # Reconstruct the old-label finite numerator/denominator from actual samples;
    # no fitted phase, reference-cache mutation, or alternate source trajectory.
    # The fixed-k incident line launches its pulse upstream of the padded domain,
    # so the pulsed grating row needs a longer record to reach the probe.
    steps = 2048 if case == "grating_pulse" else 512
    raw_execution, electric, magnetic = execution(case, device="cuda", normalized=False, steps=steps)
    raw = raw_execution.run()
    normalized_execution, _, _ = execution(case, device="cuda", normalized=True, steps=steps)
    normalized = normalized_execution.run()
    solver = normalized.solver
    assert solver._normalization_reference_provider == provider
    assert solver._normalization_label_offset == beta
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(raw.solver, name), getattr(solver, name))
        if solver.complex_fields_enabled:
            assert torch.equal(getattr(raw.solver, name + "_imag"), getattr(solver, name + "_imag"))
    trace = raw.raw_monitor("trace")
    n = np.arange(len(trace["t"]))
    theta = 2 * math.pi * 2e9 * solver.dt
    samples = np.array([normalization_signal(solver, solver._normalization_source_time,
        normalization_sample_time(solver, int(step))) for step in n])
    scale = solver.dt * 2e9 if not case.endswith("cw") else (1. if case.startswith("grating") else 2.) / len(n)
    old_denominator = np.sum(samples * np.exp(1j * theta * n)) * scale
    assert abs(old_denominator) > 1e-8
    direct = source_normalization_spectrum(solver, (2e9,), steps_run=len(n), window_type="none")[0]
    np.testing.assert_allclose(direct.cpu(), old_denominator * np.exp(1j * beta * theta), rtol=3e-6, atol=1e-12)
    for field, offset in ((electric, 0.), (magnetic, -.5)):
        values = trace["components"][field].cpu().numpy().reshape(len(n), -1)[:, 0]
        old_numerator = np.sum(values * np.exp(1j * theta * (n + offset))) * scale
        assert abs(old_numerator) > 1e-12
        np.testing.assert_allclose(raw.raw_monitor("point")["components"][field].cpu(),
            old_numerator * np.exp(1j * theta), rtol=3e-6, atol=1e-12)
        np.testing.assert_allclose(normalized.raw_monitor("point")["components"][field].cpu(),
            old_numerator / old_denominator * np.exp(1j * (1 - beta) * theta), rtol=3e-6, atol=1e-10)
        if field == electric:
            index = tuple(raw.raw_monitor("point")["field_indices"][field])
            np.testing.assert_allclose(raw.field(field, frequency=2e9)[index].cpu(),
                old_numerator * np.exp(1j * theta), rtol=3e-6, atol=1e-12)
            np.testing.assert_allclose(normalized.field(field, frequency=2e9)[index].cpu(),
                old_numerator / old_denominator * np.exp(1j * (1 - beta) * theta), rtol=3e-6, atol=1e-10)


@pytest.mark.parametrize("normalized", [False, True])
def test_reference_selection_keeps_first_compiled_separate_from_primary_tfsf(normalized):
    prepared, _, _ = execution("axis_pulse", steps=1)
    scene = prepared.simulation.scene
    plane = scene.sources[0]
    scene = scene.clone(sources=[mw.PointDipole((0., 0., 0.), source_time=plane.source_time), plane])
    solver = mw.Simulation.fdtd(scene, frequency=2e9, backend="cpu_reference", cuda_graph=False,
        full_field_dft=False, run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=normalized)).prepare().solver
    from witwin.maxwell.fdtd.runtime.spectral import source_reference_metadata
    assert solver._normalization_reference_provider == "additive_J"
    assert solver._normalization_label_offset == .5
    assert solver._compiled_source["reference_provider_kind"] == "plane_wave_axis_aligned"
    solver._normalize_source = normalized
    entry = source_reference_metadata(solver, 2e9)
    assert entry["source_reference_label_offset"] == (.5 if normalized else 1.)
