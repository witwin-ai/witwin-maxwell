"""Compiled provider labels, once-only exports, trace clocks, and resume refusal."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import resume
from witwin.maxwell.fdtd.distributed import resume as distributed_resume
from witwin.maxwell.fdtd.observers import _compute_source_spectrum
from witwin.maxwell.fdtd.postprocess import _entry_source_spectrum
from witwin.maxwell.fdtd.runtime.spectral import source_normalization_spectrum, source_reference_metadata
from witwin.maxwell.fdtd.runtime.timeplan import source_reference_label_offset
from witwin.maxwell.sources import compile_source_time


@pytest.mark.parametrize("provider,waveform,beta", [
    (provider, waveform, beta)
    for provider, pulse_beta in [
        ("additive_J", .5), ("terminal_drive", .5), ("magnetic_M", 0.), ("mixed_JM", 0.),
        ("soft_equivalent_sheet", 0.), ("mode_surface", 0.), ("custom_field_surface", 0.),
        ("plane_wave_axis_aligned", 1.), ("plane_wave_ref_x_ez", 1.), ("plane_wave_aux", 1.),
        ("plane_wave_discrete_cw", 1.), ("analytic_profile", 0.), ("plane_wave_grating_slab_cw", 0.),
    ]
    # The three corrected CW providers no longer carry a one-step phase delay.
    for waveform, beta in [("gaussian_pulse", pulse_beta),
        ("cw", 0. if provider in {"plane_wave_discrete_cw", "analytic_profile", "plane_wave_grating_slab_cw"} else pulse_beta)]
])
def test_compiled_provider_and_waveform_select_beta(provider, waveform, beta):
    source = dict(reference_provider_kind=provider, reference_waveform_kind=waveform,
                  reference_incident_kind="physical_upstream_yee_entry_plane", reference_incident_projection=-.2)
    assert source_reference_label_offset(source) == beta


def test_missing_provider_or_physical_entry_is_not_an_alpha_fallback():
    with pytest.raises(KeyError, match="reference_provider_kind"):
        source_reference_label_offset(dict(kind="plane_wave"))
    with pytest.raises(ValueError, match="physical Yee entry plane"):
        source_reference_label_offset(dict(reference_provider_kind="plane_wave_aux", reference_waveform_kind="gaussian_pulse"))


@pytest.mark.parametrize("beta,alpha", [(0., 0.), (.5, .5), (1., 0.), (1., 1.)])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("waveform", ["cw", "pulse"])
def test_direct_and_repeated_entry_exports_match_independent_finite_sum(beta, alpha, analytic, waveform):
    from witwin.maxwell.sources import evaluate_source_time_normalization
    from witwin.maxwell.fdtd.runtime.representation import spectral_scale
    f, dt, count = 2e9, 9.5238e-12, 64
    source = compile_source_time(mw.CW(f) if waveform == "cw" else mw.GaussianPulse(f, .5e9), default_frequency=f)
    solver = SimpleNamespace(dt=dt, device="cpu", _normalize_source=True,
        complex_fields_enabled=analytic, _normalization_source_time=source,
        _normalization_time_offset=alpha, _normalization_label_offset=beta,
        _normalization_reference=dict(reference_provider_kind={0.: "magnetic_M", .5: "additive_J", 1.: "plane_wave_aux"}[beta]),
        _spectral_integration_dt=None if waveform == "cw" else dt, _spectral_reference_frequency=None,
        _time_plan=dict(total_steps=count, response="cw_steady_state" if waveform == "cw" else "full_response",
                        spectral_start_step=0, source_representation="analytic_cw" if analytic and waveform == "cw" else "real_projected"))
    n = np.arange(count)
    samples = (np.exp(-2j * np.pi * f * (n + alpha) * dt) if analytic and waveform == "cw"
               else evaluate_source_time_normalization(source, torch.tensor((n + alpha) * dt)).numpy())
    scale = spectral_scale(count, frequency=f, phasor_factor=1. if analytic and waveform == "cw" else 2.,
                           integration_dt=solver._spectral_integration_dt)
    expected = np.sum(samples * np.exp(2j * np.pi * f * (n + beta) * dt)) * scale
    direct = source_normalization_spectrum(solver, (f,), steps_run=count, window_type="none")
    np.testing.assert_allclose(direct.numpy(), [expected], rtol=2e-13, atol=1e-20)
    internal = np.sum(samples * np.exp(2j * np.pi * f * (n + 1) * dt))
    for tensor in (False, True):
        entry = dict(source_dft_real=torch.tensor(internal.real) if tensor else internal.real,
                     source_dft_imag=torch.tensor(internal.imag) if tensor else internal.imag,
                     **source_reference_metadata(solver, f))
        for _ in range(2):
            np.testing.assert_allclose(_entry_source_spectrum(entry, scale), expected, rtol=2e-13, atol=1e-20)
            actual = _compute_source_spectrum([entry], [0], lambda entry: scale)
            np.testing.assert_allclose(actual, [expected], rtol=3e-6, atol=1e-20)
        assert entry["source_dft_real"] == internal.real
        assert entry["source_dft_imag"] == internal.imag


def _execution():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.05, .05),) * 3), grid=mw.GridSpec.uniform(.01),
        boundary=mw.BoundarySpec.pec(), device="cpu",
        sources=[mw.PointDipole((0., 0., 0.), source_time=mw.CW(1e9))],
        monitors=[mw.PointMonitor("point", (0., 0., 0.), fields=("Ex", "Hy")),
                  mw.FieldTimeMonitor("trace", components=("Ex", "Hy"), position=(0., 0., 0.))])
    return mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=8),
        backend="cpu_reference", cuda_graph=False, full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=True)).prepare()


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("key", ["observer_spectral_origin", "source_reference_label", "normalization_reference", "normalization_label_offset"])
def test_old_observer_checkpoint_is_refused_before_restoring(key, distributed):
    execution = _execution()
    checkpoint = execution.run_until(3)
    solver = execution.solver
    if distributed:
        solver = SimpleNamespace(shard_layouts=(), logical_scene=solver.scene,
            shards=(SimpleNamespace(rank=0, solver=solver),), _nccl=False, dt=solver.dt,
            _time_plan=solver._time_plan, _normalization_reference=solver._normalization_reference,
            _normalization_label_offset=solver._normalization_label_offset)
        checkpoint = replace(checkpoint, signature=distributed_resume._signature(solver))
    signature = dict(checkpoint.signature)
    signature.pop(key)
    signature["logical_fingerprint" if distributed else "configuration_fingerprint"] = "old fingerprint"
    before = execution.solver.Ex.clone()
    with pytest.raises(NotImplementedError, match="physical Yee observer origin"):
        resume.restore_resume_checkpoint(solver, replace(checkpoint, signature=signature), total_steps=8)
    assert torch.equal(execution.solver.Ex, before)


def test_format_five_is_refused_and_new_spectral_checkpoint_round_trips(tmp_path):
    expected = _execution().run()
    checkpoint = _execution().run_until(3)
    path = tmp_path / "new.pt"
    checkpoint.save(path)
    target = _execution().run(resume_from=mw.FDTDResumeCheckpoint.load(path))
    assert torch.equal(target.field("Ex"), expected.field("Ex"))
    trace = target.raw_monitor("trace")
    assert trace["field_time_offsets_steps"] == {"E": 0., "H": -.5}
    torch.testing.assert_close(trace["t"], torch.arange(1, 9, dtype=torch.float64) * target.solver.dt)
    torch.testing.assert_close(trace["components"]["Ex"], expected.raw_monitor("trace")["components"]["Ex"], rtol=0, atol=0)
    payload = checkpoint._snapshot()
    payload["schema_version"] = 5
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema_version 5"):
        mw.FDTDResumeCheckpoint.load(path)


def test_reference_labels_do_not_change_the_configuration_fingerprint():
    solver = _execution().solver
    stamped = resume._configuration_fingerprint(solver)
    for source in solver._compiled_sources:
        for key in tuple(source):
            if key.startswith("reference_"):
                del source[key]
    assert resume._configuration_fingerprint(solver) == stamped


@pytest.mark.parametrize("count", [1, 2, 4])
def test_public_port_execution_uses_physical_labels_and_finite_reference(count):
    import cmath
    import os
    from tests.support.observer_origin_probe import port_table_probe, port_unit_probe

    device = os.environ.get("OBSERVER_ORIGIN_DEVICE", "cpu")
    ns = port_table_probe()
    control = port_unit_probe()
    results = []
    for normalized in (False, True):
        result = mw.Simulation.fdtd(ns["scene"], frequencies=(ns["F"],),
            backend="cpu_reference" if device == "cpu" else "native", cuda_graph=False,
            field_dtype=torch.float32, full_field_dft=True,
            excitations=mw.PortExcitation("feed", source_time=mw.CW(ns["F"])),
            run_time=mw.TimeConfig(time_steps=count, dt=ns["DT"], transient_cycles=0),
            spectral_sampler=mw.SpectralSampler(window="none", normalize_source=normalized)).run()
        assert result.stats()["steps_run"] == count
        results.append(result)
    raw, normalized = results
    theta = 2 * np.pi * ns["F"] * raw.solver.dt
    n = np.arange(count)
    denominator = 2 / count * np.sum(np.cos(theta * (n + .5)) * np.exp(1j * theta * (n + .5)))
    for name in ("voltage", "current"):
        torch.testing.assert_close(getattr(normalized.port("feed"), name),
            getattr(raw.port("feed"), name) / denominator, rtol=3e-6, atol=1e-12)
    for name in ("Ex", "Hy"):
        np.testing.assert_allclose(normalized.raw_monitor("probe")["components"][name].cpu(),
            raw.raw_monitor("probe")["components"][name].cpu() / denominator, rtol=3e-6, atol=1e-12)
    if count == 1:
        for name, expected in (("voltage", control["independent_N1_V"]), ("current", control["independent_N1_I"])):
            value = complex(getattr(raw.port("feed"), name).item())
            assert abs(value / (2 * expected) - cmath.exp(.5j * theta)) < 3e-6
        electric = complex(raw.raw_monitor("probe")["components"]["Ex"].item())
        assert abs(electric / (2 * control["independent_N1_E"]) - cmath.exp(1j * theta)) < 3e-6
    if count == 2:
        magnetic = complex(raw.raw_monitor("probe")["components"]["Hy"].item())
        assert abs(magnetic / control["independent_N2_H"] - cmath.exp(1.5j * theta)) < 3e-6
