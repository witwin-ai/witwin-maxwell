"""Ordinary GPU grouping agrees with independent host-scheduled spectra."""
import numpy as np
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def simulation(*, graph, window, pulse, normalize):
    frequency = 2.45e9
    source_time = mw.GaussianPulse(frequency=frequency, fwidth=frequency) if pulse else mw.CW(frequency=frequency)
    frequencies = tuple(np.linspace(2e9, 3e9, 11))
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),) * 3),
        grid=mw.GridSpec.uniform(.01), boundary=mw.BoundarySpec.pml(num_layers=4), device="cuda",
        sources=[mw.PointDipole(position=(-.02, 0, 0), polarization="Ez", source_time=source_time)],
        monitors=[mw.PointMonitor("point", (-.01, 0, 0), fields=("Ez", "Hy")),
                  mw.PlaneMonitor("plane", axis="x", position=.01, fields=("Ez", "Hy"))])
    return mw.Simulation.fdtd(scene, frequencies=frequencies, cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=512, transient_cycles=0, steady_cycles=1), full_field_dft=False,
        spectral_sampler=mw.SpectralSampler(window=window, normalize_source=normalize))


def check(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            check(actual[key], expected[key])
    elif isinstance(expected, (np.ndarray, torch.Tensor)):
        a, b = torch.as_tensor(actual).cpu(), torch.as_tensor(expected).cpu()
        scale = b.abs().max().clamp_min(1e-30)
        torch.testing.assert_close(a / scale, b / scale, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("window", ["none", "hanning"])
@pytest.mark.parametrize("pulse", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_grouped_multifrequency_point_and_plane_match_host_schedule(window, pulse, normalize):
    expected = simulation(graph=False, window=window, pulse=pulse, normalize=normalize).run()
    prepared = simulation(graph=True, window=window, pulse=pulse, normalize=normalize).prepare()
    actual = prepared.run()
    assert prepared.solver._observer_cuda_graph_active
    for name in ("point", "plane"):
        check(actual.monitor(name), expected.monitor(name))
    for a, b in zip(prepared.solver._observer_spectral_entries, expected.solver._observer_spectral_entries):
        assert a["sample_count"] == b["sample_count"]
        assert a["window_normalization"] == pytest.approx(b["window_normalization"], rel=2e-14)


@pytest.mark.parametrize('frequencies', [(1e9,), (.8e9, 1.2e9)])
def test_small_native_point_spectra_avoid_grouping_overhead(frequencies):
    def model(graph):
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),)*3),
            grid=mw.GridSpec.uniform(.01), boundary=mw.BoundarySpec.pml(num_layers=4), device='cuda',
            sources=[mw.PointDipole(position=(0., 0., 0.), polarization='Ez',
                source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9))],
            monitors=[mw.PointMonitor('probe', (.01, 0., 0.), fields=('Ez',))])
        return mw.Simulation.fdtd(scene, frequencies=frequencies, cuda_graph=graph,
            full_field_dft=True, run_time=mw.TimeConfig(time_steps=256),
            spectral_sampler=mw.SpectralSampler(window='none'))

    expected = model(False).run()
    prepared = model(True).prepare()
    actual = prepared.run()
    assert not prepared.solver._observer_cuda_graph_active
    check(actual.monitor('probe'), expected.monitor('probe'))
    for name, expected_field in expected.fields.items():
        torch.testing.assert_close(actual.fields[name], expected_field, rtol=3e-5, atol=1e-10)
