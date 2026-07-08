import numpy as np
import pytest
import torch

import witwin.maxwell as mw


def _short_runtime():
    return mw.TimeConfig(time_steps=4)


def _run_fdtd(scene, *, frequencies=(1e9,)):
    return mw.Simulation.fdtd(
        scene,
        frequencies=list(frequencies),
        run_time=_short_runtime(),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=True,
    ).run()


def _base_scene(*, material, box_position=(0.0, 0.0, 0.0), box_size=(0.2, 0.2, 0.2)):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            name="box",
            geometry=mw.Box(position=box_position, size=box_size),
            material=material,
        )
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.5e9, amplitude=1.0),
            name="src",
        )
    )
    return scene


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_permittivity_monitor_reports_box_and_vacuum():
    scene = _base_scene(material=mw.Material(eps_r=4.0))
    scene.add_monitor(
        mw.PermittivityMonitor("perm", position=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.6))
    )
    result = _run_fdtd(scene)

    payload = result.monitor("perm")
    eps = _to_numpy(payload["eps"])
    assert eps.ndim == 3
    assert np.isreal(eps).all()
    # A fully-enclosed interior node reports the box permittivity, an exterior node vacuum.
    assert eps.max() == pytest.approx(4.0, abs=0.1)
    assert eps.min() == pytest.approx(1.0, abs=0.1)

    x = _to_numpy(payload["x"])
    y = _to_numpy(payload["y"])
    z = _to_numpy(payload["z"])
    assert eps.shape == (x.shape[0], y.shape[0], z.shape[0])
    assert _to_numpy(payload["eps_x"]).shape == eps.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_medium_monitor_reports_mu_and_conductivity():
    sigma = 0.5
    scene = _base_scene(material=mw.Material(eps_r=2.0, sigma_e=sigma))
    scene.add_monitor(
        mw.MediumMonitor("med", position=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    )
    result = _run_fdtd(scene)

    payload = result.monitor("med")
    mu = _to_numpy(payload["mu"])
    sigma_e = _to_numpy(payload["sigma_e"])
    np.testing.assert_allclose(np.real(mu), 1.0, atol=1e-5)
    # A point monitor at the box center reports the structure conductivity.
    np.testing.assert_allclose(np.real(sigma_e), sigma, rtol=1e-4, atol=1e-6)
    # eps carries the conductive (complex) contribution at the evaluated frequency.
    assert np.iscomplexobj(_to_numpy(payload["eps"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_permittivity_monitor_multi_frequency_dispersive():
    material = mw.Material(
        eps_r=2.0,
        lorentz_poles=(mw.LorentzPole(delta_eps=3.0, resonance_frequency=2e9, gamma=0.1e9),),
    )
    scene = _base_scene(material=material)
    f1, f2 = 0.8e9, 1.2e9
    scene.add_monitor(
        mw.PermittivityMonitor("perm", position=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), frequencies=[f1, f2])
    )
    result = _run_fdtd(scene)

    payload = result.monitor("perm")
    eps = _to_numpy(payload["eps"])
    assert np.iscomplexobj(eps)
    assert eps.shape[0] == 2  # leading frequency axis
    assert not np.allclose(eps[0], eps[1])

    # The reported eps must match the material's own susceptibility evaluation.
    expected_f1 = complex(material.relative_permittivity(f1))
    expected_f2 = complex(material.relative_permittivity(f2))
    center = tuple(dim // 2 for dim in eps.shape[1:])
    np.testing.assert_allclose(eps[0][center], expected_f1, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(eps[1][center], expected_f2, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_material_monitor_is_metadata_only():
    material = mw.Material(eps_r=4.0)
    extra_frequency = 9e9

    baseline_scene = _base_scene(material=material)
    baseline_scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.16, frequencies=[1e9]))
    baseline_sim = mw.Simulation.fdtd(
        baseline_scene,
        frequencies=[1e9],
        run_time=_short_runtime(),
        full_field_dft=False,
    )
    baseline_frequencies = baseline_sim._collect_fdtd_requested_frequencies()

    scene = _base_scene(material=material)
    scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.16, frequencies=[1e9]))
    scene.add_monitor(
        mw.PermittivityMonitor("perm", position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4), frequencies=[extra_frequency])
    )
    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=_short_runtime(),
        full_field_dft=False,
    )
    # The permittivity monitor must not force an extra DFT frequency.
    assert sim._collect_fdtd_requested_frequencies() == baseline_frequencies

    result = sim.run()
    assert extra_frequency not in tuple(float(f) for f in result.stats()["frequencies"])

    flux_payload = result.monitor("flux")
    assert "flux" in flux_payload
    perm_payload = result.monitor("perm")
    # The material monitor still resolves at its own requested frequency.
    assert tuple(float(f) for f in perm_payload["frequencies"]) == (extra_frequency,)
    assert _to_numpy(perm_payload["eps"]).max() == pytest.approx(4.0, abs=0.1)


def test_material_monitor_frequency_skip_is_cpu_verifiable():
    # The frequency-skip guard in _collect_fdtd_requested_frequencies is pure Python,
    # so verify it without CUDA / without running the solve: a material monitor must
    # never widen the DFT frequency set.
    def _cpu_sim(*monitors):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))),
            grid=mw.GridSpec.uniform(0.1),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cpu",
        )
        scene.add_monitor(mw.FluxMonitor("flux", axis="z", position=0.1, frequencies=[1e9]))
        for monitor in monitors:
            scene.add_monitor(monitor)
        return mw.Simulation.fdtd(scene, frequencies=[1e9], run_time=_short_runtime())

    baseline = _cpu_sim()._collect_fdtd_requested_frequencies()
    with_perm = _cpu_sim(
        mw.PermittivityMonitor("perm", position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4), frequencies=[7e9]),
        mw.MediumMonitor("med", position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2), frequencies=[8e9, 9e9]),
    )._collect_fdtd_requested_frequencies()
    assert with_perm == baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_medium_monitor_multi_frequency_volume():
    # Exercise the multi-frequency + volume path: mu/sigma_e per-axis stacking and the
    # frequency-independent sigma_e broadcast across the leading frequency axis.
    material = mw.Material(
        eps_r=2.0,
        sigma_e=0.3,
        lorentz_poles=(mw.LorentzPole(delta_eps=1.5, resonance_frequency=2e9, gamma=0.2e9),),
    )
    scene = _base_scene(material=material)
    f1, f2 = 0.7e9, 1.3e9
    scene.add_monitor(
        mw.MediumMonitor("med", position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3), frequencies=[f1, f2])
    )
    result = _run_fdtd(scene)
    payload = result.monitor("med")

    eps = _to_numpy(payload["eps"])
    mu = _to_numpy(payload["mu"])
    sigma_e = _to_numpy(payload["sigma_e"])

    # Leading frequency axis of length 2 on every field.
    assert eps.shape[0] == 2 and mu.shape[0] == 2 and sigma_e.shape[0] == 2
    x = _to_numpy(payload["x"])
    y = _to_numpy(payload["y"])
    z = _to_numpy(payload["z"])
    assert eps.shape[1:] == (x.shape[0], y.shape[0], z.shape[0])
    assert _to_numpy(payload["sigma_e_x"]).shape == sigma_e.shape
    assert _to_numpy(payload["mu_z"]).shape == mu.shape

    # sigma_e is frequency-independent: identical across the two frequency slices.
    np.testing.assert_allclose(sigma_e[0], sigma_e[1])
    np.testing.assert_allclose(np.real(mu), 1.0, atol=1e-5)
    # Dispersive eps genuinely differs between the two frequencies.
    assert np.iscomplexobj(eps)
    assert not np.allclose(eps[0], eps[1])
