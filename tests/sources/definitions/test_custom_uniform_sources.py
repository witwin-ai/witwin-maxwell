import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.sources import compile_fdtd_sources


def _make_scene(*, device="cpu"):
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device=device,
    )


# ---------------------------------------------------------------------------
# Tier A: construction, compilation, and dataset validation (CPU, no solver).
# ---------------------------------------------------------------------------


def test_uniform_current_source_compiles_box_and_polarization():
    scene = _make_scene()
    scene.add_source(
        mw.UniformCurrentSource(
            size=(0.2, 0.2, 0.2),
            polarization="Ez",
            source_time=mw.CW(frequency=1e9, amplitude=3.0),
            center=(0.1, 0.0, -0.1),
            name="uc",
        )
    )

    compiled = compile_fdtd_sources(scene, default_frequency=1e9)

    assert len(compiled) == 1
    entry = compiled[0]
    assert entry["kind"] == "uniform_current"
    assert entry["name"] == "uc"
    assert entry["center"] == pytest.approx((0.1, 0.0, -0.1))
    assert entry["size"] == pytest.approx((0.2, 0.2, 0.2))
    assert entry["polarization"] == pytest.approx((0.0, 0.0, 1.0))
    assert entry["source_time"]["kind"] == "cw"
    assert entry["source_time"]["amplitude"] == pytest.approx(3.0)


def test_uniform_current_source_normalizes_polarization_vector():
    source = mw.UniformCurrentSource(size=(0.1, 0.1, 0.1), polarization=(0.0, 3.0, 4.0))
    assert source.polarization == pytest.approx((0.0, 0.6, 0.8))


def test_uniform_current_source_rejects_negative_size():
    with pytest.raises(ValueError):
        mw.UniformCurrentSource(size=(0.1, -0.2, 0.1), polarization="Ez")


def test_current_dataset_rejects_shape_mismatch():
    coords = (np.linspace(-0.1, 0.1, 3), np.linspace(-0.1, 0.1, 3), np.linspace(-0.1, 0.1, 3))
    with pytest.raises(ValueError):
        mw.CurrentDataset(coords, {"Jz": np.ones((3, 3, 2))})


def test_current_dataset_rejects_unknown_component():
    coords = (np.linspace(-0.1, 0.1, 2), np.linspace(-0.1, 0.1, 2), np.linspace(-0.1, 0.1, 2))
    with pytest.raises(ValueError):
        mw.CurrentDataset(coords, {"Ex": np.ones((2, 2, 2))})


def test_field_dataset_rejects_non_increasing_coordinates():
    coords = (np.array([0.1, 0.0, 0.2]), np.linspace(-0.1, 0.1, 3), np.array([0.0]))
    with pytest.raises(ValueError):
        mw.FieldDataset(coords, {"Ex": np.ones((3, 3, 1))})


def test_custom_current_source_compiles_dataset():
    n = 3
    axis = np.linspace(-0.1, 0.1, n)
    dataset = mw.CurrentDataset((axis, axis, axis), {"Jz": np.ones((n, n, n)), "My": np.zeros((n, n, n))})
    scene = _make_scene()
    scene.add_source(mw.CustomCurrentSource(dataset, source_time=mw.CW(frequency=1e9), name="cc"))

    compiled = compile_fdtd_sources(scene, default_frequency=1e9)
    assert compiled[0]["kind"] == "custom_current"
    assert compiled[0]["name"] == "cc"
    assert set(compiled[0]["dataset"].components) == {"Jz", "My"}


def test_custom_current_source_rejects_wrong_dataset_type():
    with pytest.raises(TypeError):
        mw.CustomCurrentSource("not a dataset")


def test_custom_field_source_requires_planar_dataset():
    n = 3
    axis = np.linspace(-0.1, 0.1, n)
    volumetric = mw.FieldDataset((axis, axis, axis), {"Ex": np.ones((n, n, n))})
    with pytest.raises(ValueError):
        mw.CustomFieldSource(volumetric)


def test_custom_field_source_compiles_plane_normal():
    n = 4
    tangential = np.linspace(-0.2, 0.2, n)
    dataset = mw.FieldDataset(
        (tangential, tangential, np.array([0.05])),
        {"Ex": np.ones((n, n, 1)), "Hy": np.ones((n, n, 1))},
    )
    source = mw.CustomFieldSource(dataset, source_time=mw.CW(frequency=1e9), name="cf")
    assert source.normal_axis == "z"

    scene = _make_scene()
    scene.add_source(source)
    compiled = compile_fdtd_sources(scene, default_frequency=1e9)
    assert compiled[0]["kind"] == "custom_field"
    assert compiled[0]["normal_axis"] == "z"


# ---------------------------------------------------------------------------
# Tier B: real FDTD physics (CUDA only).
# ---------------------------------------------------------------------------

_ETA0 = 376.730313668


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_uniform_current_radiated_power_scales_as_amplitude_squared():
    def _flux(amplitude):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))),
            grid=mw.GridSpec.uniform(0.06),
            boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
            device="cuda",
        )
        scene.add_source(
            mw.UniformCurrentSource(
                size=(0.1, 0.1, 0.1),
                polarization="Ez",
                source_time=mw.CW(frequency=1e9, amplitude=amplitude),
                center=(0.0, 0.0, 0.0),
            )
        )
        scene.add_monitor(mw.FluxMonitor("fz", axis="z", position=0.18, frequencies=[1e9]))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[1e9],
            run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        ).run()
        return float(torch.as_tensor(result.monitor("fz")["flux"]).reshape(-1)[0].cpu())

    flux_low = _flux(1.0)
    flux_high = _flux(3.0)

    assert flux_low > 0.0
    # A linear solver makes the radiated power exactly quadratic in the drive amplitude.
    assert flux_high / flux_low == pytest.approx(9.0, rel=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_custom_current_radiated_power_scales_as_amplitude_squared():
    n = 5
    axis = np.linspace(-0.06, 0.06, n)
    current = np.ones((n, n, n))

    def _flux(amplitude):
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4))),
            grid=mw.GridSpec.uniform(0.06),
            boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
            device="cuda",
        )
        dataset = mw.CurrentDataset((axis, axis, axis), {"Jz": current})
        scene.add_source(mw.CustomCurrentSource(dataset, source_time=mw.CW(frequency=1e9, amplitude=amplitude)))
        scene.add_monitor(mw.FluxMonitor("fz", axis="z", position=0.18, frequencies=[1e9]))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[1e9],
            run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=15),
            spectral_sampler=mw.SpectralSampler(window="none"),
            full_field_dft=False,
        ).run()
        return float(torch.as_tensor(result.monitor("fz")["flux"]).reshape(-1)[0].cpu())

    flux_low = _flux(1.0)
    flux_high = _flux(2.0)

    assert flux_low > 0.0
    assert flux_high / flux_low == pytest.approx(4.0, rel=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_custom_field_source_reproduces_plane_wave():
    # Replay the tangential E/H of an x-polarized plane wave travelling toward +z
    # on a z-plane. Equivalent surface currents J = n x H, M = -n x E should launch
    # a predominantly forward-going, transversely uniform, x-polarized wave.
    n = 13
    tangential = np.linspace(-0.45, 0.45, n)
    z_source = -0.12
    dataset = mw.FieldDataset(
        (tangential, tangential, np.array([z_source])),
        {"Ex": np.ones((n, n, 1)), "Hy": (1.0 / _ETA0) * np.ones((n, n, 1))},
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))),
        grid=mw.GridSpec.uniform(0.06),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
    )
    scene.add_source(mw.CustomFieldSource(dataset, source_time=mw.CW(frequency=1e9, amplitude=1.0)))

    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=300),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
    ).prepare()
    result = prepared.run()

    ex = result.tensor("Ex").abs().cpu().numpy()
    ey = result.tensor("Ey").abs().cpu().numpy()
    ez = result.tensor("Ez").abs().cpu().numpy()
    z = prepared.solver.scene.z.cpu().numpy()
    kz = int(np.argmin(np.abs(z - z_source)))
    cx = ex.shape[0] // 2
    cy = ex.shape[1] // 2

    forward = ex[cx, cy, kz + 2 : ex.shape[2] - 7].mean()
    backward = ex[cx, cy, 6 : kz - 1].mean()
    # Predominantly forward propagation (equivalence-principle one-sidedness).
    assert forward > 1.8 * backward

    # Transverse uniformity of a downstream plane (plane-wave signature).
    plane = ex[6:-6, 6:-6, kz + 4]
    assert plane.mean() > 0.2
    assert plane.std() / plane.mean() < 0.15

    # Correct polarization: in the interior (away from finite-aperture edge
    # diffraction) Ex dominates the cross-polarized components.
    ex_interior = ex[6:-6, 6:-6, kz + 4].mean()
    ey_interior = ey[6:-6, 6:-6, kz + 4].mean()
    ez_interior = ez[6:-6, 6:-6, kz + 4].mean()
    assert ex_interior > 20.0 * ey_interior
    assert ex_interior > 5.0 * ez_interior
