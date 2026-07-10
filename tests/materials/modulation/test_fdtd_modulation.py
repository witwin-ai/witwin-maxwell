import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene


_C0 = 299_792_458.0


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Descriptor validation (CPU)
# ---------------------------------------------------------------------------


def test_modulation_spec_validation():
    spec = mw.ModulationSpec(frequency=1.0e9, amplitude=0.2, phase=0.5)
    assert spec.frequency == 1.0e9
    assert spec.amplitude == 0.2
    assert spec.phase == 0.5
    assert spec.angular_frequency == pytest.approx(2.0 * np.pi * 1.0e9)

    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=0.0, amplitude=0.2)
    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=1.0e9, amplitude=0.0)
    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=1.0e9, amplitude=0.6)
    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=1.0e9, amplitude=0.2, phase=float("inf"))
    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=1.0e9, amplitude=torch.zeros(4, 4))
    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=1.0e9, amplitude=torch.full((2, 2, 2), 0.7))
    with pytest.raises(ValueError):
        mw.ModulationSpec(frequency=1.0e9, amplitude=0.2, phase=torch.zeros(3))

    grid_spec = mw.ModulationSpec(
        frequency=1.0e9,
        amplitude=torch.full((2, 2, 2), 0.3),
        phase=torch.zeros(2, 2, 2),
    )
    assert torch.is_tensor(grid_spec.amplitude)
    assert torch.is_tensor(grid_spec.phase)


def test_material_modulation_combination_guards():
    modulation = mw.ModulationSpec(frequency=1.0e8, amplitude=0.1)

    material = mw.Material(eps_r=4.0, modulation=modulation)
    assert material.is_modulated
    assert not mw.Material(eps_r=4.0).is_modulated

    with pytest.raises(TypeError):
        mw.Material(eps_r=4.0, modulation="not-a-spec")
    # Modulation now composes with dispersion (electro-optic modulator in a dispersive
    # crystal) and with the instantaneous nonlinear channels; see
    # tests/materials/combinations/test_modulated_dispersive_nonlinear.py.
    dispersive = mw.Material(
        eps_r=4.0,
        modulation=modulation,
        debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1.0e-10),),
    )
    assert dispersive.is_modulated and dispersive.is_electric_dispersive
    nonlinear = mw.Material(eps_r=4.0, modulation=modulation, kerr_chi3=1.0e-10)
    assert nonlinear.is_modulated and nonlinear.is_nonlinear
    # Anisotropic tensor and static conductivity remain physically out of reach for the
    # scalar-modulated isotropic eps_inf update.
    with pytest.raises(NotImplementedError, match="anisotropic"):
        mw.Material(
            eps_r=4.0,
            modulation=modulation,
            epsilon_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0),
        )
    with pytest.raises(NotImplementedError, match="conductivity"):
        mw.Material(eps_r=4.0, modulation=modulation, sigma_e=1.0)
    with pytest.raises(ValueError):
        mw.Material(pec=True, modulation=modulation)


# ---------------------------------------------------------------------------
# Compile-layer rasterization (CPU)
# ---------------------------------------------------------------------------


def _compile_scene_with_modulation(modulation, *, second_material=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
            material=mw.Material(eps_r=4.0, modulation=modulation),
        )
    )
    if second_material is not None:
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.35, 0.0, 0.0), size=(0.1, 0.4, 0.4)),
                material=second_material,
            )
        )
    prepared = prepare_scene(scene)
    return prepared, prepared.compile_materials()


def test_scene_compile_rasterizes_modulation_quadrature_fields():
    frequency = 2.0e8
    amplitude = 0.2
    phase = 0.5
    scene, model = _compile_scene_with_modulation(
        mw.ModulationSpec(frequency=frequency, amplitude=amplitude, phase=phase)
    )

    assert model["modulation_frequency"] == frequency
    center = (scene.Nx // 2, scene.Ny // 2, scene.Nz // 2)
    assert float(model["modulation_cos"][center]) == pytest.approx(amplitude * np.cos(phase), rel=1e-5)
    assert float(model["modulation_sin"][center]) == pytest.approx(amplitude * np.sin(phase), rel=1e-5)
    # Far outside the structure the modulation fields vanish.
    assert float(model["modulation_cos"][1, 1, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(model["modulation_sin"][1, 1, 1]) == pytest.approx(0.0, abs=1e-6)


def test_scene_compile_without_modulation_has_no_frequency():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2))),
        grid=mw.GridSpec.uniform(0.05),
        device="cpu",
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4)),
                material=mw.Material(eps_r=4.0),
            )
        ],
    )
    model = prepare_scene(scene).compile_materials()
    assert model["modulation_frequency"] is None
    assert not torch.any(model["modulation_cos"] != 0)
    assert not torch.any(model["modulation_sin"] != 0)


def test_scene_compile_rejects_mixed_modulation_frequencies():
    with pytest.raises(NotImplementedError, match="single modulation frequency"):
        _compile_scene_with_modulation(
            mw.ModulationSpec(frequency=2.0e8, amplitude=0.2),
            second_material=mw.Material(
                eps_r=2.0, modulation=mw.ModulationSpec(frequency=3.0e8, amplitude=0.2)
            ),
        )


# ---------------------------------------------------------------------------
# FDTD runtime (GPU)
# ---------------------------------------------------------------------------


def _build_modulated_slab_scene(*, carrier_frequency, modulation, amplitude=80.0, boundary=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.16, 0.16), (-0.16, 0.16))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=boundary if boundary is not None else mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=carrier_frequency, amplitude=amplitude),
                name="pw",
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.24, 0.64, 0.64)),
                material=mw.Material(eps_r=4.0, modulation=modulation),  # modulation=None -> static slab
            )
        ],
    )
    scene.add_monitor(
        mw.FieldTimeMonitor("probe", components=("Ez",), position=(0.42, 0.0, 0.0), interval=1)
    )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_modulated_scene_prepare_enables_modulation_runtime():
    modulation = mw.ModulationSpec(frequency=2.5e8, amplitude=0.25)
    scene = _build_modulated_slab_scene(carrier_frequency=1.0e9, modulation=modulation)
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=16),
        full_field_dft=False,
    ).prepare().solver

    assert solver.modulation_enabled
    assert solver.modulation_angular_frequency == pytest.approx(2.0 * np.pi * 2.5e8)
    assert solver.mod_cos_Ez.shape == solver.eps_Ez.shape
    assert solver.mod_sin_Ez.shape == solver.eps_Ez.shape
    # Modulation no longer forces the dense CPML layout: both dense and compressed
    # (slab) psi layouts now have modulated kernel variants. For this small scene
    # the dense psi footprint is far below the auto memory limit, so "auto" still
    # selects dense here (see test_fdtd_modulated_slab_cpml.py for the slab path).
    assert solver._cpml_memory_mode == "dense"


def _sideband_powers(trace, dt, carrier_frequency, modulation_frequency):
    """Windowed FFT power at the carrier and the two first-order sidebands."""
    tail = trace[len(trace) // 4 :]
    window = np.hanning(tail.size)
    spectrum = np.fft.rfft(tail * window)
    freqs = np.fft.rfftfreq(tail.size, d=dt)
    power = np.abs(spectrum) ** 2

    def peak_power(target):
        index = int(np.argmin(np.abs(freqs - target)))
        low = max(index - 2, 0)
        high = min(index + 3, power.size)
        return float(power[low:high].max())

    return (
        peak_power(carrier_frequency),
        peak_power(carrier_frequency - modulation_frequency),
        peak_power(carrier_frequency + modulation_frequency),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_modulated_slab_generates_sidebands():
    carrier_frequency = 1.0e9
    modulation_frequency = 2.5e8
    time_steps = 4096

    def run(modulation):
        scene = _build_modulated_slab_scene(
            carrier_frequency=carrier_frequency,
            modulation=modulation,
        )
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[carrier_frequency],
            run_time=mw.TimeConfig(time_steps=time_steps),
            full_field_dft=False,
        ).run()
        payload = result.monitor("probe")
        trace = _to_numpy(payload["field"]).astype(np.float64)
        dt = float(result.solver.dt)
        del result
        torch.cuda.empty_cache()
        return trace, dt

    modulated_trace, dt = run(
        mw.ModulationSpec(frequency=modulation_frequency, amplitude=0.25)
    )
    reference_trace, reference_dt = run(None)
    assert dt == reference_dt

    carrier, lower, upper = _sideband_powers(
        modulated_trace, dt, carrier_frequency, modulation_frequency
    )
    ref_carrier, ref_lower, ref_upper = _sideband_powers(
        reference_trace, dt, carrier_frequency, modulation_frequency
    )

    assert carrier > 0.0
    assert ref_carrier > 0.0
    # The modulated slab converts carrier power into omega +/- Omega sidebands.
    assert lower > 1.0e-3 * carrier
    assert upper > 1.0e-3 * carrier
    # The unmodulated reference has no such peaks: its residual spectral content
    # at the sideband bins is orders of magnitude below the modulated run.
    assert lower > 20.0 * ref_lower
    assert upper > 20.0 * ref_upper
    assert ref_lower < 1.0e-3 * ref_carrier
    assert ref_upper < 1.0e-3 * ref_carrier


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_modulated_run_with_mur_boundaries_stays_finite():
    # Exercises the standard (non-CPML) modulated kernel path.
    modulation = mw.ModulationSpec(frequency=2.5e8, amplitude=0.25)
    scene = _build_modulated_slab_scene(
        carrier_frequency=1.0e9,
        modulation=modulation,
        boundary=mw.BoundarySpec.mur(),
    )
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=256),
        full_field_dft=False,
    ).run()
    trace = _to_numpy(result.monitor("probe")["field"])
    assert np.all(np.isfinite(trace))
    assert float(np.abs(trace).max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_modulation_rejects_bloch_runs():
    modulation = mw.ModulationSpec(frequency=2.5e8, amplitude=0.25)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.16, 0.16), (-0.16, 0.16))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            x="bloch",
            y="bloch",
            z="pml",
            bloch_wavevector=(1.0, 0.5, 0.0),
        ),
        device="cuda",
        sources=[
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.08,
                source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            )
        ],
        structures=[
            mw.Structure(
                geometry=Box(position=(0.3, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
                material=mw.Material(eps_r=4.0, modulation=modulation),
            )
        ],
    )
    with pytest.raises(NotImplementedError, match="time-modulated"):
        mw.Simulation.fdtd(
            scene,
            frequencies=[1.0e9],
            run_time=mw.TimeConfig(time_steps=8),
            full_field_dft=False,
        ).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_modulation_composes_with_dispersive_scene():
    # A modulated slab now shares a Scene with a (disjoint) dispersive material: both
    # feature flags stay enabled and the runtime no longer rejects the combination.
    # The full electro-optic-modulator physics lives in
    # tests/materials/combinations/test_modulated_dispersive_nonlinear.py.
    modulation = mw.ModulationSpec(frequency=2.5e8, amplitude=0.25)
    scene = _build_modulated_slab_scene(carrier_frequency=1.0e9, modulation=modulation)
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.4, 0.0, 0.0), size=(0.08, 0.32, 0.32)),
            material=mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=2.0e-10),
        )
    )
    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=8),
        full_field_dft=False,
    ).prepare().solver
    assert solver.modulation_enabled
    assert solver.electric_dispersive_enabled
