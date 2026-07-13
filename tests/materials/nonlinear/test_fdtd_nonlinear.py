import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plane_complex_mean(data):
    if isinstance(data, dict):
        data = data["data"]
    field = _to_cpu_numpy(data)
    crop = field[3:-3, 3:-3]
    if crop.size == 0:
        crop = field
    return complex(crop.mean())


def test_nonlinear_susceptibility_requires_nonzero_chi2():
    with pytest.raises(ValueError, match="nonzero"):
        mw.NonlinearSusceptibility(chi2=0.0)
    with pytest.raises(ValueError, match="finite"):
        mw.NonlinearSusceptibility(chi2=float("nan"))


def test_material_composes_chi2_nonlinearity():
    material = mw.Material(eps_r=2.25, nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-6))
    assert material.is_nonlinear
    assert material.nonlinear_chi2 == pytest.approx(1.0e-6)
    assert material.nonlinear_chi3 == 0.0

    stacked = mw.Material(
        eps_r=2.25,
        nonlinearity=(
            mw.NonlinearSusceptibility(chi2=1.0e-6),
            mw.NonlinearSusceptibility(chi2=2.0e-6),
        ),
    )
    assert stacked.nonlinear_chi2 == pytest.approx(3.0e-6)

    with pytest.raises(TypeError):
        mw.Material(eps_r=2.25, nonlinearity=object())


def test_nonlinear_susceptibility_chi3_folds_into_kerr_channel():
    chi3_only = mw.Material(eps_r=2.25, nonlinearity=mw.NonlinearSusceptibility(chi3=2.0e-10))
    assert chi3_only.is_nonlinear
    assert chi3_only.nonlinear_chi2 == 0.0
    assert chi3_only.nonlinear_chi3 == pytest.approx(2.0e-10)

    combined = mw.Material(
        eps_r=2.25,
        kerr_chi3=1.0e-10,
        nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-6, chi3=2.0e-10),
    )
    assert combined.nonlinear_chi3 == pytest.approx(3.0e-10)
    assert combined.nonlinear_chi2 == pytest.approx(1.0e-6)


def test_two_photon_absorption_descriptor_validation():
    with pytest.raises(ValueError, match="beta"):
        mw.TwoPhotonAbsorption(beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        mw.TwoPhotonAbsorption(beta=-1.0e-4)
    with pytest.raises(ValueError, match="n0"):
        mw.TwoPhotonAbsorption(beta=1.0e-4, n0=0.0)


def test_material_composes_two_photon_absorption():
    eps0 = 8.8541878128e-12
    c0 = 299_792_458.0
    beta = 1.0e-4

    material = mw.Material(eps_r=2.25, nonlinearity=mw.TwoPhotonAbsorption(beta=beta))
    assert material.is_nonlinear
    assert material.nonlinear_chi2 == 0.0
    assert material.nonlinear_chi3 == 0.0
    # Default n0 = sqrt(eps_r).
    expected = (4.0 / 3.0) * beta * (1.5 * eps0 * c0) ** 2
    assert material.tpa_sigma_scale == pytest.approx(expected, rel=1.0e-9)

    explicit = mw.Material(eps_r=2.25, nonlinearity=mw.TwoPhotonAbsorption(beta=beta, n0=2.0))
    expected_explicit = (4.0 / 3.0) * beta * (2.0 * eps0 * c0) ** 2
    assert explicit.tpa_sigma_scale == pytest.approx(expected_explicit, rel=1.0e-9)


def test_nonlinear_material_composes_dispersion_but_rejects_anisotropy_in_same_material():
    # Same-material nonlinearity + electric dispersion is now supported (chi2 SHG
    # needs the dispersion to set the phase-matching between w and 2w).
    dispersive_nonlinear = mw.Material(
        eps_r=2.25,
        nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-6),
        debye_poles=(mw.DebyePole(delta_eps=1.0, tau=1.0e-9),),
    )
    assert dispersive_nonlinear.is_nonlinear
    assert dispersive_nonlinear.is_electric_dispersive
    # A nonlinear anisotropic tensor still needs a coupled tensor update.
    with pytest.raises(NotImplementedError, match="nonlinear Material"):
        mw.Material(
            eps_r=2.25,
            nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-6),
            epsilon_tensor=mw.DiagonalTensor3(2.0, 2.5, 3.0),
        )


def test_scene_allows_nonlinear_and_dispersive_materials_in_separate_structures():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            name="chi2_box",
            geometry=Box(position=(-0.4, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material(eps_r=2.25, nonlinearity=mw.NonlinearSusceptibility(chi2=1.0e-6)),
        )
    )
    scene.add_structure(
        mw.Structure(
            name="kerr_box",
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material(eps_r=2.25, kerr_chi3=1.0e-10),
        )
    )
    scene.add_structure(
        mw.Structure(
            name="debye_box",
            geometry=Box(position=(0.4, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=mw.Material.debye(eps_inf=2.0, delta_eps=1.0, tau=1.0e-9),
        )
    )
    scene.add_structure(
        mw.Structure(
            name="aniso_box",
            geometry=Box(position=(0.0, 0.0, 0.25), size=(0.3, 0.3, 0.1)),
            material=mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.0, 2.5, 3.0)),
        )
    )

    model = prepare_scene(scene).compile_materials()

    assert bool(torch.any(model["chi2"] != 0))
    assert bool(torch.any(model["kerr_chi3"] != 0))
    assert model["debye_poles"] and bool(torch.any(model["debye_poles"][0]["weight"] != 0))


def _build_plane_wave_scene(
    frequency,
    *,
    amplitude,
    domain=((-1.0, 1.0), (-0.3, 0.3), (-0.3, 0.3)),
    spacing=0.02,
):
    return mw.Scene(
        domain=mw.Domain(bounds=domain),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.CW(frequency=frequency, amplitude=amplitude),
                name="pw",
            )
        ],
    )


def _clamped_gather(field, ii, jj, kk):
    ii = ii.clamp(0, field.shape[0] - 1)
    jj = jj.clamp(0, field.shape[1] - 1)
    kk = kk.clamp(0, field.shape[2] - 1)
    return field[ii, jj, kk]


def _collocate_reference(component, ex, ey, ez, target_shape):
    """Torch reference of the kernel's clamped 4-point off-axis collocation."""
    device = ex.device
    i, j, k = torch.meshgrid(
        torch.arange(target_shape[0], device=device),
        torch.arange(target_shape[1], device=device),
        torch.arange(target_shape[2], device=device),
        indexing="ij",
    )
    if component == "Ex":
        ex_value = ex
        ey_value = 0.25 * (
            _clamped_gather(ey, i, j - 1, k)
            + _clamped_gather(ey, i, j, k)
            + _clamped_gather(ey, i + 1, j - 1, k)
            + _clamped_gather(ey, i + 1, j, k)
        )
        ez_value = 0.25 * (
            _clamped_gather(ez, i, j, k - 1)
            + _clamped_gather(ez, i, j, k)
            + _clamped_gather(ez, i + 1, j, k - 1)
            + _clamped_gather(ez, i + 1, j, k)
        )
    elif component == "Ey":
        ex_value = 0.25 * (
            _clamped_gather(ex, i - 1, j, k)
            + _clamped_gather(ex, i, j, k)
            + _clamped_gather(ex, i - 1, j + 1, k)
            + _clamped_gather(ex, i, j + 1, k)
        )
        ey_value = ey
        ez_value = 0.25 * (
            _clamped_gather(ez, i, j, k - 1)
            + _clamped_gather(ez, i, j, k)
            + _clamped_gather(ez, i, j + 1, k - 1)
            + _clamped_gather(ez, i, j + 1, k)
        )
    else:
        ex_value = 0.25 * (
            _clamped_gather(ex, i - 1, j, k)
            + _clamped_gather(ex, i, j, k)
            + _clamped_gather(ex, i - 1, j, k + 1)
            + _clamped_gather(ex, i, j, k + 1)
        )
        ey_value = 0.25 * (
            _clamped_gather(ey, i, j - 1, k)
            + _clamped_gather(ey, i, j, k)
            + _clamped_gather(ey, i, j - 1, k + 1)
            + _clamped_gather(ey, i, j, k + 1)
        )
        ez_value = ez
    return ex_value, ey_value, ez_value


def _reference_nonlinear_coefficients(solver, component):
    tensors = {
        "Ex": (
            solver.eps_Ex,
            solver.cex_decay_external,
            solver.sigma_e_Ex,
            solver.nonlinear_chi2_Ex,
            solver.kerr_chi3_Ex,
            solver.tpa_sigma_Ex,
        ),
        "Ey": (
            solver.eps_Ey,
            solver.cey_decay_external,
            solver.sigma_e_Ey,
            solver.nonlinear_chi2_Ey,
            solver.kerr_chi3_Ey,
            solver.tpa_sigma_Ey,
        ),
        "Ez": (
            solver.eps_Ez,
            solver.cez_decay_external,
            solver.sigma_e_Ez,
            solver.nonlinear_chi2_Ez,
            solver.kerr_chi3_Ez,
            solver.tpa_sigma_Ez,
        ),
    }[component]
    eps, external, sigma_static, chi2, chi3, tpa_sigma = tensors
    ex_value, ey_value, ez_value = _collocate_reference(
        component, solver.Ex, solver.Ey, solver.Ez, eps.shape
    )
    field_sq = ex_value * ex_value + ey_value * ey_value + ez_value * ez_value
    own = {"Ex": ex_value, "Ey": ey_value, "Ez": ez_value}[component]
    effective = eps + solver.eps0 * (chi2 * own + chi3 * field_sq)
    effective = torch.clamp(effective, min=1.0e-12 * solver.eps0)
    sigma = torch.clamp(sigma_static + tpa_sigma * field_sq, min=0.0)
    half = 0.5 * sigma * solver.dt / effective
    denom = 1.0 + half
    decay = external * (1.0 - half) / denom
    curl = external * (solver.dt / effective) / denom
    return decay, curl


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_nonlinear_coefficient_kernel_matches_torch_reference():
    frequency = 5.0e8
    scene = _build_plane_wave_scene(
        frequency,
        amplitude=50.0,
        domain=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4)),
        spacing=0.04,
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.4, 0.4)),
            material=mw.Material(
                eps_r=2.25,
                sigma_e=0.02,
                nonlinearity=(
                    mw.NonlinearSusceptibility(chi2=1.0e-4),
                    mw.TwoPhotonAbsorption(beta=1.0e-4),
                ),
            ),
        )
    )
    # A Kerr structure elsewhere exercises the chi3 channel of the general kernel.
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(-0.6, 0.0, 0.0), size=(0.2, 0.4, 0.4)),
            material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-8),
        )
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver

    assert solver.nonlinear_enabled
    assert solver.nonlinear_general_enabled
    assert solver.tpa_enabled

    generator = torch.Generator(device="cuda").manual_seed(7)
    for field in (solver.Ex, solver.Ey, solver.Ez):
        field.copy_(torch.randn(field.shape, generator=generator, device=field.device) * 30.0)

    solver._update_nonlinear_electric_coefficients()

    for component, decay_dynamic, curl_dynamic in (
        ("Ex", solver.cex_decay_dynamic, solver.cex_curl_dynamic),
        ("Ey", solver.cey_decay_dynamic, solver.cey_curl_dynamic),
        ("Ez", solver.cez_decay_dynamic, solver.cez_curl_dynamic),
    ):
        expected_decay, expected_curl = _reference_nonlinear_coefficients(solver, component)
        assert torch.allclose(decay_dynamic, expected_decay, rtol=1.0e-5, atol=1.0e-6)
        assert torch.allclose(curl_dynamic, expected_curl, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_chi3_descriptor_matches_kerr_material_run():
    """NonlinearSusceptibility(chi3=...) is the same runtime channel as kerr_chi3."""
    frequency = 1.0e9
    chi3 = 1.0e-10

    def _run(material):
        scene = _build_plane_wave_scene(
            frequency,
            amplitude=120.0,
            domain=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4)),
            spacing=0.04,
        )
        scene.add_structure(
            mw.Structure(
                geometry=Box(position=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8)),
                material=material,
            )
        )
        scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.34, fields=("Ez",)))
        result = mw.Simulation.fdtd(
            scene,
            frequencies=[frequency],
            run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ).run()
        payload = result.monitor("post")["components"]["Ez"]
        data = payload["data"] if isinstance(payload, dict) else payload
        return data.detach().clone()

    kerr_data = _run(mw.Material(eps_r=2.25, kerr_chi3=chi3))
    descriptor_data = _run(
        mw.Material(eps_r=2.25, nonlinearity=mw.NonlinearSusceptibility(chi3=chi3))
    )

    assert torch.allclose(kerr_data, descriptor_data, rtol=1.0e-6, atol=1.0e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_general_kernel_chi3_only_matches_kerr_fast_path():
    """The general coefficient kernel reduces exactly to the Kerr curl update."""
    frequency = 1.0e9
    scene = _build_plane_wave_scene(
        frequency,
        amplitude=50.0,
        domain=((-1.0, 1.0), (-0.4, 0.4), (-0.4, 0.4)),
        spacing=0.04,
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.6, 0.4, 0.4)),
            material=mw.Material(eps_r=2.25, nonlinearity=mw.NonlinearSusceptibility(chi3=1.0e-8)),
        )
    )
    solver = mw.Simulation.fdtd(scene, frequencies=[frequency]).prepare().solver

    # chi3-only stays on the Kerr fast path.
    assert solver.kerr_enabled and not solver.nonlinear_general_enabled

    generator = torch.Generator(device="cuda").manual_seed(11)
    for field in (solver.Ex, solver.Ey, solver.Ez):
        field.copy_(torch.randn(field.shape, generator=generator, device=field.device) * 40.0)

    solver._update_nonlinear_electric_coefficients()
    kerr_curls = {
        "Ex": solver.cex_curl_dynamic.clone(),
        "Ey": solver.cey_curl_dynamic.clone(),
        "Ez": solver.cez_curl_dynamic.clone(),
    }

    for component, eps, decay, sigma, chi3, curl_reference in (
        ("Ex", solver.eps_Ex, solver.cex_decay, solver.sigma_e_Ex, solver.kerr_chi3_Ex, kerr_curls["Ex"]),
        ("Ey", solver.eps_Ey, solver.cey_decay, solver.sigma_e_Ey, solver.kerr_chi3_Ey, kerr_curls["Ey"]),
        ("Ez", solver.eps_Ez, solver.cez_decay, solver.sigma_e_Ez, solver.kerr_chi3_Ez, kerr_curls["Ez"]),
    ):
        dynamic_decay = torch.empty_like(eps)
        dynamic_curl = torch.empty_like(eps)
        solver.fdtd_module.updateNonlinearElectricCoefficients3D(
            DynamicDecay=dynamic_decay,
            DynamicCurl=dynamic_curl,
            Ex=solver.Ex,
            Ey=solver.Ey,
            Ez=solver.Ez,
            LinearPermittivity=eps,
            # Non-conductive scene: the static decay is exactly the external
            # (PML x PEC) factor the general kernel expects.
            ExternalDecay=decay,
            SigmaStatic=sigma,
            Chi2=torch.zeros_like(eps),
            Chi3=chi3,
            TpaSigma=torch.zeros_like(eps),
            component=component,
            dt=solver.dt,
            eps0=solver.eps0,
        ).launchRaw()

        assert torch.equal(dynamic_curl, curl_reference)
        assert torch.equal(dynamic_decay, decay)


def _second_harmonic_amplitude(*, chi2, amplitude, thickness):
    frequency = 5.0e8
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                # A pulsed pump keeps the turn-on smooth (an abrupt CW start
                # would drive the chi2 medium far outside the perturbative
                # regime); the narrow fwidth keeps the source's own spectral
                # content at 2f negligible.
                source_time=mw.GaussianPulse(frequency=frequency, fwidth=1.0e8, amplitude=amplitude),
                name="pw",
            )
        ],
    )
    material_kwargs = {}
    if chi2 != 0.0:
        material_kwargs["nonlinearity"] = mw.NonlinearSusceptibility(chi2=chi2)
    # Fixed entrance face so the interaction length is exactly `thickness`.
    entrance = -0.35
    scene.add_structure(
        mw.Structure(
            geometry=Box(
                position=(entrance + thickness / 2.0, 0.0, 0.0),
                size=(thickness, 0.8, 0.8),
            ),
            material=mw.Material(eps_r=2.25, **material_kwargs),
        )
    )
    scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.55, fields=("Ez",)))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency, 2.0 * frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
        spectral_sampler=mw.SpectralSampler(window="hanning", normalize_source=False),
        full_field_dft=False,
    ).run()
    payload = result.monitor("post", freq_index=1)
    return abs(_plane_complex_mean(payload["components"]["Ez"]))


def _tpa_normalized_transmission(*, beta, amplitude):
    frequency = 5.0e8
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-1.0, 1.0), (-0.3, 0.3), (-0.3, 0.3))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.pml(num_layers=8, strength=1.0),
        device="cuda",
        sources=[
            mw.PlaneWave(
                direction=(1.0, 0.0, 0.0),
                polarization=(0.0, 0.0, 1.0),
                source_time=mw.GaussianPulse(frequency=frequency, fwidth=1.0e8, amplitude=amplitude),
                name="pw",
            )
        ],
    )
    material_kwargs = {}
    if beta != 0.0:
        material_kwargs["nonlinearity"] = mw.TwoPhotonAbsorption(beta=beta)
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(-0.2, 0.0, 0.0), size=(0.3, 0.8, 0.8)),
            material=mw.Material(eps_r=2.25, **material_kwargs),
        )
    )
    scene.add_monitor(mw.PlaneMonitor("post", axis="x", position=0.55, fields=("Ez",)))
    result = mw.Simulation.fdtd(
        scene,
        frequencies=[frequency],
        run_time=mw.TimeConfig.auto(steady_cycles=8, transient_cycles=18),
        spectral_sampler=mw.SpectralSampler(window="hanning", normalize_source=False),
        full_field_dft=False,
    ).run()
    payload = result.monitor("post")["components"]["Ez"]
    return abs(_plane_complex_mean(payload)) / amplitude


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_two_photon_absorption_transmission_decreases_with_intensity():
    """TPA transmission follows the saturable 1/(1 + beta*I*L_eff) trend.

    The DFT amplitude is normalized by the incident amplitude, so a linear
    medium gives an intensity-independent normalized transmission while a TPA
    medium absorbs progressively more at higher intensity.
    """
    beta = 4.0e-5
    amplitudes = (25.0, 100.0, 400.0)

    linear_low = _tpa_normalized_transmission(beta=0.0, amplitude=amplitudes[0])
    linear_high = _tpa_normalized_transmission(beta=0.0, amplitude=amplitudes[-1])
    # Linearity check of the normalization itself.
    assert linear_high == pytest.approx(linear_low, rel=1.0e-3)

    transmissions = [
        _tpa_normalized_transmission(beta=beta, amplitude=amplitude) for amplitude in amplitudes
    ]

    # Nonlinear loss must be monotonic in intensity and significant at the top.
    assert transmissions[0] < linear_low
    assert transmissions[1] < transmissions[0]
    assert transmissions[2] < 0.7 * transmissions[1]
    assert transmissions[2] < 0.7 * transmissions[0]
    # Saturable-absorber shape: the fractional loss grows slower than the
    # intensity itself (1/(1+x) rather than exponential in x): a 16x intensity
    # increase must not suppress transmission by more than 16x.
    assert transmissions[2] > transmissions[0] / 16.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_chi2_second_harmonic_scaling():
    """Undepleted-pump SHG in a phase-matched (dispersionless) slab.

    The second-harmonic amplitude must scale linearly in chi2, linearly in the
    interaction length, and quadratically in the pump amplitude; the tolerances
    check the scaling exponents rather than absolute magnitudes.
    """
    chi2 = 1.0e-6
    amplitude = 100.0
    thickness = 0.3

    background = _second_harmonic_amplitude(chi2=0.0, amplitude=amplitude, thickness=thickness)
    base = _second_harmonic_amplitude(chi2=chi2, amplitude=amplitude, thickness=thickness)
    double_chi2 = _second_harmonic_amplitude(chi2=2.0 * chi2, amplitude=amplitude, thickness=thickness)
    double_amplitude = _second_harmonic_amplitude(chi2=chi2, amplitude=2.0 * amplitude, thickness=thickness)
    double_thickness = _second_harmonic_amplitude(chi2=chi2, amplitude=amplitude, thickness=2.0 * thickness)

    # The linear run's residual at 2f (spectral leakage) must be negligible.
    assert background < 0.1 * base

    assert 1.6 < double_chi2 / base < 2.4
    assert 3.2 < double_amplitude / base < 4.8
    assert 1.5 < double_thickness / base < 2.5
