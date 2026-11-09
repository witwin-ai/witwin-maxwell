import math

import pytest
import torch

import witwin.maxwell as mw

_C0 = 299_792_458.0
_FREQUENCIES = torch.linspace(100e12, 400e12, 121, dtype=torch.float64)


def _sample(material, frequencies=_FREQUENCIES):
    return torch.tensor(
        [material.relative_permittivity(float(f)) for f in frequencies],
        dtype=torch.complex128,
    )


def _assert_passive_and_consistent(fit, frequencies):
    report = fit.report
    assert report.passive
    assert report.stable
    assert report.passivity_margin >= -1e-12
    assert report.max_passivity_violation <= 1e-12
    assert report.unstable_poles == 0
    assert all(pole.real <= 0.0 for pole in report.poles)
    # The returned object is an ordinary Material whose own evaluation is the fit.
    dense = torch.logspace(
        math.log10(report.passivity_band[0]),
        math.log10(report.passivity_band[1]),
        257,
        dtype=torch.float64,
    )
    assert all(fit.material.relative_permittivity(float(f)).imag >= -1e-12 for f in dense)
    assert report.frequency_band == (float(frequencies.min()), float(frequencies.max()))


def test_recovers_single_lorentz_parameters():
    truth = mw.Material(
        eps_r=2.0,
        lorentz_poles=(mw.LorentzPole(delta_eps=1.5, resonance_frequency=250e12, gamma=20e12),),
    )
    fit = mw.fit_dispersion(_FREQUENCIES, _sample(truth), num_poles=2, name="fit")

    assert isinstance(fit.material, mw.Material)
    assert fit.material.name == "fit"
    assert len(fit.material.lorentz_poles) == 1
    assert not fit.material.drude_poles and not fit.material.debye_poles
    pole = fit.material.lorentz_poles[0]
    assert pole.delta_eps == pytest.approx(1.5, rel=1e-6)
    assert pole.resonance_frequency == pytest.approx(250e12, rel=1e-6)
    assert pole.gamma == pytest.approx(20e12, rel=1e-6)
    assert float(fit.material.eps_r) == pytest.approx(2.0, rel=1e-6)
    assert fit.report.relative_max_error < 1e-6
    assert fit.report.eps_inf == pytest.approx(2.0, rel=1e-6)
    _assert_passive_and_consistent(fit, _FREQUENCIES)


def test_recovers_drude_parameters():
    truth = mw.Material(eps_r=3.0, drude_poles=(mw.DrudePole(plasma_frequency=2000e12, gamma=30e12),))
    fit = mw.fit_dispersion(_FREQUENCIES, _sample(truth), num_poles=2)

    assert len(fit.material.drude_poles) == 1
    assert not fit.material.lorentz_poles and not fit.material.debye_poles
    pole = fit.material.drude_poles[0]
    assert pole.plasma_frequency == pytest.approx(2000e12, rel=1e-6)
    assert pole.gamma == pytest.approx(30e12, rel=1e-5)
    assert float(fit.material.eps_r) == pytest.approx(3.0, rel=1e-5)
    assert fit.report.drude_count == 1
    _assert_passive_and_consistent(fit, _FREQUENCIES)
    # The Drude term contributes its marginal pole at s = 0 and one at s = -gamma.
    assert 0j in fit.report.poles


def test_recovers_debye_parameters():
    tau = 1.0 / (2.0 * math.pi * 200e12)
    truth = mw.Material(eps_r=1.5, debye_poles=(mw.DebyePole(delta_eps=4.0, tau=tau),))
    fit = mw.fit_dispersion(_FREQUENCIES, _sample(truth), num_poles=2)

    assert len(fit.material.debye_poles) == 1
    assert not fit.material.lorentz_poles and not fit.material.drude_poles
    pole = fit.material.debye_poles[0]
    assert pole.delta_eps == pytest.approx(4.0, rel=1e-6)
    assert pole.tau == pytest.approx(tau, rel=1e-6)
    assert float(fit.material.eps_r) == pytest.approx(1.5, rel=1e-6)
    _assert_passive_and_consistent(fit, _FREQUENCIES)


def test_recovers_mixed_drude_and_two_lorentz_terms():
    truth = mw.Material(
        eps_r=1.2,
        drude_poles=(mw.DrudePole(plasma_frequency=1500e12, gamma=20e12),),
        lorentz_poles=(
            mw.LorentzPole(delta_eps=0.5, resonance_frequency=150e12, gamma=10e12),
            mw.LorentzPole(delta_eps=1.0, resonance_frequency=300e12, gamma=40e12),
        ),
    )
    fit = mw.fit_dispersion(_FREQUENCIES, _sample(truth), num_poles=3)

    lorentz = sorted(fit.material.lorentz_poles, key=lambda pole: pole.resonance_frequency)
    assert len(lorentz) == 2 and len(fit.material.drude_poles) == 1
    for fitted, expected in zip(lorentz, sorted(truth.lorentz_poles, key=lambda p: p.resonance_frequency)):
        assert fitted.delta_eps == pytest.approx(expected.delta_eps, rel=1e-5)
        assert fitted.resonance_frequency == pytest.approx(expected.resonance_frequency, rel=1e-6)
        assert fitted.gamma == pytest.approx(expected.gamma, rel=1e-5)
    assert fit.material.drude_poles[0].plasma_frequency == pytest.approx(1500e12, rel=1e-6)
    assert fit.report.relative_max_error < 1e-6
    _assert_passive_and_consistent(fit, _FREQUENCIES)


def test_noisy_data_stays_passive_and_finds_the_resonance():
    truth = mw.Material(
        eps_r=2.0,
        lorentz_poles=(mw.LorentzPole(delta_eps=1.5, resonance_frequency=250e12, gamma=20e12),),
    )
    clean = _sample(truth)
    generator = torch.Generator().manual_seed(1234)
    noise = torch.complex(
        torch.randn(clean.shape, generator=generator, dtype=torch.float64),
        torch.randn(clean.shape, generator=generator, dtype=torch.float64),
    )
    noisy = clean * (1.0 + 0.02 * noise)
    for num_poles in (1, 3):
        fit = mw.fit_dispersion(_FREQUENCIES, noisy, num_poles=num_poles)
        _assert_passive_and_consistent(fit, _FREQUENCIES)
        dominant = max(fit.material.lorentz_poles, key=lambda pole: pole.delta_eps)
        assert dominant.resonance_frequency == pytest.approx(250e12, rel=1e-2)
        assert dominant.gamma == pytest.approx(20e12, rel=5e-2)
        # No unresolved in-band spike: the fit tracks the clean model between samples.
        dense = torch.linspace(100e12, 400e12, 1201, dtype=torch.float64)
        reference = _sample(truth, dense)
        fitted = _sample(fit.material, dense)
        assert float(torch.max(torch.abs(fitted - reference) / torch.abs(reference))) < 0.03


def test_nk_input_path_matches_permittivity_path():
    truth = mw.Material(
        eps_r=2.25,
        lorentz_poles=(mw.LorentzPole(delta_eps=0.8, resonance_frequency=200e12, gamma=15e12),),
    )
    epsilon = _sample(truth)
    index = torch.sqrt(epsilon)
    assert bool(torch.all(index.imag >= 0.0))
    wavelengths = _C0 / _FREQUENCIES
    torch.testing.assert_close(mw.permittivity_from_nk(index.real, index.imag), epsilon)

    fit_nk = mw.fit_dispersion_nk(wavelengths, index.real, index.imag, num_poles=2)
    fit_eps = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=2)
    assert fit_nk.report.relative_max_error < 1e-6
    for frequency in (120e12, 200e12, 380e12):
        assert fit_nk.material.relative_permittivity(frequency) == pytest.approx(
            truth.relative_permittivity(frequency), rel=1e-6
        )
        assert fit_nk.material.relative_permittivity(frequency) == pytest.approx(
            fit_eps.material.relative_permittivity(frequency), rel=1e-6
        )


def test_lossless_sellmeier_data_fits_with_lossless_terms():
    silica = mw.library_material("SiO2")
    wavelengths = torch.linspace(0.4e-6, 2.0e-6, 161, dtype=torch.float64)
    frequencies = _C0 / wavelengths
    fit = mw.fit_dispersion(frequencies, _sample(silica, frequencies), num_poles=2)
    assert fit.report.relative_max_error < 1e-5
    _assert_passive_and_consistent(fit, frequencies)
    index = fit.material.relative_permittivity(_C0 / 1.55e-6) ** 0.5
    assert index.real == pytest.approx(1.4440, abs=1e-4)
    assert abs(index.imag) < 1e-5


def test_metal_lorentz_drude_data_fits_with_few_poles():
    gold = mw.library_material("Au")
    wavelengths = torch.linspace(0.4e-6, 1.0e-6, 151, dtype=torch.float64)
    frequencies = _C0 / wavelengths
    fit = mw.fit_dispersion(frequencies, _sample(gold, frequencies), num_poles=3)
    assert fit.report.relative_max_error < 5e-3
    assert fit.report.eps_inf >= 1.0
    _assert_passive_and_consistent(fit, frequencies)


def test_eps_inf_options_window_and_restricted_pole_types():
    truth = mw.Material(
        eps_r=2.0,
        lorentz_poles=(mw.LorentzPole(delta_eps=1.5, resonance_frequency=250e12, gamma=20e12),),
    )
    epsilon = _sample(truth)

    fixed = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=2, eps_inf=2.0)
    assert float(fixed.material.eps_r) == 2.0
    assert fixed.report.relative_max_error < 1e-6

    capped = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=2, eps_inf_bounds=(1.0, 1.5))
    assert 1.0 <= float(capped.material.eps_r) <= 1.5
    _assert_passive_and_consistent(capped, _FREQUENCIES)

    window = (150e12, 300e12)
    windowed = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=1, frequency_range=window)
    inside = _FREQUENCIES[(_FREQUENCIES >= window[0]) & (_FREQUENCIES <= window[1])]
    assert windowed.report.sample_count == inside.numel()
    assert windowed.report.frequency_band == (float(inside[0]), float(inside[-1]))

    debye_only = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=2, pole_types=("debye",))
    assert not debye_only.material.lorentz_poles and not debye_only.material.drude_poles
    _assert_passive_and_consistent(debye_only, _FREQUENCIES)

    weights = torch.ones_like(_FREQUENCIES)
    weights[:20] = 100.0
    weighted = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=2, weights=weights)
    assert weighted.report.relative_max_error < 1e-6


def test_conductivity_term_and_tolerance_guard():
    truth = mw.Material(
        eps_r=4.0,
        sigma_e=0.5,
        debye_poles=(mw.DebyePole(delta_eps=2.0, tau=1.0 / (2.0 * math.pi * 250e12)),),
    )
    epsilon = _sample(truth)
    fit = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=2, conductivity=True)
    assert fit.report.relative_max_error < 1e-6
    assert fit.material.sigma_e == pytest.approx(0.5, rel=1e-4)
    assert fit.report.sigma_e == pytest.approx(fit.material.sigma_e)
    _assert_passive_and_consistent(fit, _FREQUENCIES)

    lorentz = mw.Material(
        eps_r=1.2,
        lorentz_poles=(
            mw.LorentzPole(delta_eps=0.5, resonance_frequency=150e12, gamma=10e12),
            mw.LorentzPole(delta_eps=1.0, resonance_frequency=300e12, gamma=40e12),
        ),
    )
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        mw.fit_dispersion(_FREQUENCIES, _sample(lorentz), num_poles=1, pole_types=("debye",), tolerance=1e-3)


def test_fit_is_reproducible():
    # The fit has no random state; CPU LAPACK may still differ at rounding level
    # between calls, so repeated fits agree to a tight tolerance, not bitwise.
    truth = mw.Material(
        eps_r=1.2,
        drude_poles=(mw.DrudePole(plasma_frequency=1500e12, gamma=20e12),),
        lorentz_poles=(mw.LorentzPole(delta_eps=1.0, resonance_frequency=300e12, gamma=40e12),),
    )
    epsilon = _sample(truth)
    first = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=3)
    second = mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=3)
    for attribute in ("lorentz_count", "drude_count", "debye_count"):
        assert getattr(first.report, attribute) == getattr(second.report, attribute)
    assert float(first.material.eps_r) == pytest.approx(float(second.material.eps_r), rel=1e-9)
    for frequency in (100e12, 250e12, 400e12):
        assert first.material.relative_permittivity(frequency) == pytest.approx(
            second.material.relative_permittivity(frequency), rel=1e-9
        )


def test_input_validation():
    epsilon = torch.full_like(_FREQUENCIES, 2.0 + 0.1j, dtype=torch.complex128)
    with pytest.raises(ValueError, match="num_poles"):
        mw.fit_dispersion(_FREQUENCIES, epsilon, num_poles=0)
    with pytest.raises(ValueError, match="pole_types"):
        mw.fit_dispersion(_FREQUENCIES, epsilon, pole_types=("sellmeier",))
    with pytest.raises(ValueError, match="samples"):
        mw.fit_dispersion(_FREQUENCIES[:5], epsilon[:5], num_poles=3)
    with pytest.raises(ValueError, match="positive"):
        mw.fit_dispersion(-_FREQUENCIES, epsilon)
    with pytest.raises(ValueError, match="weights"):
        mw.fit_dispersion(_FREQUENCIES, epsilon, weights=torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="eps_inf_bounds"):
        mw.fit_dispersion(_FREQUENCIES, epsilon, eps_inf_bounds=(2.0, 1.0))


def test_load_dispersion_csv_wavelength_nk_and_frequency_permittivity(tmp_path):
    nk_path = tmp_path / "nk.csv"
    nk_path.write_text(
        "# tabulated optical constants\n"
        "wavelength_um,n,k\n"
        "1.0,1.5,0.01\n"
        "0.5,1.6,0.02\n"
        "2.0,1.45,0.0\n"
    )
    frequencies, epsilon = mw.load_dispersion_csv(nk_path)
    expected_f = torch.tensor([_C0 / 2.0e-6, _C0 / 1.0e-6, _C0 / 0.5e-6], dtype=torch.float64)
    torch.testing.assert_close(frequencies, expected_f)
    expected_eps = torch.complex(
        torch.tensor([1.45, 1.5, 1.6], dtype=torch.float64),
        torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64),
    ) ** 2
    torch.testing.assert_close(epsilon, expected_eps)
    assert epsilon.dtype == torch.complex128

    eps_path = tmp_path / "eps.txt"
    eps_path.write_text("f_THz eps_re eps_im\n300 2.0 0.5\n100 3.0 0.25\n")
    frequencies, epsilon = mw.load_dispersion_csv(
        eps_path, abscissa="frequency", unit="THz", quantity="permittivity"
    )
    torch.testing.assert_close(frequencies, torch.tensor([100e12, 300e12], dtype=torch.float64))
    torch.testing.assert_close(epsilon, torch.tensor([3.0 + 0.25j, 2.0 + 0.5j], dtype=torch.complex128))

    with pytest.raises(ValueError, match="unit"):
        mw.load_dispersion_csv(nk_path, unit="furlong")


def test_csv_round_trip_through_fit(tmp_path):
    truth = mw.Material(
        eps_r=2.25,
        lorentz_poles=(mw.LorentzPole(delta_eps=0.8, resonance_frequency=200e12, gamma=15e12),),
    )
    index = torch.sqrt(_sample(truth))
    wavelengths_um = _C0 / _FREQUENCIES * 1e6
    path = tmp_path / "material.csv"
    rows = [f"{w:.17g},{n:.17g},{k:.17g}" for w, n, k in zip(wavelengths_um.tolist(), index.real.tolist(), index.imag.tolist())]
    path.write_text("lambda,n,k\n" + "\n".join(rows) + "\n")
    frequencies, epsilon = mw.load_dispersion_csv(path)
    fit = mw.fit_dispersion(frequencies, epsilon, num_poles=1)
    assert fit.report.relative_max_error < 1e-6
    assert fit.material.lorentz_poles[0].resonance_frequency == pytest.approx(200e12, rel=1e-6)
