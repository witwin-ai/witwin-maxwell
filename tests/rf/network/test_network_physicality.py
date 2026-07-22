import torch

import witwin.maxwell as mw


def _delayed_one_port(*, delay: float, magnitude: float = 0.8) -> mw.NetworkData:
    frequencies = torch.linspace(0.0, 20.0, 257, dtype=torch.float64)
    laplace = -2j * torch.pi * frequencies
    scattering = (magnitude * torch.exp(-laplace * delay)).reshape(-1, 1, 1)
    return mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("port",),
    )


def test_physicality_reports_passivity_and_finite_band_causality() -> None:
    causal = _delayed_one_port(delay=0.2)
    advanced = _delayed_one_port(delay=-0.2)
    active = _delayed_one_port(delay=0.2, magnitude=1.2)

    causal_report = causal.validate_physicality(causality_tolerance=1e-2)
    advanced_report = advanced.validate_physicality(causality_tolerance=1e-2)
    active_report = active.validate_physicality(causality_tolerance=1e-2)

    assert causal_report.passive and causal_report.causal
    assert not advanced_report.causal
    assert not active_report.passive
    assert causal_report.stable is None
    assert "finite-band" in causal_report.warnings[0]


def test_physicality_marks_nonuniform_sweep_causality_indeterminate() -> None:
    frequencies = torch.logspace(-2, 1, 20, dtype=torch.float64)
    network = mw.NetworkData(
        frequencies=frequencies,
        s=torch.zeros((20, 1, 1), dtype=torch.complex128),
        z0=50.0,
        port_names=("port",),
    )

    report = network.validate_physicality()

    assert report.causal is None
    assert report.negative_time_energy_ratio is None
    assert "indeterminate" in report.warnings[0]


def test_physicality_marks_truncated_rc_causality_indeterminate() -> None:
    frequencies = torch.linspace(0.0, 20.0, 257, dtype=torch.float64)
    laplace = -2j * torch.pi * frequencies
    scattering = (1.0 / (1.0 + 0.05 * laplace)).reshape(-1, 1, 1)
    network = mw.NetworkData(
        frequencies=frequencies,
        s=scattering,
        z0=50.0,
        port_names=("port",),
    )

    report = network.validate_physicality()

    assert report.causal is None
    assert report.negative_time_energy_ratio is not None
    assert "ambiguous" in report.warnings[0]


def test_network_data_rational_fit_uses_requested_representation() -> None:
    network = _delayed_one_port(delay=0.01)

    model = network.fit_rational(
        mw.RationalFitConfig(order=6, iterations=6, relative_tolerance=1e-3),
        representation="S",
    )

    assert isinstance(model, mw.RationalModel)
    assert model.representation == "S"
    assert model.report is not None
    assert model.report.relative_max_error < 1e-3
