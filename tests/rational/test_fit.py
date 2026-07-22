import pytest
import torch

from witwin.maxwell.rational import RationalFitConfig, RationalModel


def test_shared_pole_multiresponse_fit_recovers_synthetic_model() -> None:
    poles = torch.tensor([-2.0 + 4.0j, -2.0 - 4.0j], dtype=torch.complex128)
    first = torch.tensor(
        [[0.4 + 0.1j, 0.08 - 0.03j], [0.08 - 0.03j, 0.3 + 0.05j]],
        dtype=torch.complex128,
    )
    residues = torch.stack((first, first.conj()), dim=-1)
    source = RationalModel(poles, residues, torch.eye(2, dtype=torch.float64) * 0.1)
    frequencies = torch.logspace(-2, 1, 80, dtype=torch.float64)

    fitted = RationalModel.fit(
        frequencies,
        source.evaluate(frequencies),
        RationalFitConfig(order=2, iterations=2, relative_tolerance=1e-8),
        initial_poles=poles,
    )

    assert fitted.report is not None
    assert fitted.report.relative_max_error < 1e-8
    assert fitted.report.unstable_poles == 0


@pytest.mark.parametrize("kind,order", [("rc", 1), ("rlc", 2)])
def test_analytic_rc_rlc_fit_meets_phase_one_error_gate(kind: str, order: int) -> None:
    frequencies = torch.logspace(-2, 1, 100, dtype=torch.float64)
    s = -2j * torch.pi * frequencies
    if kind == "rc":
        values = 1.0 / (1.0 + 0.2 * s)
    else:
        values = 1.0 / (1.0 + 0.1 * s + 0.01 * s * s)

    fitted = RationalModel.fit(
        frequencies,
        values,
        RationalFitConfig(order=order, iterations=5, relative_tolerance=1e-3),
    )

    assert fitted.report is not None
    assert fitted.report.relative_max_error < 1e-3
    assert torch.all(fitted.poles.real < 0.0)


def test_automatic_fit_rejects_trainable_samples_without_detaching() -> None:
    frequencies = torch.logspace(-2, 1, 20, requires_grad=True)
    values = torch.ones(20, dtype=torch.complex64)
    with pytest.raises(RuntimeError, match="does not detach"):
        RationalModel.fit(frequencies, values, RationalFitConfig(order=1))


def test_matched_transmission_line_fit_meets_phase_one_error_gate() -> None:
    frequencies = torch.linspace(0.0, 2.0, 161, dtype=torch.float64)
    s = -2j * torch.pi * frequencies
    through = 0.8 * torch.exp(-s * 0.08)
    scattering = torch.zeros((frequencies.numel(), 2, 2), dtype=torch.complex128)
    scattering[:, 0, 1] = through
    scattering[:, 1, 0] = through

    fitted = RationalModel.fit(
        frequencies,
        scattering,
        RationalFitConfig(order=6, iterations=6, relative_tolerance=1e-3),
        representation="S",
    )

    assert fitted.report is not None
    assert fitted.report.relative_max_error < 1e-3
    assert fitted.report.passivity_margin is not None
    assert fitted.report.passivity_margin > 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [({"order": 1.5}, "order"), ({"iterations": 2.5}, "iterations")],
)
def test_fit_config_requires_integer_counts(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RationalFitConfig(**kwargs)
