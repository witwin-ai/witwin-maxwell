"""Host numerical checks of the complex fixed-k Yee line and slab corrections."""
import math
from types import SimpleNamespace

import pytest
import torch

from witwin.maxwell.fdtd.boundary.cpml import DEFAULT_CPML_CONFIG
from witwin.maxwell.fdtd.excitation.fixed_k import FixedKIncidentLine
from witwin.maxwell.fdtd.excitation.tfsf_apply import apply_tfsf_e_correction, apply_tfsf_h_correction


def _line(axis=2, k=.7, polarization=(0., 1., 0.)):
    solver = SimpleNamespace(Nx=160, Ny=160, Nz=160, Ex=torch.zeros(1, dtype=torch.float64),
                             device="cpu", dt=.2, c=1., eps0=1., mu0=1.,
                             _requested_port_frequencies=(.05, .1, .2),
                             resolved_bloch_wavevector=tuple(k if i == (axis + 1) % 3 else 0. for i in range(3)),
                             cpml_config=dict(DEFAULT_CPML_CONFIG))
    source = dict(injection=dict(axis="xyz"[axis]), direction=tuple(float(i == axis) for i in range(3)),
                  polarization=polarization,
                  source_time=dict(kind="cw", frequency=.1, amplitude=0., phase=0., delay=0.))
    return FixedKIncidentLine(solver, source, dict.fromkeys("xyz", 1.), (10, 10, 10))


def test_soft_drive_keeps_integer_step_sampling(monkeypatch):
    times = []

    def waveform(source_time, time):
        times.append(time)
        return 1.0

    monkeypatch.setattr("witwin.maxwell.fdtd.excitation.fixed_k.evaluate_source_time", waveform)
    line = _line()
    for _ in range(3):
        line.advance_magnetic()
        line.advance_electric()
    assert times == [step * line.dt for step in range(3)]


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("polarization", ("TE", "TM"))
@pytest.mark.parametrize("sign", (-1, 1))
def test_line_matches_three_dimensional_yee_dispersion(axis, polarization, sign):
    tangent = (axis + 1) % 3
    other = (axis + 2) % 3
    p = tuple(float(i == (other if polarization == "TE" else tangent)) for i in range(3))
    line = _line(axis, sign * .7, p)
    k = torch.tensor(line.k, dtype=torch.float64)
    kz = sign * .9
    k[axis] = 2 * math.sin(kz / 2)
    omega = 2 / line.dt * math.asin(float(k.norm()) * line.dt / 2)
    discrete_omega = 2 / line.dt * math.sin(omega * line.dt / 2)
    if polarization == "TE":
        e = torch.tensor(p, dtype=torch.float64)
        h = torch.linalg.cross(k, e) / discrete_omega
    else:
        h = torch.zeros(3, dtype=torch.float64)
        h[other] = 1.
        e = -torch.linalg.cross(k, h) / discrete_omega
    positions = torch.arange(line.electric.shape[1], dtype=torch.float64)
    for i in range(3):
        e_positions = positions + (.5 if i == axis else 0.)
        h_positions = positions + (0. if i == axis else .5)
        line.electric[i] = e[i] * torch.exp(1j * kz * e_positions)
        line.magnetic[i] = h[i] * torch.exp(1j * (kz * h_positions + omega * line.dt / 2))
    old_e, old_h = line.electric.clone(), line.magnetic.clone()
    line.advance_magnetic()
    line.advance_electric()
    factor = complex(math.cos(omega * line.dt), -math.sin(omega * line.dt))
    torch.testing.assert_close(line.electric[:, 80:200], old_e[:, 80:200] * factor, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(line.magnetic[:, 80:200], old_h[:, 80:200] * factor, atol=1e-14, rtol=1e-14)


def test_split_bank_corrections_use_complex_line_and_profile():
    line = _line()
    line.electric[1, 80] = 2 + 3j
    line.magnetic[0, 79] = -4 + 5j
    profile = torch.tensor([[[1 + 2j]], [[3 - 4j]]], dtype=torch.complex128)
    solver = SimpleNamespace(tfsf_enabled=True, dt=.2, Ey=torch.zeros((3, 2, 2), dtype=torch.float64),
                             Hx=torch.zeros((3, 2, 2), dtype=torch.float64))
    solver.Ey_imag, solver.Hx_imag = torch.zeros_like(solver.Ey), torch.zeros_like(solver.Hx)
    solver._tfsf_state = dict(provider="plane_wave_fixed_k", auxiliary_grid=line,
        magnetic_terms=[dict(field_name="Hx", offsets=(1, 0, 0), coeff_patch=profile, component=1, sample_index=80)],
        electric_terms=[dict(field_name="Ey", offsets=(1, 0, 0), coeff_patch=profile, component=0, sample_index=79)])
    apply_tfsf_h_correction(solver, 0.)
    apply_tfsf_e_correction(solver, 0.)
    assert torch.equal(torch.complex(solver.Hx, solver.Hx_imag)[1:, :1, :1], profile * (2 + 3j))
    assert torch.equal(torch.complex(solver.Ey, solver.Ey_imag)[1:, :1, :1], profile * (-4 + 5j))


def test_line_cpml_pulse_is_bounded_and_state_round_trip_is_exact(tmp_path):
    line = _line()
    line.source_time = dict(kind="gaussian_pulse", frequency=.2, fwidth=.05, amplitude=1., phase=0., delay=20.)
    for _ in range(190):
        line.advance_magnetic()
        line.advance_electric()
    assert line.electric.abs().max() > .01
    assert line.psi_h.abs().max() > 0
    path = tmp_path / "line.pt"
    torch.save(line.state_tensors(), path)
    restored = _line()
    restored.source_time = line.source_time
    for name, value in torch.load(path, weights_only=True).items():
        restored.state_tensors()[name].copy_(value)
    peak = 0.
    for _ in range(1000):
        for current in (line, restored):
            current.advance_magnetic()
            current.advance_electric()
        peak = max(peak, float(line.electric.abs().max()))
    assert peak < 3
    for name, value in line.state_tensors().items():
        assert torch.equal(value, restored.state_tensors()[name]), name


@pytest.mark.parametrize("polarization", ("TE", "TM"))
def test_below_cutoff_mode_has_discrete_decay_and_no_temporal_growth(polarization):
    line = _line(polarization=(0., 1., 0.) if polarization == "TE" else (1., 0., 0.))
    omega = .4
    discrete_omega = 2 * math.sin(omega * line.dt / 2) / line.dt
    normal = math.sqrt(line.k[0] ** 2 - discrete_omega ** 2)
    kappa = 2 * math.asinh(normal / 2)
    k = torch.tensor((line.k[0], 0., 1j * normal), dtype=torch.complex128)
    if polarization == "TE":
        e = torch.tensor((0., 1., 0.), dtype=torch.complex128)
        h = torch.linalg.cross(k, e) / discrete_omega
    else:
        h = torch.tensor((0., 1., 0.), dtype=torch.complex128)
        e = -torch.linalg.cross(k, h) / discrete_omega
    positions = torch.arange(line.electric.shape[1], dtype=torch.float64) - 140
    for i in range(3):
        line.electric[i] = e[i] * torch.exp(-kappa * (positions + (.5 if i == 2 else 0.)))
        line.magnetic[i] = h[i] * torch.exp(-kappa * (positions + (0. if i == 2 else .5)) +
                                          1j * omega * line.dt / 2)
    before = line.electric.clone()
    line.advance_magnetic()
    line.advance_electric()
    phase = complex(math.cos(omega * line.dt), -math.sin(omega * line.dt))
    torch.testing.assert_close(line.electric[:, 140:148], before[:, 140:148] * phase, rtol=1.e-13, atol=1.e-14)
    component = 1 if polarization == "TE" else 0
    measured = math.log(float(abs(line.electric[component, 140] / line.electric[component, 145]))) / 5
    assert measured == pytest.approx(kappa, rel=1.e-13)
