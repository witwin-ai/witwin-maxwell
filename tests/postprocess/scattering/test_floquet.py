"""One CPU control for the analytic oracle, label sampling and native projection."""

import math
from types import SimpleNamespace

import torch

import witwin.maxwell as mw
from scripts.dev.fdtd.bloch_floquet_acceptance.common import (
    F0, FREQUENCIES, K0, LIGHT_LINE_MARGIN, SPECTRAL_TAIL, make_pulse, oblique_airy,
)
from witwin.maxwell.constants import C_0, EPSILON_0
from witwin.maxwell.fdtd.runtime.stepping import _electric_field_energy
from witwin.maxwell.postprocess.floquet import floquet_orders, reflection_transmission, resample_fixed_angle


def test_airy_and_fixed_angle_cpu():
    for transverse in (0., .01 * K0, .25 * K0, .55 * K0):
        pulse = make_pulse(transverse, min(FREQUENCIES), max(FREQUENCIES))
        cutoff = C_0 * transverse / (2 * math.pi) * (1 + LIGHT_LINE_MARGIN)
        spectral_tail = math.exp(-.5 * ((pulse.frequency - cutoff) / pulse.fwidth) ** 2)
        assert spectral_tail <= SPECTRAL_TAIL * (1 + 1.e-12)
        negative_lobe = math.exp(-.5 * ((pulse.frequency + cutoff) / pulse.fwidth) ** 2)
        assert spectral_tail + negative_lobe <= 2 * SPECTRAL_TAIL * (1 + 1.e-12)
        assert math.exp(-.5 * (pulse.delay / pulse.sigma_t) ** 2) < 1.e-13
    energy_solver = SimpleNamespace(complex_fields_enabled=True)
    for field in ("Ex", "Ey", "Ez"):
        setattr(energy_solver, field, torch.ones(1, dtype=torch.float64))
        setattr(energy_solver, field + "_imag", torch.ones(1, dtype=torch.float64))
        setattr(energy_solver, "eps_" + field, EPSILON_0)
    energy = _electric_field_energy(energy_solver)
    torch.testing.assert_close(energy, torch.tensor(6 * EPSILON_0, dtype=torch.float64), atol=0., rtol=1.e-14)
    for eps in (4., 12.):
        for pol in ("TE", "TM"):
            for kpar in (0., .25 * K0, .55 * K0):
                r, t = oblique_airy((.8 * F0, F0, 1.2 * F0), kpar, eps, pol)
                torch.testing.assert_close(abs(r).square() + abs(t).square(), torch.ones(3, dtype=torch.float64))
                r0, t0 = oblique_airy((F0,), kpar, eps, pol, thickness=0.)
                torch.testing.assert_close(r0, torch.zeros_like(r0))
                torch.testing.assert_close(t0, torch.ones_like(t0))
            r, t = oblique_airy((F0,), 0., eps, pol)
            index = math.sqrt(eps)
            rho = (1 - index) / (1 + index)
            phase = torch.exp(torch.tensor(2j * math.pi * index * .37, dtype=torch.complex128))
            torch.testing.assert_close(r[0], rho * (1 - phase ** 2) / (1 - rho ** 2 * phase ** 2))
            torch.testing.assert_close(t[0], (1 - rho ** 2) * phase / (1 - rho ** 2 * phase ** 2))

    kpar = torch.tensor([0., 1., 2.], dtype=torch.float64)
    omega = torch.tensor([.5, 1., 2., 3., 5.], dtype=torch.float64)
    labels = (2 * kpar[:, None] + 3j * omega[None, :]).requires_grad_()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[1, 2] = True
    sampled = resample_fixed_angle(labels, math.pi / 6, kpar=kpar, omega=omega, mask=mask, wave_speed=1.)
    torch.testing.assert_close(sampled["labels"][:2], omega[:2] + 3j * omega[:2])
    assert sampled["mask"].tolist() == [False, False, True, False, True]
    assert torch.isnan(sampled["labels"][[2, 4]]).all()
    sampled["labels"][[0, 1, 3]].real.sum().backward()
    assert torch.isfinite(labels.grad).all()
    # Exact-grid queries use only their contributing endpoint, preserving the
    # unmasked zero-k row even when an adjacent row is unavailable.
    mask[1] = True
    exact = resample_fixed_angle(labels.detach(), 0., kpar=kpar, omega=omega, mask=mask, wave_speed=1.)
    torch.testing.assert_close(exact["labels"], labels[0].detach())
    assert not exact["mask"].any()

    # A known nonzero order on staggered E coordinates, with a duplicated Bloch
    # endpoint on each primal axis, must recover both complex polarizations.
    nodes = torch.linspace(-math.pi, math.pi, 33, dtype=torch.float64)
    halves = (nodes[1:] + nodes[:-1]) / 2
    kt = torch.tensor([1.2, .1], dtype=torch.float64)
    unit = kt / torch.linalg.vector_norm(kt)
    te = torch.stack((-unit[1], unit[0]))
    cos_theta = torch.sqrt(1 - kt.square().sum() / 1.5 ** 2)
    expected = torch.tensor([2 + 3j, 1 - .5j], dtype=torch.complex128, requires_grad=True)
    electric = expected[0] * te + expected[1] * cos_theta * unit
    components = {}
    for i, (field, coords) in enumerate((("Ex", (halves, nodes)), ("Ey", (nodes, halves)))):
        a, b = coords
        values = electric[i] * torch.exp(1j * (kt[0] * a[:, None] + kt[1] * b[None, :]))
        components[field] = dict(coords=coords, data=values)
    result = SimpleNamespace(raw_monitor=lambda name, frequency: dict(axis="z", position=0., components=components))
    spectrum = floquet_orders(result, plane_monitor_name="plane", period=(2 * math.pi, 2 * math.pi),
                              bloch_wavevector=(.2, .1),
                              frequencies=(1.04 * math.hypot(.2, .1) * C_0 / (2 * math.pi), 1.5 * C_0 / (2 * math.pi)),
                              background=mw.Material(eps_r=1.))
    order = (spectrum["orders"] == torch.tensor([1, 0])).all(-1).nonzero().flatten()[0]
    torch.testing.assert_close(spectrum["amplitudes"][1, order], expected)
    assert spectrum["mask"][0].all()
    assert torch.isnan(spectrum["amplitudes"][0]).all()
    physical = floquet_orders(result, plane_monitor_name="plane", period=(2 * math.pi, 2 * math.pi),
                             bloch_wavevector=(.2, .1), frequencies=spectrum["frequencies"].tolist(),
                             background=mw.Material(eps_r=1.), light_line_margin=0.)
    zero = (physical["orders"] == 0).all(-1).nonzero().flatten()[0]
    assert not physical["mask"][0, zero]

    incident_components, total_components = {}, {}
    incident_kt = torch.tensor([.2, .1], dtype=torch.float64)
    incident_te = torch.tensor([-.1, .2], dtype=torch.float64) / incident_kt.norm()
    for i, (field, entry) in enumerate(components.items()):
        a, b = entry["coords"]
        wave = incident_te[i] * torch.exp(1j * (incident_kt[0] * a[:, None] + incident_kt[1] * b[None, :]))
        incident_components[field] = dict(coords=entry["coords"], data=wave)
        total_components[field] = dict(coords=entry["coords"], data=wave + entry["data"])
    reference = SimpleNamespace(raw_monitor=lambda name, frequency: dict(
        axis="z", position=0., components=incident_components))
    scattered = SimpleNamespace(raw_monitor=lambda name, frequency: dict(
        axis="z", position=0., components=total_components if name == "reflection" else components))
    coefficients = reflection_transmission(
        scattered, reference, reflection_monitor_name="reflection", transmission_monitor_name="transmission",
        period=(2 * math.pi, 2 * math.pi), bloch_wavevector=(.2, .1),
        frequencies=(1.5 * C_0 / (2 * math.pi),), background=mw.Material(eps_r=1.), incident_polarization="TE")
    power_scale = ((1.5 ** 2 - kt.square().sum()) / (1.5 ** 2 - incident_kt.square().sum())) ** .25
    for key in ("r", "t"):
        torch.testing.assert_close(coefficients[key][0, order], expected * power_scale)
    (spectrum["amplitudes"][1, order].abs().square().sum() +
     coefficients["r"][0, order].abs().square().sum()).backward()
    assert torch.isfinite(expected.grad).all()


def test_saved_line_reference_matches_an_incident_run_at_interpolated_planes():
    frequencies = (.9e9, 1.1e9)
    kpar = torch.tensor((2., 1.), dtype=torch.float64)
    direction = kpar / kpar.norm()
    te = torch.stack((-direction[1], direction[0]))
    ds, dt = .02, .01 / C_0
    omega = 2 * math.pi * torch.tensor(frequencies, dtype=torch.float64)
    discrete_k = torch.tensor([2 * math.sin(float(k) * ds / 2) / ds for k in kpar])
    q = 2 / ds * torch.asin(ds / 2 * torch.sqrt(
        (2 * torch.sin(omega * dt / 2) / (C_0 * dt)).square() - discrete_k.square().sum()))
    nodes = torch.linspace(-.05, .05, 9, dtype=torch.float64)
    halves = (nodes[1:] + nodes[:-1]) / 2
    expected_r, expected_t = .2 + .1j, -.4 + .3j
    weights, positions = (.4, .6), (.03, .05)
    phase = sum(w * torch.exp(1j * q * (z + .1)) for w, z in zip(weights, positions))
    for polarization in ("TE", "TM"):
        cosine = torch.sqrt(1 - kpar.square().sum() / (omega / C_0).square())
        tangential = te.expand(2, -1) if polarization == "TE" else cosine[:, None] * direction
        electric = torch.cat((tangential, torch.zeros((2, 1), dtype=torch.float64)), -1).to(torch.complex128)

        def monitor(name, frequency, *, incident=False):
            index = frequencies.index(frequency)
            components = {}
            for i, (field, coords) in enumerate((("Ex", (halves, nodes)), ("Ey", (nodes, halves)))):
                a, b = coords
                amplitude = electric[index, i] * phase[index]
                if not incident:
                    amplitude = amplitude * (1 + expected_r if name == "reflection" else expected_t)
                components[field] = dict(coords=coords, plane_weights=weights, plane_positions=positions,
                    data=amplitude * torch.exp(1j * (kpar[0] * a[:, None] + kpar[1] * b[None, :])))
            return dict(axis="z", position=.042, components=components)

        result = SimpleNamespace(raw_monitor=monitor, metadata=dict(tfsf_incident_reference=dict(
            frequencies=frequencies, electric=electric, dt=dt, ds=ds, position=-.1,
            discrete_wavevector=(*discrete_k.tolist(), 0.), direction_sign=1)))
        incident = SimpleNamespace(raw_monitor=lambda name, frequency: monitor(name, frequency, incident=True))
        kwargs = dict(reflection_monitor_name="reflection", transmission_monitor_name="transmission",
                      period=(.1, .1), bloch_wavevector=kpar.tolist(), frequencies=frequencies,
                      background=mw.Material(), incident_polarization=polarization)
        saved = reflection_transmission(result, **kwargs)
        matched = reflection_transmission(result, incident, **kwargs)
        zero = (saved["orders"] == 0).all(-1).nonzero().flatten()[0]
        pol = saved["polarizations"].index(polarization)
        for key, expected in (("r", expected_r), ("t", expected_t)):
            torch.testing.assert_close(saved[key][:, zero, pol], torch.full((2,), expected, dtype=torch.complex128))
            torch.testing.assert_close(saved[key][:, zero, pol], matched[key][:, zero, pol])

        denominators = torch.tensor((2 + 1j, 3 - 2j), dtype=torch.complex128)

        def normalized_monitor(name, frequency):
            payload = monitor(name, frequency)
            for entry in payload["components"].values():
                entry["data"] = entry["data"] / denominators[frequencies.index(frequency)]
            payload["source_normalization_spectra"] = dict.fromkeys(("Ex", "Ey"), denominators)
            payload["source_normalization_frequencies"] = frequencies
            return payload

        result.raw_monitor = normalized_monitor
        normalized = reflection_transmission(result, **kwargs)
        for key in ("r", "t"):
            torch.testing.assert_close(normalized[key][:, zero, pol], saved[key][:, zero, pol])
