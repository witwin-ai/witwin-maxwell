"""Vertical formulation-C spectral and independent reference gates."""

from pathlib import Path

import numpy as np
import pytest
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.build import build_extension
from tests.support.layered_green_oracle import C0, vertical_spectral, vertical_spatial
from tests.mom.layered.test_layered_green import rfic_stack

pytestmark = pytest.mark.cuda


def spectral_samples(stack, frequencies, q, sources, observers):
    media = (stack.bottom, *stack.layers, stack.top)
    def tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device="cuda")
    return build_extension().vertical_spectral(
        tensor([complex(m.eps_r) if m != "pec" else 1 for m in media], torch.complex128),
        tensor([m.sigma if m != "pec" else 0 for m in media], torch.float64),
        tensor(stack.interfaces(), torch.float64), tensor(frequencies, torch.float64),
        tensor(q, torch.complex128), tensor(sources, torch.float64),
        tensor(observers, torch.float64), stack.bottom == "pec", stack.top == "pec")


def test_vertical_spectral_impedance_transfer(record_property):
    stack = rfic_stack()
    frequencies = (1e9, 100e9)
    sources = (300.3e-6, 302.4e-6, 305.04e-6, 307e-6)
    observers = (304e-6, 305.07e-6, 300.8e-6, 303e-6)
    q = np.array([2 * np.pi * f / C0 * (np.geomspace(.3, 300, 21) - .12j)
                  for f in frequencies])
    values = spectral_samples(stack, frequencies, q, sources, observers)
    actual = values.cpu().numpy()
    expected = np.array([[[vertical_spectral(stack, f, node, zs, zo)
                           for zs, zo in zip(sources, observers)] for node in q[fi]]
                         for fi, f in enumerate(frequencies)])
    errors = np.max(abs(actual - expected), axis=(0, 1, 2)) / np.max(abs(expected), axis=(0, 1, 2))
    record_property("spectral_component_relative_errors", errors.tolist())
    assert np.all(errors <= 1e-12)
    reverse = spectral_samples(stack, frequencies, q, observers, sources).cpu().numpy()
    np.testing.assert_allclose(actual[..., 0], reverse[..., 0], rtol=1e-12)
    np.testing.assert_allclose(actual[..., 1], -reverse[..., 2], rtol=1e-12)
    np.testing.assert_allclose(actual[..., 3], reverse[..., 3], rtol=1e-12)
    for _ in range(2):
        assert torch.equal(values, spectral_samples(stack, frequencies, q, sources, observers))


def test_vertical_spectral_homogeneous_and_pec(record_property):
    f, epsilon = 10e9, 4.1 + .04j
    q = np.array([2 * np.pi * f / C0 * (np.geomspace(.3, 100, 31) - .1j)])
    zs, zo = .2e-3, .7e-3
    errors = []
    for bottom in (HalfSpace(epsilon), "pec"):
        stack = LayerStack((Layer("uniform", 1e-3, epsilon),), bottom, HalfSpace(epsilon))
        actual = spectral_samples(stack, (f,), q, (zs,), (zo,)).cpu().numpy()[0, :, 0]
        k = np.sqrt((2 * np.pi * f / C0)**2 * epsilon - q[0]**2)
        direct = .5j / k * np.exp(1j * k * (zo - zs))
        image = .5j / k * np.exp(1j * k * (zo + zs)) if bottom == "pec" else 0
        expected = np.stack((direct + image, np.zeros_like(direct), np.zeros_like(direct),
                             (direct - image) / epsilon), axis=-1)
        scale = np.max(abs(expected))
        error = np.max(abs(actual - expected)) / scale
        errors.append(error)
        assert error <= 1e-9
    record_property("homogeneous_and_ground_spectral_errors", errors)


def test_vertical_external_oracle_convention(record_property):
    # The external reference has its own adaptive quadrature and is intentionally
    # a looser gate than the impedance-transfer oracle.
    data = np.load(Path(__file__).parent / "data/strata_ling_jin_30ghz.npz")
    sigma = float(data["sigma"][1])
    stack = LayerStack(tuple(Layer(f"L{i}", h * 1e-3, e, sigma)
                             for i, (_, h, e) in enumerate(data["layers"][::-1])),
                       "pec", HalfSpace())
    errors = []
    for pair in (0, 2):
        zs, zo = data["pairs"][pair]
        indices = (0, 8, 16, 24)
        actual = np.stack([vertical_spatial(stack, float(data["frequency"]),
                                           float(data["rho"][i]), zs, zo)[0] for i in indices], axis=-1)
        expected = data["vertical_values"][1, pair][:, indices]
        error = np.max(abs(actual - expected), axis=-1) / np.max(abs(expected), axis=-1)
        errors.append(error.tolist())
        assert np.all(error <= 1e-3)
    record_property("external_oracle_component_errors", errors)


def test_vertical_homogeneous_table(record_property):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    f, epsilon = 10e9, 4.1 + .04j
    stack = LayerStack((Layer("lower", .5e-3, epsilon),
                        Layer("upper", .5e-3, epsilon * (1 + 1e-13))),
                       HalfSpace(epsilon), HalfSpace(epsilon * (1 + 1e-13)))
    table = LayeredGreenTable.build(stack, (.2e-3, .4e-3), (f,), 2e-3,
                                   device="cuda", via_spans=((.1e-3, .4e-3),))
    rho = torch.tensor(np.geomspace(1e-6, 2e-3, 31), dtype=torch.float64, device="cuda")
    source, observer = .17e-3, .33e-3
    actual = table.evaluate_vertical(rho, source, observer)[0].cpu().numpy()
    R = np.hypot(rho.cpu().numpy(), observer - source)
    g = np.exp(1j * 2 * np.pi * f / C0 * np.sqrt(epsilon) * R) / (4 * np.pi * R)
    expected = np.array((g, np.zeros_like(g), np.zeros_like(g), g / epsilon))
    errors = np.max(abs(actual - expected), axis=-1) / np.max(abs(g))
    record_property("table_component_errors", errors.tolist())
    assert np.all(errors <= 1e-9)
    with pytest.raises(RuntimeError, match="via spans"):
        table.evaluate_vertical(rho, .09e-3, observer)
    with pytest.raises(RuntimeError, match="rho"):
        table.evaluate_vertical(rho * 2, source, observer)


@pytest.mark.parametrize("frequency", (1e9, 100e9))
def test_vertical_lossy_table(frequency, record_property):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    stack = rfic_stack()
    torch.cuda.reset_peak_memory_stats()
    table = LayeredGreenTable.build(stack, (302.5e-6, 305.05e-6), (frequency,), 40e-6,
                                   device="cuda", extract_distance=5e-6,
                                   via_spans=((302.5e-6, 305.05e-6),))
    rho = torch.tensor((2e-6, 10e-6, 30e-6), dtype=torch.float64, device="cuda")
    source, observer = 303.7e-6, 305.03e-6
    actual = table.evaluate_vertical(rho, source, observer)[0].cpu().numpy()
    expected = np.stack([vertical_spatial(stack, frequency, r, source, observer)[0]
                         for r in rho.cpu().numpy()], axis=-1)
    errors = np.max(abs(actual - expected), axis=-1) / np.max(abs(expected), axis=-1)
    record_property("lossy_table_component_errors", errors.tolist())
    record_property("peak_reserved_bytes", torch.cuda.max_memory_reserved())
    record_property("z_admission", table.vertical.z_interpolation_error.item())
    record_property("rho_admission", table.vertical.rho_interpolation_error.item())
    record_property("tail_estimate_max", table.vertical.tail_error_estimate.max().item())
    assert torch.cuda.max_memory_reserved() < 12 * 1024**3
    assert np.all(errors <= 1e-6)


@pytest.mark.parametrize("sigma_index", (0, 1))
@pytest.mark.parametrize("pair_index", (0, 1, 2))
def test_vertical_external_table(sigma_index, pair_index, record_property):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    data = np.load(Path(__file__).parent / "data/strata_ling_jin_30ghz.npz")
    sigma = float(data["sigma"][sigma_index])
    stack = LayerStack(tuple(Layer(f"L{i}", h * 1e-3, e, sigma)
                             for i, (_, h, e) in enumerate(data["layers"][::-1])),
                       "pec", HalfSpace())
    source, observer = data["pairs"][pair_index]
    spans = tuple((float(z - 1e-5), float(z + 1e-5)) for z in sorted({source, observer}))
    table = LayeredGreenTable.build(stack, (float(source), float(observer)),
                                   (float(data["frequency"]),), 2.1e-2, device="cuda",
                                   via_spans=spans, extract_distance=.5e-3)
    rho = torch.tensor(data["rho"], dtype=torch.float64, device="cuda")
    actual = table.evaluate_vertical(rho, float(source), float(observer))[0].cpu().numpy()
    expected = data["vertical_values"][sigma_index, pair_index]
    errors = np.max(abs(actual - expected), axis=-1) / np.max(abs(expected), axis=-1)
    record_property("external_table_component_errors", errors.tolist())
    assert np.all(errors <= 1e-3)


def test_vertical_ground_table(record_property):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    f, epsilon = 10e9, 3.2 + .02j
    stack = LayerStack((Layer("uniform", 1e-3, epsilon),), "pec", HalfSpace(epsilon))
    table = LayeredGreenTable.build(stack, (.2e-3, .4e-3), (f,), 2e-3,
                                   device="cuda", via_spans=((.1e-3, .4e-3),))
    radial = np.geomspace(1e-6, 2e-3, 31)
    rho = torch.tensor(radial, dtype=torch.float64, device="cuda")
    source, observer = .17e-3, .33e-3
    actual = table.evaluate_vertical(rho, source, observer)[0].cpu().numpy()
    k = 2 * np.pi * f / C0 * np.sqrt(epsilon)
    R, Ri = np.hypot(radial, observer-source), np.hypot(radial, observer+source)
    direct, image = np.exp(1j*k*R)/(4*np.pi*R), np.exp(1j*k*Ri)/(4*np.pi*Ri)
    expected = np.array((direct+image, np.zeros_like(R), np.zeros_like(R), (direct-image)/epsilon))
    errors = np.max(abs(actual-expected), axis=-1)/np.max(abs(direct+image))
    record_property("ground_table_component_errors", errors.tolist())
    assert np.all(errors <= 1e-9)


def test_vertical_three_builds_and_horizontal_identity(record_property):
    from dataclasses import fields
    from witwin.maxwell.mom.layered import LayeredGreenTable
    stack = rfic_stack()
    plain = LayeredGreenTable.build(stack, (302.5e-6, 305.05e-6), (100e9,), 40e-6,
                                   device="cuda", extract_distance=5e-6)
    tables = [LayeredGreenTable.build(stack, (302.5e-6, 305.05e-6), (100e9,), 40e-6,
                                      device="cuda", extract_distance=5e-6,
                                      via_spans=((302.5e-6,305.05e-6),)) for _ in range(3)]
    for table in tables:
        for field in fields(plain):
            if isinstance(getattr(plain, field.name), torch.Tensor):
                assert torch.equal(getattr(plain, field.name), getattr(table, field.name)), field.name
    for table in tables[1:]:
        for field in fields(table.vertical):
            if isinstance(getattr(table.vertical, field.name), torch.Tensor):
                assert torch.equal(getattr(tables[0].vertical, field.name), getattr(table.vertical, field.name)), field.name
    rho = torch.tensor((1e-6, 30e-6), dtype=torch.float64, device="cuda")
    actual = tables[0].evaluate_vertical(rho, 303.7e-6, 305.03e-6)
    reverse = tables[0].evaluate_vertical(rho, 305.03e-6, 303.7e-6)
    assert torch.equal(actual[:, 0], reverse[:, 0])
    assert torch.equal(actual[:, 1], -reverse[:, 2])
    record_property("z_admission", tables[0].vertical.z_interpolation_error.item())
    record_property("rho_admission", tables[0].vertical.rho_interpolation_error.item())


def test_vertical_z_admission_refuses(monkeypatch):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    ext = build_extension()
    monkeypatch.setattr(ext, "vertical_error", lambda actual, expected: torch.tensor(2e-6, device="cuda"))
    stack = LayerStack((Layer("uniform", 1e-3),), HalfSpace(), HalfSpace())
    with pytest.raises(RuntimeError, match="Vertical z interpolation admission failed"):
        LayeredGreenTable.build(stack, (.2e-3, .4e-3), (10e9,), 2e-3, device="cuda",
                                via_spans=((.1e-3, .4e-3),))


def test_complex_order_one_bessel():
    from scipy.special import jv, hankel1, hankel2
    ext = build_extension()
    z = np.r_[np.linspace(0, 32, 129), np.linspace(32.001, 2048, 101)].astype(complex)
    z = np.r_[z, z - .1j, z - 8j]
    q = torch.tensor(z[None], dtype=torch.complex128, device="cuda")
    weights = torch.ones_like(q)
    rho = torch.ones(1, dtype=torch.float64, device="cuda")
    kinds = torch.zeros(q.shape, dtype=torch.int32, device="cuda")
    actual = ext.bessel_one(q, weights, kinds, rho, 0, 1, False).cpu().numpy()[0, 0]
    np.testing.assert_allclose(actual, jv(1,z)*z/(2*np.pi), rtol=1e-13, atol=1e-14)
    for kind, fn in ((1,hankel1),(-1,hankel2)):
        z = np.linspace(32.001, 2048, 101) + kind*8j
        q = torch.tensor(z[None], dtype=torch.complex128, device="cuda")
        kinds = torch.full(q.shape, kind, dtype=torch.int32, device="cuda")
        actual = ext.bessel_one(q, torch.ones_like(q), kinds, rho, 0, 1, False).cpu().numpy()[0, 0]
        np.testing.assert_allclose(actual, fn(1,z)*z/(2*np.pi), rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("frequency", (1e9, 100e9))
@pytest.mark.parametrize("epsilon", (1., 4.1+.04j))
def test_vertical_forced_sommerfeld(frequency, epsilon, record_property):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    wavelength = C0/(frequency*np.sqrt(complex(epsilon).real))
    stack = LayerStack((Layer("lower", .2*wavelength, epsilon),
                        Layer("upper", .2*wavelength, epsilon*(1+1e-13))),
                       HalfSpace(epsilon), HalfSpace(epsilon*(1+1e-13)))
    table = LayeredGreenTable.build(stack, (.05*wavelength, .15*wavelength), (frequency,),
                                   5*wavelength, device="cuda",
                                   via_spans=((.05*wavelength,.15*wavelength),))
    assert table.exact_mode == 0
    radii = np.geomspace(.001, 5, 41)*wavelength
    rho = torch.tensor(radii, dtype=torch.float64, device="cuda")
    source, observer = .073*wavelength, .129*wavelength
    actual = table.evaluate_vertical(rho, source, observer)[0].cpu().numpy()
    R = np.hypot(radii, observer-source)
    g = np.exp(1j*2*np.pi*frequency/C0*np.sqrt(complex(epsilon))*R)/(4*np.pi*R)
    expected = np.array((g, np.zeros_like(g), np.zeros_like(g), g/epsilon))
    local = max(np.max(abs((actual[0]-g)/g)),
                np.max(abs((actual[3]-g/epsilon)/(g/epsilon))))
    cross = np.max(abs(actual[1:3]))/np.max(abs(g))
    record_property("forced_local_error", float(local))
    record_property("forced_cross_error", float(cross))
    assert local <= 1e-9
    assert cross <= 1e-9


def test_vertical_span_ends_on_ground(record_property):
    from witwin.maxwell.mom.layered import LayeredGreenTable
    epsilon, frequency = 3.2+.02j, 10e9
    stack = LayerStack((Layer("uniform", 1e-3, epsilon),), "pec", HalfSpace(epsilon))
    table = LayeredGreenTable.build(stack, (0., .4e-3), (frequency,), 2e-3,
                                   device="cuda", via_spans=((0., .4e-3),))
    radial = np.geomspace(1e-6, 2e-3, 31)
    rho = torch.tensor(radial, dtype=torch.float64, device="cuda")
    actual = table.evaluate_vertical(rho, 0., .2e-3)[0].cpu().numpy()
    R = np.hypot(radial, .2e-3)
    g = np.exp(1j*2*np.pi*frequency/C0*np.sqrt(epsilon)*R)/(4*np.pi*R)
    expected = np.array((2*g, np.zeros_like(g), np.zeros_like(g), np.zeros_like(g)))
    errors = np.max(abs(actual-expected), axis=-1)/np.max(abs(2*g))
    record_property("ground_endpoint_component_errors", errors.tolist())
    assert np.all(errors <= 1e-9)
    record_property("ground_endpoint_z_admission", table.vertical.z_interpolation_error.item())
