"""R1 admission gates. Run on CUDA under the workstation memory-cap plugin."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from scipy.special import jv
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered import LayeredGreenTable
from witwin.maxwell.mom.layered.build import build_extension
from tests.support.layered_green_oracle import C0, homogeneous, spatial, spectral


pytestmark = pytest.mark.cuda


def rfic_stack():
    return LayerStack((Layer("silicon", 300e-6, 11.9, 10.0),
                       Layer("oxide1", 2e-6, 4.1), Layer("oxide2", 3e-6, 4.1),
                       Layer("nitride", .1e-6, 7.5), Layer("oxide3", 4e-6, 4.1)),
                      HalfSpace(11.9, 10.0), HalfSpace())


def rfic_planes(stack):
    edges = stack.interfaces()
    return (edges[1], edges[3], edges[5])


@pytest.fixture(scope="module")
def rfic():
    stack = rfic_stack()
    return LayeredGreenTable.build(stack, rfic_planes(stack), (1e9, 100e9), 40e-6, device="cuda")


def cuda_rho(values):
    return torch.tensor(values, dtype=torch.float64, device="cuda")


def test_homogeneous():
    for epsilon in (1.0, 4.1 + .04j):
        f = 10e9
        wavelength = C0 / (f * np.sqrt(epsilon.real))
        stack = LayerStack((Layer("uniform", wavelength, epsilon),),
                           HalfSpace(epsilon), HalfSpace(epsilon))
        planes = (.1 * wavelength, .3 * wavelength)
        rho = np.geomspace(1e-3 * wavelength, 10 * wavelength, 61)
        table = LayeredGreenTable.build(stack, planes, (f,), rho[-1], device="cuda")
        for a, b in ((0, 0), (0, 1), (1, 1)):
            actual = table.evaluate(planes[a], planes[b], cuda_rho(rho))[0].cpu().numpy()
            expected = homogeneous(epsilon, f, rho, planes[a], planes[b])
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
        assert table.remainder.is_contiguous()
        assert table.interpolation_error.item() <= 1e-6
    with pytest.raises(FrozenInstanceError):
        stack.layers = ()
    with pytest.raises(ValueError, match="mu_r"):
        Layer("magnetic", 1e-3, mu_r=2)
    with pytest.raises(ValueError, match="mu_r"):
        HalfSpace(mu_r=2)


def test_ground_image():
    epsilon, f = 3.2 + .02j, 10e9
    stack = LayerStack((Layer("uniform", .01, epsilon),), "pec", HalfSpace(epsilon))
    planes = (.001, .003)
    rho = np.geomspace(1e-5, .1, 41)
    table = LayeredGreenTable.build(stack, planes, (f,), rho[-1], device="cuda")
    for a, b in ((0, 0), (0, 1)):
        actual = table.evaluate(planes[a], planes[b], cuda_rho(rho))[0].cpu().numpy()
        expected = homogeneous(epsilon, f, rho, planes[a], planes[b], ground=0)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)


def test_spectral_and_complex_bessel(rfic):
    rng = np.random.default_rng(7102)
    q = np.stack([2 * np.pi * f / C0 * (np.exp(rng.uniform(np.log(.3), np.log(300), 21)) -
                  1j * rng.uniform(0, .2, 21)) for f in (1e9, 100e9)])
    actual = rfic.spectral_samples(torch.tensor(q, dtype=torch.complex128, device="cuda"))
    actual = actual.cpu().numpy().reshape(2, 21, -1, 2)
    stack = rfic_stack()
    planes = rfic_planes(stack)
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    for fi, f in enumerate((1e9, 100e9)):
        for qi, node in enumerate(q[fi]):
            for pi, (i, j) in enumerate(pairs):
                expected = spectral(stack, f, node, planes[i], planes[j])
                np.testing.assert_allclose(actual[fi, qi, pi], expected, rtol=1e-12, atol=0)
    z = np.r_[np.linspace(0, 32, 129), np.linspace(32.001, 2048, 101)]
    z = np.r_[z, z - .1j, z - 8j].astype(np.complex128)
    actual = build_extension().j0(torch.tensor(z, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(actual, jv(0, z), rtol=1e-13, atol=0)


def test_lossy_stacks_against_unextracted_oracle(rfic):
    # The positive imaginary permittivity is passive for exp(-i*w*t).
    slab = LayerStack((Layer("slab", .5e-3, 9.8 + .1j),), "pec", HalfSpace())
    planes, frequencies = (.3e-3, .5e-3), (10e9, 60e9)
    rho = np.geomspace(1e-5, 3e-3, 7)
    table = LayeredGreenTable.build(slab, planes, frequencies, rho[-1], device="cuda")
    actual = table.evaluate(*planes, cuda_rho(rho)).cpu().numpy()
    for fi, f in enumerate(frequencies):
        expected = np.stack([spatial(slab, f, r, *planes)[0] for r in rho], axis=-1)
        assert np.all(np.max(abs(actual[fi] - expected), axis=-1) <=
                      1e-6 * np.max(abs(expected), axis=-1))
    stack = rfic_stack()
    planes = rfic_planes(stack)
    rho = np.array((2e-6, 10e-6, 30e-6))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        actual = rfic.evaluate(planes[i], planes[j], cuda_rho(rho)).cpu().numpy()
        for fi, f in enumerate((1e9, 100e9)):
            expected = np.stack([spatial(stack, f, r, planes[i], planes[j])[0] for r in rho], axis=-1)
            assert np.all(np.max(abs(actual[fi] - expected), axis=-1) <=
                          1e-6 * np.max(abs(expected), axis=-1))


def test_coplanar_interface_static_limit():
    stack = LayerStack((Layer("dielectric", 1e-3, 11.9),), HalfSpace(11.9), HalfSpace())
    plane = stack.interfaces()[-1]
    rho = np.geomspace(1e-6, 1e-5, 17)
    table = LayeredGreenTable.build(stack, (plane,), (1e6,), 1e-4, device="cuda")
    actual = table.evaluate(plane, plane, cuda_rho(rho))[0].cpu().numpy()
    g = 1 / (4 * np.pi * rho)
    np.testing.assert_allclose(actual, np.array((g, 2 * g / (1 + 11.9))), rtol=1e-4, atol=0)


def test_reciprocity_and_request_order(rfic):
    stack = rfic_stack()
    planes = rfic_planes(stack)
    reverse = LayeredGreenTable.build(stack, planes[::-1], (1e9, 100e9), 40e-6, device="cuda")
    rho = cuda_rho((1e-6, 5e-6, 31e-6))
    a = rfic.evaluate(planes[0], planes[2], rho)
    assert torch.equal(a, rfic.evaluate(planes[2], planes[0], rho))
    assert torch.equal(a, reverse.evaluate(planes[0], planes[2], rho))


def test_three_builds_bitwise_identical(rfic):
    stack = rfic_stack()
    for _ in range(2):
        again = LayeredGreenTable.build(stack, rfic_planes(stack), (1e9, 100e9), 40e-6, device="cuda")
        for field in fields(rfic):
            a, b = getattr(rfic, field.name), getattr(again, field.name)
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), field.name


def test_fail_closed(rfic):
    planes = rfic_planes(rfic_stack())
    with pytest.raises(RuntimeError, match="rho"):
        rfic.evaluate(planes[0], planes[1], cuda_rho((41e-6,)))
    with pytest.raises(RuntimeError, match="Unknown metal plane"):
        rfic.evaluate(planes[0] + 1e-9, planes[1], cuda_rho((1e-6,)))
    with pytest.raises(RuntimeError, match="rho"):
        rfic.evaluate(planes[0], planes[1], rfic.rho[-1:].contiguous())


def test_forced_sommerfeld_homogeneous():
    for epsilon in (1.0, 4.1 + .04j):
        for frequency in (1e9, 100e9):
            wavelength = C0 / (frequency * np.sqrt(epsilon.real))
            stack = LayerStack((Layer("lower", .2 * wavelength, epsilon),
                                Layer("upper", .2 * wavelength, epsilon * (1 + 1e-13))),
                               HalfSpace(epsilon), HalfSpace(epsilon * (1 + 1e-13)))
            planes = tuple(np.array((.1, .2, .3)) * wavelength)
            rho = np.geomspace(1e-3, 5, 41) * wavelength
            table = LayeredGreenTable.build(stack, planes, (frequency,), rho[-1], device="cuda")
            assert table.exact_mode == 0
            for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
                actual = table.evaluate(planes[i], planes[j], cuda_rho(rho))[0].cpu().numpy()
                expected = homogeneous(epsilon, frequency, rho, planes[i], planes[j])
                assert np.max(abs((actual - expected) / expected)) <= 1e-9


def test_rfic_envelope(record_property):
    stack = rfic_stack()
    edges = stack.interfaces()
    planes = (100e-6, 200e-6, edges[1], edges[2], 303e-6, edges[3], edges[4], edges[5])
    frequencies = np.geomspace(.1e9, 200e9, 9)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    table = LayeredGreenTable.build(stack, planes, frequencies, 2e-3, device="cuda")
    assert table.interpolation_error.item() <= 1e-6
    assert torch.cuda.max_memory_reserved() < 12 * 1024**3
    rho = np.array((1e-6, 1e-4, 1e-3, 2e-3))
    oracle_seconds = 0
    for source, observer in ((edges[1], edges[5]), (edges[3], edges[4]), (100e-6, 200e-6)):
        actual = table.evaluate(source, observer, cuda_rho(rho)).cpu().numpy()
        for index in (0, -1):
            start = perf_counter()
            expected = np.stack([spatial(stack, frequencies[index], r, source, observer)[0]
                                 for r in rho], axis=-1)
            oracle_seconds += perf_counter() - start
            assert np.all(np.max(abs(actual[index] - expected), axis=-1) <=
                          1e-6 * np.max(abs(expected), axis=-1))
    record_property("oracle_seconds", oracle_seconds)
    record_property("peak_reserved_bytes", torch.cuda.max_memory_reserved())


def test_external_layered_reference():
    # X2: open-source formulation-C library data (scripts/dev/rfic/make_strata_reference.py), conjugated
    # to exp(-i*omega*t). Its default adaptive integration agrees to ~3e-4, so this gate checks
    # convention, normalization and layering, not the last digits (O2 covers those).
    data = np.load(Path(__file__).parent / "data" / "strata_ling_jin_30ghz.npz")
    mm = 1e-3
    for s, sigma in enumerate(data["sigma"]):
        layers = tuple(Layer(f"L{i}", h * mm, e, sigma) for i, (_, h, e) in enumerate(data["layers"][::-1]))
        stack = LayerStack(layers, "pec", HalfSpace())
        planes = tuple(sorted({z for pair in data["pairs"] for z in pair}))
        table = LayeredGreenTable.build(stack, planes, (float(data["frequency"]),), 2.1e-2, device="cuda")
        for p, (source, observer) in enumerate(data["pairs"]):
            actual = table.evaluate(float(source), float(observer), cuda_rho(data["rho"]))[0].cpu().numpy()
            expected = data["values"][s, p]
            assert np.all(np.max(abs(actual - expected), axis=-1) <= 1e-3 * np.max(abs(expected), axis=-1))


def test_single_transform_admission(monkeypatch):
    ext = build_extension()
    bessel = ext.bessel
    calls = []

    def record(q, weights, kinds, rho, begin, end, midpoint):
        calls.append((end - begin, q.shape[1], midpoint))
        return bessel(q, weights, kinds, rho, begin, end, midpoint)

    monkeypatch.setattr(ext, 'bessel', record)
    stack = rfic_stack()
    table = LayeredGreenTable.build(stack, rfic_planes(stack), (100e9,), 40e-6,
                                    device='cuda', extract_distance=5e-6)
    assert all(not midpoint for _, _, midpoint in calls)
    assert sum(rows * nodes for rows, nodes, _ in calls) == sum(
        (plan.end - plan.begin) * plan.nodes for plan in table.quadrature_bins)
    assert torch.equal(table.rho[1::2], (table.rho[:-1:2] + table.rho[2::2]) * .5)
    coarse = table.remainder[..., ::2].contiguous()
    grid = table.rho[::2].contiguous()
    error = ext.admission(coarse, ext.slopes(coarse, grid),
                          table.remainder[..., 1::2].contiguous(), grid,
                          table.image_offsets, table.image_distances, table.image_coefficients)
    assert torch.equal(error, table.interpolation_error)
    assert table.remainder.is_contiguous()


def test_image_budget_keeps_the_green_function(rfic):
    stack = rfic_stack()
    planes = rfic_planes(stack)
    budget = LayeredGreenTable.build(stack, planes, (1e9, 100e9), 40e-6, device="cuda", extract_distance=5e-6)
    counts = budget.image_offsets.diff()
    assert counts.max() < rfic.image_offsets.diff().max()
    rho = cuda_rho(np.geomspace(1e-7, 40e-6, 33))
    for i in range(3):
        for j in range(i, 3):
            full, kept = rfic.evaluate(planes[i], planes[j], rho), budget.evaluate(planes[i], planes[j], rho)
            scale = full.abs().amax(-1)
            assert ((kept - full).abs().amax(-1) <= 1e-10 * scale).all()
    with pytest.raises(ValueError, match="thinnest layer"):
        LayeredGreenTable.build(stack, planes, (1e9,), 40e-6, device="cuda", extract_distance=.05e-6)
