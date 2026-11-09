import math

import pytest
import torch
from scipy.signal import windows

import witwin.maxwell as mw

pp = mw.postprocess


def _normalized(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return tensor / tensor.abs().max()


@pytest.mark.filterwarnings("ignore:This window is not suitable")
@pytest.mark.parametrize("n", [7, 8, 16, 33])
@pytest.mark.parametrize(
    ("window", "kwargs", "reference"),
    [
        ("hamming", {}, lambda n: windows.hamming(n)),
        ("hann", {}, lambda n: windows.hann(n)),
        ("blackman", {}, lambda n: windows.blackman(n)),
        ("blackman_harris", {}, lambda n: windows.blackmanharris(n)),
        ("kaiser", {"beta": 5.0}, lambda n: windows.kaiser(n, 5.0)),
        ("chebyshev", {"sidelobe_db": 35.0}, lambda n: windows.chebwin(n, 35.0)),
        ("taylor", {"sidelobe_db": 35.0, "nbar": 5}, lambda n: windows.taylor(n, 5, 35.0)),
    ],
)
def test_windows_match_standard_definitions(n, window, kwargs, reference):
    weights = pp.taper_weights(n, window, **kwargs)
    assert weights.dtype == torch.float64 and weights.shape == (n,)
    torch.testing.assert_close(weights, _normalized(reference(n)), rtol=0.0, atol=1e-13)
    assert float(weights.max()) == pytest.approx(1.0)
    torch.testing.assert_close(weights, weights.flip(0), rtol=0.0, atol=1e-14)


def test_uniform_and_single_element_tapers_are_ones():
    torch.testing.assert_close(pp.taper_weights(5), torch.ones(5, dtype=torch.float64))
    torch.testing.assert_close(
        pp.taper_weights(1, "chebyshev", sidelobe_db=30.0), torch.ones(1, dtype=torch.float64)
    )


@pytest.mark.parametrize("n", [6, 9, 14])
def test_dolph_chebyshev_weights_realize_the_chebyshev_polynomial(n):
    sidelobe_db = 25.0
    weights = pp.taper_weights(n, "chebyshev", sidelobe_db=sidelobe_db)
    ratio = 10.0 ** (sidelobe_db / 20.0)
    x0 = math.cosh(math.acosh(ratio) / (n - 1))
    psi = torch.linspace(-math.pi, math.pi, 257, dtype=torch.float64)
    offsets = torch.arange(n, dtype=torch.float64) - 0.5 * (n - 1)
    af = torch.sum(weights[None, :] * torch.cos(offsets[None, :] * psi[:, None]), dim=1)
    x = x0 * torch.cos(0.5 * psi)
    chebyshev = torch.where(
        x.abs() <= 1.0,
        torch.cos((n - 1) * torch.arccos(x.clamp(-1.0, 1.0))),
        torch.cosh((n - 1) * torch.arccosh(x.abs().clamp_min(1.0))),
    )
    scale = af[128] / chebyshev[128]
    torch.testing.assert_close(af, scale * chebyshev, rtol=0.0, atol=1e-12 * float(af.abs().max()))


def test_rectangular_taper_is_separable_and_matches_lattice_order():
    taper = pp.rectangular_taper(4, 3, pp.Taper("taylor", sidelobe_db=30.0, nbar=3), "hamming")
    wx = pp.taper_weights(4, "taylor", sidelobe_db=30.0, nbar=3)
    wy = pp.taper_weights(3, "hamming")
    torch.testing.assert_close(taper, wx[:, None] * wy[None, :])
    positions = pp.rectangular_lattice(4, 3, 0.1, 0.2)
    flat = taper.reshape(-1)
    # Row-major (ix, iy): element 3 sits at ix=1, iy=0.
    assert flat[3] == pytest.approx(float(wx[1] * wy[0]))
    assert positions[3, 0] == pytest.approx(-0.05) and positions[3, 1] == pytest.approx(-0.2)
    torch.testing.assert_close(pp.rectangular_taper(3, 3, "hann"), pp.rectangular_taper(3, 3, "hann", "hann"))


def test_taper_efficiency_orders_tapers():
    uniform = pp.taper_efficiency(pp.taper_weights(16))
    chebyshev = pp.taper_efficiency(pp.taper_weights(16, "chebyshev", sidelobe_db=30.0))
    blackman = pp.taper_efficiency(pp.taper_weights(16, "blackman"))
    assert float(uniform) == pytest.approx(1.0)
    assert 1.0 > float(chebyshev) > float(blackman) > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": "triangle"},
        {"window": "chebyshev"},
        {"window": "chebyshev", "sidelobe_db": -10.0},
        {"window": "taylor", "sidelobe_db": 30.0},
        {"window": "taylor", "sidelobe_db": 30.0, "nbar": 0},
        {"window": "kaiser"},
    ],
)
def test_invalid_taper_specifications_raise(kwargs):
    with pytest.raises(ValueError):
        pp.Taper(**kwargs)
