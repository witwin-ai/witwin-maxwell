"""Amplitude taper windows for analytic antenna arrays.

All windows are symmetric, real, evaluated from their analytic definitions in
torch (``float64`` by default) and scaled to a peak weight of one. Index ``k``
runs over ``0..N-1`` along one array axis. Two-dimensional rectangular arrays
use separable tapers ``w[ix, iy] = wx[ix] * wy[iy]`` whose flattened row-major
order matches :func:`rectangular_lattice`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_COSINE_SUM_COEFFICIENTS = {
    "hamming": (0.54, 0.46),
    "hann": (0.5, 0.5),
    "blackman": (0.42, 0.5, 0.08),
    "blackman_harris": (0.35875, 0.48829, 0.14128, 0.01168),
}
TAPER_WINDOWS = (
    "uniform",
    "hamming",
    "hann",
    "blackman",
    "blackman_harris",
    "kaiser",
    "chebyshev",
    "taylor",
)


def _cosine_sum(n: int, coefficients, *, device, dtype) -> torch.Tensor:
    # Symmetric generalized cosine window: sum_m (-1)^m a_m cos(2 pi m k / (N-1)).
    x = 2.0 * math.pi * torch.arange(n, device=device, dtype=dtype) / (n - 1)
    window = torch.zeros(n, device=device, dtype=dtype)
    for order, coefficient in enumerate(coefficients):
        window = window + ((-1.0) ** order) * coefficient * torch.cos(order * x)
    return window


def _kaiser(n: int, beta: float, *, device, dtype) -> torch.Tensor:
    ratio = 2.0 * torch.arange(n, device=device, dtype=dtype) / (n - 1) - 1.0
    argument = beta * torch.sqrt(torch.clamp_min(1.0 - ratio.square(), 0.0))
    beta_tensor = torch.tensor(beta, device=device, dtype=dtype)
    return torch.special.i0(argument) / torch.special.i0(beta_tensor)


def _chebyshev_polynomial(order: int, x: torch.Tensor) -> torch.Tensor:
    inside = torch.cos(order * torch.arccos(torch.clamp(x, -1.0, 1.0)))
    magnitude = torch.clamp_min(torch.abs(x), 1.0)
    outside = torch.cosh(order * torch.arccosh(magnitude))
    outside = torch.where((x < 0.0) & (order % 2 == 1), -outside, outside)
    return torch.where(torch.abs(x) <= 1.0, inside, outside)


def _dolph_chebyshev(n: int, sidelobe_db: float, *, device, dtype) -> torch.Tensor:
    # The array factor of the symmetric N-element Dolph-Chebyshev array is
    # AF(psi) = T_{N-1}(x0 cos(psi / 2)) with T_{N-1}(x0) = R = 10^(SLL/20).
    # With AF(psi) = sum_k w_k exp(j (k - (N-1)/2) psi), the N samples
    # psi_m = 2 pi m / N determine the weights exactly through an inverse DFT.
    order = n - 1
    ratio = 10.0 ** (sidelobe_db / 20.0)
    x0 = math.cosh(math.acosh(ratio) / order)
    m = torch.arange(n, device=device, dtype=dtype)
    psi = 2.0 * math.pi * m / n
    samples = _chebyshev_polynomial(order, x0 * torch.cos(0.5 * psi))
    offsets = torch.arange(n, device=device, dtype=dtype) - 0.5 * order
    weights = torch.sum(
        samples[None, :] * torch.cos(offsets[:, None] * psi[None, :]), dim=1
    ) / n
    return weights


def _taylor(n: int, nbar: int, sidelobe_db: float, *, device, dtype) -> torch.Tensor:
    # Taylor nbar distribution sampled at the element centres
    # x_k = (k - (N-1)/2) / N of a continuous aperture of unit length:
    # w(x) = 1 + 2 sum_{m=1}^{nbar-1} F_m cos(2 pi m x).
    ratio = 10.0 ** (sidelobe_db / 20.0)
    a = math.acosh(ratio) / math.pi
    sigma_sq = nbar**2 / (a**2 + (nbar - 0.5) ** 2)
    orders = torch.arange(1, nbar, device=device, dtype=dtype)
    coefficients = []
    for m in range(1, nbar):
        numerator = (-1.0) ** (m + 1) * torch.prod(
            1.0 - m**2 / (sigma_sq * (a**2 + (orders - 0.5) ** 2))
        )
        others = orders[orders != m]
        denominator = 2.0 * torch.prod(1.0 - m**2 / others.square())
        coefficients.append(numerator / denominator)
    x = (torch.arange(n, device=device, dtype=dtype) - 0.5 * (n - 1)) / n
    window = torch.ones(n, device=device, dtype=dtype)
    for order, coefficient in zip(range(1, nbar), coefficients):
        window = window + 2.0 * coefficient * torch.cos(2.0 * math.pi * order * x)
    return window


@dataclass(frozen=True)
class Taper:
    """One-axis amplitude taper specification.

    ``window`` is one of ``TAPER_WINDOWS``. ``sidelobe_db`` is the positive
    design sidelobe attenuation for ``"chebyshev"`` and ``"taylor"`` (``30``
    designs -30 dB sidelobes), ``nbar`` the Taylor transition index and
    ``beta`` the Kaiser shape parameter. The symmetric Hann and Blackman
    definitions have (numerically) zero end weights, matching the standard
    symmetric window definitions.
    """

    window: str = "uniform"
    sidelobe_db: float | None = None
    nbar: int | None = None
    beta: float | None = None

    def __post_init__(self):
        window = str(self.window).lower().replace("-", "_")
        if window not in TAPER_WINDOWS:
            raise ValueError(f"Unknown taper window {self.window!r}; expected one of {TAPER_WINDOWS}.")
        object.__setattr__(self, "window", window)
        if window in ("chebyshev", "taylor"):
            if self.sidelobe_db is None:
                raise ValueError(f"{window} taper requires sidelobe_db.")
            if not math.isfinite(self.sidelobe_db) or self.sidelobe_db <= 0.0:
                raise ValueError("sidelobe_db must be a positive finite attenuation in dB.")
        if window == "taylor":
            if self.nbar is None:
                raise ValueError("taylor taper requires nbar.")
            if int(self.nbar) != self.nbar or self.nbar < 1:
                raise ValueError("nbar must be a positive integer.")
            object.__setattr__(self, "nbar", int(self.nbar))
        if window == "kaiser":
            if self.beta is None:
                raise ValueError("kaiser taper requires beta.")
            if not math.isfinite(self.beta) or self.beta < 0.0:
                raise ValueError("beta must be a non-negative finite number.")

    def weights(
        self,
        n: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Return the ``[n]`` real taper weights scaled to a peak of one."""

        if int(n) != n or n < 1:
            raise ValueError("n must be a positive integer.")
        n = int(n)
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point dtype.")
        if n == 1 or self.window == "uniform":
            return torch.ones(n, device=device, dtype=dtype)
        if self.window in _COSINE_SUM_COEFFICIENTS:
            window = _cosine_sum(
                n, _COSINE_SUM_COEFFICIENTS[self.window], device=device, dtype=dtype
            )
        elif self.window == "kaiser":
            window = _kaiser(n, float(self.beta), device=device, dtype=dtype)
        elif self.window == "chebyshev":
            window = _dolph_chebyshev(n, float(self.sidelobe_db), device=device, dtype=dtype)
        else:
            window = _taylor(n, self.nbar, float(self.sidelobe_db), device=device, dtype=dtype)
        return window / torch.amax(torch.abs(window))


def _resolve_taper(value) -> Taper:
    if isinstance(value, Taper):
        return value
    if isinstance(value, str):
        return Taper(value)
    raise TypeError("taper must be a Taper instance or a window name.")


def taper_weights(
    n: int,
    window: str = "uniform",
    *,
    sidelobe_db: float | None = None,
    nbar: int | None = None,
    beta: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``[n]`` peak-normalized taper weights for one array axis."""

    taper = Taper(window, sidelobe_db=sidelobe_db, nbar=nbar, beta=beta)
    return taper.weights(n, device=device, dtype=dtype)


def rectangular_taper(
    nx: int,
    ny: int,
    taper_x: Taper | str = "uniform",
    taper_y: Taper | str | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the separable ``[nx, ny]`` taper ``wx[:, None] * wy[None, :]``.

    ``taper_y`` defaults to ``taper_x``. ``reshape(-1)`` gives the weight
    order of :func:`rectangular_lattice` positions.
    """

    resolved_x = _resolve_taper(taper_x)
    resolved_y = resolved_x if taper_y is None else _resolve_taper(taper_y)
    wx = resolved_x.weights(nx, device=device, dtype=dtype)
    wy = resolved_y.weights(ny, device=device, dtype=dtype)
    return wx[:, None] * wy[None, :]


def taper_efficiency(weights: torch.Tensor) -> torch.Tensor:
    """Aperture taper efficiency ``|sum w|^2 / (N sum |w|^2)`` over all elements."""

    if not isinstance(weights, torch.Tensor) or weights.numel() == 0:
        raise ValueError("weights must be a non-empty tensor.")
    flat = weights.reshape(-1)
    power = torch.sum(torch.abs(flat).square())
    if not bool(power > 0.0):
        raise ValueError("weights must not be identically zero.")
    return torch.abs(torch.sum(flat)).square() / (flat.numel() * power)


__all__ = [
    "TAPER_WINDOWS",
    "Taper",
    "rectangular_taper",
    "taper_efficiency",
    "taper_weights",
]
