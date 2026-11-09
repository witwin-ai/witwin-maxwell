"""Resonance extraction from time signals by harmonic inversion.

The filter-diagonalization method (FDM) of Mandelshtam and Taylor, in the
Wall-Neuhauser closed form, fits a uniformly sampled signal inside a chosen
frequency window by a finite sum of damped complex exponentials

    y(t) = sum_k a_k * exp(-i * omega_k * (t - t0)),
    omega_k = 2 * pi * f_k - i * gamma_k,

where ``t0`` is the time of the first analyzed sample. The time convention is
the repository phasor convention ``exp(-i omega t)``: a positive frequency
``f_k`` is a component ``exp(-i 2 pi f_k t)`` and a positive decay rate
``gamma_k`` is an amplitude envelope ``exp(-gamma_k t)`` (energy decays as
``exp(-2 gamma_k t)``). The quality factor is ``Q = Re(omega) / (2 gamma) =
pi * f / gamma``.

A real signal contains each resonance as a conjugate pair at ``+f`` and
``-f``; only the members inside the requested window are reported, so for a
real input ``A * exp(-gamma t) * cos(2 pi f t - phi)`` the reported mode at
``+f`` has ``amplitude = (A / 2) * exp(i phi)``.

The whole path is torch-native complex128 on the device of the input signal;
it uses no SciPy or NumPy numerics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

_CHUNK_ELEMENTS = 1 << 22


@dataclass(frozen=True)
class ResonanceModes:
    """Resonances extracted by harmonic inversion, sorted by frequency.

    All per-mode tensors have shape ``(num_modes,)`` and live on the device of
    the analyzed signal.

    Attributes:
        frequency: Real frequencies ``f_k`` in Hz (``float64``).
        decay_rate: Amplitude decay rates ``gamma_k`` in 1/s (``float64``);
            negative values are growing modes.
        quality_factor: ``Q_k = pi * f_k / gamma_k`` (``float64``); ``inf`` for
            an undamped mode, negative for a growing one.
        amplitude: Complex amplitudes ``a_k`` at ``t = t0`` (``complex128``).
        error: Fractional frequency-error estimate ``|delta omega| / |omega|``
            from the consistency of the first- and second-order FDM
            eigenproblems (``float64``); genuine modes of a clean signal reach
            roundoff level, spurious noise modes are large.
        t0: Time of the first analyzed sample, the amplitude/phase reference.
        dt: Sample spacing in seconds.
        num_basis: Number of FDM basis frequencies used.
        rank: Numerical rank of the regularized basis overlap matrix.
    """

    frequency: torch.Tensor
    decay_rate: torch.Tensor
    quality_factor: torch.Tensor
    amplitude: torch.Tensor
    error: torch.Tensor
    t0: float
    dt: float
    num_basis: int
    rank: int

    def __len__(self) -> int:
        return int(self.frequency.numel())

    @property
    def magnitude(self) -> torch.Tensor:
        return self.amplitude.abs()

    @property
    def phase(self) -> torch.Tensor:
        """Phase ``arg(a_k)`` in radians at ``t = t0``."""
        return torch.angle(self.amplitude)

    @property
    def angular_frequency(self) -> torch.Tensor:
        """Complex angular frequencies ``omega_k = 2 pi f_k - i gamma_k``."""
        return torch.complex(2.0 * math.pi * self.frequency, -self.decay_rate)

    def reconstruct(self, t, *, real: bool = False) -> torch.Tensor:
        """Evaluate the fitted model at times ``t``.

        With ``real=True`` the model is returned as ``2 * Re(sum_k ...)``,
        which reproduces a real signal when only one member of each conjugate
        pair (and no zero-frequency mode) lies in the window.
        """

        times = torch.as_tensor(t, dtype=torch.float64, device=self.frequency.device)
        phase = torch.exp(-1j * (times[..., None] - self.t0) * self.angular_frequency)
        values = (phase * self.amplitude).sum(dim=-1)
        return 2.0 * values.real if real else values


def _basis_count(frequency_window, dt, half_length, num_basis, density, max_basis):
    if num_basis is not None:
        count = int(num_basis)
        if count < 1:
            raise ValueError("num_basis must be >= 1.")
        return count
    if density <= 0.0:
        raise ValueError("density must be positive.")
    fmin, fmax = frequency_window
    # The natural FDM resolution is 1 / (M dt); density scales the basis
    # spacing relative to it.
    count = math.ceil(density * (fmax - fmin) * half_length * dt)
    return max(8, min(int(max_basis), count))


def _fdm_matrices(signal, phi, half_length):
    """Return ``U^(0..2)`` and the amplitude projections ``f_0(z_j)``.

    ``U^(p)_{jk} = sum_{n,m=0}^{M} z_j^{-n} z_k^{-m} c_{n+m+p}`` with basis
    points ``z_j = exp(-i phi_j)`` on the unit circle, evaluated in closed form
    from chunked ``O(J N)`` sums.
    """

    M = half_length
    device = signal.device
    num_basis = phi.numel()
    s = torch.arange(2 * M + 1, device=device, dtype=torch.float64)
    weight = (M + 1 - (M - s).abs()).to(torch.complex128)
    head = (s <= M).to(torch.complex128)
    tail = 1.0 - head
    columns = []
    for p in range(3):
        c = signal[p : p + 2 * M + 1]
        columns.extend((c * head, c * tail, c * weight))
    rhs = torch.stack(columns, dim=1)

    sums = torch.zeros((num_basis, rhs.shape[1]), dtype=torch.complex128, device=device)
    chunk = max(1, _CHUNK_ELEMENTS // num_basis)
    for begin in range(0, 2 * M + 1, chunk):
        end = min(begin + chunk, 2 * M + 1)
        powers = torch.exp(1j * phi[:, None] * s[None, begin:end])
        sums += powers @ rhs[begin:end]

    z = torch.exp(-1j * phi)
    z_inv_m = torch.exp(1j * phi * M)
    z_m1 = torch.exp(-1j * phi * (M + 1))
    zj = z[:, None]
    zk = z[None, :]
    denominator = zj - zk
    eye = torch.eye(num_basis, dtype=torch.bool, device=device)
    denominator = torch.where(eye, torch.ones_like(denominator), denominator)
    matrices = []
    for p in range(3):
        f = sums[:, 3 * p]
        g = z_m1 * sums[:, 3 * p + 1]
        diagonal = sums[:, 3 * p + 2]
        numerator = (
            zj * f[None, :]
            - zk * f[:, None]
            + z_inv_m[None, :] * g[:, None]
            - z_inv_m[:, None] * g[None, :]
        )
        matrix = numerator / denominator
        matrices.append(torch.where(eye, torch.diag_embed(diagonal), matrix))
    return matrices, sums[:, 0]


def harmonic_inversion(
    signal,
    dt: float,
    frequency_window: tuple[float, float],
    *,
    t0: float = 0.0,
    num_basis: int | None = None,
    density: float = 1.1,
    max_basis: int = 300,
    svd_cutoff: float = 1e-10,
    max_error: float = 1e-4,
    min_relative_amplitude: float = 0.0,
) -> ResonanceModes:
    """Extract resonances from a uniformly sampled signal by filter diagonalization.

    Args:
        signal: 1-D real or complex tensor ``y(t0 + n dt)``. Computation runs
            in ``complex128`` on the signal's device.
        dt: Sample spacing in seconds.
        frequency_window: ``(fmin, fmax)`` in Hz, inside the Nyquist band
            ``[-1/(2 dt), 1/(2 dt)]``. Negative frequencies select
            ``exp(+i 2 pi |f| t)`` components of complex signals.
        t0: Time of the first sample; amplitudes and phases refer to it.
        num_basis: Number of basis frequencies uniformly spaced over the
            window. By default ``ceil(density * (fmax - fmin) * M * dt)``
            clamped to ``[8, max_basis]`` with ``M = (N - 3) // 2``.
        density: Basis density relative to the FDM resolution ``1 / (M dt)``.
        max_basis: Upper bound of the default basis count.
        svd_cutoff: Singular values of the basis overlap ``U^(0)`` below
            ``svd_cutoff * sigma_max`` are discarded before the generalized
            eigenproblem ``U^(1) b = u U^(0) b`` is reduced to a standard one.
        max_error: Modes with fractional frequency-error estimate above this
            value are rejected as spurious. Physical modes of clean or mildly
            noisy signals sit many orders of magnitude below the default.
        min_relative_amplitude: Modes with ``|a_k|`` below this fraction of the
            largest accepted ``|a_k|`` are rejected.

    Returns:
        A :class:`ResonanceModes` record of the accepted modes inside the
        window, sorted by increasing frequency.
    """

    values = torch.as_tensor(signal)
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {tuple(values.shape)}.")
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError("dt must be positive.")
    fmin, fmax = (float(value) for value in frequency_window)
    nyquist = 0.5 / dt
    if not fmin < fmax:
        raise ValueError("frequency_window must satisfy fmin < fmax.")
    if fmin < -nyquist or fmax > nyquist:
        raise ValueError(f"frequency_window must lie inside the Nyquist band +/-{nyquist:.6g} Hz.")
    num_samples = values.numel()
    if num_samples < 9:
        raise ValueError("harmonic inversion needs at least 9 samples.")

    c = values.to(torch.complex128)
    device = c.device
    scale = float(c.abs().max())
    if scale == 0.0 or not math.isfinite(scale):
        raise ValueError("signal must be finite and not identically zero.")
    c = c / scale

    half_length = (num_samples - 3) // 2
    count = _basis_count((fmin, fmax), dt, half_length, num_basis, density, max_basis)
    if count == 1:
        basis_frequencies = torch.tensor([0.5 * (fmin + fmax)], dtype=torch.float64, device=device)
    else:
        basis_frequencies = torch.linspace(fmin, fmax, count, dtype=torch.float64, device=device)
    phi = 2.0 * math.pi * dt * basis_frequencies

    (u0, u1, u2), projection = _fdm_matrices(c, phi, half_length)

    left, sigma, right_h = torch.linalg.svd(u0)
    keep = sigma > float(svd_cutoff) * sigma[0]
    rank = int(keep.sum())
    inv_sqrt = sigma[keep].rsqrt().to(torch.complex128)
    left = left[:, keep]
    right = right_h[keep].mH
    reduced = inv_sqrt[:, None] * (left.mH @ u1 @ right) * inv_sqrt[None, :]
    eigenvalues, eigenvectors = torch.linalg.eig(reduced)
    vectors = right @ (inv_sqrt[:, None] * eigenvectors)

    norm = (vectors * (u0 @ vectors)).sum(dim=0)
    second = (vectors * (u2 @ vectors)).sum(dim=0) / norm
    amplitude = (vectors * projection[:, None]).sum(dim=0).square() / norm * scale

    log_u = torch.log(eigenvalues)
    frequency = -log_u.imag / (2.0 * math.pi * dt)
    decay_rate = -log_u.real / dt
    omega_abs = torch.hypot(2.0 * math.pi * frequency, decay_rate)
    error = torch.log(second / eigenvalues.square()).abs() / (2.0 * dt) / omega_abs

    accepted = (
        torch.isfinite(frequency)
        & torch.isfinite(decay_rate)
        & torch.isfinite(amplitude.abs())
        & (frequency >= fmin)
        & (frequency <= fmax)
        & (error <= float(max_error))
    )
    if min_relative_amplitude > 0.0 and bool(accepted.any()):
        largest = amplitude.abs()[accepted].max()
        accepted &= amplitude.abs() >= float(min_relative_amplitude) * largest

    frequency = frequency[accepted]
    order = torch.argsort(frequency)
    frequency = frequency[order]
    decay_rate = decay_rate[accepted][order]
    return ResonanceModes(
        frequency=frequency,
        decay_rate=decay_rate,
        quality_factor=math.pi * frequency / decay_rate,
        amplitude=amplitude[accepted][order],
        error=error[accepted][order],
        t0=float(t0),
        dt=dt,
        num_basis=count,
        rank=rank,
    )


def _uniform_spacing(t: torch.Tensor) -> float:
    if t.numel() < 2:
        raise ValueError("FieldTimeMonitor payload needs at least two time samples.")
    steps = t[1:] - t[:-1]
    dt = float(steps.mean())
    if not dt > 0.0 or float((steps - dt).abs().max()) > 1e-6 * dt:
        raise ValueError("FieldTimeMonitor samples must be uniformly spaced in time.")
    return dt


def find_resonances(
    payload: Mapping,
    frequency_window: tuple[float, float],
    *,
    component: str | None = None,
    sample: tuple[int, ...] | None = None,
    reduce: str | None = None,
    weights: torch.Tensor | None = None,
    start_time: float | None = None,
    **options,
) -> ResonanceModes:
    """Run :func:`harmonic_inversion` on a ``FieldTimeMonitor`` payload.

    Args:
        payload: The mapping returned by ``Result.monitor(name)`` for a
            ``FieldTimeMonitor`` (``kind == "field_time"``), holding the
            sample times ``t`` and per-component buffers ``components`` of
            shape ``(num_samples, *spatial)``.
        frequency_window: ``(fmin, fmax)`` in Hz.
        component: Recorded component such as ``"Ez"``; optional when the
            monitor recorded a single component.
        sample: Spatial index into a plane or volume buffer selecting one
            point trace.
        reduce: ``"sum"`` or ``"mean"`` over all spatial samples.
        weights: Tensor broadcastable to the spatial shape; the analyzed
            signal is ``sum(weights * field)`` over the spatial axes. Using
            several weighted points avoids missing modes that vanish at a
            single probe location.
        start_time: Discard samples with ``t < start_time``, e.g. the
            source-on transient, so only the free ringdown is fitted.
        **options: Forwarded to :func:`harmonic_inversion`.

    Point payloads need none of ``sample``/``reduce``/``weights``; plane and
    volume payloads need exactly one. Amplitudes refer to the first retained
    sample time. The payload clock ``t = n dt`` is the E-field clock; H traces
    sit half a step earlier on the Yee grid, which shifts only their phases.
    """

    if not isinstance(payload, Mapping) or payload.get("kind") != "field_time":
        raise ValueError("find_resonances expects a FieldTimeMonitor payload from Result.monitor(...).")
    buffers = payload["components"]
    if component is None:
        if len(buffers) != 1:
            raise ValueError(f"component is required; recorded components are {tuple(buffers)}.")
        component = next(iter(buffers))
    name = str(component)
    if name not in buffers:
        name = name[:1].upper() + name[1:].lower()
    if name not in buffers:
        raise KeyError(f"Component {component!r} not recorded; available: {tuple(buffers)}.")
    field = buffers[name]
    t = torch.as_tensor(payload["t"], dtype=torch.float64, device=field.device)

    spatial_ndim = field.ndim - 1
    selectors = sum(option is not None for option in (sample, reduce, weights))
    if selectors > 1:
        raise ValueError("Pass at most one of sample, reduce, weights.")
    if spatial_ndim == 0:
        if selectors:
            raise ValueError("Point FieldTimeMonitor traces take no spatial selection.")
        trace = field
    elif sample is not None:
        index = tuple(int(i) for i in sample)
        if len(index) != spatial_ndim:
            raise ValueError(f"sample needs {spatial_ndim} indices for spatial shape {tuple(field.shape[1:])}.")
        trace = field[(slice(None),) + index]
    elif reduce is not None:
        dims = tuple(range(1, field.ndim))
        if reduce == "sum":
            trace = field.sum(dim=dims)
        elif reduce == "mean":
            trace = field.mean(dim=dims)
        else:
            raise ValueError("reduce must be 'sum' or 'mean'.")
    elif weights is not None:
        weight = torch.as_tensor(weights, device=field.device)
        trace = (field * weight).sum(dim=tuple(range(1, field.ndim)))
    else:
        raise ValueError(
            f"Plane/volume FieldTimeMonitor traces of spatial shape {tuple(field.shape[1:])} "
            "need one of sample, reduce, weights."
        )

    if start_time is not None:
        keep = t >= float(start_time)
        t = t[keep]
        trace = trace[keep]
    dt = _uniform_spacing(t)
    return harmonic_inversion(trace, dt, frequency_window, t0=float(t[0]), **options)


__all__ = ["ResonanceModes", "find_resonances", "harmonic_inversion"]
