"""Main-lobe, sidelobe, null, and front-to-back metrics of radiation patterns.

Index bookkeeping (peak, lobe boundaries, local extrema) runs on one detached
host copy of the pattern; every reported value is then recomputed from the
input tensor with torch interpolation formulas, so levels and angles stay on
the input device and remain differentiable with respect to the pattern while
the sampled lobe structure is unchanged.

Angles are radians. Levels in dB are relative to the (interpolated) main-lobe
peak, so sidelobe and null levels are negative and the front-to-back ratio is
positive. Quantities that a pattern does not define (no half-power crossing
before the pattern edge, no null, no sample outside the main lobe, ...) are
returned as NaN rather than raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

_TWO_PI = 2.0 * math.pi
_HALF_POWER_AMPLITUDE = math.sqrt(0.5)


@dataclass(frozen=True)
class LobeMetrics:
    """Lobe metrics of one pattern cut.

    ``peak_value`` is the interpolated peak in the input power units.
    ``half_power_angles`` and ``first_null_angles`` are ``[left, right]``
    boundaries of the main lobe; their differences are the -3 dB
    (``half_power_beamwidth``) and first-null (``first_null_beamwidth``)
    beamwidths. ``first_sidelobe_db`` is the larger of the two sidelobes
    adjacent to the main lobe; ``max_sidelobe_db`` the largest pattern level
    outside the first nulls. ``null_angles`` lists every interior local
    minimum of the cut with its sampled level ``null_levels_db``. On periodic
    cuts reported angles are wrapped into ``[angles[0], angles[0] + 2*pi)``.
    """

    peak_angle: torch.Tensor
    peak_value: torch.Tensor
    half_power_angles: torch.Tensor
    half_power_beamwidth: torch.Tensor
    first_null_angles: torch.Tensor
    first_null_beamwidth: torch.Tensor
    first_sidelobe_db: torch.Tensor
    max_sidelobe_db: torch.Tensor
    max_sidelobe_angle: torch.Tensor
    sidelobe_angles: torch.Tensor
    sidelobe_levels_db: torch.Tensor
    null_angles: torch.Tensor
    null_levels_db: torch.Tensor
    front_to_back_db: torch.Tensor
    periodic: bool

    @property
    def half_power_beamwidth_deg(self) -> torch.Tensor:
        return torch.rad2deg(self.half_power_beamwidth)

    @property
    def first_null_beamwidth_deg(self) -> torch.Tensor:
        return torch.rad2deg(self.first_null_beamwidth)


@dataclass(frozen=True)
class PatternLobeMetrics:
    """Lobe metrics of a full-sphere ``[T, P]`` pattern.

    ``elevation`` is the great-circle cut through the peak in the plane
    ``phi = peak_phi``, parameterized by the signed polar angle
    ``alpha in [-pi, pi)`` (``alpha = theta`` on the ``peak_phi`` half-plane and
    ``alpha = -theta`` on the opposite half-plane, taken at the sampled phi
    nearest ``peak_phi + pi``). ``azimuth`` is the conical cut at
    ``theta = peak_theta`` parameterized by phi; its widths are phi spans, not
    great-circle arcs. ``main_lobe_mask`` marks the samples reachable from the
    peak along non-increasing grid paths; ``max_sidelobe_db`` is the largest
    level outside it over the whole sphere. ``front_to_back_db`` compares the
    peak with the antipodal direction along the elevation great circle.
    """

    peak_theta: torch.Tensor
    peak_phi: torch.Tensor
    peak_value: torch.Tensor
    main_lobe_mask: torch.Tensor
    max_sidelobe_db: torch.Tensor
    max_sidelobe_theta: torch.Tensor
    max_sidelobe_phi: torch.Tensor
    front_to_back_db: torch.Tensor
    elevation: LobeMetrics
    azimuth: LobeMetrics


def _to_power(pattern, scale: str) -> torch.Tensor:
    if not isinstance(pattern, torch.Tensor):
        raise TypeError("pattern must be a torch.Tensor.")
    if pattern.is_complex():
        return torch.abs(pattern).square()
    if not pattern.dtype.is_floating_point:
        raise TypeError("pattern must be a floating-point or complex tensor.")
    if scale == "power":
        return pattern
    if scale == "field":
        return pattern.square()
    if scale == "db":
        return torch.pow(10.0, pattern / 10.0)
    raise ValueError("scale must be 'power', 'field', or 'db'.")


def _validate_power(power: torch.Tensor) -> None:
    if not bool(torch.all(torch.isfinite(power))):
        raise ValueError("pattern must contain only finite values.")
    if bool(torch.any(power < 0.0)):
        raise ValueError("power pattern must be non-negative.")
    if not bool(torch.amax(power) > 0.0):
        raise ValueError("pattern must not be identically zero.")


def _db(ratio: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(ratio)


def _safe_log_power(power: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(torch.clamp_min(power, torch.finfo(power.dtype).tiny))


def _parabolic_vertex(x0, x1, x2, f0, f1, f2):
    # Vertex of the parabola through three (possibly non-uniform) samples.
    d1 = x1 - x0
    d2 = x1 - x2
    numerator = d1.square() * (f1 - f2) - d2.square() * (f1 - f0)
    denominator = d1 * (f1 - f2) - d2 * (f1 - f0)
    concave = denominator > 0.0
    safe_denominator = torch.where(concave, denominator, torch.ones_like(denominator))
    vertex = torch.where(concave, x1 - 0.5 * numerator / safe_denominator, x1)
    vertex = torch.minimum(torch.maximum(vertex, torch.minimum(x0, x2)), torch.maximum(x0, x2))
    value = (
        f0 * (vertex - x1) * (vertex - x2) / ((x0 - x1) * (x0 - x2))
        + f1 * (vertex - x0) * (vertex - x2) / ((x1 - x0) * (x1 - x2))
        + f2 * (vertex - x0) * (vertex - x1) / ((x2 - x0) * (x2 - x1))
    )
    value = torch.where(concave, torch.maximum(value, f1), f1)
    return vertex, value


def _refined_maxima(angles, log_power, index: np.ndarray, interior: np.ndarray):
    """Parabolic (dB-domain) refinement of local maxima; edge maxima stay sampled."""

    device = angles.device
    center = torch.as_tensor(index, device=device, dtype=torch.long)
    left = torch.as_tensor(np.where(interior, index - 1, index), device=device, dtype=torch.long)
    right = torch.as_tensor(np.where(interior, index + 1, index), device=device, dtype=torch.long)
    x1, f1 = angles[center], log_power[center]
    if not interior.any():
        return x1, f1
    mask = torch.as_tensor(interior, device=device)
    # Substitute a harmless symmetric stencil on edge samples to avoid 0/0.
    x0 = torch.where(mask, angles[left], x1 - 1.0)
    x2 = torch.where(mask, angles[right], x1 + 1.0)
    f0 = torch.where(mask, log_power[left], f1 - 1.0)
    f2 = torch.where(mask, log_power[right], f1 - 1.0)
    vertex, value = _parabolic_vertex(x0, x1, x2, f0, f1, f2)
    return torch.where(mask, vertex, x1), torch.where(mask, value, f1)


def _quadratic_root(nodes, target, estimate, lower, upper, iterations: int = 4):
    """Newton solve of ``q(x) = target`` for the quadratic through three ``(x, y)`` nodes."""

    for _ in range(iterations):
        value = torch.zeros_like(estimate)
        slope = torch.zeros_like(estimate)
        for i, (xi, yi) in enumerate(nodes):
            xa, xb = (xj for j, (xj, _) in enumerate(nodes) if j != i)
            denominator = (xi - xa) * (xi - xb)
            value = value + yi * (estimate - xa) * (estimate - xb) / denominator
            slope = slope + yi * (2.0 * estimate - xa - xb) / denominator
        sloped = slope != 0.0
        safe_slope = torch.where(sloped, slope, torch.ones_like(slope))
        step = torch.where(sloped, (value - target) / safe_slope, torch.zeros_like(slope))
        estimate = torch.minimum(torch.maximum(estimate - step, lower), upper)
    return estimate


def _refined_minima(angles, amplitude, index: np.ndarray):
    """Refined directions of sampled local minima of the field amplitude.

    Two local models are compared: a genuine null, where the field changes
    sign across the gap next to the smaller neighbour (zero of the quadratic
    through the signed samples), and a shallow non-zero minimum (vertex of
    the quadratic through the unsigned amplitudes). The model with the
    smaller second divided difference, i.e. the smoother interpretation,
    is used. Both are exact for locally quadratic data.
    """

    device = angles.device
    center = torch.as_tensor(index, device=device, dtype=torch.long)
    x_left, x_mid, x_right = angles[center - 1], angles[center], angles[center + 1]
    a_left, a_mid, a_right = amplitude[center - 1], amplitude[center], amplitude[center + 1]
    right_gap = a_right < a_left
    signed_mid = torch.where(right_gap, a_mid, -a_mid)
    lower = torch.where(right_gap, x_mid, x_left)
    upper = torch.where(right_gap, x_right, x_mid)
    near = torch.where(right_gap, a_mid, a_left)
    far = torch.where(right_gap, a_right, a_mid)
    estimate = lower + near / torch.clamp_min(near + far, torch.finfo(near.dtype).tiny) * (upper - lower)
    nodes = ((x_left, a_left), (x_mid, signed_mid), (x_right, -a_right))
    zero = _quadratic_root(nodes, 0.0, estimate, lower, upper)
    vertex, _ = _parabolic_vertex(x_left, x_mid, x_right, -a_left, -a_mid, -a_right)

    def curvature(y_left, y_mid, y_right):
        return torch.abs(
            ((y_right - y_mid) / (x_right - x_mid) - (y_mid - y_left) / (x_mid - x_left)) / (x_right - x_left)
        )

    genuine = curvature(a_left, signed_mid, -a_right) <= curvature(a_left, a_mid, a_right)
    return torch.where(genuine, zero, vertex)


def _crossing(angles, amplitude, low: int, high: int, extra: int | None):
    """Half-power crossing between a sample below (``low``) and above (``high``).

    With a third sample ``extra`` the field amplitude is interpolated by the
    quadratic through all three samples; otherwise linearly.
    """

    x_low, x_high = angles[low], angles[high]
    a_low, a_high = amplitude[low], amplitude[high]
    estimate = x_low + (_HALF_POWER_AMPLITUDE - a_low) / (a_high - a_low) * (x_high - x_low)
    if extra is None:
        return estimate
    nodes = ((x_low, a_low), (x_high, a_high), (angles[extra], amplitude[extra]))
    lower, upper = torch.minimum(x_low, x_high), torch.maximum(x_low, x_high)
    return _quadratic_root(nodes, _HALF_POWER_AMPLITUDE, estimate, lower, upper)


def _interpolate(angles, power, value: torch.Tensor) -> torch.Tensor:
    position = int(torch.searchsorted(angles.detach(), value.detach().reshape(1)).item())
    position = min(max(position, 1), angles.numel() - 1)
    x0, x1 = angles[position - 1], angles[position]
    fraction = (value - x0) / (x1 - x0)
    return power[position - 1] + fraction * (power[position] - power[position - 1])


def _local_maxima(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Indices of local maxima in ``values[lo:hi+1]``; plateaus report their last sample."""

    if hi < lo:
        return np.empty(0, dtype=np.int64)
    index = np.arange(lo, hi + 1)
    left = np.where(index > 0, values[np.maximum(index - 1, 0)], -np.inf)
    right = np.where(index < values.size - 1, values[np.minimum(index + 1, values.size - 1)], -np.inf)
    return index[(values[index] >= left) & (values[index] > right)]


def lobe_metrics(
    angles: torch.Tensor,
    pattern: torch.Tensor,
    *,
    scale: str = "power",
    periodic: bool = False,
) -> LobeMetrics:
    """Measure the main lobe, sidelobes, nulls and front-to-back ratio of a cut.

    ``angles`` must be a strictly increasing 1D tensor matching ``pattern``.
    Complex patterns are field phasors; real patterns are interpreted by
    ``scale`` (``"power"``, ``"field"`` amplitude, or ``"db"``). With
    ``periodic=True`` the samples cover one full circle; a duplicated
    endpoint at ``angles[0] + 2*pi`` is dropped. The main lobe extends from
    the global peak (first maximum on ties) along non-increasing samples to
    the first local minimum on each side. Peak and sidelobe levels use
    parabolic interpolation in dB, half-power crossings use quadratic
    interpolation of the field amplitude, and nulls use the zero of a
    quadratic through the signed field amplitude (or the amplitude minimum
    for shallow nulls). A rising sample inside a noisy main-lobe flank ends
    the main lobe there.
    """

    if not isinstance(angles, torch.Tensor):
        raise TypeError("angles must be a torch.Tensor.")
    power = _to_power(pattern, scale)
    if angles.ndim != 1 or power.shape != angles.shape:
        raise ValueError("angles and pattern must be matching 1D tensors.")
    if angles.is_complex() or not angles.dtype.is_floating_point:
        raise TypeError("angles must be a real floating-point tensor.")
    if angles.device != power.device:
        raise ValueError("angles and pattern must share a device.")
    if angles.numel() < 3:
        raise ValueError("A pattern cut needs at least three samples.")
    if not bool(torch.all(angles[1:] > angles[:-1])):
        raise ValueError("angles must be strictly increasing.")
    _validate_power(power)
    angles = angles.to(power.dtype)

    if periodic:
        span = float((angles[-1] - angles[0]).item())
        tolerance = 1e-9 * _TWO_PI
        if span > _TWO_PI + tolerance:
            raise ValueError("periodic cuts must span at most 2*pi.")
        if abs(span - _TWO_PI) <= tolerance:
            angles, power = angles[:-1], power[:-1]
        if angles.numel() < 3:
            raise ValueError("A periodic cut needs at least three distinct samples.")
    count = angles.numel()
    if periodic:
        ext_angles = torch.cat((angles - _TWO_PI, angles, angles + _TWO_PI))
        ext_power = torch.cat((power, power, power))
    else:
        ext_angles, ext_power = angles, power

    host = ext_power.detach().to("cpu", torch.float64).numpy()
    base_peak = int(np.argmax(host[count : 2 * count] if periodic else host))
    center = base_peak + count if periodic else base_peak
    lower = center - (count - 1) if periodic else 0
    upper = center + (count - 1) if periodic else count - 1

    log_power = _safe_log_power(ext_power)
    interior_peak = np.array([lower < center < upper])
    peak_angle, peak_log = _refined_maxima(
        ext_angles, log_power, np.array([center]), interior_peak
    )
    peak_angle, peak_log = peak_angle[0], peak_log[0]
    peak_value = torch.pow(10.0, peak_log / 10.0)
    amplitude = torch.sqrt(
        torch.clamp_min(ext_power / peak_value, torch.finfo(power.dtype).tiny)
    )
    host_amplitude = np.sqrt(np.maximum(host / float(peak_value.detach().item()), 0.0))

    left_null = center
    while left_null > lower and host[left_null - 1] <= host[left_null]:
        left_null -= 1
    right_null = center
    while right_null < upper and host[right_null + 1] <= host[right_null]:
        right_null += 1
    covers_all = periodic and (right_null - left_null) >= count - 1
    left_found = left_null > lower and not covers_all
    right_found = right_null < upper and not covers_all

    nan = torch.full((), float("nan"), device=power.device, dtype=power.dtype)

    def half_power_side(step: int, stop: int):
        index = center
        while index != stop:
            nxt = index + step
            if host_amplitude[nxt] < _HALF_POWER_AMPLITUDE:
                extra = index - step
                if not lower <= extra <= upper:
                    extra = None
                return _crossing(ext_angles, amplitude, nxt, index, extra)
            index = nxt
        return nan

    half_left = half_power_side(-1, left_null)
    half_right = half_power_side(+1, right_null)
    half_power_angles = torch.stack((half_left, half_right))

    null_left = _refined_minima(ext_angles, amplitude, np.array([left_null]))[0] if left_found else nan
    null_right = _refined_minima(ext_angles, amplitude, np.array([right_null]))[0] if right_found else nan
    first_null_angles = torch.stack((null_left, null_right))

    # Sidelobes: local maxima strictly outside the first nulls.
    if covers_all:
        sidelobe_index = np.empty(0, dtype=np.int64)
    elif periodic:
        sidelobe_index = _local_maxima(host, right_null + 1, left_null + count - 1)
    else:
        sidelobe_index = np.concatenate(
            (_local_maxima(host, 0, left_null - 1), _local_maxima(host, right_null + 1, count - 1))
        )
    sidelobe_index = sidelobe_index.astype(np.int64)
    edge_limit = ext_angles.numel() - 1
    interior = (sidelobe_index > 0) & (sidelobe_index < edge_limit)
    if sidelobe_index.size:
        sidelobe_angles, sidelobe_log = _refined_maxima(ext_angles, log_power, sidelobe_index, interior)
        sidelobe_levels = sidelobe_log - peak_log
        strongest = int(torch.argmax(sidelobe_levels.detach()).item())
        max_sidelobe_db = sidelobe_levels[strongest]
        max_sidelobe_angle = sidelobe_angles[strongest]
        first_candidates = []
        if periodic:
            first_candidates = [0, sidelobe_index.size - 1]
        else:
            before = np.nonzero(sidelobe_index < left_null)[0]
            after = np.nonzero(sidelobe_index > right_null)[0]
            if before.size:
                first_candidates.append(int(before[-1]))
            if after.size:
                first_candidates.append(int(after[0]))
        first_sidelobe_db = torch.amax(sidelobe_levels[first_candidates])
    else:
        sidelobe_angles = power.new_zeros(0)
        sidelobe_levels = power.new_zeros(0)
        max_sidelobe_db = nan
        max_sidelobe_angle = nan
        first_sidelobe_db = nan

    # Interior local minima over one period (or the whole open cut).
    if periodic:
        span_index = np.arange(count, 2 * count)
    else:
        span_index = np.arange(1, count - 1)
    is_min = (host[span_index] < host[span_index - 1]) & (host[span_index] <= host[span_index + 1])
    null_index = span_index[is_min].astype(np.int64)
    if null_index.size:
        null_angles = _refined_minima(ext_angles, amplitude, null_index)
        null_levels = _db(ext_power[torch.as_tensor(null_index, device=power.device)] / peak_value)
    else:
        null_angles = power.new_zeros(0)
        null_levels = power.new_zeros(0)

    back_angle = peak_angle + math.pi
    if periodic:
        start = angles[0]
        back_angle = start + torch.remainder(back_angle - start, _TWO_PI)
        back_power = _interpolate(ext_angles, ext_power, back_angle)
    else:
        back_power = None
        for candidate in (peak_angle + math.pi, peak_angle - math.pi):
            if float(angles[0].item()) <= float(candidate.item()) <= float(angles[-1].item()):
                back_power = _interpolate(ext_angles, ext_power, candidate)
                break
    if back_power is None:
        front_to_back_db = nan
    else:
        front_to_back_db = _db(peak_value / torch.clamp_min(back_power, 0.0))

    def wrap(value: torch.Tensor) -> torch.Tensor:
        if not periodic:
            return value
        start = angles[0]
        return start + torch.remainder(value - start, _TWO_PI)

    return LobeMetrics(
        peak_angle=wrap(peak_angle),
        peak_value=peak_value,
        half_power_angles=wrap(half_power_angles),
        half_power_beamwidth=half_right - half_left,
        first_null_angles=wrap(first_null_angles),
        first_null_beamwidth=null_right - null_left,
        first_sidelobe_db=first_sidelobe_db,
        max_sidelobe_db=max_sidelobe_db,
        max_sidelobe_angle=wrap(max_sidelobe_angle),
        sidelobe_angles=wrap(sidelobe_angles),
        sidelobe_levels_db=sidelobe_levels,
        null_angles=wrap(null_angles),
        null_levels_db=null_levels,
        front_to_back_db=front_to_back_db,
        periodic=bool(periodic),
    )


def _sphere_vectors(theta, phi, shape) -> tuple[torch.Tensor, torch.Tensor]:
    if theta.ndim == 1 and phi.ndim == 1:
        theta_vector, phi_vector = theta, phi
    elif theta.ndim == 2 and phi.ndim == 2 and theta.shape == phi.shape:
        theta_vector, phi_vector = theta[:, 0], phi[0, :]
        if not torch.allclose(theta, theta_vector[:, None]) or not torch.allclose(phi, phi_vector[None, :]):
            raise ValueError("theta/phi grids must be meshgrid(theta, phi, indexing='ij').")
    else:
        raise ValueError("theta and phi must be 1D vectors or matching 2D grids.")
    if tuple(shape) != (theta_vector.numel(), phi_vector.numel()):
        raise ValueError("pattern must have shape [T, P] matching theta and phi.")
    if theta_vector.numel() < 3 or phi_vector.numel() < 3:
        raise ValueError("A sphere pattern needs at least three theta and phi samples.")
    if not bool(torch.all(theta_vector[1:] > theta_vector[:-1])) or not bool(
        torch.all(phi_vector[1:] > phi_vector[:-1])
    ):
        raise ValueError("theta and phi samples must be strictly increasing.")
    tolerance = 1e-9
    if abs(float(theta_vector[0].item())) > tolerance or abs(float(theta_vector[-1].item()) - math.pi) > tolerance:
        raise ValueError("theta must span [0, pi].")
    return theta_vector, phi_vector


def _main_lobe_region(power: np.ndarray, peak: tuple[int, int], tolerance: float) -> np.ndarray:
    """Grid samples reachable from ``peak`` along non-increasing neighbour steps.

    ``power`` is an open periodic-in-phi ``[T, P]`` host array whose first and
    last rows are the poles.
    """

    region = np.zeros(power.shape, dtype=bool)
    region[peak] = True
    limit = power * (1.0 + tolerance)
    while True:
        grow = region.copy()
        for shift in (1, -1):
            grow |= np.roll(region, shift, axis=1) & (power <= np.roll(limit, shift, axis=1))
        grow[1:] |= region[:-1] & (power[1:] <= limit[:-1])
        grow[:-1] |= region[1:] & (power[:-1] <= limit[1:])
        for row in (0, -1):
            # Each pole row is one physical direction.
            if grow[row].any():
                grow[row] = True
        if np.array_equal(grow, region):
            return region
        region = grow


def pattern_lobe_metrics(
    theta: torch.Tensor,
    phi: torch.Tensor,
    pattern: torch.Tensor,
    *,
    scale: str = "power",
    descent_tolerance: float = 0.0,
) -> PatternLobeMetrics:
    """Lobe metrics of a full-sphere pattern on a ``(theta, phi)`` grid.

    ``theta`` spans ``[0, pi]``; ``phi`` covers one full circle, either with a
    duplicated ``2*pi`` endpoint (the ``AntennaData`` convention) or as an
    open periodic grid. ``pattern`` has shape ``[T, P]`` and follows the same
    ``scale`` rules as :func:`lobe_metrics`. ``descent_tolerance`` lets the
    main-lobe flood fill climb relative bumps up to that fraction, which
    makes the sphere-wide sidelobe search robust to sampling noise. A peak on
    a pole yields a constant azimuth cut whose widths are NaN.
    """

    power = _to_power(pattern, scale)
    if power.ndim != 2:
        raise ValueError("pattern must have shape [T, P].")
    theta_vector, phi_vector = _sphere_vectors(theta, phi, power.shape)
    theta_vector = theta_vector.to(device=power.device, dtype=power.dtype)
    phi_vector = phi_vector.to(device=power.device, dtype=power.dtype)
    _validate_power(power)
    if descent_tolerance < 0.0 or not math.isfinite(descent_tolerance):
        raise ValueError("descent_tolerance must be a non-negative finite number.")

    phi_span = float((phi_vector[-1] - phi_vector[0]).item())
    duplicated = abs(phi_span - _TWO_PI) <= 1e-9 * _TWO_PI
    if phi_span > _TWO_PI * (1.0 + 1e-9):
        raise ValueError("phi must span at most 2*pi.")
    open_phi = phi_vector[:-1] if duplicated else phi_vector
    open_power = power[:, :-1] if duplicated else power

    flat = int(torch.argmax(open_power.detach()).item())
    peak = (flat // open_power.shape[1], flat % open_power.shape[1])
    peak_theta = theta_vector[peak[0]]
    peak_phi = open_phi[peak[1]]

    opposite = torch.remainder(open_phi - peak_phi, _TWO_PI) - math.pi
    back_column = int(torch.argmin(torch.abs(opposite.detach())).item())
    alpha = torch.cat((-theta_vector.flip(0)[:-1], theta_vector))
    elevation_cut = torch.cat((open_power[:, back_column].flip(0)[:-1], open_power[:, peak[1]]))
    elevation = lobe_metrics(alpha, elevation_cut, periodic=True)
    azimuth = lobe_metrics(open_phi, open_power[peak[0], :], periodic=True)

    host = open_power.detach().to("cpu", torch.float64).numpy()
    region = _main_lobe_region(host, peak, descent_tolerance)
    outside = ~region
    if outside.any():
        side_flat = int(np.argmax(np.where(outside, host, -1.0)))
        side = (side_flat // host.shape[1], side_flat % host.shape[1])
        max_sidelobe_db = _db(open_power[side] / open_power[peak])
        max_sidelobe_theta = theta_vector[side[0]]
        max_sidelobe_phi = open_phi[side[1]]
    else:
        max_sidelobe_db = torch.full((), float("nan"), device=power.device, dtype=power.dtype)
        max_sidelobe_theta = max_sidelobe_db
        max_sidelobe_phi = max_sidelobe_db
    region = torch.as_tensor(region, device=power.device)
    if duplicated:
        region = torch.cat((region, region[:, :1]), dim=1)

    return PatternLobeMetrics(
        peak_theta=peak_theta,
        peak_phi=peak_phi,
        peak_value=open_power[peak],
        main_lobe_mask=region,
        max_sidelobe_db=max_sidelobe_db,
        max_sidelobe_theta=max_sidelobe_theta,
        max_sidelobe_phi=max_sidelobe_phi,
        front_to_back_db=elevation.front_to_back_db,
        elevation=elevation,
        azimuth=azimuth,
    )


__all__ = ["LobeMetrics", "PatternLobeMetrics", "lobe_metrics", "pattern_lobe_metrics"]
