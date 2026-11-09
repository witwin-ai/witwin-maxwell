"""Analytic array factor and pattern multiplication for antenna arrays.

The array factor follows the solver far-field convention (peak phasors with
``exp(-i*omega*t)`` time dependence and outgoing ``exp(+i*k*r)`` waves): an
element at ``r_n`` contributes ``exp(-i*k*r_hat.r_n)``, so

    AF(theta, phi) = sum_n w_n exp(-i k r_hat(theta, phi) . r_n).

Progressive-phase steering toward ``r_hat_0`` multiplies the weights by
``exp(+i*k*r_hat_0.r_n)``, the same convention as
``BeamCodebook.from_scan_angles``. Total patterns use pattern multiplication
with one shared element pattern and neglect mutual coupling; the coupled
full-wave array workflow is ``ArrayBasisData``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..antenna import AntennaData
from ..constants import C_0, ETA_0
from .antenna import _angular_grid, _trapz_weights
from .lobes import PatternLobeMetrics, pattern_lobe_metrics


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _validate_positions(positions) -> torch.Tensor:
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor of shape [N, 3].")
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("positions must have shape [N, 3] with N >= 1.")
    if positions.is_complex() or not positions.dtype.is_floating_point:
        raise TypeError("positions must be a real floating-point tensor.")
    return positions


def _real(value, *, name: str, device, dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError(f"{name} must be real.")
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}.")
        return value.to(dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _wave_number(frequency, *, device, dtype) -> torch.Tensor:
    frequency = _real(frequency, name="frequency", device=device, dtype=dtype)
    if frequency.ndim > 1:
        raise ValueError("frequency must be a scalar or a 1D [F] tensor.")
    if not bool(torch.all(frequency > 0.0)):
        raise ValueError("frequency must be strictly positive.")
    return 2.0 * math.pi * frequency / C_0


def _unit_directions(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    return torch.stack(
        (sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)), dim=-1
    )


def rectangular_lattice(
    nx: int,
    ny: int = 1,
    dx: float = 0.0,
    dy: float | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``[nx*ny, 3]`` element positions of a centred rectangular lattice.

    Elements lie in the ``z = 0`` plane with spacings ``dx`` / ``dy`` (metres,
    ``dy`` defaults to ``dx``) and row-major ``(ix, iy)`` order, so
    ``rectangular_taper(nx, ny, ...).reshape(-1)`` lines up with the rows. A
    linear array along ``x`` is ``rectangular_lattice(n, 1, d)``.
    """

    for name, value in (("nx", nx), ("ny", ny)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    dy = dx if dy is None else dy
    if not math.isfinite(dx) or not math.isfinite(dy) or dx < 0.0 or dy < 0.0:
        raise ValueError("dx and dy must be non-negative finite spacings.")
    x = (torch.arange(int(nx), device=device, dtype=dtype) - 0.5 * (nx - 1)) * dx
    y = (torch.arange(int(ny), device=device, dtype=dtype) - 0.5 * (ny - 1)) * dy
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros_like(grid_x).reshape(-1)), dim=-1)


def steering_weights(
    positions: torch.Tensor,
    theta0,
    phi0,
    *,
    frequency,
) -> torch.Tensor:
    """Progressive phases ``exp(+i k r_hat_0 . r_n)`` that co-phase ``(theta0, phi0)``.

    Returns ``[N]`` for a scalar ``frequency`` and ``[F, N]`` for a 1D one.
    Steering angles may be tensors carrying gradients.
    """

    positions = _validate_positions(positions)
    device, dtype = positions.device, positions.dtype
    k = _wave_number(frequency, device=device, dtype=dtype)
    theta0 = _real(theta0, name="theta0", device=device, dtype=dtype)
    phi0 = _real(phi0, name="phi0", device=device, dtype=dtype)
    if theta0.ndim or phi0.ndim:
        raise ValueError("theta0 and phi0 must be scalars.")
    projection = positions @ _unit_directions(theta0, phi0)
    phase = k[..., None] * projection if k.ndim else k * projection
    return torch.exp(1j * phase.to(_complex_dtype(dtype)))


def _resolve_weights(weights, *, count: int, k: torch.Tensor, device, dtype) -> torch.Tensor:
    complex_dtype = _complex_dtype(dtype)
    if weights is None:
        weights = torch.ones(count, device=device, dtype=complex_dtype)
    elif isinstance(weights, torch.Tensor):
        if weights.device != device:
            raise ValueError(f"weights must be on device {device}.")
        weights = weights.to(complex_dtype)
    else:
        weights = torch.as_tensor(weights, device=device).to(complex_dtype)
    if weights.shape[-1:] != (count,):
        raise ValueError("weights must have the element axis last with length N.")
    if k.ndim == 0 and weights.ndim != 1:
        raise ValueError("weights must have shape [N] for a scalar frequency.")
    if k.ndim == 1 and weights.shape not in ((count,), (k.numel(), count)):
        raise ValueError("weights must have shape [N] or [F, N] for F frequencies.")
    return weights


def array_factor(
    positions: torch.Tensor,
    weights,
    theta,
    phi,
    *,
    frequency,
    steering: tuple | None = None,
) -> torch.Tensor:
    """Evaluate the complex array factor on arbitrary broadcast ``theta/phi`` shapes.

    ``positions`` is ``[N, 3]`` in metres, ``weights`` complex ``[N]`` (or
    ``[F, N]`` with a 1D ``frequency``; ``None`` means uniform unit weights).
    ``steering=(theta0, phi0)`` multiplies the weights by
    :func:`steering_weights`. The result has the broadcast angle shape, with
    a leading ``F`` axis when ``frequency`` is 1D. Differentiable with respect
    to weights, positions and steering angles.
    """

    positions = _validate_positions(positions)
    device, dtype = positions.device, positions.dtype
    k = _wave_number(frequency, device=device, dtype=dtype)
    weights = _resolve_weights(weights, count=positions.shape[0], k=k, device=device, dtype=dtype)
    if steering is not None:
        weights = weights * steering_weights(positions, *steering, frequency=frequency)
    theta = _real(theta, name="theta", device=device, dtype=dtype)
    phi = _real(phi, name="phi", device=device, dtype=dtype)
    theta, phi = torch.broadcast_tensors(theta, phi)
    projection = _unit_directions(theta, phi) @ positions.transpose(0, 1)
    if k.ndim == 0:
        phase = k * projection
        return torch.sum(torch.exp(-1j * phase.to(weights.dtype)) * weights, dim=-1)
    phase = k.reshape(-1, *([1] * projection.ndim)) * projection[None]
    if weights.ndim == 2:
        weights = weights.reshape(weights.shape[0], *([1] * theta.ndim), weights.shape[1])
    return torch.sum(torch.exp(-1j * phase.to(weights.dtype)) * weights, dim=-1)


@dataclass(frozen=True)
class ArrayPatternData:
    """Pattern-multiplied analytic array pattern on a full-sphere grid.

    Pattern tensors use ``[F, T, P]`` order with an explicit frequency axis.
    ``element_kind`` is ``"isotropic"`` (unit radiation intensity per unit
    weight), ``"scalar"`` (user field pattern ``g``, ``U = |AF g|^2``) or
    ``"vector"`` (``E_theta/E_phi`` from an antenna record,
    ``U = r^2 (|AF E_theta|^2 + |AF E_phi|^2) / (2 eta)``). ``gain`` and
    ``realized_gain`` are only defined for an ``AntennaData`` element, under
    the isolated-element (no mutual coupling) assumption: the array accepts
    ``sum_n |w_n|^2 * p_accepted`` and is driven with
    ``sum_n |w_n|^2 * p_incident``.
    """

    frequencies: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    positions: torch.Tensor
    weights: torch.Tensor
    array_factor: torch.Tensor
    e_theta: torch.Tensor | None
    e_phi: torch.Tensor | None
    radiation_intensity: torch.Tensor
    p_rad: torch.Tensor
    directivity: torch.Tensor
    gain: torch.Tensor | None
    realized_gain: torch.Tensor | None
    element_kind: str

    @property
    def device(self) -> torch.device:
        return self.frequencies.device

    @property
    def directivity_db(self) -> torch.Tensor:
        return 10.0 * torch.log10(self.directivity)

    @property
    def directivity_max(self) -> torch.Tensor:
        return torch.amax(self.directivity, dim=(-2, -1))

    @property
    def array_factor_db(self) -> torch.Tensor:
        """``|AF|^2`` in dB relative to its per-frequency peak."""

        power = torch.abs(self.array_factor).square()
        return 10.0 * torch.log10(power / torch.amax(power, dim=(-2, -1), keepdim=True))

    def lobe_metrics(
        self,
        frequency_index: int = 0,
        *,
        quantity: str = "directivity",
        descent_tolerance: float = 0.0,
    ) -> PatternLobeMetrics:
        """Sphere lobe metrics of ``directivity``, ``radiation_intensity`` or ``array_factor``."""

        if quantity not in ("directivity", "radiation_intensity", "array_factor"):
            raise ValueError("quantity must be 'directivity', 'radiation_intensity', or 'array_factor'.")
        pattern = getattr(self, quantity)[frequency_index]
        return pattern_lobe_metrics(
            self.theta, self.phi, pattern, descent_tolerance=descent_tolerance
        )


def _element_from_mapping(element: Mapping, device):
    for key in ("theta", "phi", "E_theta", "E_phi", "frequency"):
        if key not in element:
            raise ValueError(f"far-field record is missing {key!r}.")
    e_theta, e_phi = element["E_theta"], element["E_phi"]
    if not isinstance(e_theta, torch.Tensor) or not isinstance(e_phi, torch.Tensor):
        raise TypeError("far-field E_theta/E_phi must be torch tensors.")
    if e_theta.ndim == 2:
        e_theta, e_phi = e_theta[None], e_phi[None]
    frequency = torch.as_tensor(element["frequency"], device=device)
    frequencies = frequency.reshape(-1).to(e_theta.real.dtype)
    radius = element.get("radius", 1.0)
    impedance = element.get("wave_impedance", ETA_0)
    return frequencies, element["theta"], element["phi"], e_theta, e_phi, radius, impedance


def compute_array_pattern(
    positions: torch.Tensor,
    weights=None,
    *,
    element=None,
    theta=None,
    phi=None,
    frequencies=None,
    steering: tuple | None = None,
) -> ArrayPatternData:
    """Array factor times element pattern, radiated power and directivity.

    ``element`` selects the element pattern:

    - ``None``: isotropic elements; ``theta``, ``phi`` and ``frequencies``
      are required.
    - a real or complex tensor: a scalar element field pattern broadcastable
      to ``[F, T, P]`` (pass ``sqrt`` of a power pattern); ``theta``, ``phi``
      and ``frequencies`` are required.
    - an :class:`AntennaData`: its grid, frequencies, ``E_theta/E_phi`` (or
      ``sqrt(radiation_intensity)`` when fields are absent), observation
      radius and wave impedance are used, and gain/realized gain are derived.
    - a far-field mapping as returned by ``NearFieldFarFieldTransformer.transform``
      with ``theta``, ``phi``, ``E_theta``, ``E_phi``, ``frequency`` and
      optional ``radius`` / ``wave_impedance`` (default vacuum).

    Element patterns are referenced to their own phase centre; ``positions``
    are the offsets of those phase centres. The angular grid must cover the
    full sphere (``theta`` in ``[0, pi]``, ``phi`` spanning ``2*pi``) and
    radiated power uses the same trapezoidal ``sin(theta)`` quadrature as
    ``compute_antenna_data``.
    """

    positions = _validate_positions(positions)
    device, dtype = positions.device, positions.dtype
    element_kind = "isotropic"
    element_field = None
    vector_fields = None
    radius = impedance = None
    accepted = incident = None
    if isinstance(element, AntennaData) or isinstance(element, Mapping):
        if theta is not None or phi is not None or frequencies is not None:
            raise ValueError("theta, phi and frequencies come from the element record.")
        if isinstance(element, AntennaData):
            frequencies, theta, phi = element.frequencies, element.theta, element.phi
            accepted, incident = element.p_accepted, element.p_incident
            if element.e_theta is not None:
                vector_fields = (element.e_theta, element.e_phi)
                radius, impedance = element.observation_radius, element.wave_impedance
                element_kind = "vector"
            else:
                element_field = torch.sqrt(element.radiation_intensity)
                element_kind = "scalar"
        else:
            frequencies, theta, phi, e_theta, e_phi, radius, impedance = _element_from_mapping(
                element, device
            )
            vector_fields = (e_theta, e_phi)
            element_kind = "vector"
    elif element is not None:
        if not isinstance(element, torch.Tensor):
            raise TypeError("element must be None, a tensor, AntennaData, or a far-field mapping.")
        element_field = element
        element_kind = "scalar"
    if theta is None or phi is None or frequencies is None:
        raise ValueError("theta, phi and frequencies are required without an element record.")

    frequencies = _real(frequencies, name="frequencies", device=device, dtype=dtype).reshape(-1)
    theta = _real(theta, name="theta", device=device, dtype=dtype)
    phi = _real(phi, name="phi", device=device, dtype=dtype)
    theta_grid, phi_grid, theta_vector, phi_vector = _angular_grid(theta, phi, device=device)
    shape = (frequencies.numel(), *theta_grid.shape)

    k = _wave_number(frequencies, device=device, dtype=dtype)
    resolved = _resolve_weights(weights, count=positions.shape[0], k=k, device=device, dtype=dtype)
    if steering is not None:
        resolved = resolved * steering_weights(positions, *steering, frequency=frequencies)
    resolved = torch.broadcast_to(resolved, (frequencies.numel(), positions.shape[0]))
    factor = array_factor(positions, resolved, theta_grid, phi_grid, frequency=frequencies)

    e_theta = e_phi = None
    if vector_fields is not None:
        e_theta = factor * torch.broadcast_to(vector_fields[0].to(device), shape)
        e_phi = factor * torch.broadcast_to(vector_fields[1].to(device), shape)
        radius_t = torch.broadcast_to(_real(radius, name="radius", device=device, dtype=dtype), shape)
        impedance_t = torch.broadcast_to(
            _real(impedance, name="wave_impedance", device=device, dtype=dtype), shape
        )
        intensity = radius_t.square() * (
            torch.abs(e_theta).square() + torch.abs(e_phi).square()
        ) / (2.0 * impedance_t)
    elif element_field is not None:
        if element_field.device != device:
            raise ValueError(f"element pattern must be on device {device}.")
        field = torch.broadcast_to(element_field, shape)
        intensity = torch.abs(factor * field).square()
    else:
        intensity = torch.abs(factor).square()

    solid_angle = torch.sin(theta_grid) * _trapz_weights(theta_vector)[:, None] * _trapz_weights(phi_vector)[None, :]
    p_rad = torch.sum(intensity * solid_angle, dim=(-2, -1))
    if not bool(torch.all(p_rad > 0.0)):
        raise ValueError("The array pattern must radiate positive power at every frequency.")
    directivity = 4.0 * math.pi * intensity / p_rad[:, None, None]

    gain = realized_gain = None
    if accepted is not None:
        drive = torch.sum(torch.abs(resolved).square(), dim=-1)
        gain = 4.0 * math.pi * intensity / (drive * accepted.to(dtype))[:, None, None]
        realized_gain = 4.0 * math.pi * intensity / (drive * incident.to(dtype))[:, None, None]

    return ArrayPatternData(
        frequencies=frequencies,
        theta=theta_grid,
        phi=phi_grid,
        positions=positions,
        weights=resolved,
        array_factor=factor,
        e_theta=e_theta,
        e_phi=e_phi,
        radiation_intensity=intensity,
        p_rad=p_rad,
        directivity=directivity,
        gain=gain,
        realized_gain=realized_gain,
        element_kind=element_kind,
    )


__all__ = [
    "ArrayPatternData",
    "array_factor",
    "compute_array_pattern",
    "rectangular_lattice",
    "steering_weights",
]
