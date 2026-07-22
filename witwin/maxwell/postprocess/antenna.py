from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..antenna import AntennaData, Ludwig3, _power_normalized_antenna_metrics
from ..constants import ETA_0
from ..network import PortData


# Public alias for the central vacuum wave impedance constant.
FREE_SPACE_IMPEDANCE = ETA_0


def _validate_real_tensor(
    value, *, name: str, device: torch.device | None = None
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.is_complex() or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be a real floating-point tensor.")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on device {device}.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _validate_complex_field(
    value,
    *,
    name: str,
    shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_complex():
        raise TypeError(f"{name} must be a complex tensor.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape [F, T, P].")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}.")
    if not bool(torch.all(torch.isfinite(value.real))) or not bool(
        torch.all(torch.isfinite(value.imag))
    ):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _angular_grid(
    theta: torch.Tensor,
    phi: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = _validate_real_tensor(theta, name="theta", device=device)
    phi = _validate_real_tensor(phi, name="phi", device=device)
    if theta.dtype != phi.dtype:
        raise ValueError("theta and phi must have the same dtype.")

    if theta.ndim == 1 and phi.ndim == 1:
        theta_vector = theta
        phi_vector = phi
        theta_grid, phi_grid = torch.meshgrid(theta_vector, phi_vector, indexing="ij")
    elif theta.ndim == 2 and phi.ndim == 2 and theta.shape == phi.shape:
        theta_grid = theta
        phi_grid = phi
        theta_vector = theta[:, 0]
        phi_vector = phi[0, :]
        if not torch.allclose(theta_grid, theta_vector[:, None]):
            raise ValueError("theta must be constant along the phi axis.")
        if not torch.allclose(phi_grid, phi_vector[None, :]):
            raise ValueError("phi must be constant along the theta axis.")
    else:
        raise ValueError("theta and phi must both be 1D vectors or matching 2D grids.")

    if theta_vector.numel() < 3 or phi_vector.numel() < 2:
        raise ValueError(
            "The angular grid needs at least three theta and two phi samples."
        )
    if not bool(torch.all(theta_vector[1:] > theta_vector[:-1])):
        raise ValueError("theta samples must be strictly increasing.")
    if not bool(torch.all(phi_vector[1:] > phi_vector[:-1])):
        raise ValueError("phi samples must be strictly increasing.")

    atol = 128.0 * torch.finfo(theta.dtype).eps
    zero = torch.zeros((), device=device, dtype=theta.dtype)
    pi = torch.full((), math.pi, device=device, dtype=theta.dtype)
    two_pi = torch.full((), 2.0 * math.pi, device=device, dtype=theta.dtype)
    if not torch.isclose(theta_vector[0], zero, atol=atol, rtol=0.0):
        raise ValueError("theta must start at 0 radians for full-sphere integration.")
    if not torch.isclose(theta_vector[-1], pi, atol=atol, rtol=0.0):
        raise ValueError("theta must end at pi radians for full-sphere integration.")
    if not torch.isclose(phi_vector[-1] - phi_vector[0], two_pi, atol=atol, rtol=0.0):
        raise ValueError(
            "phi must span exactly 2*pi radians for full-sphere integration."
        )
    return theta_grid, phi_grid, theta_vector, phi_vector


def _trapz_weights(points: torch.Tensor) -> torch.Tensor:
    differences = points[1:] - points[:-1]
    weights = torch.empty_like(points)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    if points.numel() > 2:
        weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return weights


def _broadcast_positive_parameter(
    value,
    *,
    name: str,
    shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise ValueError(f"{name} must be on device {device}.")
        if value.is_complex() or not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be real floating-point data.")
        if value.dtype != dtype:
            raise TypeError(f"{name} tensor must use dtype {dtype}.")
        parameter = value
    else:
        parameter = torch.as_tensor(value, device=device, dtype=dtype)

    if parameter.ndim == 1 and parameter.shape == (shape[0],):
        parameter = parameter[:, None, None]
    elif parameter.ndim == 2 and parameter.shape == shape[1:]:
        parameter = parameter[None, ...]
    try:
        parameter = torch.broadcast_to(parameter, shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must be scalar or broadcastable to [F, T, P]; a [F] tensor is allowed."
        ) from exc
    if not bool(torch.all(torch.isfinite(parameter))):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool(torch.all(parameter > 0.0)):
        raise ValueError(f"{name} must be strictly positive.")
    return parameter


def _coordinate_provenance(
    *,
    phase_center,
    frame,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if phase_center is None:
        resolved_center = torch.zeros(3, device=device, dtype=dtype)
    elif isinstance(phase_center, torch.Tensor):
        if phase_center.device != device:
            raise ValueError(f"phase_center must be on device {device}.")
        if phase_center.is_complex() or not phase_center.dtype.is_floating_point:
            raise TypeError("phase_center must be real floating-point data.")
        if phase_center.dtype != dtype:
            raise TypeError(f"phase_center tensor must use dtype {dtype}.")
        resolved_center = phase_center
    else:
        resolved_center = torch.as_tensor(phase_center, device=device, dtype=dtype)
    if resolved_center.shape != (3,):
        raise ValueError("phase_center must have shape [3].")
    if not bool(torch.all(torch.isfinite(resolved_center))):
        raise ValueError("phase_center must contain only finite values.")

    if frame is None:
        resolved_frame = torch.eye(3, device=device, dtype=dtype)
    elif isinstance(frame, torch.Tensor):
        if frame.device != device:
            raise ValueError(f"frame must be on device {device}.")
        if frame.is_complex() or not frame.dtype.is_floating_point:
            raise TypeError("frame must be real floating-point data.")
        if frame.dtype != dtype:
            raise TypeError(f"frame tensor must use dtype {dtype}.")
        resolved_frame = frame
    else:
        resolved_frame = torch.as_tensor(frame, device=device, dtype=dtype)
    if resolved_frame.shape != (3, 3):
        raise ValueError("frame must have shape [3, 3].")
    if not bool(torch.all(torch.isfinite(resolved_frame))):
        raise ValueError("frame must contain only finite values.")
    identity = torch.eye(3, device=device, dtype=dtype)
    tolerance = 256.0 * torch.finfo(dtype).eps
    if not torch.allclose(
        resolved_frame.transpose(0, 1) @ resolved_frame,
        identity,
        atol=tolerance,
        rtol=tolerance,
    ) or not bool(torch.linalg.det(resolved_frame) > 0.0):
        raise ValueError("frame columns must form a right-handed orthonormal basis.")
    return resolved_center, resolved_frame


def _axial_ratio(e_theta: torch.Tensor, e_phi: torch.Tensor) -> torch.Tensor:
    theta_power = torch.abs(e_theta).square()
    phi_power = torch.abs(e_phi).square()
    total_power = theta_power + phi_power
    linear_stokes = theta_power - phi_power
    diagonal_stokes = 2.0 * torch.real(e_theta * torch.conj(e_phi))
    polarized_power = torch.sqrt(
        torch.clamp_min(linear_stokes.square() + diagonal_stokes.square(), 0.0)
    )
    major_power = 0.5 * (total_power + polarized_power)
    minor_power = torch.clamp_min(0.5 * (total_power - polarized_power), 0.0)
    resolved_minor = minor_power > torch.finfo(minor_power.dtype).eps * total_power
    safe_minor_power = torch.where(
        resolved_minor,
        minor_power,
        torch.ones_like(minor_power),
    )
    finite_ratio = torch.sqrt(major_power / safe_minor_power)
    return torch.where(
        resolved_minor,
        finite_ratio,
        torch.full_like(minor_power, torch.inf),
    )


def compute_antenna_data(
    *,
    frequencies: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    driven_port: PortData,
    e_theta: torch.Tensor | None = None,
    e_phi: torch.Tensor | None = None,
    radiation_intensity: torch.Tensor | None = None,
    radius=1.0,
    wave_impedance=FREE_SPACE_IMPEDANCE,
    polarization: Ludwig3 | None = None,
    phase_center=None,
    frame=None,
    surface_currents: tuple[object, ...] | None = None,
) -> AntennaData:
    """Build antenna metrics from full-sphere far-field data and one driven port.

    Fields and radiation intensity use explicit ``[F, T, P]`` order. Electric
    fields are peak phasors at ``radius`` and are converted to radiation
    intensity with a positive real ``wave_impedance``. Supplying radiation
    intensity directly leaves polarization fields and axial ratio unavailable.
    ``frame`` stores its Cartesian basis vectors as columns.
    """

    if not isinstance(driven_port, PortData):
        raise TypeError("driven_port must be a PortData instance.")
    frequencies = _validate_real_tensor(frequencies, name="frequencies")
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(frequencies > 0.0)):
        raise ValueError("frequencies must be strictly positive.")
    device = frequencies.device
    if driven_port.frequencies.device != device:
        raise ValueError("frequencies and driven_port must be on the same device.")
    if driven_port.voltage.ndim != 1:
        raise ValueError(
            "driven_port must describe one excitation with signal shape [F]."
        )
    if (
        driven_port.frequencies.shape != frequencies.shape
        or driven_port.frequencies.dtype != frequencies.dtype
        or not torch.equal(driven_port.frequencies, frequencies)
    ):
        raise ValueError("driven_port frequencies must match frequencies exactly.")

    theta_grid, phi_grid, theta_vector, phi_vector = _angular_grid(
        theta,
        phi,
        device=device,
    )
    shape = (frequencies.numel(), *theta_grid.shape)

    has_fields = e_theta is not None or e_phi is not None
    if has_fields and (e_theta is None or e_phi is None):
        raise ValueError("e_theta and e_phi must be supplied together.")
    if has_fields == (radiation_intensity is not None):
        raise ValueError(
            "Supply either e_theta/e_phi or radiation_intensity, but not both."
        )

    resolved_e_theta = None
    resolved_e_phi = None
    radius_tensor = None
    impedance_tensor = None
    if has_fields:
        resolved_e_theta = _validate_complex_field(
            e_theta,
            name="e_theta",
            shape=shape,
            device=device,
        )
        resolved_e_phi = _validate_complex_field(
            e_phi,
            name="e_phi",
            shape=shape,
            device=device,
        )
        if resolved_e_theta.dtype != resolved_e_phi.dtype:
            raise ValueError("e_theta and e_phi must have the same dtype.")
        field_real_dtype = resolved_e_theta.real.dtype
        radius_tensor = _broadcast_positive_parameter(
            radius,
            name="radius",
            shape=shape,
            device=device,
            dtype=field_real_dtype,
        )
        impedance_tensor = _broadcast_positive_parameter(
            wave_impedance,
            name="wave_impedance",
            shape=shape,
            device=device,
            dtype=field_real_dtype,
        )
        normalization = (
            "peak phasor; U=r^2*(|E_theta|^2+|E_phi|^2)/(2*eta); "
            "gain referenced to accepted power; realized gain referenced to incident power"
        )
    else:
        intensity = _validate_real_tensor(
            radiation_intensity,
            name="radiation_intensity",
            device=device,
        )
        if intensity.shape != shape:
            raise ValueError("radiation_intensity must have shape [F, T, P].")
        if not bool(torch.all(intensity >= 0.0)):
            raise ValueError("radiation_intensity must be non-negative.")
        normalization = (
            "radiation intensity supplied in W/sr; gain referenced to accepted power; "
            "realized gain referenced to incident power"
        )

    theta_weights = _trapz_weights(theta_vector)
    phi_weights = _trapz_weights(phi_vector)
    solid_angle_weights = (
        torch.sin(theta_grid) * theta_weights[:, None] * phi_weights[None, :]
    )
    p_incident = driven_port.incident_power
    p_accepted = driven_port.accepted_power
    if not bool(torch.all(torch.isfinite(p_incident))) or not bool(
        torch.all(torch.isfinite(p_accepted))
    ):
        raise ValueError("Driven-port powers must be finite.")
    if not bool(torch.all(p_incident > 0.0)):
        raise ValueError("Driven-port incident power must be strictly positive.")
    if not bool(torch.all(p_accepted > 0.0)):
        raise ValueError("Driven-port accepted power must be strictly positive.")

    if resolved_e_theta is not None:
        metrics = _power_normalized_antenna_metrics(
            e_theta=resolved_e_theta,
            e_phi=resolved_e_phi,
            observation_radius=radius_tensor,
            wave_impedance=impedance_tensor,
            solid_angle_weights=solid_angle_weights,
            incident_power=p_incident,
            accepted_power=p_accepted,
        )
        intensity = metrics["radiation_intensity"]
    else:
        p_rad = torch.sum(intensity * solid_angle_weights[None, ...], dim=(-2, -1))
        radiation_valid = torch.isfinite(p_rad) & (p_rad > 0.0)
        accepted_valid = torch.isfinite(p_accepted) & (p_accepted > 0.0)
        incident_valid = torch.isfinite(p_incident) & (p_incident > 0.0)
        safe_p_rad = torch.where(radiation_valid, p_rad, torch.ones_like(p_rad))
        safe_accepted = torch.where(accepted_valid, p_accepted, torch.ones_like(p_accepted))
        safe_incident = torch.where(incident_valid, p_incident, torch.ones_like(p_incident))
        four_pi_intensity = 4.0 * math.pi * intensity
        metrics = {
            "p_rad": p_rad,
            "directivity": four_pi_intensity / safe_p_rad[:, None, None],
            "gain": four_pi_intensity / safe_accepted[:, None, None],
            "realized_gain": four_pi_intensity / safe_incident[:, None, None],
            "radiation_efficiency": p_rad / safe_accepted,
            "mismatch_efficiency": p_accepted / safe_incident,
            "system_efficiency": p_rad / safe_incident,
            "eirp": torch.amax(four_pi_intensity, dim=(-2, -1)),
            "radiation_valid": radiation_valid,
            "accepted_power_valid": accepted_valid,
            "incident_power_valid": incident_valid,
        }
    if not bool(torch.all(metrics["radiation_valid"])):
        raise ValueError("Integrated radiated power must be finite and strictly positive.")
    p_rad = metrics["p_rad"]
    directivity = metrics["directivity"]
    gain = metrics["gain"]
    realized_gain = metrics["realized_gain"]
    radiation_efficiency = metrics["radiation_efficiency"]
    mismatch_efficiency = metrics["mismatch_efficiency"]
    system_efficiency = metrics["system_efficiency"]
    eirp = metrics["eirp"]

    basis = Ludwig3() if polarization is None else polarization
    if not isinstance(basis, Ludwig3):
        raise TypeError("polarization must be a Ludwig3 instance.")
    co_polarized = None
    cross_polarized = None
    axial_ratio = None
    if resolved_e_theta is not None and resolved_e_phi is not None:
        co_polarized, cross_polarized = basis.project(
            resolved_e_theta,
            resolved_e_phi,
            phi_grid,
        )
        axial_ratio = _axial_ratio(resolved_e_theta, resolved_e_phi)

    resolved_center, resolved_frame = _coordinate_provenance(
        phase_center=phase_center,
        frame=frame,
        device=device,
        dtype=theta_grid.dtype,
    )
    return AntennaData(
        frequencies=frequencies,
        theta=theta_grid,
        phi=phi_grid,
        e_theta=resolved_e_theta,
        e_phi=resolved_e_phi,
        observation_radius=radius_tensor,
        wave_impedance=impedance_tensor,
        radiation_intensity=intensity,
        p_rad=p_rad,
        p_accepted=p_accepted,
        p_incident=p_incident,
        directivity=directivity,
        gain=gain,
        realized_gain=realized_gain,
        radiation_efficiency=radiation_efficiency,
        mismatch_efficiency=mismatch_efficiency,
        system_efficiency=system_efficiency,
        eirp=eirp,
        co_polarized=co_polarized,
        cross_polarized=cross_polarized,
        axial_ratio=axial_ratio,
        phase_center=resolved_center,
        frame=resolved_frame,
        polarization_basis=basis,
        driven_port_name=driven_port.port_name,
        surface_currents=surface_currents,
        power_normalization=normalization,
    )


def _configuration_tensor(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise ValueError(f"configuration tensor must be on device {device}.")
        if value.is_complex() or not value.dtype.is_floating_point:
            raise TypeError("configuration tensor must be real floating-point data.")
        if value.dtype != dtype:
            raise TypeError(f"configuration tensor must use dtype {dtype}.")
        return value
    return torch.as_tensor(value, device=device, dtype=dtype)


def _select_frequency_parameter(value, index: int, frequency_count: int):
    if not isinstance(value, torch.Tensor):
        return value
    if value.shape == (frequency_count,) or (
        value.ndim >= 3 and value.shape[0] == frequency_count
    ):
        return value[index]
    return value


def _far_fields_from_result(
    result,
    *,
    surface: str,
    frequencies: torch.Tensor,
    theta: torch.Tensor | Sequence[float] | None = None,
    phi: torch.Tensor | Sequence[float] | None = None,
    theta_points: int = 181,
    phi_points: int = 361,
    radius=1.0,
    phase_center=None,
    frame=None,
    batch_size: int = 1024,
) -> dict[str, object]:
    """Transform a closed-surface result without imposing port-power validity."""

    from .nfft import NearFieldFarFieldTransformer
    from .stratton_chu import equivalent_surface_currents_from_monitor

    if not isinstance(surface, str) or not surface:
        raise ValueError("surface must be a non-empty monitor name.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not isinstance(frequencies, torch.Tensor):
        raise TypeError("frequencies must be a torch.Tensor.")
    if (
        frequencies.ndim != 1
        or frequencies.is_complex()
        or not frequencies.dtype.is_floating_point
    ):
        raise TypeError("frequencies must be a real floating-point tensor with shape [F].")
    if frequencies.numel() == 0:
        raise ValueError("frequencies must not be empty.")
    device = frequencies.device
    dtype = frequencies.dtype
    frequency_count = int(frequencies.numel())
    result_frequencies = tuple(float(value) for value in result.frequencies)
    if len(result_frequencies) != frequency_count:
        raise ValueError(
            "The requested far-field frequencies and Result must expose the same explicit frequency axis."
        )
    expected = torch.as_tensor(result_frequencies, device=device, dtype=dtype)
    if not torch.allclose(frequencies, expected, rtol=1e-6, atol=0.0):
        raise ValueError("The requested far-field frequencies must match the Result frequencies.")

    if theta is None:
        if theta_points < 3:
            raise ValueError("theta_points must be at least three.")
        theta_tensor = torch.linspace(
            0.0, math.pi, int(theta_points), device=device, dtype=dtype
        )
    else:
        theta_tensor = _configuration_tensor(theta, device=device, dtype=dtype)
    if phi is None:
        if phi_points < 2:
            raise ValueError("phi_points must be at least two.")
        phi_tensor = torch.linspace(
            0.0, 2.0 * math.pi, int(phi_points), device=device, dtype=dtype
        )
    else:
        phi_tensor = _configuration_tensor(phi, device=device, dtype=dtype)
    theta_grid, phi_grid, _, _ = _angular_grid(theta_tensor, phi_tensor, device=device)

    resolved_center, resolved_frame = _coordinate_provenance(
        phase_center=phase_center,
        frame=frame,
        device=device,
        dtype=dtype,
    )
    local_direction = torch.stack(
        (
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
            torch.cos(theta_grid),
        ),
        dim=-1,
    )
    local_theta = torch.stack(
        (
            torch.cos(theta_grid) * torch.cos(phi_grid),
            torch.cos(theta_grid) * torch.sin(phi_grid),
            -torch.sin(theta_grid),
        ),
        dim=-1,
    )
    local_phi = torch.stack(
        (
            -torch.sin(phi_grid),
            torch.cos(phi_grid),
            torch.zeros_like(phi_grid),
        ),
        dim=-1,
    )
    global_direction = torch.einsum("ij,...j->...i", resolved_frame, local_direction)
    global_theta = torch.einsum("ij,...j->...i", resolved_frame, local_theta)
    global_phi = torch.einsum("ij,...j->...i", resolved_frame, local_phi)

    e_theta_fields = []
    e_phi_fields = []
    current_surfaces = []
    impedances = []
    for index, frequency in enumerate(result_frequencies):
        currents = equivalent_surface_currents_from_monitor(
            result,
            surface,
            frequency=frequency,
        )
        if torch.device(currents.device) != device:
            raise ValueError(
                "The Huygens surface and driven-port data must be on the same device."
            )
        transformer = NearFieldFarFieldTransformer(
            currents,
            solver=result.solver,
            device=device,
        )
        impedance = complex(transformer.eta)
        wavenumber = complex(transformer.k)
        tolerance = 1e-10 * max(abs(impedance.real), abs(wavenumber.real), 1.0)
        if (
            impedance.real <= 0.0
            or abs(impedance.imag) > tolerance
            or abs(wavenumber.imag) > tolerance
        ):
            raise NotImplementedError(
                "Antenna gain currently requires a homogeneous lossless exterior."
            )
        selected_radius = _select_frequency_parameter(radius, index, frequency_count)
        transformed = transformer.transform_directions(
            global_direction.reshape(-1, 3),
            radius=selected_radius,
            batch_size=batch_size,
        )
        e_field = transformed["E"].reshape(theta_grid.shape + (3,))
        phase_shift = torch.exp(
            1j
            * wavenumber.real
            * torch.sum(global_direction * resolved_center, dim=-1)
        ).to(dtype=e_field.dtype)
        e_field = e_field * phase_shift[..., None]
        e_theta_fields.append(
            torch.sum(e_field * global_theta.to(dtype=e_field.dtype), dim=-1)
        )
        e_phi_fields.append(
            torch.sum(e_field * global_phi.to(dtype=e_field.dtype), dim=-1)
        )
        current_surfaces.append(currents)
        impedances.append(impedance.real)

    # The far field inherits its precision from the monitor payload (the solve
    # dtype), while the angular grid, radius, impedance, and driven-port powers
    # all live at ``dtype`` (the requested frequency dtype). Promote the far field
    # to the matching complex dtype so the whole metric pipeline runs at a single
    # real precision -- the driven-port powers must stay at ``dtype`` for the
    # mismatch-ratio bookkeeping, and ``compute_antenna_data`` now rejects mixed
    # real dtypes outright rather than silently casting.
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    e_theta = torch.stack(e_theta_fields, dim=0).to(dtype=complex_dtype)
    e_phi = torch.stack(e_phi_fields, dim=0).to(dtype=complex_dtype)
    impedance_tensor = torch.as_tensor(impedances, device=device, dtype=dtype)
    resolved_radius = radius
    return {
        "frequencies": frequencies,
        "theta": theta_grid,
        "phi": phi_grid,
        "e_theta": e_theta,
        "e_phi": e_phi,
        "radius": resolved_radius,
        "wave_impedance": impedance_tensor,
        "phase_center": resolved_center,
        "frame": resolved_frame,
        "surface_currents": tuple(current_surfaces),
    }


def antenna_data_from_result(
    result,
    *,
    surface: str,
    driven_port: str | PortData,
    polarization: Ludwig3 | None = None,
    theta: torch.Tensor | Sequence[float] | None = None,
    phi: torch.Tensor | Sequence[float] | None = None,
    theta_points: int = 181,
    phi_points: int = 361,
    radius=1.0,
    phase_center=None,
    frame=None,
    batch_size: int = 1024,
) -> AntennaData:
    """Build antenna results from a closed Huygens monitor and a driven port.

    The angular coordinates are expressed in ``frame``. Its columns are the
    local x/y/z basis vectors in the simulation coordinates. ``phase_center``
    is expressed in simulation coordinates and is applied to the far-field
    phase, rather than being stored as provenance only.
    """

    if isinstance(driven_port, str):
        port = result.port(driven_port)
    elif isinstance(driven_port, PortData):
        port = driven_port
    else:
        raise TypeError("driven_port must be a port name or PortData.")
    transformed = _far_fields_from_result(
        result,
        surface=surface,
        frequencies=port.frequencies,
        theta=theta,
        phi=phi,
        theta_points=theta_points,
        phi_points=phi_points,
        radius=radius,
        phase_center=phase_center,
        frame=frame,
        batch_size=batch_size,
    )
    return compute_antenna_data(
        frequencies=transformed["frequencies"],
        theta=transformed["theta"],
        phi=transformed["phi"],
        driven_port=port,
        e_theta=transformed["e_theta"],
        e_phi=transformed["e_phi"],
        radius=transformed["radius"],
        wave_impedance=transformed["wave_impedance"],
        polarization=polarization,
        phase_center=transformed["phase_center"],
        frame=transformed["frame"],
        surface_currents=transformed["surface_currents"],
    )


__all__ = [
    "FREE_SPACE_IMPEDANCE",
    "_far_fields_from_result",
    "antenna_data_from_result",
    "compute_antenna_data",
]
