from __future__ import annotations

import hashlib
from typing import Sequence

import torch

from ..antenna import Ludwig3
from ..array import ArrayBasisData, EmbeddedElementPatternData
from ..array_execution import ArrayRunData
from .antenna import _far_fields_from_result
from .stratton_chu import EquivalentCurrentsSurface


def _frequency_parameter(value, index: int, count: int):
    if isinstance(value, torch.Tensor):
        if value.shape == (count,) or (value.ndim >= 3 and value.shape[0] == count):
            return value[index]
    return value


def _basis_fingerprint(
    request,
    network,
    incident: torch.Tensor,
    embedded_patterns: EmbeddedElementPatternData,
    radiated_power_matrix: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    contract = (
        request.monitor_name,
        request.monitor_bounds,
        request.monitor_faces,
        request.frequencies,
        request.port_names,
        request.physical_port_names,
        request.phase_center_source,
        request.run_manifest_fingerprint,
        network.phasor_convention,
        network.power_wave_convention,
        type(embedded_patterns.polarization_basis).__name__,
        embedded_patterns.polarization_basis.reference_angle,
        embedded_patterns.phase_center_source,
        embedded_patterns.field_basis,
        embedded_patterns.power_normalization,
        embedded_patterns.phasor_convention,
        embedded_patterns.power_wave_convention,
        embedded_patterns.field_units,
        embedded_patterns.port_names,
        "array-basis-content-v2",
        "derived solver phasors promoted only from complex64 to NetworkData complex128",
    )
    digest.update(repr(contract).encode("utf-8"))
    for tensor in (
        network.frequencies,
        network.s,
        network.z0,
        incident,
        embedded_patterns.frequencies,
        embedded_patterns.theta,
        embedded_patterns.phi,
        embedded_patterns.e_theta,
        embedded_patterns.e_phi,
        embedded_patterns.phase_center,
        embedded_patterns.frame,
        embedded_patterns.observation_radius,
        embedded_patterns.wave_impedance,
        radiated_power_matrix,
    ):
        detached = tensor.detach().contiguous().cpu()
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(repr(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _derived_phasor_at_network_precision(
    value: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_complex():
        raise TypeError(f"{name} must be a complex torch.Tensor.")
    if value.device != device:
        raise ValueError(f"{name} must be on NetworkData device {device}.")
    if value.dtype == dtype:
        return value
    if value.dtype == torch.complex64 and dtype == torch.complex128:
        return value.to(dtype=torch.complex128)
    raise TypeError(
        f"{name} uses {value.dtype}; only explicit complex64-to-complex128 "
        "promotion to NetworkData precision is supported."
    )


def _transform_column(
    results,
    *,
    request,
    frequencies: torch.Tensor,
    radius,
    batch_size: int,
):
    frequency_count = int(frequencies.numel())
    if len(results) == 1:
        transformed = _far_fields_from_result(
            results[0],
            surface=request.monitor_name,
            frequencies=frequencies,
            theta=request.theta,
            phi=request.phi,
            radius=radius,
            phase_center=request.phase_center,
            frame=request.frame,
            batch_size=batch_size,
        )
        return transformed
    if len(results) != frequency_count:
        raise RuntimeError(
            "Array basis execution retained neither one broadband Result nor one Result per frequency."
        )

    transformed_by_frequency = []
    for index, (column_result, frequency) in enumerate(zip(results, frequencies)):
        transformed_by_frequency.append(
            _far_fields_from_result(
                column_result,
                surface=request.monitor_name,
                frequencies=frequency.reshape(1),
                theta=request.theta,
                phi=request.phi,
                radius=_frequency_parameter(radius, index, frequency_count),
                phase_center=request.phase_center,
                frame=request.frame,
                batch_size=batch_size,
            )
        )
    reference = transformed_by_frequency[0]
    for transformed in transformed_by_frequency[1:]:
        for name in ("theta", "phi", "phase_center", "frame"):
            if not torch.equal(transformed[name], reference[name]):
                raise RuntimeError(f"Array basis {name} changed between frequency runs.")
    return {
        "frequencies": frequencies,
        "theta": reference["theta"],
        "phi": reference["phi"],
        "e_theta": torch.cat(
            [transformed["e_theta"] for transformed in transformed_by_frequency], dim=0
        ),
        "e_phi": torch.cat(
            [transformed["e_phi"] for transformed in transformed_by_frequency], dim=0
        ),
        "radius": radius,
        "wave_impedance": torch.cat(
            [transformed["wave_impedance"] for transformed in transformed_by_frequency],
            dim=0,
        ),
        "phase_center": reference["phase_center"],
        "frame": reference["frame"],
        "surface_currents": tuple(
            transformed["surface_currents"][0]
            for transformed in transformed_by_frequency
        ),
    }


def _radiated_power_matrix(
    columns: list[dict[str, object]],
    incident: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the closed-surface complex-Poynting power operator [F, N, N]."""

    frequency_count, port_count = incident.shape
    matrix = torch.zeros(
        (frequency_count, port_count, port_count),
        device=incident.device,
        dtype=dtype,
    )
    for frequency_index in range(frequency_count):
        current_sets = [
            column["surface_currents"][frequency_index] for column in columns
        ]
        if any(not isinstance(currents, EquivalentCurrentsSurface) for currents in current_sets):
            raise RuntimeError(
                "Absolute array power requires a closed equivalent-current surface."
            )
        reference = current_sets[0]
        if any(len(currents.surfaces) != len(reference.surfaces) for currents in current_sets):
            raise RuntimeError("Array closed-surface topology differs between excitation columns.")

        frequency_matrix = torch.zeros(
            (port_count, port_count), device=incident.device, dtype=dtype
        )
        for surface_index, reference_surface in enumerate(reference.surfaces):
            surfaces = [currents.surfaces[surface_index] for currents in current_sets]
            for surface in surfaces[1:]:
                if (
                    surface.axis != reference_surface.axis
                    or surface.normal_direction != reference_surface.normal_direction
                    or surface.quadrature_rule != reference_surface.quadrature_rule
                    or not torch.equal(surface.u, reference_surface.u)
                    or not torch.equal(surface.v, reference_surface.v)
                    or not torch.equal(
                        surface.weights_2d(), reference_surface.weights_2d()
                    )
                ):
                    raise RuntimeError(
                        "Array closed-surface sampling differs between excitation columns."
                    )

            normal = reference_surface.normal.to(dtype=dtype)
            electric_fields = []
            magnetic_fields = []
            for port_index, surface in enumerate(surfaces):
                scale = incident[frequency_index, port_index]
                expanded_normal = normal.expand_as(surface.M)
                surface_m = _derived_phasor_at_network_precision(
                    surface.M,
                    name="surface M",
                    device=incident.device,
                    dtype=dtype,
                )
                surface_j = _derived_phasor_at_network_precision(
                    surface.J,
                    name="surface J",
                    device=incident.device,
                    dtype=dtype,
                )
                electric_fields.append(
                    torch.cross(expanded_normal, surface_m, dim=-1) / scale
                )
                magnetic_fields.append(
                    -torch.cross(expanded_normal, surface_j, dim=-1) / scale
                )
            electric = torch.stack(electric_fields, dim=0)
            magnetic = torch.stack(magnetic_fields, dim=0)
            cross_power = torch.cross(
                electric.unsqueeze(0),
                torch.conj(magnetic).unsqueeze(1),
                dim=-1,
            )
            normal_power = torch.sum(cross_power * normal, dim=-1)
            area_weights = reference_surface.weights_2d().to(
                device=incident.device,
                dtype=incident.real.dtype,
            )
            frequency_matrix = frequency_matrix + 0.5 * torch.sum(
                normal_power * area_weights,
                dim=(-2, -1),
            )
        matrix[frequency_index] = 0.5 * (frequency_matrix + frequency_matrix.mH)
    return matrix


def array_basis_from_result(
    result,
    *,
    monitor: str,
    polarization: Ludwig3 | None = None,
    theta: torch.Tensor | Sequence[float] | None = None,
    phi: torch.Tensor | Sequence[float] | None = None,
    theta_points: int = 181,
    phi_points: int = 361,
    radius=1.0,
    phase_center=None,
    frame=None,
    batch_size: int = 1024,
) -> ArrayBasisData:
    """Build unit-incident-power embedded patterns from retained PortSweep columns."""

    if result.method != "fdtd":
        raise NotImplementedError("Array basis extraction is FDTD-only.")
    network = result.network
    if network is None or not network.is_complete:
        raise RuntimeError("Result.array_basis() requires a complete PortSweep NetworkData.")
    run_data = getattr(result, "_array_run_data", None)
    if not isinstance(run_data, ArrayRunData):
        raise RuntimeError(
            "This Result does not retain in-memory full-wave PortSweep columns; run an "
            "FDTD PortSweep and call array_basis() before Result.save()."
        )
    if run_data.incident.shape != (network.s.shape[0], network.s.shape[1]):
        raise RuntimeError("Retained incident-wave data does not match NetworkData shape [F, N].")
    if run_data.incident.device != network.s.device:
        raise RuntimeError("Retained incident waves and NetworkData must share a device.")
    if isinstance(radius, torch.Tensor):
        if radius.device != network.s.device:
            raise ValueError(f"radius tensor must be on device {network.s.device}.")
        if radius.is_complex() or radius.dtype != network.frequencies.dtype:
            raise TypeError(
                f"radius tensor must use real dtype {network.frequencies.dtype}."
            )

    requested_theta = theta
    requested_phi = phi
    if requested_theta is None:
        requested_theta = torch.linspace(
            0.0,
            torch.pi,
            int(theta_points),
            device=network.s.device,
            dtype=network.frequencies.dtype,
        )
    if requested_phi is None:
        requested_phi = torch.linspace(
            0.0,
            2.0 * torch.pi,
            int(phi_points),
            device=network.s.device,
            dtype=network.frequencies.dtype,
        )
    (request,) = result.scene.compile_array_monitors(
        monitor=monitor,
        frequencies=tuple(float(value) for value in result.frequencies),
        theta=requested_theta,
        phi=requested_phi,
        phase_center=phase_center,
        frame=frame,
        device=network.s.device,
        dtype=network.frequencies.dtype,
        run_manifest=run_data.manifest,
    )
    if request.port_names != network.port_names:
        raise RuntimeError("Compiled array port order does not match NetworkData port order.")

    incident = _derived_phasor_at_network_precision(
        run_data.incident,
        name="retained incident waves",
        device=network.s.device,
        dtype=network.s.dtype,
    )
    magnitude = torch.abs(incident)
    threshold = torch.clamp(
        torch.amax(magnitude, dim=0) * 1.0e-6,
        min=torch.finfo(magnitude.dtype).tiny,
    )
    weak = magnitude < threshold.unsqueeze(0)
    if bool(torch.any(weak)):
        indices = torch.nonzero(weak).detach().cpu().tolist()
        raise RuntimeError(f"Array basis incident power wave is below threshold at [F, N] indices {indices}.")

    columns = []
    reference = None
    for column_results in run_data.column_results:
        transformed = _transform_column(
            column_results,
            request=request,
            frequencies=network.frequencies,
            radius=radius,
            batch_size=batch_size,
        )
        if reference is None:
            reference = transformed
        else:
            for name in ("theta", "phi", "phase_center", "frame", "wave_impedance"):
                if not torch.equal(transformed[name], reference[name]):
                    raise RuntimeError(f"Array basis {name} differs between excitation columns.")
        columns.append(transformed)

    e_theta = torch.stack(
        [
            _derived_phasor_at_network_precision(
                column["e_theta"],
                name="embedded E_theta column",
                device=network.s.device,
                dtype=network.s.dtype,
            )
            for column in columns
        ],
        dim=1,
    ) / incident[:, :, None, None]
    e_phi = torch.stack(
        [
            _derived_phasor_at_network_precision(
                column["e_phi"],
                name="embedded E_phi column",
                device=network.s.device,
                dtype=network.s.dtype,
            )
            for column in columns
        ],
        dim=1,
    ) / incident[:, :, None, None]
    radiated_power_matrix = _radiated_power_matrix(
        columns,
        incident,
        dtype=network.s.dtype,
    )
    embedded_patterns = EmbeddedElementPatternData(
        frequencies=network.frequencies,
        port_names=network.port_names,
        theta=reference["theta"],
        phi=reference["phi"],
        e_theta=e_theta,
        e_phi=e_phi,
        phase_center=reference["phase_center"],
        frame=reference["frame"],
        observation_radius=reference["radius"],
        wave_impedance=reference["wave_impedance"],
        polarization_basis=Ludwig3() if polarization is None else polarization,
        phase_center_source=request.phase_center_source,
    )
    basis = ArrayBasisData(
        network=network,
        embedded_patterns=embedded_patterns,
        fingerprint=_basis_fingerprint(
            request,
            network,
            incident,
            embedded_patterns,
            radiated_power_matrix,
        ),
        radiated_power_matrix=radiated_power_matrix,
        metadata={
            "monitor": request.monitor_name,
            "monitor_bounds": request.monitor_bounds,
            "monitor_faces": request.monitor_faces,
            "run_manifest": dict(request.run_manifest_metadata),
            "run_manifest_fingerprint": request.run_manifest_fingerprint,
            "execution": "retained_fdtd_port_sweep_columns",
            "normalization_incident_wave": "measured a_n(f)",
            "measured_incident_waves": incident.detach().clone(),
            "solver_rerun": False,
            "radiated_power_source": "closed_surface_complex_poynting_quadratic",
            "far_field_amplitude_fitting": False,
            "precision_policy": (
                "derived solver phasors retain NetworkData device and are promoted "
                "only from complex64 to NetworkData complex128"
            ),
        },
    )
    return basis


__all__ = ["array_basis_from_result"]
