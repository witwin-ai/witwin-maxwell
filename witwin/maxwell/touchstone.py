from __future__ import annotations

import math
import re
from pathlib import Path

import torch

from .network import NetworkData


_FREQUENCY_SCALES = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}
_SNP_SUFFIX = re.compile(r"\.s([1-9][0-9]*)p", re.IGNORECASE)


def _validate_output_path(path: str | Path, *, port_count: int) -> tuple[Path, bool]:
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    match = _SNP_SUFFIX.fullmatch(suffix)
    if match is not None:
        suffix_port_count = int(match.group(1))
        if suffix_port_count != port_count:
            raise ValueError(
                f"Touchstone suffix {suffix!r} declares {suffix_port_count} ports, "
                f"but network contains {port_count}."
            )
        return output_path, False
    if suffix == ".ts":
        return output_path, True
    raise ValueError("Touchstone path must end in '.sNp' with the matching port count or '.ts'.")


def _resolve_version(version, *, uniform_z0: bool, ts_suffix: bool) -> str:
    normalized = str(version).lower()
    if normalized == "auto":
        return "2.0" if ts_suffix or not uniform_z0 else "1.0"
    if normalized not in {"1.0", "2.0"}:
        raise ValueError("version must be 'auto', '1.0', or '2.0'.")
    if normalized == "1.0" and ts_suffix:
        raise ValueError("The '.ts' suffix requires Touchstone version 2.0.")
    return normalized


def _format_pair(value: complex, data_format: str) -> tuple[float, float]:
    if data_format == "ri":
        return value.real, value.imag
    magnitude = abs(value)
    angle = math.degrees(math.atan2(value.imag, value.real))
    if data_format == "ma":
        return magnitude, angle
    return 20.0 * math.log10(magnitude) if magnitude > 0.0 else -math.inf, angle


def _format_number(value: float) -> str:
    return format(value, ".17g")


def _data_lines(frequency: float, matrix: torch.Tensor, data_format: str) -> list[str]:
    port_count = matrix.shape[0]
    if port_count == 2:
        entries = (matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1])
        values = [_format_number(frequency)]
        for entry in entries:
            values.extend(_format_number(item) for item in _format_pair(complex(entry), data_format))
        return [" ".join(values)]

    lines = []
    for row in range(port_count):
        values = [_format_number(frequency)] if row == 0 else []
        for column in range(port_count):
            values.extend(
                _format_number(item)
                for item in _format_pair(complex(matrix[row, column]), data_format)
            )
        lines.append(" ".join(values))
    return lines


def write_touchstone(
    network: NetworkData,
    path: str | Path,
    format: str = "ri",
    frequency_unit: str = "hz",
    version: str = "auto",
) -> None:
    """Write complete 1-, 2-, or 3-port scattering data as Touchstone text.

    Export is an inference boundary: tensors are explicitly detached and copied
    to CPU before validation or file I/O, and no autograd graph is preserved.
    """

    if not isinstance(network, NetworkData):
        raise TypeError("network must be a NetworkData instance.")

    data_format = str(format).lower()
    if data_format not in {"ri", "ma", "db"}:
        raise ValueError("format must be 'ri', 'ma', or 'db'.")
    unit = str(frequency_unit).lower()
    if unit not in _FREQUENCY_SCALES:
        raise ValueError("frequency_unit must be 'hz', 'khz', 'mhz', or 'ghz'.")

    frequencies = network.frequencies.detach().cpu()
    scattering = network.s.detach().cpu()
    reference = network.z0.detach().cpu()
    valid_columns = network.valid_columns.detach().cpu()
    frequency_count, port_count, _ = scattering.shape
    if port_count not in {1, 2, 3}:
        raise ValueError("Touchstone export currently supports 1-, 2-, and 3-port networks.")
    if not bool(torch.all(valid_columns)):
        raise ValueError("Touchstone export requires complete excitation columns.")
    if frequency_count > 1 and not bool(torch.all(torch.diff(frequencies) > 0.0)):
        raise ValueError("Touchstone frequencies must be strictly increasing.")
    if not bool(torch.all(torch.isfinite(scattering.real))) or not bool(
        torch.all(torch.isfinite(scattering.imag))
    ):
        raise ValueError("Touchstone scattering data must contain only finite values.")
    if not bool(torch.all(reference.imag == 0.0)):
        raise ValueError("Touchstone export does not support complex reference impedances.")
    real_reference = reference.real
    if frequency_count > 1 and not torch.equal(
        real_reference,
        real_reference[0].unsqueeze(0).expand_as(real_reference),
    ):
        raise ValueError("Touchstone export requires frequency-invariant reference impedances.")

    output_path, ts_suffix = _validate_output_path(path, port_count=port_count)
    per_port_reference = real_reference[0]
    uniform_z0 = bool(torch.all(per_port_reference == per_port_reference[0]))
    resolved_version = _resolve_version(version, uniform_z0=uniform_z0, ts_suffix=ts_suffix)
    if resolved_version == "1.0" and not uniform_z0:
        raise ValueError("Touchstone 1.0 requires one uniform real reference impedance.")

    unit_scale = _FREQUENCY_SCALES[unit]
    option_reference = _format_number(float(per_port_reference[0]))
    lines = []
    if resolved_version == "2.0":
        lines.append("[Version] 2.0")
    lines.append(f"# {unit.upper()} S {data_format.upper()} R {option_reference}")
    if resolved_version == "2.0":
        lines.append(f"[Number of Ports] {port_count}")
        if port_count == 2:
            lines.append("[Two-Port Data Order] 21_12")
        lines.extend(
            (
                f"[Number of Frequencies] {frequency_count}",
                "[Reference] "
                + " ".join(_format_number(float(value)) for value in per_port_reference),
                "[Matrix Format] Full",
                "[Network Data]",
            )
        )

    scaled_frequencies = frequencies / unit_scale
    for index in range(frequency_count):
        lines.extend(
            _data_lines(
                float(scaled_frequencies[index]),
                scattering[index],
                data_format,
            )
        )
    if resolved_version == "2.0":
        lines.append("[End]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return output_path


__all__ = ["write_touchstone"]
