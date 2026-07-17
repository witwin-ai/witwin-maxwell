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
_KEYWORD = re.compile(r"^\[([A-Za-z]+(?:(?:-| )[A-Za-z]+)*)\](?:[ \t]+(.*))?$")
_PORT_NAME_COMMENT = re.compile(r"^\s*Port\[([1-9][0-9]*)\]\s*=\s*(.*?)\s*$", re.IGNORECASE)
_DB_ZERO_FLOOR = -400.0


class TouchstoneParseError(ValueError):
    """A strict Touchstone parse error with source-line context."""

    def __init__(self, message: str, *, path: str | Path, line_number: int):
        self.path = Path(path)
        self.line_number = int(line_number)
        super().__init__(f"{self.path}: line {self.line_number}: {message}")


def _parse_error(message: str, *, path: Path, line_number: int):
    raise TouchstoneParseError(message, path=path, line_number=line_number)


def _source_lines(
    path: Path,
) -> tuple[list[tuple[int, str]], tuple[str, ...], dict[int, tuple[str, int]]]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read Touchstone file {path}: {exc}") from exc
    try:
        text = raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        line_number = raw_bytes[: exc.start].count(b"\n") + 1
        _parse_error(
            "Touchstone text must contain only ASCII characters",
            path=path,
            line_number=line_number,
        )
        raise AssertionError from exc

    content: list[tuple[int, str]] = []
    comments: list[str] = []
    port_names: dict[int, tuple[str, int]] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if any(character not in "\t" and not (" " <= character <= "~") for character in raw):
            _parse_error(
                "Touchstone text contains a disallowed control character",
                path=path,
                line_number=line_number,
            )
        code, marker, comment = raw.partition("!")
        if marker:
            original = "!" + comment
            comments.append(original)
            match = _PORT_NAME_COMMENT.fullmatch(comment)
            if match is not None:
                index = int(match.group(1))
                name = match.group(2).strip()
                if not name:
                    _parse_error("port name comments must not be empty", path=path, line_number=line_number)
                if index in port_names:
                    _parse_error(
                        f"duplicate port name comment for port {index}",
                        path=path,
                        line_number=line_number,
                    )
                port_names[index] = (name, line_number)
        stripped = code.strip()
        if stripped:
            content.append((line_number, stripped))
    return content, tuple(comments), port_names


def _split_keyword(code: str) -> tuple[str, str] | None:
    match = _KEYWORD.fullmatch(code)
    if match is None:
        return None
    return match.group(1).lower(), (match.group(2) or "").strip()


def _float_token(token: str, *, path: Path, line_number: int, description: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        _parse_error(
            f"invalid numeric token {token!r} in {description}",
            path=path,
            line_number=line_number,
        )
        raise AssertionError from exc
    if math.isnan(value):
        _parse_error(f"NaN is not permitted in {description}", path=path, line_number=line_number)
    return value


def _positive_float(token: str, *, path: Path, line_number: int, description: str) -> float:
    value = _float_token(token, path=path, line_number=line_number, description=description)
    if not math.isfinite(value) or value <= 0.0:
        _parse_error(f"{description} must be finite and positive", path=path, line_number=line_number)
    return value


def _nonnegative_float(token: str, *, path: Path, line_number: int, description: str) -> float:
    value = _float_token(token, path=path, line_number=line_number, description=description)
    if not math.isfinite(value) or value < 0.0:
        _parse_error(f"{description} must be finite and non-negative", path=path, line_number=line_number)
    return value


def _parse_option_line(code: str, *, path: Path, line_number: int) -> dict[str, object]:
    if not code.startswith("#"):
        _parse_error("expected the Touchstone option line", path=path, line_number=line_number)
    tokens = code[1:].split()
    options: dict[str, object] = {
        "frequency_unit": "ghz",
        "parameter": "s",
        "format": "ma",
        "reference": 50.0,
    }
    seen: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index].lower()
        if token in _FREQUENCY_SCALES:
            category, value = "frequency unit", token
        elif token in {"s", "y", "z"}:
            category, value = "parameter", token
        elif token in {"ri", "ma", "db"}:
            category, value = "format", token
        elif token == "r":
            category = "reference"
            index += 1
            if index >= len(tokens):
                _parse_error("option-line R requires a value", path=path, line_number=line_number)
            value = _positive_float(
                tokens[index],
                path=path,
                line_number=line_number,
                description="reference resistance",
            )
        else:
            _parse_error(
                f"unsupported option-line token {tokens[index]!r}",
                path=path,
                line_number=line_number,
            )
        if category in seen:
            _parse_error(f"duplicate option-line {category}", path=path, line_number=line_number)
        seen.add(category)
        key = "frequency_unit" if category == "frequency unit" else category
        options[key] = value
        index += 1
    return options


def _port_count_from_suffix(path: Path) -> int | None:
    match = _SNP_SUFFIX.fullmatch(path.suffix)
    return None if match is None else int(match.group(1))


def _positive_int(argument: str, *, path: Path, line_number: int, description: str) -> int:
    try:
        value = int(argument)
    except ValueError as exc:
        _parse_error(f"{description} must be a positive integer", path=path, line_number=line_number)
        raise AssertionError from exc
    if value <= 0:
        _parse_error(f"{description} must be a positive integer", path=path, line_number=line_number)
    return value


def _pair_value(
    first_token: str,
    second_token: str,
    *,
    data_format: str,
    path: Path,
    line_number: int,
) -> complex:
    first = _float_token(
        first_token,
        path=path,
        line_number=line_number,
        description="network data",
    )
    second = _float_token(
        second_token,
        path=path,
        line_number=line_number,
        description="network data",
    )
    if data_format == "ri":
        if not math.isfinite(first) or not math.isfinite(second):
            _parse_error("RI data must be finite", path=path, line_number=line_number)
        return complex(first, second)
    if not math.isfinite(second):
        _parse_error("network-data angles must be finite", path=path, line_number=line_number)
    if data_format == "ma":
        if not math.isfinite(first) or first < 0.0:
            _parse_error("MA magnitudes must be finite and non-negative", path=path, line_number=line_number)
        magnitude = first
    else:
        if not math.isfinite(first):
            _parse_error("DB magnitudes must be finite", path=path, line_number=line_number)
        magnitude = 10.0 ** (first / 20.0)
    angle = math.radians(second)
    return magnitude * complex(math.cos(angle), math.sin(angle))


def _matrix_indices(port_count: int, matrix_format: str, two_port_order: str) -> list[tuple[int, int]]:
    if matrix_format == "lower":
        return [(row, column) for row in range(port_count) for column in range(row + 1)]
    if matrix_format == "upper":
        return [(row, column) for row in range(port_count) for column in range(row, port_count)]
    if port_count == 2 and two_port_order == "21_12":
        return [(0, 0), (1, 0), (0, 1), (1, 1)]
    return [(row, column) for row in range(port_count) for column in range(port_count)]


def _validate_v1_data_layout(
    lines: list[tuple[int, str]],
    *,
    port_count: int,
    path: Path,
) -> None:
    if port_count <= 2:
        expected = 1 + 2 * port_count * port_count
        for line_number, code in lines:
            if len(code.split()) != expected:
                _parse_error(
                    f"incomplete network data: Touchstone 1.x {port_count}-port records require exactly {expected} values per line",
                    path=path,
                    line_number=line_number,
                )
        return

    cursor = 0
    while cursor < len(lines):
        for row in range(port_count):
            remaining = port_count
            first_chunk = True
            while remaining:
                if cursor >= len(lines):
                    _parse_error(
                        f"incomplete Touchstone 1.x matrix row {row + 1}",
                        path=path,
                        line_number=lines[-1][0] if lines else 1,
                    )
                line_number, code = lines[cursor]
                token_count = len(code.split())
                has_frequency = row == 0 and first_chunk
                data_tokens = token_count - 1 if has_frequency else token_count
                if data_tokens <= 0 or data_tokens % 2:
                    _parse_error(
                        "frequency values must occur only at the start of a complete matrix record",
                        path=path,
                        line_number=line_number,
                    )
                pair_count = data_tokens // 2
                if pair_count > 4 or pair_count > remaining:
                    _parse_error(
                        "Touchstone 1.x permits at most four parameter pairs per line",
                        path=path,
                        line_number=line_number,
                    )
                if port_count <= 4 and pair_count != port_count:
                    _parse_error(
                        f"Touchstone 1.x matrix row {row + 1} requires {port_count} pairs on one line",
                        path=path,
                        line_number=line_number,
                    )
                remaining -= pair_count
                first_chunk = False
                cursor += 1


def _parse_data_tokens(
    lines: list[tuple[int, str]],
    *,
    port_count: int,
    frequency_count: int | None,
    frequency_unit: str,
    data_format: str,
    matrix_format: str,
    two_port_order: str,
    version: str,
    path: Path,
    eof_line: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if version == "1.0":
        _validate_v1_data_layout(lines, port_count=port_count, path=path)
    tokens: list[tuple[str, int, int]] = []
    for line_number, code in lines:
        for position, token in enumerate(code.split()):
            tokens.append((token, line_number, position))
    indices = _matrix_indices(port_count, matrix_format, two_port_order)
    group_size = 1 + 2 * len(indices)
    if not tokens:
        _parse_error("network data is empty", path=path, line_number=eof_line)
    if len(tokens) % group_size:
        _parse_error(
            f"incomplete network data: expected groups of {group_size} numeric values",
            path=path,
            line_number=tokens[-1][1],
        )
    parsed_count = len(tokens) // group_size
    if frequency_count is not None and parsed_count != frequency_count:
        _parse_error(
            f"[Number of Frequencies] declares {frequency_count}, found {parsed_count}",
            path=path,
            line_number=eof_line,
        )

    scale = _FREQUENCY_SCALES[frequency_unit]
    frequencies: list[float] = []
    matrices: list[list[list[complex]]] = []
    for group_index in range(parsed_count):
        group = tokens[group_index * group_size : (group_index + 1) * group_size]
        if group[0][2] != 0:
            _parse_error(
                "frequency values must begin a new network-data line",
                path=path,
                line_number=group[0][1],
            )
        raw_frequency = _nonnegative_float(
            group[0][0],
            path=path,
            line_number=group[0][1],
            description="frequency",
        )
        frequency = raw_frequency * scale
        if frequencies and frequency <= frequencies[-1]:
            _parse_error(
                "frequencies must be strictly increasing without duplicates",
                path=path,
                line_number=group[0][1],
            )
        frequencies.append(frequency)
        matrix = [[0j for _ in range(port_count)] for _ in range(port_count)]
        offset = 1
        for row, column in indices:
            value = _pair_value(
                group[offset][0],
                group[offset + 1][0],
                data_format=data_format,
                path=path,
                line_number=group[offset][1],
            )
            matrix[row][column] = value
            if matrix_format in {"lower", "upper"} and row != column:
                matrix[column][row] = value
            offset += 2
        matrices.append(matrix)
    return (
        torch.tensor(frequencies, dtype=torch.float64),
        torch.tensor(matrices, dtype=torch.complex128),
    )


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
    return 20.0 * math.log10(magnitude) if magnitude > 0.0 else _DB_ZERO_FLOOR, angle


def _format_number(value: float) -> str:
    return format(value, ".17g")


def _data_lines(
    frequency: float,
    matrix: torch.Tensor,
    data_format: str,
    *,
    max_pairs_per_line: int | None = None,
) -> list[str]:
    port_count = matrix.shape[0]
    if port_count == 2:
        entries = (matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1])
        values = [_format_number(frequency)]
        for entry in entries:
            values.extend(_format_number(item) for item in _format_pair(complex(entry), data_format))
        return [" ".join(values)]

    lines: list[str] = []
    for row in range(port_count):
        pairs: list[list[str]] = []
        for column in range(port_count):
            pairs.append(
                [
                    _format_number(item)
                    for item in _format_pair(complex(matrix[row, column]), data_format)
                ]
            )
        chunk_size = port_count if max_pairs_per_line is None else max_pairs_per_line
        for start in range(0, port_count, chunk_size):
            values = [_format_number(frequency)] if row == 0 and start == 0 else []
            values.extend(
                item
                for pair in pairs[start : start + chunk_size]
                for item in pair
            )
            lines.append(" ".join(values))
    return lines


def read_touchstone(
    path: str | Path,
    *,
    device=None,
    dtype: torch.dtype = torch.complex128,
) -> NetworkData:
    """Read common Touchstone 1.x/2.0 S, Z, or Y network data strictly."""

    if dtype not in {torch.complex64, torch.complex128}:
        raise TypeError("dtype must be torch.complex64 or torch.complex128.")
    input_path = Path(path)
    suffix_port_count = _port_count_from_suffix(input_path)
    if suffix_port_count is None and input_path.suffix.lower() != ".ts":
        raise ValueError("Touchstone path must end in '.sNp' or '.ts'.")
    content, comments, comment_port_names = _source_lines(input_path)
    if not content:
        _parse_error("Touchstone file is empty", path=input_path, line_number=1)

    warnings: list[str] = []
    first_line, first_code = content[0]
    first_keyword = _split_keyword(first_code)
    version = "1.0"
    cursor = 0
    if first_keyword is not None and first_keyword[0] == "version":
        version = first_keyword[1]
        if version != "2.0":
            _parse_error(
                "[Version] must contain the supported value 2.0",
                path=input_path,
                line_number=first_line,
            )
        cursor = 1
    elif first_keyword is not None:
        _parse_error(
            "[Version] 2.0 must be the first non-comment line",
            path=input_path,
            line_number=first_line,
        )
    if cursor >= len(content):
        _parse_error("missing option line", path=input_path, line_number=first_line)
    option_line, option_code = content[cursor]
    options = _parse_option_line(option_code, path=input_path, line_number=option_line)
    cursor += 1

    port_count = suffix_port_count
    declared_frequency_count: int | None = None
    matrix_format = "full"
    two_port_order = "21_12"
    reference_values: list[float] | None = None
    reference_line: int | None = None
    network_data_line: int | None = None
    end_line: int | None = None
    data_lines: list[tuple[int, str]] = []

    if version == "1.0":
        if port_count is None:
            _parse_error(
                "Touchstone 1.x requires an '.sNp' suffix to declare the port count",
                path=input_path,
                line_number=option_line,
            )
        for line_number, code in content[cursor:]:
            if code.startswith("#"):
                warnings.append(f"line {line_number}: ignored additional option line")
                continue
            if _split_keyword(code) is not None:
                _parse_error(
                    "Touchstone 2.0 keywords are not permitted in a 1.x file",
                    path=input_path,
                    line_number=line_number,
                )
            data_lines.append((line_number, code))
        network_data_line = option_line
        end_line = data_lines[-1][0] if data_lines else option_line
    else:
        seen: set[str] = set()
        pending_reference = False
        while cursor < len(content):
            line_number, code = content[cursor]
            keyword = _split_keyword(code)
            if pending_reference and keyword is None:
                assert reference_values is not None
                for token in code.split():
                    reference_values.append(
                        _positive_float(
                            token,
                            path=input_path,
                            line_number=line_number,
                            description="[Reference] impedance",
                        )
                    )
                if port_count is not None and len(reference_values) >= port_count:
                    if len(reference_values) != port_count:
                        _parse_error(
                            f"[Reference] requires exactly {port_count} values",
                            path=input_path,
                            line_number=line_number,
                        )
                    pending_reference = False
                cursor += 1
                continue
            if pending_reference:
                _parse_error(
                    f"[Reference] requires exactly {port_count} values",
                    path=input_path,
                    line_number=line_number,
                )
            if keyword is None:
                _parse_error(
                    "expected a Touchstone 2.0 keyword before [Network Data]",
                    path=input_path,
                    line_number=line_number,
                )
            name, argument = keyword
            if name in seen:
                _parse_error(f"duplicate [{name}] keyword", path=input_path, line_number=line_number)
            if name == "number of ports":
                if seen:
                    _parse_error(
                        "[Number of Ports] must be the first keyword after the option line",
                        path=input_path,
                        line_number=line_number,
                    )
                port_count = _positive_int(
                    argument,
                    path=input_path,
                    line_number=line_number,
                    description="[Number of Ports]",
                )
                if suffix_port_count is not None and port_count != suffix_port_count:
                    _parse_error(
                        f"file suffix declares {suffix_port_count} ports but [Number of Ports] declares {port_count}",
                        path=input_path,
                        line_number=line_number,
                    )
            elif name == "number of frequencies":
                declared_frequency_count = _positive_int(
                    argument,
                    path=input_path,
                    line_number=line_number,
                    description="[Number of Frequencies]",
                )
            elif name == "two-port data order":
                normalized = argument.lower()
                if normalized not in {"21_12", "12_21"}:
                    _parse_error(
                        "[Two-Port Data Order] must be 21_12 or 12_21",
                        path=input_path,
                        line_number=line_number,
                    )
                two_port_order = normalized
            elif name == "matrix format":
                normalized = argument.lower()
                if normalized not in {"full", "lower", "upper"}:
                    _parse_error(
                        "[Matrix Format] must be Full, Lower, or Upper",
                        path=input_path,
                        line_number=line_number,
                    )
                matrix_format = normalized
            elif name == "reference":
                if port_count is None:
                    _parse_error(
                        "[Reference] must follow [Number of Ports]",
                        path=input_path,
                        line_number=line_number,
                    )
                reference_line = line_number
                reference_values = []
                for token in argument.split():
                    reference_values.append(
                        _positive_float(
                            token,
                            path=input_path,
                            line_number=line_number,
                            description="[Reference] impedance",
                        )
                    )
                if len(reference_values) > port_count:
                    _parse_error(
                        f"[Reference] requires exactly {port_count} values",
                        path=input_path,
                        line_number=line_number,
                    )
                pending_reference = len(reference_values) < port_count
            elif name == "network data":
                if argument:
                    _parse_error(
                        "[Network Data] does not accept an argument",
                        path=input_path,
                        line_number=line_number,
                    )
                network_data_line = line_number
                seen.add(name)
                cursor += 1
                break
            elif name in {
                "mixed-mode order",
                "number of noise frequencies",
                "noise data",
                "begin information",
                "end information",
            }:
                _parse_error(
                    f"[{name}] is outside the supported single-ended network subset",
                    path=input_path,
                    line_number=line_number,
                )
            else:
                _parse_error(f"unsupported keyword [{name}]", path=input_path, line_number=line_number)
            seen.add(name)
            cursor += 1

        if pending_reference:
            _parse_error(
                f"[Reference] requires exactly {port_count} values",
                path=input_path,
                line_number=reference_line or option_line,
            )
        if port_count is None:
            _parse_error("missing [Number of Ports]", path=input_path, line_number=option_line)
        if declared_frequency_count is None:
            _parse_error("missing [Number of Frequencies]", path=input_path, line_number=option_line)
        if network_data_line is None:
            _parse_error("missing [Network Data]", path=input_path, line_number=content[-1][0])
        if port_count == 2 and "two-port data order" not in seen:
            _parse_error(
                "2-port Touchstone 2.0 data requires [Two-Port Data Order]",
                path=input_path,
                line_number=network_data_line,
            )
        if port_count != 2 and "two-port data order" in seen:
            _parse_error(
                "[Two-Port Data Order] is only valid for a 2-port file",
                path=input_path,
                line_number=network_data_line,
            )

        while cursor < len(content):
            line_number, code = content[cursor]
            keyword = _split_keyword(code)
            if keyword is None:
                data_lines.append((line_number, code))
                cursor += 1
                continue
            name, argument = keyword
            if name != "end" or argument:
                _parse_error(
                    "only [End] may follow network data in the supported subset",
                    path=input_path,
                    line_number=line_number,
                )
            end_line = line_number
            cursor += 1
            break
        if end_line is None:
            _parse_error("missing [End]", path=input_path, line_number=content[-1][0])
        if cursor != len(content):
            _parse_error(
                "unexpected content after [End]",
                path=input_path,
                line_number=content[cursor][0],
            )

    assert port_count is not None
    frequency_unit = str(options["frequency_unit"])
    parameter = str(options["parameter"])
    data_format = str(options["format"])
    reference = float(options["reference"])
    if reference_values is None:
        reference_values = [reference] * port_count
    if len(reference_values) != port_count:
        _parse_error(
            f"reference impedance count must equal {port_count}",
            path=input_path,
            line_number=reference_line or option_line,
        )

    frequencies, matrix = _parse_data_tokens(
        data_lines,
        port_count=port_count,
        frequency_count=declared_frequency_count,
        frequency_unit=frequency_unit,
        data_format=data_format,
        matrix_format=matrix_format,
        two_port_order=two_port_order,
        version=version,
        path=input_path,
        eof_line=end_line or network_data_line or option_line,
    )
    if version == "1.0" and parameter == "z":
        matrix = matrix * reference
    elif version == "1.0" and parameter == "y":
        matrix = matrix / reference

    if comment_port_names:
        expected_indices = set(range(1, port_count + 1))
        if set(comment_port_names) != expected_indices:
            warnings.append("incomplete Port[n] comment names were ignored")
            names = tuple(str(index) for index in range(1, port_count + 1))
        else:
            resolved_names: list[str] = []
            name_lines: dict[str, int] = {}
            for index in range(1, port_count + 1):
                name, line_number = comment_port_names[index]
                if name in name_lines:
                    _parse_error(
                        f"duplicate resolved port name {name!r}",
                        path=input_path,
                        line_number=line_number,
                    )
                name_lines[name] = line_number
                resolved_names.append(name)
            names = tuple(resolved_names)
    else:
        names = tuple(str(index) for index in range(1, port_count + 1))

    target_device = torch.device("cpu" if device is None else device)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    frequencies = frequencies.to(device=target_device, dtype=real_dtype)
    matrix = matrix.to(device=target_device, dtype=dtype)
    z0 = torch.tensor(reference_values, device=target_device, dtype=real_dtype).to(dtype=dtype)
    metadata = {
        "touchstone": {
            "version": version,
            "format": data_format,
            "parameter": parameter,
            "frequency_unit": frequency_unit,
            "matrix_format": matrix_format,
            "two_port_data_order": two_port_order if port_count == 2 else None,
            "port_order": names,
            "reference_impedances": tuple(reference_values),
            "comments": comments,
            "parser_warnings": tuple(warnings),
            "source_path": str(input_path),
        }
    }
    if parameter == "s":
        return NetworkData(
            frequencies=frequencies,
            s=matrix,
            z0=z0,
            port_names=names,
            metadata=metadata,
        )
    factory = NetworkData.from_z if parameter == "z" else NetworkData.from_y
    return factory(
        frequencies=frequencies,
        **{parameter: matrix},
        z0=z0,
        port_names=names,
        metadata=metadata,
    )


def write_touchstone(
    network: NetworkData,
    path: str | Path,
    format: str = "ri",
    frequency_unit: str = "hz",
    version: str = "auto",
    parameter: str = "s",
) -> Path:
    """Write complete N-port S, Z, or Y data as Touchstone text.

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
    parameter_kind = str(parameter).lower()
    if parameter_kind not in {"s", "z", "y"}:
        raise ValueError("parameter must be 's', 'z', or 'y'.")

    frequencies = network.frequencies.detach().cpu()
    reference = network.z0.detach().cpu()
    valid_columns = network.valid_columns.detach().cpu()
    if parameter_kind == "s":
        matrix = network.s.detach().cpu()
    elif parameter_kind == "z":
        matrix = network.to_z().detach().cpu()
    else:
        matrix = network.to_y().detach().cpu()
    frequency_count, port_count, _ = matrix.shape
    if not bool(torch.all(valid_columns)):
        raise ValueError("Touchstone export requires complete excitation columns.")
    if frequency_count > 1 and not bool(torch.all(torch.diff(frequencies) > 0.0)):
        raise ValueError("Touchstone frequencies must be strictly increasing.")
    if not bool(torch.all(torch.isfinite(matrix.real))) or not bool(
        torch.all(torch.isfinite(matrix.imag))
    ):
        raise ValueError("Touchstone network data must contain only finite values.")
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
    if resolved_version == "1.0" and parameter_kind == "z":
        matrix = matrix / per_port_reference[0]
    elif resolved_version == "1.0" and parameter_kind == "y":
        matrix = matrix * per_port_reference[0]

    unit_scale = _FREQUENCY_SCALES[unit]
    option_reference = _format_number(float(per_port_reference[0]))
    lines: list[str] = []
    for index, name in enumerate(network.port_names, start=1):
        if any(character in "\r\n" or not (" " <= character <= "~") for character in name):
            raise ValueError("Touchstone port names must contain only printable ASCII characters.")
        lines.append(f"! Port[{index}] = {name}")
    if resolved_version == "2.0":
        lines.append("[Version] 2.0")
    lines.append(
        f"# {unit.upper()} {parameter_kind.upper()} {data_format.upper()} R {option_reference}"
    )
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
                matrix[index],
                data_format,
                max_pairs_per_line=4 if resolved_version == "1.0" else None,
            )
        )
    if resolved_version == "2.0":
        lines.append("[End]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return output_path


__all__ = ["TouchstoneParseError", "read_touchstone", "write_touchstone"]
