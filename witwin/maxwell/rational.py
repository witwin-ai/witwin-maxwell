from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .constants import complex_dtype_for, real_dtype_for


_SCHEMA_VERSION = 1


def _finite_complex(value: torch.Tensor) -> bool:
    return bool(torch.all(torch.isfinite(value.real))) and bool(
        torch.all(torch.isfinite(value.imag))
    )


def _requires_grad(*values: torch.Tensor | None) -> bool:
    return any(value is not None and value.requires_grad for value in values)


@dataclass(frozen=True)
class RationalFitConfig:
    """Configuration for shared-pole rational fitting.

    Automatic pole relocation and passivity projection are intentionally
    non-differentiable. Trainable workflows should construct a
    :class:`RationalModel` from pre-fitted poles and optimize its residues and
    direct term instead.
    """

    order: int = 8
    band: tuple[float, float] | None = None
    weights: torch.Tensor | None = None
    iterations: int = 6
    enforce_stability: bool = True
    enforce_passivity: bool = False
    relative_tolerance: float = 1e-3
    passivity_tolerance: float = 1e-9
    enforcement_tolerance: float = 1e-2
    proportional: bool = False
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 1:
            raise ValueError("order must be a positive integer.")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations < 0
        ):
            raise ValueError("iterations must be a non-negative integer.")
        if self.band is not None:
            if len(self.band) != 2 or not (0.0 <= self.band[0] < self.band[1]):
                raise ValueError("band must be an increasing non-negative frequency pair.")
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive.")
        if self.passivity_tolerance < 0.0:
            raise ValueError("passivity_tolerance must be non-negative.")
        if self.enforcement_tolerance < 0.0:
            raise ValueError("enforcement_tolerance must be non-negative.")
        if self.weights is not None:
            if not isinstance(self.weights, torch.Tensor):
                raise TypeError("weights must be a torch.Tensor when provided.")
            if self.weights.ndim != 1 or self.weights.is_complex():
                raise ValueError("weights must be a real one-dimensional tensor.")
            if not bool(torch.all(torch.isfinite(self.weights))) or not bool(
                torch.all(self.weights > 0.0)
            ):
                raise ValueError("weights must contain finite positive values.")


@dataclass(frozen=True)
class FitReport:
    """Auditable numerical report for a rational fit or projection."""

    rms_error: float
    max_error: float
    relative_rms_error: float
    relative_max_error: float
    frequency_band: tuple[float, float]
    order: int
    iterations: int
    unstable_poles: int = 0
    passivity_margin: float | None = None
    max_passivity_violation: float | None = None
    condition_numbers: tuple[float, ...] = ()
    enforcement_change: float = 0.0
    passivity_enforced: bool = False
    differentiable_parameters: tuple[str, ...] = ("residues", "direct")
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkFitReport(FitReport):
    """Multiport specialization of :class:`FitReport`."""

    port_count: int = 0
    delay_seconds: tuple[float, ...] = ()
    delay_estimation_rank: int | None = None
    delay_equation_count: int = 0
    delay_residual_seconds: float | None = None
    delay_phase_error_degrees: float | None = None
    delay_reembedding_max_error: float | None = None


@dataclass(frozen=True)
class PassivityReport:
    passive: bool
    margin: float
    max_violation: float
    tolerance: float
    sample_count: int
    enforcement_change: float = 0.0
    certified: bool = False
    method: str = "sampled"


def _report_from_payload(payload: Mapping[str, Any] | None) -> FitReport | None:
    if payload is None:
        return None
    kind = payload.get("kind", "FitReport")
    values = dict(payload["values"])
    if kind == "NetworkFitReport":
        return NetworkFitReport(**values)
    if kind != "FitReport":
        raise ValueError(f"Unsupported rational fit report type {kind!r}.")
    return FitReport(**values)


def _report_payload(report: FitReport | None) -> Mapping[str, Any] | None:
    if report is None:
        return None
    return {"kind": type(report).__name__, "values": asdict(report)}


def _conjugate_groups(
    poles: torch.Tensor,
    *,
    tolerance: float = 1e-7,
) -> list[tuple[int, int | None]]:
    scale = max(1.0, float(torch.max(torch.abs(poles)).item()))
    threshold = tolerance * scale
    unused = set(range(poles.numel()))
    groups: list[tuple[int, int | None]] = []
    while unused:
        index = min(unused)
        unused.remove(index)
        pole = poles[index]
        if abs(float(pole.imag.item())) <= threshold:
            groups.append((index, None))
            continue
        candidates = list(unused)
        if not candidates:
            raise ValueError("Every non-real pole must have a conjugate partner.")
        distances = torch.abs(poles[candidates] - torch.conj(pole))
        location = int(torch.argmin(distances).item())
        partner = candidates[location]
        if float(distances[location].item()) > threshold:
            raise ValueError("Every non-real pole must have a conjugate partner.")
        unused.remove(partner)
        if float(pole.imag.item()) < 0.0:
            index, partner = partner, index
        groups.append((index, partner))
    return groups


@dataclass(frozen=True)
class RationalModel:
    """Shared-pole real rational transfer matrix.

    ``residues`` has shape ``[Nout, Nin, order]``. Scalar responses may use
    shape ``[order]`` and are normalized to a 1x1 transfer matrix. Frequency
    evaluation follows the repository convention ``s = -i 2 pi f``.
    """

    poles: torch.Tensor
    residues: torch.Tensor
    direct: torch.Tensor | float | complex = 0.0
    proportional: torch.Tensor | float | complex | None = None
    representation: str = "Y"
    report: FitReport | None = None

    def __post_init__(self) -> None:
        poles = torch.as_tensor(self.poles)
        residues = torch.as_tensor(self.residues, device=poles.device)
        dtype = complex_dtype_for(torch.promote_types(poles.dtype, residues.dtype))
        poles = poles.to(dtype=dtype)
        residues = residues.to(dtype=dtype)
        if poles.ndim != 1 or poles.numel() == 0:
            raise ValueError("poles must have non-empty shape [order].")
        if residues.ndim == 1:
            residues = residues.reshape(1, 1, -1)
        if residues.ndim != 3 or residues.shape[-1] != poles.numel():
            raise ValueError("residues must have shape [Nout, Nin, order].")

        direct = torch.as_tensor(self.direct, device=poles.device).to(dtype=dtype)
        if direct.ndim == 0:
            direct = direct.expand(residues.shape[:2])
        if direct.shape != residues.shape[:2]:
            raise ValueError("direct must be scalar or have shape [Nout, Nin].")
        if self.proportional is None:
            proportional = torch.zeros_like(direct)
        else:
            proportional = torch.as_tensor(self.proportional, device=poles.device).to(
                dtype=dtype
            )
            if proportional.ndim == 0:
                proportional = proportional.expand(residues.shape[:2])
            if proportional.shape != residues.shape[:2]:
                raise ValueError("proportional must be scalar or have shape [Nout, Nin].")

        if not _finite_complex(poles) or not _finite_complex(residues):
            raise ValueError("poles and residues must contain only finite values.")
        if not _finite_complex(direct) or not _finite_complex(proportional):
            raise ValueError("direct and proportional must contain only finite values.")
        groups = _conjugate_groups(poles)
        tolerance = 128.0 * torch.finfo(real_dtype_for(dtype)).eps
        for first, second in groups:
            if second is None:
                if float(torch.max(torch.abs(residues[..., first].imag)).item()) > tolerance:
                    raise ValueError("Residues at real poles must be real.")
            elif float(
                torch.max(torch.abs(residues[..., second] - residues[..., first].conj())).item()
            ) > tolerance:
                raise ValueError("Residues at conjugate poles must be conjugates.")
        if float(torch.max(torch.abs(direct.imag)).item()) > tolerance:
            raise ValueError("direct must be real for a real realization.")
        if float(torch.max(torch.abs(proportional.imag)).item()) > tolerance:
            raise ValueError("proportional must be real for a real realization.")

        representation = self.representation.upper()
        if representation not in {"Y", "Z", "S"}:
            raise ValueError("representation must be 'Y', 'Z', or 'S'.")
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "direct", direct)
        object.__setattr__(self, "proportional", proportional)
        object.__setattr__(self, "representation", representation)

    @property
    def order(self) -> int:
        return self.poles.numel()

    @property
    def output_count(self) -> int:
        return self.residues.shape[0]

    @property
    def input_count(self) -> int:
        return self.residues.shape[1]

    @property
    def is_stable(self) -> bool:
        return bool(torch.all(self.poles.real < 0.0))

    def evaluate(self, frequencies: torch.Tensor | Sequence[float]) -> torch.Tensor:
        frequencies = torch.as_tensor(
            frequencies,
            device=self.poles.device,
            dtype=real_dtype_for(self.poles.dtype),
        )
        if frequencies.ndim != 1 or frequencies.numel() == 0:
            raise ValueError("frequencies must have non-empty shape [F].")
        if not bool(torch.all(torch.isfinite(frequencies))) or not bool(
            torch.all(frequencies >= 0.0)
        ):
            raise ValueError("frequencies must be finite and non-negative.")
        s = torch.complex(torch.zeros_like(frequencies), -2.0 * torch.pi * frequencies)
        basis = 1.0 / (s[:, None] - self.poles[None, :])
        response = torch.einsum("fk,oik->foi", basis, self.residues)
        return response + self.direct[None, ...] + s[:, None, None] * self.proportional

    def to_state_space(
        self,
        *,
        port_order: Sequence[str] | None = None,
        require_stable: bool = True,
    ) -> StateSpaceNetwork:
        if self.output_count != self.input_count:
            raise ValueError("A network state-space realization must be square.")
        if bool(torch.any(torch.abs(self.proportional) > 0.0)):
            raise ValueError(
                "A nonzero proportional term has no finite standard A/B/C/D realization."
            )
        if require_stable and not self.is_stable:
            raise ValueError("A stable state-space realization requires Re(pole) < 0.")

        groups = _conjugate_groups(self.poles)
        real_dtype = real_dtype_for(self.poles.dtype)
        device = self.poles.device
        port_count = self.input_count
        state_per_input = self.order
        blocks: list[torch.Tensor] = []
        basis_inputs: list[torch.Tensor] = []
        residue_columns: list[list[torch.Tensor]] = [[] for _ in range(port_count)]
        for first, second in groups:
            pole = self.poles[first]
            if second is None:
                blocks.append(pole.real.reshape(1, 1))
                basis_inputs.append(torch.ones((1, 1), dtype=real_dtype, device=device))
                for input_index in range(port_count):
                    residue_columns[input_index].append(
                        self.residues[:, input_index, first].real[:, None]
                    )
            else:
                a = pole.real
                b = pole.imag
                blocks.append(torch.stack((torch.stack((a, -b)), torch.stack((b, a)))))
                basis_inputs.append(
                    torch.tensor([[1.0], [0.0]], dtype=real_dtype, device=device)
                )
                for input_index in range(port_count):
                    residue = 0.5 * (
                        self.residues[:, input_index, first]
                        + self.residues[:, input_index, second].conj()
                    )
                    residue_columns[input_index].append(
                        torch.stack((2.0 * residue.real, -2.0 * residue.imag), dim=1)
                    )
        pole_block = torch.block_diag(*blocks)
        input_block = torch.cat(basis_inputs, dim=0)
        if pole_block.shape != (state_per_input, state_per_input):
            raise RuntimeError("Internal pole realization has an inconsistent state count.")
        A = torch.block_diag(*(pole_block for _ in range(port_count)))
        B = torch.zeros(
            (port_count * state_per_input, port_count), dtype=real_dtype, device=device
        )
        C_blocks: list[torch.Tensor] = []
        for input_index in range(port_count):
            start = input_index * state_per_input
            B[start : start + state_per_input, input_index : input_index + 1] = input_block
            C_blocks.append(torch.cat(residue_columns[input_index], dim=1))
        C = torch.cat(C_blocks, dim=1)
        D = self.direct.real
        names = tuple(port_order) if port_order is not None else tuple(
            str(index + 1) for index in range(port_count)
        )
        return StateSpaceNetwork(
            A=A,
            B=B,
            C=C,
            D=D,
            representation=self.representation,
            port_order=names,
            report=self.report,
            passivity_margin=(
                None if self.report is None else self.report.passivity_margin
            ),
        )

    def check_passivity(
        self,
        frequencies: torch.Tensor | Sequence[float],
        *,
        tolerance: float = 1e-9,
    ) -> PassivityReport:
        return _check_rational_band_passivity(self, frequencies, tolerance=tolerance)

    def enforce_passivity(
        self,
        frequencies: torch.Tensor | Sequence[float],
        *,
        tolerance: float = 1e-9,
    ) -> tuple[RationalModel, PassivityReport]:
        if _requires_grad(self.poles, self.residues, self.direct, self.proportional):
            raise RuntimeError(
                "Passivity enforcement is non-differentiable; use pre-fitted fixed poles "
                "and optimize residues/direct without projection."
            )
        before = self.check_passivity(frequencies, tolerance=tolerance)
        if before.passive and before.certified:
            return self, before
        if before.passive:
            raise RuntimeError(
                "Passivity could not be certified over the requested band; "
                "enforcement will not accept an unresolved sampled result."
            )
        if self.representation not in {"Y", "Z"}:
            raise ValueError("Automatic sampled enforcement is only defined for Y or Z models.")
        if self.input_count != self.output_count:
            raise ValueError("Positive-real enforcement requires a square transfer matrix.")
        headroom = max(tolerance, 1e-2 * max(before.max_violation, 1.0))
        shift = before.max_violation + headroom
        identity = torch.eye(
            self.input_count, dtype=self.direct.dtype, device=self.direct.device
        )
        updated_direct = self.direct + shift * identity
        denominator = max(float(torch.linalg.vector_norm(self.direct).item()), 1.0)
        change = float(torch.linalg.vector_norm(updated_direct - self.direct).item()) / denominator
        updated = replace(self, direct=updated_direct)
        after = updated.check_passivity(frequencies, tolerance=tolerance)
        if not after.passive or not after.certified:
            raise RuntimeError(
                "Passivity enforcement did not produce a certified in-band model."
            )
        report = replace(after, enforcement_change=change)
        return updated, report

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "data_type": "RationalModel",
            "poles": self.poles.detach().cpu(),
            "residues": self.residues.detach().cpu(),
            "direct": self.direct.detach().cpu(),
            "proportional": self.proportional.detach().cpu(),
            "representation": self.representation,
            "report": _report_payload(self.report),
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> RationalModel:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or payload.get("data_type") != "RationalModel":
            raise ValueError("Persisted file does not contain a RationalModel.")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported RationalModel persistence schema.")
        return cls(
            poles=payload["poles"],
            residues=payload["residues"],
            direct=payload["direct"],
            proportional=payload["proportional"],
            representation=payload["representation"],
            report=_report_from_payload(payload.get("report")),
        )

    @classmethod
    def fit(
        cls,
        frequencies: torch.Tensor,
        values: torch.Tensor,
        config: RationalFitConfig | None = None,
        *,
        representation: str = "Y",
        initial_poles: torch.Tensor | None = None,
    ) -> RationalModel:
        return fit_rational(
            frequencies,
            values,
            config=config,
            representation=representation,
            initial_poles=initial_poles,
        )


@dataclass(frozen=True)
class StateSpaceNetwork:
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    representation: str = "Y"
    port_order: tuple[str, ...] = ()
    passivity_margin: float | None = None
    report: FitReport | None = None

    def __post_init__(self) -> None:
        tensors = (self.A, self.B, self.C, self.D)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("A, B, C, and D must be torch.Tensor instances.")
        if any(value.is_complex() or not value.dtype.is_floating_point for value in tensors):
            raise TypeError("A, B, C, and D must be real floating-point tensors.")
        if len({value.device for value in tensors}) != 1 or len(
            {value.dtype for value in tensors}
        ) != 1:
            raise ValueError("A, B, C, and D must share device and dtype.")
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square.")
        state_count = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != state_count:
            raise ValueError("B must have shape [Nstate, Nin].")
        if self.C.ndim != 2 or self.C.shape[1] != state_count:
            raise ValueError("C must have shape [Nout, Nstate].")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D must have shape [Nout, Nin].")
        if not all(bool(torch.all(torch.isfinite(value))) for value in tensors):
            raise ValueError("A, B, C, and D must contain only finite values.")
        if self.port_order and len(self.port_order) != self.B.shape[1]:
            raise ValueError("port_order length must match the number of inputs.")

    @property
    def state_count(self) -> int:
        return self.A.shape[0]

    @property
    def input_count(self) -> int:
        return self.B.shape[1]

    @property
    def output_count(self) -> int:
        return self.C.shape[0]

    def evaluate(self, frequencies: torch.Tensor | Sequence[float]) -> torch.Tensor:
        frequencies = torch.as_tensor(
            frequencies, device=self.A.device, dtype=self.A.dtype
        )
        if frequencies.ndim != 1 or frequencies.numel() == 0:
            raise ValueError("frequencies must have non-empty shape [F].")
        if not bool(torch.all(torch.isfinite(frequencies))) or not bool(
            torch.all(frequencies >= 0.0)
        ):
            raise ValueError("frequencies must be finite and non-negative.")
        if self.state_count == 0:
            return self.D.to(dtype=complex_dtype_for(self.D.dtype))[None, ...].expand(
                frequencies.numel(), -1, -1
            )
        s = torch.complex(torch.zeros_like(frequencies), -2.0 * torch.pi * frequencies)
        complex_dtype = complex_dtype_for(self.A.dtype)
        A = self.A.to(dtype=complex_dtype)
        B = self.B.to(dtype=complex_dtype)
        C = self.C.to(dtype=complex_dtype)
        D = self.D.to(dtype=complex_dtype)
        identity = torch.eye(self.state_count, dtype=complex_dtype, device=self.A.device)
        system = s[:, None, None] * identity[None, ...] - A[None, ...]
        solved = torch.linalg.solve(system, B.expand(frequencies.numel(), -1, -1))
        return torch.einsum("os,fsi->foi", C, solved) + D[None, ...]

    def check_passivity(
        self,
        frequencies: torch.Tensor | Sequence[float],
        *,
        tolerance: float = 1e-9,
    ) -> PassivityReport:
        """Certify passivity between samples with resolvent interval bounds."""

        return _check_state_space_band_passivity(
            self,
            frequencies,
            tolerance=tolerance,
        )

    def discretize(
        self,
        dt: float,
        *,
        stability_margin: float = 1e-7,
    ) -> DiscreteStateSpaceNetwork:
        return bilinear_discretize(self, dt, stability_margin=stability_margin)


@dataclass(frozen=True)
class DiscreteStateSpaceNetwork:
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    dt: float
    representation: str = "Y"
    port_order: tuple[str, ...] = ()
    pole_radius: float = 0.0
    passivity_margin: float | None = None
    report: FitReport | None = None

    @property
    def state_count(self) -> int:
        return self.A.shape[0]

    def step(
        self,
        state: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = self.A @ state + self.B @ value
        output = self.C @ state + self.D @ value
        return next_state, output


def _sample_passivity_margins(
    response: torch.Tensor,
    *,
    representation: str,
) -> torch.Tensor:
    if representation in {"Y", "Z"}:
        hermitian = 0.5 * (response + response.mH)
        return torch.min(torch.linalg.eigvalsh(hermitian).real, dim=-1).values
    if representation == "S":
        return 1.0 - torch.max(torch.linalg.svdvals(response), dim=-1).values
    raise ValueError("representation must be 'Y', 'Z', or 'S'.")


def _interval_lipschitz_bound(
    model: RationalModel,
    intervals: torch.Tensor,
) -> torch.Tensor:
    """Bound ``||dH/df||_2`` over each interval from pole distances."""

    lower = intervals[:, 0]
    upper = intervals[:, 1]
    omega_lower = -2.0 * torch.pi * upper
    omega_upper = -2.0 * torch.pi * lower
    pole_imag = model.poles.imag[None, :]
    closest_imag = torch.minimum(
        torch.maximum(pole_imag, omega_lower[:, None]),
        omega_upper[:, None],
    )
    distance_squared = (
        model.poles.real[None, :] ** 2 + (pole_imag - closest_imag) ** 2
    )
    residue_norms = torch.linalg.matrix_norm(
        model.residues.permute(2, 0, 1),
        ord=2,
    )
    pole_bound = 2.0 * torch.pi * torch.sum(
        residue_norms[None, :] / distance_squared,
        dim=1,
    )
    proportional_bound = 2.0 * torch.pi * torch.linalg.matrix_norm(
        model.proportional,
        ord=2,
    )
    return pole_bound + proportional_bound


def _check_rational_band_passivity(
    model: RationalModel,
    frequencies: torch.Tensor | Sequence[float],
    *,
    tolerance: float,
    max_depth: int = 48,
    max_intervals: int = 8192,
) -> PassivityReport:
    """Use pole-aware interval bounds to verify or refute in-band passivity."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    real_dtype = real_dtype_for(model.poles.dtype)
    values = torch.as_tensor(frequencies, device=model.poles.device, dtype=real_dtype)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(torch.isfinite(values))) or not bool(torch.all(values >= 0.0)):
        raise ValueError("frequencies must be finite and non-negative.")
    if not model.is_stable:
        return PassivityReport(
            passive=False,
            margin=float("-inf"),
            max_violation=float("inf"),
            tolerance=float(tolerance),
            sample_count=0,
            certified=True,
            method="stability gate",
        )

    lower = torch.min(values)
    upper = torch.max(values)
    resonances = -model.poles.imag / (2.0 * torch.pi)
    resonances = resonances[(resonances > lower) & (resonances < upper)]
    points = torch.unique(torch.cat((values, resonances.to(dtype=real_dtype))), sorted=True)
    point_margins = _sample_passivity_margins(
        model.evaluate(points),
        representation=model.representation,
    )
    minimum_margin = torch.min(point_margins)
    sample_count = points.numel()
    if bool(minimum_margin < -tolerance):
        margin = float(minimum_margin.item())
        return PassivityReport(
            passive=False,
            margin=margin,
            max_violation=max(0.0, -margin),
            tolerance=float(tolerance),
            sample_count=sample_count,
            certified=True,
            method="pole-aware interval bound",
        )
    if points.numel() == 1:
        margin = float(minimum_margin.item())
        return PassivityReport(
            passive=margin >= -tolerance,
            margin=margin,
            max_violation=max(0.0, -margin),
            tolerance=float(tolerance),
            sample_count=sample_count,
            certified=False,
            method="single-frequency sample",
        )

    intervals = torch.stack((points[:-1], points[1:]), dim=1)
    unresolved = intervals
    certified = True
    for _ in range(max_depth):
        if unresolved.numel() == 0:
            break
        if unresolved.shape[0] > max_intervals:
            certified = False
            break
        midpoint = torch.mean(unresolved, dim=1)
        margins = _sample_passivity_margins(
            model.evaluate(midpoint),
            representation=model.representation,
        )
        sample_count += midpoint.numel()
        minimum_margin = torch.minimum(minimum_margin, torch.min(margins))
        if bool(torch.any(margins < -tolerance)):
            margin = float(minimum_margin.item())
            return PassivityReport(
                passive=False,
                margin=margin,
                max_violation=max(0.0, -margin),
                tolerance=float(tolerance),
                sample_count=sample_count,
                certified=True,
                method="pole-aware interval bound",
            )
        radius = 0.5 * (unresolved[:, 1] - unresolved[:, 0])
        lower_bound = margins - _interval_lipschitz_bound(model, unresolved) * radius
        needs_split = lower_bound < -tolerance
        if not bool(torch.any(needs_split)):
            unresolved = unresolved[:0]
            break
        selected = unresolved[needs_split]
        selected_midpoint = midpoint[needs_split]
        left = torch.stack((selected[:, 0], selected_midpoint), dim=1)
        right = torch.stack((selected_midpoint, selected[:, 1]), dim=1)
        unresolved = torch.cat((left, right), dim=0)
    else:
        certified = False

    margin = float(minimum_margin.item())
    return PassivityReport(
        passive=margin >= -tolerance,
        margin=margin,
        max_violation=max(0.0, -margin),
        tolerance=float(tolerance),
        sample_count=sample_count,
        certified=certified and unresolved.numel() == 0,
        method="pole-aware interval bound",
    )


def _state_space_interval_lipschitz_bound(
    model: StateSpaceNetwork,
    intervals: torch.Tensor,
) -> torch.Tensor:
    """Bound ``||dH/df||_2`` with a uniform resolvent Neumann bound."""

    if model.state_count == 0:
        return torch.zeros(
            intervals.shape[0],
            dtype=intervals.dtype,
            device=intervals.device,
        )
    midpoint = torch.mean(intervals, dim=1)
    radius_hz = 0.5 * (intervals[:, 1] - intervals[:, 0])
    omega = -2.0 * torch.pi * midpoint
    complex_dtype = complex_dtype_for(model.A.dtype)
    identity = torch.eye(
        model.state_count,
        dtype=complex_dtype,
        device=model.A.device,
    )
    operator = torch.complex(torch.zeros_like(omega), omega)
    system = (
        operator[:, None, None] * identity[None, ...]
        - model.A.to(dtype=complex_dtype)[None, ...]
    )
    resolvent = torch.linalg.solve(
        system,
        identity.expand(intervals.shape[0], -1, -1),
    )
    midpoint_norm = torch.linalg.matrix_norm(resolvent, ord=2)
    neumann_ratio = 2.0 * torch.pi * radius_hz * midpoint_norm
    denominator = 1.0 - neumann_ratio
    uniform_norm = torch.where(
        denominator > 0.0,
        midpoint_norm / denominator,
        torch.full_like(midpoint_norm, torch.inf),
    )
    input_norm = torch.linalg.matrix_norm(model.B, ord=2)
    output_norm = torch.linalg.matrix_norm(model.C, ord=2)
    return 2.0 * torch.pi * output_norm * input_norm * uniform_norm.square()


def _check_state_space_band_passivity(
    model: StateSpaceNetwork,
    frequencies: torch.Tensor | Sequence[float],
    *,
    tolerance: float,
    max_depth: int = 48,
    max_intervals: int = 8192,
) -> PassivityReport:
    """Verify or refute in-band state-space passivity between input samples."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    values = torch.as_tensor(
        frequencies,
        device=model.A.device,
        dtype=model.A.dtype,
    )
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(torch.isfinite(values))) or not bool(torch.all(values >= 0.0)):
        raise ValueError("frequencies must be finite and non-negative.")
    if model.state_count and bool(
        torch.any(torch.linalg.eigvals(model.A.clone()).real >= 0.0)
    ):
        return PassivityReport(
            passive=False,
            margin=float("-inf"),
            max_violation=float("inf"),
            tolerance=float(tolerance),
            sample_count=0,
            certified=True,
            method="stability gate",
        )

    points = torch.unique(values, sorted=True)
    point_margins = _sample_passivity_margins(
        model.evaluate(points),
        representation=model.representation,
    )
    minimum_margin = torch.min(point_margins)
    sample_count = points.numel()
    if bool(minimum_margin < -tolerance):
        margin = float(minimum_margin.item())
        return PassivityReport(
            passive=False,
            margin=margin,
            max_violation=max(0.0, -margin),
            tolerance=float(tolerance),
            sample_count=sample_count,
            certified=True,
            method="state-space resolvent interval bound",
        )
    if points.numel() == 1:
        margin = float(minimum_margin.item())
        return PassivityReport(
            passive=margin >= -tolerance,
            margin=margin,
            max_violation=max(0.0, -margin),
            tolerance=float(tolerance),
            sample_count=sample_count,
            certified=False,
            method="single-frequency sample",
        )

    unresolved = torch.stack((points[:-1], points[1:]), dim=1)
    certified = True
    for _ in range(max_depth):
        if unresolved.numel() == 0:
            break
        if unresolved.shape[0] > max_intervals:
            certified = False
            break
        midpoint = torch.mean(unresolved, dim=1)
        margins = _sample_passivity_margins(
            model.evaluate(midpoint),
            representation=model.representation,
        )
        sample_count += midpoint.numel()
        minimum_margin = torch.minimum(minimum_margin, torch.min(margins))
        if bool(torch.any(margins < -tolerance)):
            margin = float(minimum_margin.item())
            return PassivityReport(
                passive=False,
                margin=margin,
                max_violation=max(0.0, -margin),
                tolerance=float(tolerance),
                sample_count=sample_count,
                certified=True,
                method="state-space resolvent interval bound",
            )
        radius = 0.5 * (unresolved[:, 1] - unresolved[:, 0])
        lower_bound = (
            margins
            - _state_space_interval_lipschitz_bound(model, unresolved) * radius
        )
        needs_split = lower_bound < -tolerance
        if not bool(torch.any(needs_split)):
            unresolved = unresolved[:0]
            break
        selected = unresolved[needs_split]
        selected_midpoint = midpoint[needs_split]
        unresolved = torch.cat(
            (
                torch.stack((selected[:, 0], selected_midpoint), dim=1),
                torch.stack((selected_midpoint, selected[:, 1]), dim=1),
            ),
            dim=0,
        )
    else:
        certified = False

    margin = float(minimum_margin.item())
    return PassivityReport(
        passive=margin >= -tolerance,
        margin=margin,
        max_violation=max(0.0, -margin),
        tolerance=float(tolerance),
        sample_count=sample_count,
        certified=certified and unresolved.numel() == 0,
        method="state-space resolvent interval bound",
    )


def bilinear_discretize(
    system: StateSpaceNetwork,
    dt: float,
    *,
    stability_margin: float = 1e-7,
) -> DiscreteStateSpaceNetwork:
    """Discretize a continuous realization using the bilinear transform."""

    if not isinstance(dt, (float, int)) or isinstance(dt, bool) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar.")
    if not torch.isfinite(torch.tensor(float(dt))):
        raise ValueError("dt must be finite.")
    if not 0.0 < stability_margin < 1.0:
        raise ValueError("stability_margin must lie strictly between zero and one.")
    identity = torch.eye(system.state_count, dtype=system.A.dtype, device=system.A.device)
    left = identity - 0.5 * float(dt) * system.A
    right = identity + 0.5 * float(dt) * system.A
    Ad = torch.linalg.solve(left, right)
    Bd = torch.linalg.solve(left, float(dt) * system.B)
    Cd = torch.linalg.solve(left.T, system.C.T).T
    Dd = system.D + 0.5 * float(dt) * (Cd @ system.B)
    # Some CUDA eigensolver paths use an in-place Schur workspace.  Keep the
    # transition matrix isolated from that workspace because it is returned to
    # the time-step runtime below.
    poles = torch.linalg.eigvals(Ad.clone())
    radius = 0.0 if poles.numel() == 0 else float(torch.max(torch.abs(poles)).item())
    if radius >= 1.0 - stability_margin:
        raise ValueError(
            "Bilinear discretization failed the strict pole gate: "
            f"max |z|={radius:.9g}, required < {1.0 - stability_margin:.9g}."
        )
    return DiscreteStateSpaceNetwork(
        A=Ad,
        B=Bd,
        C=Cd,
        D=Dd,
        dt=float(dt),
        representation=system.representation,
        port_order=system.port_order,
        pole_radius=radius,
        passivity_margin=system.passivity_margin,
        report=system.report,
    )


def check_sampled_passivity(
    response: torch.Tensor,
    *,
    representation: str = "Y",
    tolerance: float = 1e-9,
) -> PassivityReport:
    """Check positive-real or scattering passivity at sampled frequencies."""

    if not isinstance(response, torch.Tensor) or not response.is_complex():
        raise TypeError("response must be a complex torch.Tensor.")
    if response.ndim != 3 or response.shape[1] != response.shape[2]:
        raise ValueError("response must have shape [F, N, N].")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    representation = representation.upper()
    if representation in {"Y", "Z"}:
        hermitian = 0.5 * (response + response.mH)
        margin_tensor = torch.min(torch.linalg.eigvalsh(hermitian).real)
        margin = float(margin_tensor.item())
    elif representation == "S":
        largest = torch.max(torch.linalg.svdvals(response))
        margin = float((1.0 - largest).item())
    else:
        raise ValueError("representation must be 'Y', 'Z', or 'S'.")
    violation = max(0.0, -margin)
    return PassivityReport(
        passive=margin >= -tolerance,
        margin=margin,
        max_violation=violation,
        tolerance=float(tolerance),
        sample_count=response.shape[0],
    )


def _parameter_basis(
    s: torch.Tensor,
    poles: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[int, int | None]]]:
    groups = _conjugate_groups(poles)
    columns: list[torch.Tensor] = []
    for first, second in groups:
        first_basis = 1.0 / (s - poles[first])
        if second is None:
            columns.append(first_basis)
        else:
            second_basis = 1.0 / (s - poles[second])
            columns.append(first_basis + second_basis)
            columns.append(1j * (first_basis - second_basis))
    return torch.stack(columns, dim=1), groups


def _parameters_to_residues(
    parameters: torch.Tensor,
    groups: list[tuple[int, int | None]],
    order: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    leading = parameters.shape[:-1]
    residues = torch.zeros((*leading, order), dtype=dtype, device=parameters.device)
    cursor = 0
    for first, second in groups:
        if second is None:
            residues[..., first] = parameters[..., cursor].to(dtype=dtype)
            cursor += 1
        else:
            value = torch.complex(parameters[..., cursor], parameters[..., cursor + 1])
            residues[..., first] = value
            residues[..., second] = value.conj()
            cursor += 2
    if cursor != parameters.shape[-1]:
        raise RuntimeError("Internal rational parameter count is inconsistent.")
    return residues


def _scaled_real_lstsq(
    matrix: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    real_matrix = torch.cat((matrix.real, matrix.imag), dim=0)
    real_target = torch.cat((target.real, target.imag), dim=0)
    scales = torch.linalg.vector_norm(real_matrix, dim=0)
    if bool(torch.any(scales == 0.0)):
        raise RuntimeError("Rational fit produced an unconstrained coefficient column.")
    scaled = real_matrix / scales[None, :]
    solution = torch.linalg.lstsq(scaled, real_target).solution / scales
    condition = float(torch.linalg.cond(scaled).item())
    return solution, condition


def _initial_poles(
    order: int,
    frequencies: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    positive = frequencies[frequencies > 0.0]
    if positive.numel() == 0:
        raise ValueError("Rational fitting requires at least one positive frequency.")
    lower = torch.min(positive)
    upper = torch.max(positive)
    if lower == upper:
        lower = lower * 0.5
        upper = upper * 2.0
    pair_count = order // 2
    poles: list[torch.Tensor] = []
    if order % 2:
        center = 2.0 * torch.pi * torch.sqrt(lower * upper)
        poles.append(torch.complex(-center, torch.zeros_like(center)))
    if pair_count:
        samples = torch.logspace(
            torch.log10(lower),
            torch.log10(upper),
            pair_count,
            dtype=frequencies.dtype,
            device=frequencies.device,
        )
        omega = 2.0 * torch.pi * samples
        damping = torch.maximum(0.05 * omega, 2.0 * torch.pi * lower * 1e-3)
        for real, imag in zip(-damping, omega):
            positive_pole = torch.complex(real, imag)
            poles.extend((positive_pole, positive_pole.conj()))
    return torch.stack(poles).to(dtype=dtype)


def _canonicalize_relocated_poles(
    poles: torch.Tensor,
    *,
    enforce_stability: bool,
) -> torch.Tensor:
    scale = max(1.0, float(torch.max(torch.abs(poles)).item()))
    imag_tolerance = 1e-6 * scale
    unused = set(range(poles.numel()))
    real_poles: list[torch.Tensor] = []
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    while unused:
        index = min(unused)
        unused.remove(index)
        pole = poles[index]
        if abs(float(pole.imag.item())) <= imag_tolerance:
            real_poles.append(pole.real)
            continue
        candidates = list(unused)
        if not candidates:
            real_poles.append(pole.real)
            continue
        distances = torch.abs(poles[candidates] - pole.conj())
        partner = candidates[int(torch.argmin(distances).item())]
        unused.remove(partner)
        mate = poles[partner]
        real = 0.5 * (pole.real + mate.real)
        imag = 0.5 * (torch.abs(pole.imag) + torch.abs(mate.imag))
        pairs.append((real, imag))

    floor = torch.tensor(1e-10 * scale, dtype=poles.real.dtype, device=poles.device)

    def stable_real(value: torch.Tensor) -> torch.Tensor:
        if not enforce_stability:
            return value
        return -torch.maximum(torch.abs(value), floor)

    output: list[torch.Tensor] = []
    for value in sorted(real_poles, key=lambda item: abs(float(item.item()))):
        real = stable_real(value)
        output.append(torch.complex(real, torch.zeros_like(real)))
    for real_value, imag_value in sorted(
        pairs, key=lambda item: abs(float(item[1].item()))
    ):
        real = stable_real(real_value)
        positive = torch.complex(real, imag_value)
        output.extend((positive, positive.conj()))
    if len(output) != poles.numel():
        raise RuntimeError("Pole relocation changed the rational model order.")
    return torch.stack(output).to(dtype=poles.dtype)


def _relocate_poles(
    s: torch.Tensor,
    values: torch.Tensor,
    poles: torch.Tensor,
    weights: torch.Tensor,
    *,
    proportional: bool,
    enforce_stability: bool,
) -> tuple[torch.Tensor, float]:
    basis, groups = _parameter_basis(s, poles)
    sample_count, response_count = values.shape
    basis_count = basis.shape[1]
    numerator_count = basis_count + 1 + int(proportional)
    column_count = response_count * numerator_count + basis_count
    matrix = torch.zeros(
        (sample_count * response_count, column_count),
        dtype=poles.dtype,
        device=poles.device,
    )
    target = torch.zeros(
        sample_count * response_count, dtype=poles.dtype, device=poles.device
    )
    for response_index in range(response_count):
        rows = slice(response_index * sample_count, (response_index + 1) * sample_count)
        columns = slice(
            response_index * numerator_count,
            (response_index + 1) * numerator_count,
        )
        pieces = [basis, torch.ones_like(s[:, None])]
        if proportional:
            pieces.append(s[:, None])
        matrix[rows, columns] = torch.cat(pieces, dim=1)
        matrix[rows, response_count * numerator_count :] = (
            -values[:, response_index, None] * basis
        )
        target[rows] = values[:, response_index]
    repeated_weights = weights.repeat(response_count)
    matrix = matrix * repeated_weights[:, None]
    target = target * repeated_weights
    solution, condition = _scaled_real_lstsq(matrix, target)
    denominator_parameters = solution[response_count * numerator_count :]
    denominator_residues = _parameters_to_residues(
        denominator_parameters,
        groups,
        poles.numel(),
        dtype=poles.dtype,
    )
    relocation_matrix = torch.diag(poles) - torch.ones(
        (poles.numel(), 1), dtype=poles.dtype, device=poles.device
    ) @ denominator_residues[None, :]
    relocated = torch.linalg.eigvals(relocation_matrix)
    return (
        _canonicalize_relocated_poles(
            relocated,
            enforce_stability=enforce_stability,
        ),
        condition,
    )


def _fit_fixed_poles(
    s: torch.Tensor,
    values: torch.Tensor,
    poles: torch.Tensor,
    weights: torch.Tensor,
    *,
    proportional: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, ...]]:
    basis, groups = _parameter_basis(s, poles)
    pieces = [basis, torch.ones_like(s[:, None])]
    if proportional:
        pieces.append(s[:, None])
    matrix = torch.cat(pieces, dim=1) * weights[:, None]
    response_count = values.shape[1]
    parameter_rows: list[torch.Tensor] = []
    conditions: list[float] = []
    for response_index in range(response_count):
        parameters, condition = _scaled_real_lstsq(
            matrix,
            values[:, response_index] * weights,
        )
        parameter_rows.append(parameters)
        conditions.append(condition)
    parameters = torch.stack(parameter_rows, dim=0)
    basis_count = basis.shape[1]
    residues = _parameters_to_residues(
        parameters[:, :basis_count],
        groups,
        poles.numel(),
        dtype=poles.dtype,
    )
    direct = parameters[:, basis_count].to(dtype=poles.dtype)
    if proportional:
        proportional_values = parameters[:, basis_count + 1].to(dtype=poles.dtype)
    else:
        proportional_values = torch.zeros_like(direct)
    return residues, direct, proportional_values, tuple(conditions)


def fit_rational(
    frequencies: torch.Tensor,
    values: torch.Tensor,
    config: RationalFitConfig | None = None,
    *,
    representation: str = "Y",
    initial_poles: torch.Tensor | None = None,
) -> RationalModel:
    """Fit a real shared-pole transfer matrix by iterative vector fitting."""

    config = config or RationalFitConfig()
    if not isinstance(frequencies, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise TypeError("frequencies and values must be torch.Tensor instances.")
    if frequencies.requires_grad or values.requires_grad:
        raise RuntimeError(
            "Automatic rational fitting is non-differentiable and does not detach inputs. "
            "Fit fixed poles first, then optimize RationalModel residues/direct."
        )
    if config.weights is not None and config.weights.requires_grad:
        raise RuntimeError("Automatic rational fitting does not accept trainable weights.")
    if frequencies.ndim != 1 or frequencies.numel() < 2:
        raise ValueError("frequencies must have shape [F] with at least two samples.")
    if frequencies.is_complex() or not frequencies.dtype.is_floating_point:
        raise TypeError("frequencies must be a real floating-point tensor.")
    if not values.is_complex():
        raise TypeError("values must be a complex tensor.")
    if values.shape[0] != frequencies.numel():
        raise ValueError("values must use frequency as its first dimension.")
    if values.ndim == 1:
        values = values.reshape(-1, 1, 1)
    if values.ndim != 3:
        raise ValueError("values must have shape [F] or [F, Nout, Nin].")
    if not bool(torch.all(torch.isfinite(frequencies))) or not bool(
        torch.all(frequencies >= 0.0)
    ):
        raise ValueError("frequencies must be finite and non-negative.")
    if not _finite_complex(values):
        raise ValueError("values must contain only finite values.")
    if values.device != frequencies.device:
        raise ValueError("frequencies and values must share a device.")

    if config.band is not None:
        mask = (frequencies >= config.band[0]) & (frequencies <= config.band[1])
        if int(torch.count_nonzero(mask).item()) < config.order + 2:
            raise ValueError("The fit band does not contain enough samples for the requested order.")
        fit_frequencies = frequencies[mask]
        fit_values = values[mask]
        if config.weights is not None:
            if config.weights.numel() != frequencies.numel():
                raise ValueError("weights length must match frequencies.")
            weights = config.weights.to(device=frequencies.device, dtype=frequencies.dtype)[mask]
        else:
            weights = torch.ones_like(fit_frequencies)
    else:
        fit_frequencies = frequencies
        fit_values = values
        if config.weights is not None:
            if config.weights.numel() != frequencies.numel():
                raise ValueError("weights length must match frequencies.")
            weights = config.weights.to(device=frequencies.device, dtype=frequencies.dtype)
        else:
            weights = torch.ones_like(fit_frequencies)
    if fit_frequencies.numel() < config.order + 2:
        raise ValueError("At least order + 2 frequency samples are required.")

    dtype = complex_dtype_for(torch.promote_types(values.dtype, frequencies.dtype))
    s = torch.complex(
        torch.zeros_like(fit_frequencies),
        -2.0 * torch.pi * fit_frequencies,
    ).to(dtype=dtype)
    flat_values = fit_values.reshape(fit_values.shape[0], -1).to(dtype=dtype)
    weights = weights / torch.max(weights)
    if initial_poles is None:
        poles = _initial_poles(
            config.order,
            fit_frequencies,
            dtype=dtype,
        )
    else:
        if initial_poles.requires_grad:
            raise RuntimeError("Automatic pole relocation does not accept trainable poles.")
        poles = torch.as_tensor(initial_poles, device=frequencies.device).to(dtype=dtype)
        if poles.shape != (config.order,):
            raise ValueError("initial_poles must have shape [config.order].")
        _conjugate_groups(poles)
        if config.enforce_stability:
            poles = _canonicalize_relocated_poles(poles, enforce_stability=True)

    relocation_conditions: list[float] = []
    for _ in range(config.iterations):
        poles, condition = _relocate_poles(
            s,
            flat_values,
            poles,
            weights,
            proportional=config.proportional,
            enforce_stability=config.enforce_stability,
        )
        relocation_conditions.append(condition)
    residue_rows, direct_rows, proportional_rows, final_conditions = _fit_fixed_poles(
        s,
        flat_values,
        poles,
        weights,
        proportional=config.proportional,
    )
    response_shape = fit_values.shape[1:]
    model = RationalModel(
        poles=poles,
        residues=residue_rows.reshape(*response_shape, config.order),
        direct=direct_rows.reshape(response_shape),
        proportional=proportional_rows.reshape(response_shape),
        representation=representation,
    )

    passivity = model.check_passivity(
        fit_frequencies,
        tolerance=config.passivity_tolerance,
    ) if model.input_count == model.output_count else None
    enforcement_change = 0.0
    passivity_enforced = False
    if config.enforce_passivity:
        model, enforced = model.enforce_passivity(
            fit_frequencies,
            tolerance=config.passivity_tolerance,
        )
        passivity = enforced
        enforcement_change = enforced.enforcement_change
        passivity_enforced = enforcement_change > 0.0

    fitted = model.evaluate(fit_frequencies)
    error = fitted - fit_values.to(dtype=fitted.dtype)
    max_error = float(torch.max(torch.abs(error)).item())
    rms_error = float(torch.sqrt(torch.mean(torch.abs(error) ** 2)).item())
    maximum_scale = max(float(torch.max(torch.abs(fit_values)).item()), torch.finfo(fit_frequencies.dtype).eps)
    rms_scale = max(
        float(torch.sqrt(torch.mean(torch.abs(fit_values) ** 2)).item()),
        torch.finfo(fit_frequencies.dtype).eps,
    )
    relative_max = max_error / maximum_scale
    relative_rms = rms_error / rms_scale
    unstable = int(torch.count_nonzero(model.poles.real >= 0.0).item())
    warnings: list[str] = []
    if unstable:
        warnings.append(f"{unstable} fitted poles are unstable.")
    if passivity is not None and not passivity.passive:
        warnings.append(
            f"Sampled passivity violation is {passivity.max_violation:.6g}."
        )
    report = NetworkFitReport(
        rms_error=rms_error,
        max_error=max_error,
        relative_rms_error=relative_rms,
        relative_max_error=relative_max,
        frequency_band=(
            float(torch.min(fit_frequencies).item()),
            float(torch.max(fit_frequencies).item()),
        ),
        order=config.order,
        iterations=config.iterations,
        unstable_poles=unstable,
        passivity_margin=None if passivity is None else passivity.margin,
        max_passivity_violation=None if passivity is None else passivity.max_violation,
        condition_numbers=tuple(relocation_conditions) + final_conditions,
        enforcement_change=enforcement_change,
        passivity_enforced=passivity_enforced,
        warnings=tuple(warnings),
        port_count=model.input_count if model.input_count == model.output_count else 0,
    )
    model = replace(model, report=report)
    if unstable:
        raise RuntimeError("Rational fitting produced unstable poles.")
    if enforcement_change > config.enforcement_tolerance and not config.diagnostic_only:
        raise RuntimeError(
            "Passivity enforcement relative change "
            f"{enforcement_change:.6g} exceeds tolerance "
            f"{config.enforcement_tolerance:.6g}."
        )
    if relative_max > config.relative_tolerance and not config.diagnostic_only:
        raise RuntimeError(
            "Rational fit relative max error "
            f"{relative_max:.6g} exceeds tolerance {config.relative_tolerance:.6g}."
        )
    return model
