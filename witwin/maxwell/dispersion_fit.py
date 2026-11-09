"""Passive pole fitting of tabulated permittivity into ordinary ``Material`` objects.

The fitter converts complex relative permittivity samples ``eps(f)`` (or complex
refractive index ``n + i*k`` samples) into a ``Material`` carrying ``DebyePole``,
``DrudePole`` and ``LorentzPole`` terms, using the repository convention
``exp(-i*omega*t)`` in which absorption is ``Im(eps) > 0``.

Fitting has no random state and runs in torch ``float64`` / ``complex128`` on the
device of the input samples. For every vector-fitting order ``1 .. 2*num_poles``:

1. Pole identification with the shared vector fitter
   (:func:`witwin.maxwell.rational.fit_rational`), which returns stable real and
   complex-conjugate poles in the Laplace variable ``s = -i*omega``.
2. Physical projection: every complex pair ``p, conj(p)`` becomes a Lorentz
   candidate ``omega0^2 / (s^2 + gamma*s + omega0^2)`` with ``omega0 = |p|`` and
   ``gamma = -2*Re(p)``; every real pole ``-g`` becomes a Debye candidate
   ``g / (s + g)`` and a Drude candidate ``1 / (s*(s + g))``, and every pair of
   real poles an overdamped Lorentz candidate. The candidate strengths and
   ``eps_inf`` are solved by non-negative least squares (keeping at most
   ``num_poles`` terms), so each retained term is passive by construction
   (positive strength, non-negative damping). A general pole-residue pair carries
   an ``s`` numerator term that a ``LorentzPole`` cannot represent; the
   projection drops it.
3. Joint refinement of all retained term parameters (strengths, resonances,
   dampings, ``eps_inf``) with L-BFGS in a positivity-preserving log
   parameterization, which recovers the accuracy lost by the projection, followed
   by pruning of negligible terms and of in-band resonances narrower than the
   sample spacing (unsupported by the data, typically fitted noise).

The lowest order whose weighted error is within 5% of the best order wins.
The report records the in-band relative error, the sampled passivity margin
``min Im(eps)`` over a band extended well beyond the fit window, and the
s-domain poles of the final model.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .constants import C_0, EPSILON_0
from .media import DebyePole, DrudePole, LorentzPole, Material
from .rational import FitReport, RationalFitConfig, fit_rational

_POLE_TYPES = ("lorentz", "drude", "debye")
_WAVELENGTH_UNITS = {"m": 1.0, "mm": 1.0e-3, "um": 1.0e-6, "nm": 1.0e-9}
_FREQUENCY_UNITS = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9, "THz": 1.0e12}


@dataclass(frozen=True)
class DispersionFitReport(FitReport):
    """Auditable report of a passive permittivity pole fit.

    ``relative_rms_error`` / ``relative_max_error`` are pointwise relative errors
    ``|eps_fit - eps| / |eps|`` over the fitted samples; ``rms_error`` /
    ``max_error`` are the corresponding absolute errors. ``passivity_margin`` is
    the minimum of ``Im(eps_fit)`` over ``passivity_band`` (non-negative for a
    passive medium), and ``max_passivity_violation`` is ``max(0, -margin)``.
    ``poles`` lists the s-domain poles (``s = -i*omega``, rad/s) of the fitted
    susceptibility; a Drude term contributes its marginal pole at ``s = 0``.
    """

    eps_inf: float = 1.0
    sigma_e: float = 0.0
    lorentz_count: int = 0
    drude_count: int = 0
    debye_count: int = 0
    passive: bool = True
    stable: bool = True
    passivity_band: tuple[float, float] = (0.0, 0.0)
    poles: tuple[complex, ...] = ()
    sample_count: int = 0
    refined: bool = False
    vector_fit_relative_rms_error: float | None = None


@dataclass(frozen=True)
class DispersionFit:
    """A fitted ``Material`` together with its :class:`DispersionFitReport`."""

    material: Material
    report: DispersionFitReport


@dataclass(frozen=True)
class _TermSet:
    """Dimensionless pole parameters; frequencies are normalized by ``omega_ref``."""

    eps_inf: torch.Tensor
    lorentz: torch.Tensor  # [L, 3]: delta_eps, omega0, gamma
    drude: torch.Tensor  # [D, 2]: omega_p, gamma
    debye: torch.Tensor  # [B, 2]: delta_eps, rate (= 1 / tau)
    conductivity: torch.Tensor  # scalar c of the term c / s


def _susceptibility_terms(terms: _TermSet, s: torch.Tensor) -> torch.Tensor:
    response = torch.zeros_like(s)
    if terms.lorentz.shape[0]:
        delta, omega0, gamma = terms.lorentz.unbind(dim=1)
        response = response + torch.sum(
            (delta * omega0 * omega0)[None, :]
            / (s[:, None] ** 2 + gamma[None, :] * s[:, None] + (omega0 * omega0)[None, :]),
            dim=1,
        )
    if terms.drude.shape[0]:
        omega_p, gamma = terms.drude.unbind(dim=1)
        response = response + torch.sum(
            (omega_p * omega_p)[None, :] / (s[:, None] * (s[:, None] + gamma[None, :])),
            dim=1,
        )
    if terms.debye.shape[0]:
        delta, rate = terms.debye.unbind(dim=1)
        response = response + torch.sum((delta * rate)[None, :] / (s[:, None] + rate[None, :]), dim=1)
    return response + terms.conductivity / s


def _evaluate_terms(terms: _TermSet, s: torch.Tensor) -> torch.Tensor:
    return terms.eps_inf + _susceptibility_terms(terms, s)


def _nonnegative_least_squares(matrix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lawson-Hanson active-set NNLS ``min ||A x - b||, x >= 0`` in torch."""

    rows, columns = matrix.shape
    solution = torch.zeros(columns, dtype=matrix.dtype, device=matrix.device)
    passive = torch.zeros(columns, dtype=torch.bool, device=matrix.device)
    eps = torch.finfo(matrix.dtype).eps
    tolerance = 10.0 * eps * float(torch.linalg.matrix_norm(matrix, ord=1)) * max(rows, columns)
    for _ in range(3 * columns + 1):
        gradient = matrix.T @ (target - matrix @ solution)
        candidates = torch.where(passive, torch.full_like(gradient, -math.inf), gradient)
        if bool(torch.all(passive)) or float(torch.max(candidates)) <= tolerance:
            break
        passive[int(torch.argmax(candidates))] = True
        while True:
            trial = torch.zeros_like(solution)
            trial[passive] = torch.linalg.lstsq(matrix[:, passive], target[:, None]).solution[:, 0]
            if bool(torch.all(trial[passive] > 0.0)):
                solution = trial
                break
            blocking = passive & (trial <= 0.0)
            step = torch.min(solution[blocking] / (solution[blocking] - trial[blocking]))
            solution = solution + step * (trial - solution)
            passive = passive & (solution > tolerance)
            solution = torch.where(passive, solution, torch.zeros_like(solution))
    return solution


def _pole_candidates(
    poles: torch.Tensor,
    *,
    pole_types: tuple[str, ...],
    omega_min: float,
) -> list[tuple[str, tuple[float, ...]]]:
    """Map s-domain poles to passive Lorentz / Drude / Debye candidate terms."""

    candidates: list[tuple[str, tuple[float, ...]]] = []
    real_rates: list[float] = []
    scale = max(1.0, float(torch.max(torch.abs(poles))))
    for pole in poles.tolist():
        if abs(pole.imag) > 1e-6 * scale:
            if pole.imag > 0.0 and "lorentz" in pole_types:
                candidates.append(("lorentz", (abs(pole), max(-2.0 * pole.real, 0.0))))
            continue
        rate = max(-pole.real, 0.0)
        real_rates.append(rate)
        if "drude" in pole_types:
            candidates.append(("drude", (rate,)))
        if "debye" in pole_types and rate > 0.0:
            candidates.append(("debye", (rate,)))
    if "lorentz" in pole_types:
        # Two real poles -g1, -g2 also form an overdamped Lorentz denominator
        # (s + g1)(s + g2) = s^2 + (g1 + g2) s + g1 g2. A rate far below the band
        # makes that a Drude term, which the Drude candidate already covers.
        floor = 1e-3 * omega_min
        for first in range(len(real_rates)):
            for second in range(first + 1, len(real_rates)):
                g1, g2 = real_rates[first], real_rates[second]
                if g1 > floor and g2 > floor:
                    candidates.append(("lorentz", (math.sqrt(g1 * g2), g1 + g2)))
    return candidates


def _unique_candidates(candidates):
    unique: dict[tuple, tuple[str, tuple[float, ...]]] = {}
    for kind, params in candidates:
        key = (kind, *(float(f"{value:.9e}") for value in params))
        unique.setdefault(key, (kind, params))
    return list(unique.values())


def _candidate_column(kind: str, params: tuple[float, ...], s: torch.Tensor) -> torch.Tensor:
    if kind == "lorentz":
        omega0, gamma = params
        return omega0 * omega0 / (s * s + gamma * s + omega0 * omega0)
    if kind == "drude":
        return 1.0 / (s * (s + params[0]))
    if kind == "debye":
        return params[0] / (s + params[0])
    return 1.0 / s


def _project_candidates(
    candidates: list[tuple[str, tuple[float, ...]]],
    s: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_terms: int,
    eps_inf_fixed: float | None,
    eps_inf_bounds: tuple[float, float | None],
) -> _TermSet:
    """Non-negative strengths for candidate terms, keeping at most ``max_terms`` poles."""

    lower, upper = eps_inf_bounds
    columns = [_candidate_column(kind, params, s) for kind, params in candidates]

    def solve(active: list[int], offset: float, free_eps_inf: bool) -> torch.Tensor:
        pieces = [columns[index] for index in active]
        if free_eps_inf:
            pieces.append(torch.ones_like(s))
        design = torch.stack(pieces, dim=1) * weights[:, None]
        target = (values - offset) * weights
        stacked = torch.cat((design.real, design.imag), dim=0)
        stacked_target = torch.cat((target.real, target.imag), dim=0)
        norms = torch.linalg.vector_norm(stacked, dim=0)
        norms = torch.where(norms > 0.0, norms, torch.ones_like(norms))
        return _nonnegative_least_squares(stacked / norms[None, :], stacked_target) / norms

    def solve_bounded(active: list[int]) -> tuple[torch.Tensor, float]:
        if eps_inf_fixed is not None:
            return solve(active, eps_inf_fixed, False), eps_inf_fixed
        solution = solve(active, lower, True)
        eps_inf = lower + float(solution[-1])
        if upper is not None and eps_inf > upper:
            return solve(active, upper, False), upper
        return solution[:-1], eps_inf

    active = list(range(len(candidates)))
    if not active and eps_inf_fixed is not None:
        raise ValueError("No pole candidates are available for a fixed eps_inf fit.")
    strengths, eps_inf = solve_bounded(active) if active else (torch.zeros(0, dtype=s.real.dtype, device=s.device), None)
    if eps_inf is None:
        mean = torch.sum(weights**2 * values.real) / torch.sum(weights**2)
        eps_inf = min(max(float(mean), lower), upper if upper is not None else math.inf)
    selected = [index for index in active if float(strengths[index]) > 0.0]
    pole_terms = [index for index in selected if candidates[index][0] != "conductivity"]
    if len(pole_terms) > max_terms:
        magnitude = {
            index: float(torch.max(torch.abs(strengths[index] * columns[index] * weights)))
            for index in pole_terms
        }
        keep = sorted(pole_terms, key=lambda index: -magnitude[index])[:max_terms]
        active = sorted(keep + [index for index in selected if candidates[index][0] == "conductivity"])
        strengths_active, eps_inf = solve_bounded(active)
        strengths = torch.zeros(len(candidates), dtype=s.real.dtype, device=s.device)
        strengths[active] = strengths_active
        selected = [index for index in active if float(strengths[index]) > 0.0]

    lorentz, drude, debye = [], [], []
    conductivity = 0.0
    for index in selected:
        kind, params = candidates[index]
        strength = float(strengths[index])
        if kind == "lorentz":
            lorentz.append((strength, *params))
        elif kind == "drude":
            drude.append((math.sqrt(strength), params[0]))
        elif kind == "debye":
            debye.append((strength, params[0]))
        else:
            conductivity += strength

    def table(rows, width):
        return torch.tensor(rows, dtype=s.real.dtype, device=s.device).reshape(-1, width)

    return _TermSet(
        eps_inf=torch.tensor(eps_inf, dtype=s.real.dtype, device=s.device),
        lorentz=table(lorentz, 3),
        drude=table(drude, 2),
        debye=table(debye, 2),
        conductivity=torch.tensor(conductivity, dtype=s.real.dtype, device=s.device),
    )


def _vector_fit(normalized: torch.Tensor, values: torch.Tensor, weights: torch.Tensor, *, order: int, iterations: int):
    """Shared vector fitter, one relocation per call, returning the best iterate."""

    best = None
    poles = None
    for _ in range(max(iterations, 1)):
        model = fit_rational(
            normalized / (2.0 * math.pi),
            values,
            RationalFitConfig(
                order=order,
                iterations=1,
                weights=weights,
                enforce_stability=True,
                diagnostic_only=True,
            ),
            representation="Z",
            initial_poles=poles,
        )
        poles = model.poles
        if best is None or model.report.relative_rms_error < best.report.relative_rms_error:
            best = model
        if model.report.relative_rms_error < 1e-13:
            break
    return best


def _weighted_loss(terms: _TermSet, s: torch.Tensor, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    residual = (_evaluate_terms(terms, s) - values) * weights
    return torch.sum(residual.real**2 + residual.imag**2)


def _refine_terms(
    terms: _TermSet,
    s: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps_inf_fixed: float | None,
    eps_inf_bounds: tuple[float, float | None],
    max_iterations: int,
) -> _TermSet:
    """Jointly refine all retained parameters with L-BFGS in log coordinates."""

    floor = 1e-12

    def log_of(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(tensor, min=floor)).detach().clone().requires_grad_(True)

    lorentz_log = log_of(terms.lorentz)
    drude_log = log_of(terms.drude)
    debye_log = log_of(terms.debye)
    conductivity_log = log_of(terms.conductivity) if float(terms.conductivity) > 0.0 else None
    lower, upper = eps_inf_bounds
    eps_inf_param = None
    if eps_inf_fixed is None:
        if upper is None:
            eps_inf_param = log_of(terms.eps_inf - lower)
        else:
            fraction = torch.clamp((terms.eps_inf - lower) / (upper - lower), 1e-9, 1.0 - 1e-9)
            eps_inf_param = torch.logit(fraction).detach().clone().requires_grad_(True)

    def assemble() -> _TermSet:
        if eps_inf_param is None:
            eps_inf = terms.eps_inf
        elif upper is None:
            eps_inf = lower + torch.exp(eps_inf_param)
        else:
            eps_inf = lower + (upper - lower) * torch.sigmoid(eps_inf_param)
        return _TermSet(
            eps_inf=eps_inf,
            lorentz=torch.exp(lorentz_log),
            drude=torch.exp(drude_log),
            debye=torch.exp(debye_log),
            conductivity=terms.conductivity if conductivity_log is None else torch.exp(conductivity_log),
        )

    parameters = [p for p in (lorentz_log, drude_log, debye_log, conductivity_log, eps_inf_param) if p is not None and p.numel()]
    if not parameters:
        return terms
    initial_loss = float(_weighted_loss(terms, s, values, weights))
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=max_iterations,
        history_size=50,
        tolerance_grad=1e-15,
        tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = _weighted_loss(assemble(), s, values, weights) / max(initial_loss, 1e-300)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        refined = assemble()
        refined = _TermSet(
            eps_inf=refined.eps_inf.detach(),
            lorentz=refined.lorentz.detach(),
            drude=refined.drude.detach(),
            debye=refined.debye.detach(),
            conductivity=refined.conductivity.detach(),
        )
        refined_loss = float(_weighted_loss(refined, s, values, weights))
    if not math.isfinite(refined_loss) or refined_loss > initial_loss:
        return terms
    return refined


def _prune_terms(terms: _TermSet, s: torch.Tensor, values: torch.Tensor, tolerance: float) -> _TermSet:
    """Drop negligible terms and unresolved in-band Lorentz resonances.

    A term is negligible when its in-band susceptibility stays below
    ``tolerance * max|eps|``. Frequencies in ``s`` must be sorted ascending.
    """

    scale = float(torch.max(torch.abs(values)))
    empty = _TermSet(
        eps_inf=terms.eps_inf,
        lorentz=terms.lorentz[:0],
        drude=terms.drude[:0],
        debye=terms.debye[:0],
        conductivity=torch.zeros_like(terms.conductivity),
    )

    omega = -s.imag
    gaps = omega[1:] - omega[:-1]

    def unresolved(row: torch.Tensor) -> bool:
        # A Lorentz resonance inside the sampled band that is narrower than the
        # local sample spacing is not supported by the data (typically fitted noise)
        # and would put an unsampled spike between samples.
        _, omega0, gamma = row.tolist()
        if not float(omega[0]) < omega0 < float(omega[-1]):
            return False
        index = int(torch.searchsorted(omega, torch.tensor(omega0, dtype=omega.dtype, device=omega.device)))
        return gamma < float(gaps[min(max(index - 1, 0), gaps.numel() - 1)])

    def keep(table: torch.Tensor, field: str) -> torch.Tensor:
        rows = []
        for row in table:
            if field == "lorentz" and unresolved(row):
                continue
            single = _TermSet(**{**empty.__dict__, field: row[None, :]})
            if float(torch.max(torch.abs(_susceptibility_terms(single, s)))) > tolerance * scale:
                rows.append(row)
        return torch.stack(rows) if rows else table[:0]

    conductivity = terms.conductivity
    if float(torch.max(torch.abs(conductivity / s))) <= tolerance * scale:
        conductivity = torch.zeros_like(conductivity)
    return _TermSet(
        eps_inf=terms.eps_inf,
        lorentz=keep(terms.lorentz, "lorentz"),
        drude=keep(terms.drude, "drude"),
        debye=keep(terms.debye, "debye"),
        conductivity=conductivity,
    )


def _term_count(terms: _TermSet) -> int:
    return (
        terms.lorentz.shape[0]
        + terms.drude.shape[0]
        + terms.debye.shape[0]
        + int(float(terms.conductivity) > 0.0)
    )


def _terms_to_material(terms: _TermSet, omega_ref: float, name: str | None) -> Material:
    two_pi = 2.0 * math.pi
    lorentz_poles = tuple(
        LorentzPole(
            delta_eps=delta,
            resonance_frequency=omega0 * omega_ref / two_pi,
            gamma=gamma * omega_ref / two_pi,
        )
        for delta, omega0, gamma in terms.lorentz.tolist()
    )
    drude_poles = tuple(
        DrudePole(plasma_frequency=omega_p * omega_ref / two_pi, gamma=gamma * omega_ref / two_pi)
        for omega_p, gamma in terms.drude.tolist()
    )
    debye_poles = tuple(
        DebyePole(delta_eps=delta, tau=1.0 / (rate * omega_ref)) for delta, rate in terms.debye.tolist()
    )
    return Material(
        eps_r=float(terms.eps_inf),
        sigma_e=float(terms.conductivity) * omega_ref * EPSILON_0,
        name=name,
        lorentz_poles=lorentz_poles,
        drude_poles=drude_poles,
        debye_poles=debye_poles,
    )


def _terms_poles(terms: _TermSet, omega_ref: float) -> tuple[complex, ...]:
    poles: list[complex] = []
    for _, omega0, gamma in terms.lorentz.tolist():
        root = complex(gamma * gamma / 4.0 - omega0 * omega0) ** 0.5
        poles.extend(((-gamma / 2.0 + root) * omega_ref, (-gamma / 2.0 - root) * omega_ref))
    for _, gamma in terms.drude.tolist():
        poles.extend((0j, complex(-gamma * omega_ref)))
    for _, rate in terms.debye.tolist():
        poles.append(complex(-rate * omega_ref))
    if float(terms.conductivity) > 0.0:
        poles.append(0j)
    return tuple(poles)


def permittivity_from_nk(n, k=None) -> torch.Tensor:
    """Complex relative permittivity ``(n + i*k)^2`` (``k >= 0`` is absorbing)."""

    n = torch.as_tensor(n, dtype=torch.float64)
    k = torch.zeros_like(n) if k is None else torch.as_tensor(k, dtype=torch.float64, device=n.device)
    return torch.complex(n, k) ** 2


def wavelength_to_frequency(wavelengths) -> torch.Tensor:
    """Vacuum wavelengths [m] to frequencies [Hz]."""

    wavelengths = torch.as_tensor(wavelengths, dtype=torch.float64)
    if not bool(torch.all(wavelengths > 0.0)):
        raise ValueError("wavelengths must be positive.")
    return C_0 / wavelengths


def load_dispersion_csv(
    path: str | Path,
    *,
    abscissa: str = "wavelength",
    unit: str = "um",
    quantity: str = "nk",
    delimiter: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load tabulated optical constants and return ``(frequencies [Hz], eps)``.

    Each data row holds ``x, a[, b]`` where ``x`` is a wavelength
    (``abscissa="wavelength"``, ``unit`` in ``m/mm/um/nm``) or a frequency
    (``abscissa="frequency"``, ``unit`` in ``Hz/kHz/MHz/GHz/THz``), and ``a, b``
    are ``n, k`` (``quantity="nk"``, ``k`` optional) or ``Re(eps), Im(eps)``
    (``quantity="permittivity"``). Values follow ``exp(-i*omega*t)``: absorption
    is ``k > 0`` / ``Im(eps) > 0``. Rows whose first field is not numeric (headers,
    comments) are skipped; the delimiter defaults to comma, else whitespace. The
    result is sorted by ascending frequency.
    """

    if abscissa not in {"wavelength", "frequency"}:
        raise ValueError("abscissa must be 'wavelength' or 'frequency'.")
    units = _WAVELENGTH_UNITS if abscissa == "wavelength" else _FREQUENCY_UNITS
    if unit not in units:
        raise ValueError(f"unit must be one of {sorted(units)} for abscissa={abscissa!r}.")
    if quantity not in {"nk", "permittivity"}:
        raise ValueError("quantity must be 'nk' or 'permittivity'.")
    rows: list[list[float]] = []
    with open(path, newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if delimiter is not None:
                fields = next(csv.reader([stripped], delimiter=delimiter))
            elif "," in stripped:
                fields = next(csv.reader([stripped]))
            else:
                fields = stripped.split()
            fields = [field for field in fields if field.strip()]
            try:
                numbers = [float(field) for field in fields]
            except ValueError:
                continue
            if len(numbers) not in (2, 3):
                raise ValueError(f"Expected 2 or 3 numeric columns per row, got {len(numbers)}: {line!r}.")
            rows.append(numbers + [0.0] * (3 - len(numbers)))
    if len(rows) < 2:
        raise ValueError("The table must contain at least two numeric rows.")
    table = torch.tensor(rows, dtype=torch.float64)
    coordinate = table[:, 0] * units[unit]
    frequencies = wavelength_to_frequency(coordinate) if abscissa == "wavelength" else coordinate
    if quantity == "nk":
        permittivity = permittivity_from_nk(table[:, 1], table[:, 2])
    else:
        permittivity = torch.complex(table[:, 1], table[:, 2])
    order = torch.argsort(frequencies)
    return frequencies[order], permittivity[order]


def fit_dispersion(
    frequencies,
    permittivity,
    *,
    num_poles: int = 3,
    frequency_range: tuple[float, float] | None = None,
    weights=None,
    eps_inf: float | None = None,
    eps_inf_bounds: tuple[float, float | None] = (1.0, None),
    pole_types: Sequence[str] = _POLE_TYPES,
    conductivity: bool = False,
    refine: bool = True,
    max_refine_iterations: int = 400,
    prune_tolerance: float = 1e-6,
    vector_fit_iterations: int = 10,
    passivity_band_factor: float = 100.0,
    tolerance: float | None = None,
    name: str | None = None,
) -> DispersionFit:
    """Fit complex permittivity samples to a passive Debye/Drude/Lorentz ``Material``.

    Parameters
    ----------
    frequencies:
        Sample frequencies [Hz], shape ``[F]``.
    permittivity:
        Complex relative permittivity at ``frequencies`` (``exp(-i*omega*t)``,
        absorption ``Im(eps) > 0``), shape ``[F]``.
    num_poles:
        Maximum number of Lorentz / Drude / Debye terms in the model (a static
        conductivity term is not counted). Vector fitting is tried at every order
        ``1 .. 2 * num_poles``.
    frequency_range:
        Optional ``(f_min, f_max)`` [Hz] fit window; samples outside are ignored.
    weights:
        Optional positive per-sample weights multiplying the default relative-error
        weighting ``1 / |eps|``.
    eps_inf:
        Fix the high-frequency permittivity; ``None`` fits it within ``eps_inf_bounds``.
    eps_inf_bounds:
        ``(lower, upper)`` bounds on a fitted ``eps_inf`` (``upper=None`` is unbounded).
        The default lower bound 1 keeps the FDTD Courant limit of vacuum valid.
    pole_types:
        Subset of ``("lorentz", "drude", "debye")`` allowed in the model.
    conductivity:
        Also fit a static conductivity term ``sigma_e`` (``i*sigma/(omega*eps0)``).
    refine:
        Run the joint L-BFGS refinement after the passive projection.
    prune_tolerance:
        Terms whose largest in-band susceptibility is below ``prune_tolerance *
        max|eps|`` are dropped (and the rest re-refined).
    passivity_band_factor:
        The passivity check spans ``[f_min / factor, f_max * factor]``.
    tolerance:
        If given, raise ``RuntimeError`` when the in-band relative max error exceeds it.
    name:
        Name of the returned ``Material``.
    """

    if not isinstance(num_poles, int) or isinstance(num_poles, bool) or num_poles < 1:
        raise ValueError("num_poles must be a positive integer.")
    pole_types = tuple(pole_types)
    if not pole_types or any(kind not in _POLE_TYPES for kind in pole_types):
        raise ValueError(f"pole_types must be a non-empty subset of {_POLE_TYPES}.")
    if prune_tolerance < 0.0:
        raise ValueError("prune_tolerance must be non-negative.")
    if passivity_band_factor < 1.0:
        raise ValueError("passivity_band_factor must be >= 1.")
    lower, upper = eps_inf_bounds
    if lower <= 0.0 or (upper is not None and upper <= lower):
        raise ValueError("eps_inf_bounds must satisfy 0 < lower < upper.")
    if eps_inf is not None and eps_inf <= 0.0:
        raise ValueError("eps_inf must be positive.")

    frequencies = torch.as_tensor(frequencies)
    if frequencies.is_complex():
        raise TypeError("frequencies must be real.")
    frequencies = frequencies.to(torch.float64)
    values = torch.as_tensor(permittivity, device=frequencies.device).to(torch.complex128)
    if frequencies.ndim != 1 or values.shape != frequencies.shape:
        raise ValueError("frequencies and permittivity must be one-dimensional with equal length.")
    if not bool(torch.all(torch.isfinite(frequencies))) or not bool(torch.all(frequencies > 0.0)):
        raise ValueError("frequencies must be finite and positive.")
    if not bool(torch.all(torch.isfinite(values.real) & torch.isfinite(values.imag))):
        raise ValueError("permittivity must be finite.")
    sample_weights = torch.ones_like(frequencies)
    if weights is not None:
        sample_weights = torch.as_tensor(weights, dtype=torch.float64, device=frequencies.device)
        if sample_weights.shape != frequencies.shape or not bool(torch.all(sample_weights > 0.0)):
            raise ValueError("weights must be positive with the same shape as frequencies.")
    ascending = torch.argsort(frequencies)
    frequencies, values, sample_weights = frequencies[ascending], values[ascending], sample_weights[ascending]
    if frequency_range is not None:
        f_lo, f_hi = (float(v) for v in frequency_range)
        if not 0.0 < f_lo < f_hi:
            raise ValueError("frequency_range must be an increasing positive pair.")
        mask = (frequencies >= f_lo) & (frequencies <= f_hi)
        frequencies, values, sample_weights = frequencies[mask], values[mask], sample_weights[mask]
    if frequencies.numel() < 2 * num_poles + 2:
        raise ValueError(f"At least {2 * num_poles + 2} samples are required for num_poles={num_poles}.")

    omega_ref = 2.0 * math.pi * math.sqrt(float(frequencies[0]) * float(frequencies[-1]))
    weights_rel = sample_weights / torch.abs(values).clamp_min(torch.finfo(torch.float64).tiny)
    weights_rel = weights_rel / torch.max(weights_rel)

    normalized = frequencies * (2.0 * math.pi / omega_ref)
    s = torch.complex(torch.zeros_like(normalized), -normalized)
    weight_norm = float(torch.sqrt(torch.sum(torch.abs(values * weights_rel) ** 2)))

    def score(candidate_terms: _TermSet) -> float:
        return math.sqrt(float(_weighted_loss(candidate_terms, s, values, weights_rel))) / weight_norm

    # Each vector-fitting order 1 .. 2 * num_poles proposes its own pole set, which
    # is projected, refined and scored separately; the most parsimonious order whose
    # score is within a small margin of the best one wins. Excess-order vector
    # fitting of near-exact data is ill-posed (pole-zero cancellations), so each
    # order keeps its best relocation iterate rather than the last one.
    fits: list[tuple[float, _TermSet, bool]] = []
    vector_errors: list[float] = []
    condition_numbers: list[float] = []
    for order in range(1, 2 * num_poles + 1):
        vector_model = _vector_fit(normalized, values, weights_rel, order=order, iterations=vector_fit_iterations)
        vector_errors.append(vector_model.report.relative_rms_error)
        condition_numbers.extend(vector_model.report.condition_numbers)
        candidates = _pole_candidates(vector_model.poles, pole_types=pole_types, omega_min=float(normalized[0]))
        if conductivity:
            candidates.append(("conductivity", ()))
        terms = _project_candidates(
            _unique_candidates(candidates),
            s,
            values,
            weights_rel,
            max_terms=num_poles,
            eps_inf_fixed=eps_inf,
            eps_inf_bounds=(lower, upper),
        )
        refined = False
        for _ in range(2):
            if refine:
                candidate = _refine_terms(
                    terms,
                    s,
                    values,
                    weights_rel,
                    eps_inf_fixed=eps_inf,
                    eps_inf_bounds=(lower, upper),
                    max_iterations=max_refine_iterations,
                )
                refined = refined or candidate is not terms
                terms = candidate
            pruned = _prune_terms(terms, s, values, prune_tolerance)
            if _term_count(pruned) == _term_count(terms):
                break
            terms = pruned
        fits.append((score(terms), terms, refined))
        if fits[-1][0] < 1e-10:
            break
    best_score = min(entry[0] for entry in fits)
    _, terms, refined = next(entry for entry in fits if entry[0] <= 1.05 * best_score + 1e-12)

    fitted = _evaluate_terms(terms, s)
    error = torch.abs(fitted - values)
    relative = error / torch.abs(values).clamp_min(torch.finfo(torch.float64).tiny)
    band = (float(frequencies[0]), float(frequencies[-1]))
    check_frequencies = torch.logspace(
        math.log10(band[0] / passivity_band_factor),
        math.log10(band[1] * passivity_band_factor),
        4096,
        dtype=torch.float64,
        device=frequencies.device,
    )
    check_s = torch.complex(torch.zeros_like(check_frequencies), -check_frequencies * (2.0 * math.pi / omega_ref))
    check_values = _evaluate_terms(terms, check_s)
    margin = float(torch.min(check_values.imag))
    passivity_tolerance = 1e-12 * float(torch.max(torch.abs(check_values)))
    poles = _terms_poles(terms, omega_ref)
    stable = all(pole.real <= 0.0 for pole in poles)
    passive = margin >= -passivity_tolerance
    relative_max = float(torch.max(relative))
    warnings: list[str] = []
    if not passive:
        warnings.append(f"Sampled passivity violation min Im(eps) = {margin:.6g}.")
    report = DispersionFitReport(
        rms_error=float(torch.sqrt(torch.mean(error**2))),
        max_error=float(torch.max(error)),
        relative_rms_error=float(torch.sqrt(torch.mean(relative**2))),
        relative_max_error=relative_max,
        frequency_band=band,
        order=2 * num_poles,
        iterations=vector_fit_iterations,
        unstable_poles=sum(1 for pole in poles if pole.real > 0.0),
        passivity_margin=margin,
        max_passivity_violation=max(0.0, -margin),
        condition_numbers=tuple(condition_numbers),
        differentiable_parameters=(),
        warnings=tuple(warnings),
        eps_inf=float(terms.eps_inf),
        sigma_e=float(terms.conductivity) * omega_ref * EPSILON_0,
        lorentz_count=int(terms.lorentz.shape[0]),
        drude_count=int(terms.drude.shape[0]),
        debye_count=int(terms.debye.shape[0]),
        passive=passive,
        stable=stable,
        passivity_band=(band[0] / passivity_band_factor, band[1] * passivity_band_factor),
        poles=poles,
        sample_count=int(frequencies.numel()),
        refined=refined,
        vector_fit_relative_rms_error=min(vector_errors),
    )
    if tolerance is not None and relative_max > tolerance:
        raise RuntimeError(
            f"Dispersion fit relative max error {relative_max:.6g} exceeds tolerance {tolerance:.6g}."
        )
    return DispersionFit(material=_terms_to_material(terms, omega_ref, name), report=report)


def fit_dispersion_nk(wavelengths, n, k=None, **options) -> DispersionFit:
    """Fit ``n + i*k`` samples at vacuum ``wavelengths`` [m]; see :func:`fit_dispersion`."""

    return fit_dispersion(wavelength_to_frequency(wavelengths), permittivity_from_nk(n, k), **options)
