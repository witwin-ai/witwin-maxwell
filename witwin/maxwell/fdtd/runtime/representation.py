"""The temporal representation shared by complex sources and spectral outputs."""
from __future__ import annotations

import cmath
import math

from ..boundary import has_complex_fields
from ..excitation.temporal import auxiliary_electric_time, injection_time


def require_absolute_spectra(mixed_sources, normalize_source=False):
    """Reject the undefined absolute scale while retaining samples for ratios."""
    if mixed_sources and not normalize_source:
        raise ValueError(
            "Absolute spectra of a run mixing ongoing (CW / undamped / periodic / DC) "
            "and transient sources have no horizon-independent scale; use "
            "normalize_source=True, ratios such as S/Z, or a run with only ongoing "
            "or only transient sources."
        )


def uses_analytic_cw(solver):
    """Pure CW Bloch runs carry exp(-i*omega*t), rather than real cosines."""
    if not has_complex_fields(solver):
        return False
    plan = getattr(solver, "_time_plan", None)
    if plan is not None and "source_representation" in plan:
        return plan["source_representation"] == "analytic_cw"
    sources = getattr(solver, "_compiled_sources", ())
    return bool(sources) and all(source["source_time"]["kind"] == "cw" for source in sources)


def spectral_phasor_factor(solver):
    return 1.0 if uses_analytic_cw(solver) else 2.0


def spectral_scale(weight_sum, *, frequency, phasor_factor=2.0, integration_dt=None, reference_frequency=None):
    """Scale an exp(+iwt) sum by the reference period, or as a CW phasor.

    Pulses default to the per-output-frequency period: Fourier transform times
    f, independent of the other requested frequencies. An explicit reference
    frequency gives Fourier transform times that constant. Tensor frequencies
    must broadcast over the output's frequency axis. CW scaling is unchanged.
    """
    if integration_dt is not None:
        return integration_dt * (frequency if reference_frequency is None else reference_frequency)
    return phasor_factor / weight_sum


def normalization_signal(solver, source_time, time):
    """Unit-envelope reference; declared physical amplitude and phase remain."""
    if uses_analytic_cw(solver):
        frequency = (source_time["frequency"] if isinstance(source_time, dict)
                     else source_time.frequency)
        return cmath.exp(-2j * math.pi * float(frequency) * float(time))
    from ...sources import evaluate_source_time_normalization
    return complex(evaluate_source_time_normalization(source_time, time))


def normalization_sample_time(solver, step):
    """Sample the reference envelope at its source's injection instant.

    Auxiliary TFSF uses its post-step E sample, alpha=beta=1; current and
    equivalent-sheet references retain their own injection instants.
    """
    offset = (solver._normalization_time_offset if solver._normalize_source
              else solver._source_time_offset)
    if offset == 1.0:
        return auxiliary_electric_time(step, solver.dt)
    kind = "electric_current" if offset else "incident_reference"
    return injection_time(kind, step * solver.dt, solver.dt)


def energy_representation(solver):
    if has_complex_fields(solver):
        return "combined_real_imaginary_algorithmic_state"
    return "physical_real_field"
