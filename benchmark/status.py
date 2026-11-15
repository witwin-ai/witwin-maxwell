"""Status classes for benchmark rows in RESULTS.md.

The metric limits, scenario-specific sentinel limits, investigation rows and
quarantined references are read from the benchmark suite of
``scripts/dev/integration_suites.toml``, the same manifest the integration batch
scores against. This module only classifies; it never changes a limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import tomllib

from benchmark.paths import ROOT


MANIFEST = ROOT.parent / "scripts" / "dev" / "integration_suites.toml"
LIMITS_DOC = "docs/assessments/benchmark-metric-limits-2026-11-01.md"

PASS_GLOBAL = "(i) pass, global limits"
PASS_SENTINEL = "(ii) pass via scenario sentinel limit"
NOT_SCORED = "(iii) unconverged / not scored"
KNOWN_OPEN = "(iv) known-open residual"
REFERENCE_CONVENTION = "(v) reference-convention row"
FAIL = "fail, no waiver"
STATUS_CLASSES = (PASS_GLOBAL, PASS_SENTINEL, NOT_SCORED, KNOWN_OPEN, REFERENCE_CONVENTION, FAIL)

# Column index in the 18-cell scenario row, manifest key, and whether larger is better.
SCORED_METRICS = ((4, "field_l2", False), (6, "field_linf", False), (7, "field_corr_min", True), (9, "flux_error", False))

# Rows whose residual has a documented cause that the manifest does not record as an
# investigation row. Each entry names its evidence; the measured numbers stay in the tables.
DECLARED_STATUS = {
    # Class (c) volume-current rows of the limits assessment: qualified by continuum
    # convergence, not by the coarse cached reference; no snapshot allowance.
    "custom_current_source": (KNOWN_OPEN, f"class (c) volume-current support/quadrature residual ({LIMITS_DOC})"),
    "magnetic_current_vacuum": (KNOWN_OPEN, f"class (c) volume-current support/quadrature residual ({LIMITS_DOC})"),
    "uniform_current": (KNOWN_OPEN, f"class (c) volume-current support/quadrature residual ({LIMITS_DOC})"),
    # Raw field L2 near 2 with a small Shape L2: the field matches the reference up to a
    # sign / absolute-phase gauge that the raw metric does not remove (0.6.0 release batch).
    **{
        name: (REFERENCE_CONVENTION, "sign / absolute-phase gauge: raw Field L2 near 2 while Shape L2 is small")
        for name in (
            "grating_diffraction", "mode_monitor_straight_wg", "mode_monitor_two_planes", "mode_source_wg",
            "periodic_grating", "rcs_dielectric_box", "rcs_pec_sphere", "ring_resonator_s21", "sphere_rcs",
            "tfsf_vacuum", "waveguide_s_matrix",
        )
    },
    # The TFSF gauge of tfsf_vacuum (Maxwell to reference 1.081 exp(3.04i) at 1.5% shape residual);
    # the scorer has no convention mapping, so the raw metrics keep it. The rest is the eps 4 sphere
    # beyond the Yee band edge of the 25 mm grid (diag17, bench-respec 2026-11-12).
    # Oblique s-polarized Bloch row (a historical row mover covers it, so it cannot be a manifest
    # investigation row): same time convention, Bloch phase sign and transverse phase ramp; one
    # global factor uniform over the reflected, total-field and transmitted regions whose phase is
    # periodic_grating's TFSF gauge (2.857 vs 2.855 rad) but whose magnitude puts Maxwell 7.5% above
    # the reference (power 15.6%, the flux error), against 1.1% for bloch_oblique_te (bench-respec
    # 2026-11-12, cached reference).
    "bloch_oblique": (KNOWN_OPEN, "unresolved oblique s-polarized TFSF amplitude: uniform 7.5% field (15.6% "
                      "power) excess over the reference on top of the TFSF phase gauge; not a sign or phase convention"),
    # Oblique p-polarized Bloch row: the time convention, the Bloch phase sign and the transverse
    # phase ramp agree; the cached-reference field differs by one global factor, uniform over the
    # reflected, total-field and transmitted regions (phase -1.245 rad, magnitude within 1.1%,
    # flux error 0.008), as periodic_grating does (bench-respec 2026-11-12).
    "bloch_oblique_te": (REFERENCE_CONVENTION, "TFSF-slab absolute-phase gauge: one uniform global phase, "
                         "matching time convention and Bloch phase sign; magnitude and flux agree"),
    "tfsf_dielectric_sphere": (
        REFERENCE_CONVENTION,
        "TFSF gauge of tfsf_vacuum (1.081 exp(3.04i)) plus the band-edge residual of the 25 mm grid "
        "(3.0 cells per wavelength in the sphere); accuracy is scored against Mie instead",
    ),
    "ricker_axis_x_anisotropic": (
        REFERENCE_CONVENTION,
        "reference exports the Ricker wavelet as GaussianPulse(fwidth=f0); fixed amplitude/phase offset, shape matches",
    ),
}


ORACLE_EVIDENCE = "<evidence-root>/fdtd-coord-20261111/analytic-oracles/README.md"
# Diagnoses from analytic oracles, appended to a row's basis whatever its class.
ROW_NOTES = {
    "full_tensor_slab": (
        "analytic 4x4 oracle: exact T=0.790/R=0.210 for the launched eigenpolarization; FDTD on this grid "
        "(about 6 cells per vacuum wavelength, 3.5 in the slab) gives T=0.985 and converges at second order "
        f"under refinement; the full-tensor path is not the cause ({ORACLE_EVIDENCE})"
    ),
    # Rows beyond the Yee band edge of the 25 mm grid, scored on a refined grid by
    # `python -m benchmark analytic <row>` (limits in benchmark/analytic/benchmark_rows.py).
    **{
        name: f"analytic Mie oracle on the dx/5 grid: display-plane shape L2 {value} (bench-respec 2026-11-12)"
        for name, value in (("dielectric_sphere", "0.0369"), ("planewave_dielectric_sphere", "0.0369"),
                            ("tfsf_dielectric_sphere", "0.110 (TF/SF composite)"))
    },
    "custom_grid_slab": (
        "analytic Airy slab on the dx/4 grid: T 0.8071 vs 0.8137, R 0.1951 vs 0.1863, calibrated line "
        "field 0.0455 (bench-respec 2026-11-12)"
    ),
    "lorentz_resonator": f"analytic Mie oracle: residual from the unresolved skin depth, dx/delta=8 ({ORACLE_EVIDENCE})",
    "sellmeier_sphere": (
        "analytic Mie oracle: solver defect, dispersive pole weights are volume-averaged under "
        f"averaging='polarized' ({ORACLE_EVIDENCE})"
    ),
}


@dataclass(frozen=True)
class SuiteLimits:
    global_limits: dict
    overrides: dict
    override_notes: dict
    investigation: dict
    quarantine: dict
    scalar_limits: dict = field(default_factory=dict)


def load_suite_limits(path=MANIFEST) -> SuiteLimits | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    suite = next(s for s in tomllib.loads(text)["suites"] if s["id"] == "benchmark")
    overrides = suite.get("scenario_metric_limits", {})
    notes = {}
    for line in text.splitlines():
        match = re.match(r"^(\w+)\s*=\s*\{[^}]*\}\s*#\s*(.+)$", line)
        if match and match.group(1) in overrides:
            notes[match.group(1)] = match.group(2).strip()
    return SuiteLimits(suite["metric_limits"], overrides, notes,
                       suite.get("scenario_investigation", {}), suite.get("scenario_quarantine", {}),
                       suite.get("scenario_scalar_limits", {}))


def _passes(value, limit, larger_is_better):
    return value > limit if larger_is_better else value < limit


def classify_row(row: list[str], limits: SuiteLimits | None, scalars=()) -> tuple[str, str, str]:
    """Return (status class, batch-scorer outcome, detail) for one 18-cell row.

    ``scalars`` are the row's 8-cell scalar-observable rows, scored when the manifest gives
    the scenario scalar limits.
    """
    status, outcome, detail = _classify(row, limits, scalars)
    note = ROW_NOTES.get(row[0])
    return status, outcome, "; ".join(part for part in (detail, note) if part)


def _classify(row: list[str], limits: SuiteLimits | None, scalars=()) -> tuple[str, str, str]:
    name = row[0]
    if limits is None:
        return "unclassified", "-", "suite manifest not found"
    if name in limits.quarantine:
        return NOT_SCORED, "SKIP", "quarantined reference: " + limits.quarantine[name]
    overrides = limits.overrides.get(name, {})
    unconverged, global_fail, failed, scored = [], [], [], []
    for index, key, larger in SCORED_METRICS:
        cell = row[index]
        if cell == "-" or "not scored" in cell:
            continue
        if "unconverged" in cell:
            unconverged.append(key)
            continue
        value = float(cell)
        scored.append(key)
        if not _passes(value, limits.global_limits[key], larger):
            global_fail.append(key)
        if not _passes(value, overrides.get(key, limits.global_limits[key]), larger):
            failed.append(key)
    for scalar in scalars:
        limit = limits.scalar_limits.get(name, {}).get(scalar[2])
        if limit is None:
            continue
        scored.append(scalar[2])
        if not float(scalar[5]) < limit:
            failed.append(scalar[2])
    outcome = "FAIL" if failed else "SKIP" if unconverged or not scored else "PASS"
    details = []
    if failed:
        details.append("fails " + ", ".join(failed))
    if unconverged:
        details.append("unconverged (not scored): " + ", ".join(unconverged))
    if name in limits.investigation:
        return KNOWN_OPEN, outcome, "; ".join([limits.investigation[name], *details])
    if name in DECLARED_STATUS:
        status, reason = DECLARED_STATUS[name]
        return status, outcome, "; ".join([reason, *details])
    if failed:
        return FAIL, outcome, "; ".join(details)
    if unconverged or not scored:
        return NOT_SCORED, outcome, "; ".join(details) or "no scored metric"
    if global_fail:
        note = limits.override_notes.get(name, f"class (a) snapshot budget ({LIMITS_DOC})")
        limits_text = ", ".join(
            f"{key} {overrides[key]} (global {limits.global_limits[key]})" for key in global_fail
        )
        return PASS_SENTINEL, outcome, f"{limits_text}; {note}"
    scalar_scored = [key for key in scored if key in limits.scalar_limits.get(name, {})]
    return PASS_GLOBAL, outcome, ("scored on " + ", ".join(scalar_scored)) if scalar_scored else ""
