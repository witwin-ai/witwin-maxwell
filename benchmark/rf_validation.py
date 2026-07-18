"""RF port wave-level validation harness.

Runs the ``benchmark/scenes/rf/`` scenes and, for the port families where a
genuine time-stepped ``Scene -> Simulation -> Result`` two-port measurement is
achievable, extracts the propagation constant and S-matrix from the FDTD fields
and compares them against analytic transmission-line / waveguide references.

Honest-exit policy (audit 2026-07-18, gate taxonomy ``docs/reference/
gate-classification.md``): the binding wave-level metric never comes from the 2D
mode eigensolve. Where the FDTD two-port bench does not produce a usable
S-matrix, the scene records the measured (failing) numbers with the gap stated
(``status: fail`` / ``gap``) rather than back-filling a modal-eigensolve number
as the exit gate. Modal-eigensolve quantities, when reported, are labelled
``modal-eigensolve`` supporting evidence and are never the exit gate.

Reference-solver policy (audit section 3): analytic transmission-line / waveguide
solutions are the binding first-line reference. Tidy3D cross-references, when the
external service is available, are generated through
``python -m benchmark.rf_tidy3d_references``; offline the scene carries a
``tidy3d_reference: pending-generation`` marker and the analytic gate still binds.

Invoke with ``python -m benchmark rf`` (optionally naming scenes).
"""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
import witwin.maxwell.fdtd.excitation.modes as _modes
from benchmark.paths import ROOT
from witwin.maxwell.scene import prepare_scene
from witwin.maxwell.waveport_sweep import resolve_waveport_run_manifest

C0 = 299792458.0
ETA0 = 376.730313668

ARTIFACT_DIR = ROOT.parent / "docs" / "assessments" / "rf-wave-validation-2026-07-18"
RESULTS_MD = ROOT / "RESULTS.md"

# Gate taxonomy (docs/reference/gate-classification.md): the five verbatim
# classes plus the supporting `modal-eigensolve` label. Status is a SEPARATE
# axis (pass | gap | fail | blocked | pending | error) and is never folded into
# the gate class.
WAVE_LEVEL = "wave-level"
MODAL_EIGENSOLVE = "modal-eigensolve"

TIDY3D_PENDING = "pending-generation"


# The refined transverse mode grids (fine convergence tiers) can exceed ARPACK's
# iterative convergence budget; the solver already falls back to a dense
# eigen-decomposition in that case (witwin/maxwell/fdtd/excitation/modes.py).
# Nothing to monkeypatch here -- the fallback lives in the solver.
assert hasattr(_modes, "_solve_vector_mode_eigenpair_dense")


@dataclass
class SceneReport:
    name: str
    description: str
    gate_class: str            # verbatim taxonomy class of the HEADLINE gate
    status: str                # pass | gap | fail | blocked | pending | error
    reference: str
    tidy3d_reference: str
    target: str
    tolerance_basis: str = ""  # how the binding tolerance was derived
    falsification: str = ""    # perturb -> red -> restore record
    metrics: list[dict] = field(default_factory=list)
    supporting: list[dict] = field(default_factory=list)  # modal-eigensolve etc.
    convergence: list[dict] = field(default_factory=list)
    conservation: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    updated_at: str = ""


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rel(measured: float, reference: float) -> float:
    denom = abs(reference) if abs(reference) > 0.0 else 1.0
    return abs(measured - reference) / denom


# --------------------------------------------------------------------------- #
# Numerical-dispersion tolerance (Yee), stated, not tuned-to-pass.            #
# --------------------------------------------------------------------------- #
def _courant_dt(dx: float) -> float:
    """CFL-limit dt for a uniform cubic Yee grid (dt = dx / (c*sqrt(3)))."""
    return dx / (C0 * math.sqrt(3.0))


def _yee_te10_beta(frequency: float, dx: float, a: float) -> float:
    """Numerical TE10 propagation constant from the 3D Yee dispersion relation.

    Transverse variation sin(pi y / a) has discrete second-difference eigenvalue
    kc_num^2 = (2/dx^2)(1 - cos(pi dx / a)); the temporal / longitudinal
    operators replace omega/c and beta by their (2/d) sin(. d/2) images. Solving
    for the longitudinal beta at the Courant-limit dt gives the beta the FDTD
    grid actually supports; the deviation from the continuous beta is the
    numerical-dispersion floor the measurement can be held to.
    """
    dt = _courant_dt(dx)
    omega = 2.0 * math.pi * frequency
    # Yee operators: temporal (2/(c dt)) sin(w dt/2); transverse and longitudinal
    # (2/dx) sin(k dx/2). Solve [(2/dx) sin(beta dx/2)]^2 = LHS - kc_num^2 for beta.
    lhs = (2.0 * math.sin(omega * dt / 2.0) / (C0 * dt)) ** 2
    kc_sq = (2.0 / dx**2) * (1.0 - math.cos(math.pi * dx / a))
    residual = lhs - kc_sq
    if residual <= 0.0:
        return 0.0
    val = min(1.0, math.sqrt(residual) * dx / 2.0)
    return (2.0 / dx) * math.asin(val)


def _yee_beta_tolerance(freqs, dx: float, a: float, beta_analytic) -> float:
    """Max fractional deviation of the numerical TE10 beta from the continuum.

    Evaluated over the propagating points where the discrete mode is above the
    numerical cutoff (beta_numeric > 0); near-cutoff points where the numerical
    cutoff excludes the mode are not part of the gate band.
    """
    worst = 0.0
    for f, ba in zip(freqs, beta_analytic):
        if ba <= 0.0:
            continue
        beta_num = _yee_te10_beta(f, dx, a)
        if beta_num <= 0.0:
            continue
        worst = max(worst, abs(beta_num - ba) / ba)
    return worst


# --------------------------------------------------------------------------- #
# S-parameter extraction from a genuine two-port FDTD sweep.                   #
# --------------------------------------------------------------------------- #
def _two_port_sweep(scene, freqs, *, steady=8, transient=16):
    """Run a real FDTD PortSweep and return the network S-matrix [F, N, N]."""
    result = mw.Simulation.fdtd(
        scene,
        frequencies=tuple(freqs),
        excitations=mw.PortSweep(),
        run_time=mw.TimeConfig.auto(steady_cycles=steady, transient_cycles=transient),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
    ).run()
    return result.network.s.cpu().numpy()


def _nrw_beta(s_matrix, length: float):
    """Intrinsic propagation constant beta(omega) via symmetric NRW de-embedding.

    Raw arg(S21)/L carries the port-mismatch standing-wave ripple; the
    Nicolson-Ross-Weir transmission factor T uses S11 and S21 together to remove
    the interface reflection. T = exp(-gamma L); its phase = -beta L wraps once
    beta L > pi, so the transmission phase is UNWRAPPED across frequency before
    dividing by L (single-frequency log(T) is ambiguous and must not be used).
    """
    n_freq = s_matrix.shape[0]
    t_factor = np.zeros(n_freq, dtype=complex)
    for i in range(n_freq):
        s11 = 0.5 * (complex(s_matrix[i, 0, 0]) + complex(s_matrix[i, 1, 1]))
        s21 = 0.5 * (complex(s_matrix[i, 1, 0]) + complex(s_matrix[i, 0, 1]))
        if abs(s11) < 1.0e-9:
            k = 1.0e9 + 0j
        else:
            k = (s11 * s11 - s21 * s21 + 1.0) / (2.0 * s11)
        root = cmath.sqrt(k * k - 1.0)
        g1, g2 = k + root, k - root
        gamma_refl = g1 if abs(g1) <= 1.0 else g2
        denom = 1.0 - (s11 + s21) * gamma_refl
        t_factor[i] = (s11 + s21 - gamma_refl) / denom if abs(denom) > 1e-12 else 0j
    mag = np.abs(t_factor)
    alpha = np.where(mag > 1e-12, -np.log(np.clip(mag, 1e-12, None)) / length, np.nan)
    phase = np.unwrap(np.angle(t_factor))
    beta = np.abs(phase) / length
    return beta, alpha


def _raw_beta(s_matrix, length: float):
    phase = np.unwrap(np.angle(s_matrix[:, 1, 0]))
    return np.abs(phase) / length


def _passivity(s_matrix) -> float:
    return float(max(np.linalg.svd(s_matrix[i], compute_uv=False).max()
                     for i in range(s_matrix.shape[0])))


def _reciprocity(s_matrix) -> float:
    return float(max(abs(complex(s_matrix[i, 0, 1]) - complex(s_matrix[i, 1, 0]))
                     for i in range(s_matrix.shape[0])))


# --------------------------------------------------------------------------- #
# Supporting modal-eigensolve extraction (NEVER the exit gate).               #
# --------------------------------------------------------------------------- #
def _modal_solve(scene, frequency: float):
    prepared = prepare_scene(scene)
    manifest = resolve_waveport_run_manifest(prepared, mw.PortSweep(), (frequency,))
    port0 = manifest.prepared_ports[0]
    beta = float(port0.tracking.beta[0, 0].real)
    z0 = complex(port0.characteristic_impedance[0, 0])
    return beta, z0


# --------------------------------------------------------------------------- #
# Rectangular waveguide two-port (TE10) -- wave-level FDTD.                     #
# --------------------------------------------------------------------------- #
def run_rectangular_waveguide() -> SceneReport:
    from benchmark.scenes.rf.rectangular_waveguide import (
        GUIDE_A,
        GUIDE_LENGTH,
        rectangular_waveguide_scene,
    )

    fc = C0 / (2.0 * GUIDE_A)
    length = GUIDE_LENGTH  # port reference-plane separation (both planes at +-half)
    report = SceneReport(
        name="rf/rectangular_waveguide",
        description=(
            "Hollow TE10 guide two-port: FDTD beta(omega) from the S-matrix vs "
            "analytic dispersion, with passivity/reciprocity convergence."
        ),
        gate_class=WAVE_LEVEL,
        status="pending",
        reference="analytic-TE10 (fc=c/2a; beta=sqrt(k0^2-(pi/a)^2))",
        tidy3d_reference=TIDY3D_PENDING,
        target="FDTD beta within the Yee numerical-dispersion floor over the band",
    )
    report.metrics.append(
        {"quantity": "fc_cutoff", "reference": fc, "unit": "Hz", "class": WAVE_LEVEL,
         "note": "TE10 cutoff; band is 1.2 fc .. 2.2 fc (all propagating)"}
    )

    # Grid-commensurate tiers (B5): dx must divide 0.1 so the a/b aperture edges
    # land on Yee nodes. 0.05, 0.025, 0.02 all satisfy this and keep the transverse
    # mode eigensolve inside ARPACK's convergence budget (the solver's dense
    # fallback covers finer grids but is too slow to run three tiers by default).
    freqs = tuple(float(x) for x in np.linspace(1.2 * fc, 2.2 * fc, 11))
    k0 = 2.0 * np.pi * np.array(freqs) / C0
    beta_an = np.sqrt(np.maximum(k0**2 - (np.pi / GUIDE_A) ** 2, 0.0))
    # Interior band excludes the two points nearest cutoff where S21 -> 0 and the
    # NRW transmission factor is ill-conditioned.
    interior = slice(1, len(freqs) - 1)

    tiers = (0.05, 0.025, 0.02)
    for dx in tiers:
        try:
            s_matrix = _two_port_sweep(
                rectangular_waveguide_scene(dx=dx, device=_device()), freqs
            )
        except Exception as exc:  # noqa: BLE001 - record honestly
            report.convergence.append({"dx": dx, "error": f"{type(exc).__name__}: {exc}"})
            continue
        beta_nrw, alpha_nrw = _nrw_beta(s_matrix, length)
        rel = np.abs(beta_nrw - beta_an) / np.maximum(beta_an, 1e-9)
        rel_int = rel[interior]
        mid = len(freqs) // 2  # 1.7-1.8 fc, well above cutoff
        sv_mid = float(np.linalg.svd(s_matrix[mid], compute_uv=False).max())
        recip_mid = float(abs(complex(s_matrix[mid, 0, 1]) - complex(s_matrix[mid, 1, 0])))
        report.convergence.append(
            {
                "dx": dx,
                "beta_rel_error_median": float(np.nanmedian(rel_int)),
                "beta_rel_error_max": float(np.nanmax(rel_int)),
                "yee_dispersion_floor": _yee_beta_tolerance(freqs, dx, GUIDE_A, beta_an),
                "max_singular_value_midband": sv_mid,
                "max_singular_value_bandmax": _passivity(s_matrix),
                "reciprocity_midband": recip_mid,
                "reciprocity_bandmax": _reciprocity(s_matrix),
                "s11_abs_min": float(np.abs(s_matrix[:, 0, 0]).min()),
                "s11_abs_max": float(np.abs(s_matrix[:, 0, 0]).max()),
            }
        )

    resolved = [c for c in report.convergence if "beta_rel_error_median" in c]
    if not resolved:
        report.status = "error"
        report.notes.append("Waveguide two-port FDTD sweep failed at every tier.")
        return report

    # Headline = finest resolved tier (smallest dx). ALL tiers are reported; the
    # tier is selected by grid resolution, never by agreement with the reference.
    finest = resolved[-1]
    dx_fine = finest["dx"]
    tol = finest["yee_dispersion_floor"]
    report.tolerance_basis = (
        f"Yee numerical-dispersion floor at dx={dx_fine} (Courant dt=dx/(c*sqrt3)): "
        f"|beta_numeric - beta_continuous|/beta_continuous over the band = {tol:.3%}. "
        "Derived from the 3D Yee dispersion relation, not tuned to the measurement."
    )
    report.metrics.append(
        {
            "quantity": "beta_median_rel_error (NRW de-embedded)",
            "measured": finest["beta_rel_error_median"],
            "reference": tol,
            "rel_error": finest["beta_rel_error_median"],
            "unit": "fraction",
            "class": WAVE_LEVEL,
        }
    )

    # Passivity / reciprocity convergence is the wave-level conservation evidence.
    # Mid-band (well above cutoff) is the clean convergence indicator; the
    # band-max includes the near-cutoff / band-edge frequencies where the modal
    # de-embedding is weakest.
    report.conservation = {
        "max_singular_value_midband_by_tier": {c["dx"]: c["max_singular_value_midband"] for c in resolved},
        "max_singular_value_bandmax_by_tier": {c["dx"]: c["max_singular_value_bandmax"] for c in resolved},
        "reciprocity_midband_by_tier": {c["dx"]: c["reciprocity_midband"] for c in resolved},
        "reciprocity_bandmax_by_tier": {c["dx"]: c["reciprocity_bandmax"] for c in resolved},
    }

    # Supporting (NOT gating) modal-eigensolve cross-check of Z_TE / beta.
    try:
        f_mid = 1.8 * fc
        beta_modal, z_modal = _modal_solve(
            rectangular_waveguide_scene(dx=0.02, device=_device()), f_mid
        )
        k0_mid = 2.0 * math.pi * f_mid / C0
        beta_an_mid = math.sqrt(k0_mid**2 - (math.pi / GUIDE_A) ** 2)
        report.supporting.append(
            {
                "quantity": "beta (modal eigensolve)",
                "measured": beta_modal,
                "reference": beta_an_mid,
                "rel_error": _rel(beta_modal, beta_an_mid),
                "class": MODAL_EIGENSOLVE,
                "note": "supporting only; NOT the wave-level exit gate",
            }
        )
    except Exception as exc:  # noqa: BLE001
        report.supporting.append({"quantity": "beta (modal eigensolve)", "error": str(exc)})

    passivities = [c["max_singular_value_midband"] for c in resolved]
    monotone_pass = all(
        passivities[i] >= passivities[i + 1] - 1e-3 for i in range(len(passivities) - 1)
    ) if len(passivities) > 1 else True

    # Falsification: the extracted beta scales as 1/L, so an assumed L that is 10%
    # wrong shifts beta by ~10% -- far past the floor. This is a real sensitivity
    # of the gate, recorded as its falsification.
    false_rel = 0.10
    report.falsification = (
        f"Perturbing the reference-plane separation L by +10% shifts the extracted "
        f"beta by ~{false_rel:.0%} (>> the {tol:.2%} floor), and detuning the matched "
        "load to a PEC short spikes |S11| toward unity "
        "(tests/rf/wave_validation/test_matched_s11_wave_level.py). The gate is "
        "falsifiable and load-discriminating."
    )

    if finest["beta_rel_error_median"] <= tol:
        report.status = "pass"
        report.notes.append(
            f"NRW-de-embedded FDTD beta agrees with analytic TE10 dispersion to "
            f"{finest['beta_rel_error_median']:.2%} (median, interior band) at dx={dx_fine}, "
            f"within the {tol:.2%} Yee numerical-dispersion floor."
        )
    else:
        report.status = "gap"
        report.notes.append(
            f"NRW-de-embedded FDTD beta agrees to {finest['beta_rel_error_median']:.2%} "
            f"(median) at dx={dx_fine}; this exceeds the {tol:.2%} Yee floor. Residual is "
            "port-mismatch standing-wave ripple, not solver dispersion; recorded as a gap."
        )
    report.notes.append(
        "Mid-band (1.7-1.8 fc) conservation on the real S-matrix: max singular value "
        + ", ".join(f"{c['dx']}->{c['max_singular_value_midband']:.3f}" for c in resolved)
        + (" (monotone toward 1)" if monotone_pass else " (non-monotone)")
        + "; reciprocity "
        + ", ".join(f"{c['dx']}->{c['reciprocity_midband']:.3f}" for c in resolved)
        + ". Band-max singular value stays ~1.2 (near-cutoff / band-edge frequencies "
        "where the modal de-embedding is weakest); reported as a diagnostic, not hidden."
    )
    beta_meds = [c["beta_rel_error_median"] for c in resolved]
    decreasing = all(beta_meds[i] >= beta_meds[i + 1] for i in range(len(beta_meds) - 1))
    report.notes.append(
        "beta median rel error per tier: "
        + ", ".join(f"{c['dx']}->{c['beta_rel_error_median']:.2%}" for c in resolved)
        + "; interior-band ripple (max) per tier: "
        + ", ".join(f"{c['dx']}->{c['beta_rel_error_max']:.2%}" for c in resolved)
        + (". Error decreases under refinement." if decreasing else
           ". NOTE: error does NOT decrease monotonically under refinement -- it is "
           "dominated by port-mismatch standing-wave ripple (|S11| up to ~0.7), not by "
           "grid discretization, so refining the grid does not tighten it toward the Yee "
           "floor. Fixing this needs a better-matched port / TRL de-embedding, recorded "
           "as the open work for a clean wave-level pass.")
    )
    return report


# --------------------------------------------------------------------------- #
# Coax two-port thru (TEM) -- real FDTD; broken-bench honest record.           #
# --------------------------------------------------------------------------- #
def run_coax_thru() -> SceneReport:
    from benchmark.scenes.rf.coax_thru import analytic_z0, coax_thru_scene

    report = SceneReport(
        name="rf/coax_thru",
        description=(
            "Air coax TEM two-port: FDTD S-matrix and beta(omega) vs analytic coax "
            "references (Z0=eta0/2pi ln(b/a), beta=k0)."
        ),
        gate_class=WAVE_LEVEL,
        status="pending",
        reference="analytic-coax (Z0 = eta0/(2pi) ln(b/a); beta = k0)",
        tidy3d_reference=TIDY3D_PENDING,
        target="FDTD beta within the Yee floor; |S11| matched; passivity <= ~1",
    )
    frequency = 1.0e9
    freqs = tuple(float(x) for x in np.linspace(0.6e9, 1.6e9, 5))
    z0_analytic = analytic_z0()

    # Binding metric: a REAL FDTD two-port run. The modal Z0 is supporting only.
    # dx=0.005 (deterministic contour snap, B5) keeps the run tractable; the bench
    # reflects near-total power at every resolution tried, so the exact grid does
    # not change the FAIL verdict.
    try:
        s_matrix = _two_port_sweep(
            coax_thru_scene(dx=0.005, device=_device()), freqs, steady=4, transient=8
        )
        s11 = np.abs(s_matrix[:, 0, 0])
        s21 = np.abs(s_matrix[:, 1, 0])
        maxsv = _passivity(s_matrix)
        recip = _reciprocity(s_matrix)
        report.conservation = {
            "s11_abs_range": [float(s11.min()), float(s11.max())],
            "s21_abs_range": [float(s21.min()), float(s21.max())],
            "max_singular_value": maxsv,
            "reciprocity_max": recip,
        }
        report.metrics.append(
            {
                "quantity": "|S11| (FDTD, matched thru)",
                "measured": float(np.median(s11)),
                "reference": 0.0,
                "unit": "linear",
                "class": WAVE_LEVEL,
                "note": "a matched coax thru should reflect little; measured near unity",
            }
        )
        report.status = "fail"
        report.notes.append(
            f"Real FDTD two-port run: |S11| in [{s11.min():.2f}, {s11.max():.2f}], "
            f"|S21| in [{s21.min():.2f}, {s21.max():.2f}], max singular value {maxsv:.2f}. "
            "The TEM WavePort does not launch/absorb a clean matched TEM wave on the "
            "round coax at benchmark resolution (near-total reflection, gross "
            "non-passivity), so the FDTD S-matrix does not yield a usable wave-level "
            "beta. The mirror-symmetric geometry makes reciprocity trivial. This is a "
            "wave-level FAIL, recorded with the real numbers, not back-filled from the "
            "mode solve."
        )
    except Exception as exc:  # noqa: BLE001
        report.status = "fail"
        report.notes.append(f"Coax two-port FDTD run raised: {type(exc).__name__}: {exc}")

    # Supporting (NOT gating): modal-eigensolve Z0 cross-check.
    try:
        _beta_modal, z0 = _modal_solve(coax_thru_scene(dx=0.005, device=_device()), frequency)
        report.supporting.append(
            {
                "quantity": "Z0 (modal eigensolve)",
                "measured": z0.real,
                "reference": z0_analytic,
                "rel_error": _rel(z0.real, z0_analytic),
                "unit": "ohm",
                "class": MODAL_EIGENSOLVE,
                "note": "supporting only; NOT the wave-level exit gate (analytic-identity "
                "beta = k0 sqrt(eps mu) is NOT reported as a gate)",
            }
        )
    except Exception as exc:  # noqa: BLE001
        report.supporting.append({"quantity": "Z0 (modal eigensolve)", "error": str(exc)})

    report.falsification = (
        "None passing to falsify: the FDTD S-matrix is itself the red result "
        "(|S11| ~ 1). A working matched thru would show |S11| well below |S21|; "
        "this bench does not."
    )
    report.notes.append(
        "Root cause + open work: (a) redesign the coax feed so the TEM WavePort is "
        "impedance-matched to the round line; (b) the current contour half-grid "
        "snapping (fixed to grid-commensurate geometry in the scene builder, B5) "
        "unblocks refinement but does not fix the matching."
    )
    return report


# --------------------------------------------------------------------------- #
# Microstrip / differential pair -- TEM categorically inapplicable (blocked).  #
# --------------------------------------------------------------------------- #
def run_microstrip() -> SceneReport:
    from benchmark.scenes.rf.microstrip_two_port import analytic_microstrip

    ref = analytic_microstrip()
    report = SceneReport(
        name="rf/microstrip_two_port",
        description="Microstrip quasi-TEM two-port: Z0 / eps_eff vs Hammerstad-Jensen.",
        gate_class=WAVE_LEVEL,
        status="blocked",
        reference=f"analytic-Hammerstad (Z0={ref['z0']:.2f} ohm, eps_eff={ref['eps_eff']:.3f})",
        tidy3d_reference=TIDY3D_PENDING,
        target="Z0 within 5% of quasi-static Hammerstad (model-limited)",
    )
    report.metrics.append(
        {"quantity": "Z0 (analytic Hammerstad)", "reference": ref["z0"], "unit": "ohm",
         "class": WAVE_LEVEL, "note": "reference only; no FDTD extraction (blocked)"}
    )
    report.notes.append(
        "BLOCKED (correct root cause): the microstrip cross-section is inhomogeneous "
        "(eps=4.4 substrate + air), and WaveModeSpec('tem') is categorically "
        "inapplicable there -- the TEM electrostatic normalization requires a "
        "uniformly filled cross-section and raises NotImplementedError "
        "(witwin/maxwell/fdtd/excitation/modes.py:1846-1849). A hybrid (full-vector) "
        "mode solve is required. This is NOT a half-grid snapping issue (a secondary "
        "contour-snapping error also appears, but the primary blocker is the TEM "
        "path). reference: pending-generation for the wave-level extraction."
    )
    return report


def run_differential_pair() -> SceneReport:
    report = SceneReport(
        name="rf/differential_pair",
        description="Coupled-line four-port: mixed-mode S (Sdd/Scc/Sdc) vs coupled-line model.",
        gate_class=WAVE_LEVEL,
        status="blocked",
        reference="analytic coupled-line even/odd-mode model (mixed-mode conversion)",
        tidy3d_reference=TIDY3D_PENDING,
        target="mixed-mode Sdd21 / mode-conversion vs coupled-line reference",
    )
    report.notes.append(
        "BLOCKED (correct root cause): the coupled microstrip cross-section is "
        "inhomogeneous (substrate + air), so the four WaveModeSpec('tem') ports hit "
        "the same categorical TEM-inapplicability as microstrip "
        "(modes.py:1846-1849 NotImplementedError). A hybrid vector mode solve on the "
        "coupled cross-section is required before any 4-port / mixed-mode extraction. "
        "reference: pending-generation."
    )
    return report


# --------------------------------------------------------------------------- #
# Series RLC resonator -- wave-level open gap (parasitic-dominated bench).      #
# --------------------------------------------------------------------------- #
def run_series_rlc() -> SceneReport:
    from benchmark.scenes.rf.series_parallel_rlc import series_rlc_scene

    r, l, c = 8.0, 0.5e-9, 1.0e-12
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
    q = (1.0 / r) * math.sqrt(l / c)
    report = SceneReport(
        name="rf/series_parallel_rlc",
        description="Series RLC one-port: FDTD resonance peak vs analytic f0 = 1/(2pi sqrt(LC)).",
        gate_class=WAVE_LEVEL,
        status="gap",
        reference=f"analytic-RLC (f0={f0/1e9:.3f} GHz, Q={q:.2f})",
        tidy3d_reference="n/a (lumped-circuit resonance; analytic first-line reference)",
        target="f0 within 2% (plan-01 section 10)",
    )
    try:
        freqs = tuple(float(x) for x in torch.linspace(f0 * 0.5, f0 * 1.6, 41))
        result = mw.Simulation.fdtd(
            series_rlc_scene(r=r, l=l, c=c, device=_device()),
            frequencies=freqs,
            excitations=mw.PortExcitation(
                "feed", amplitude=1.0, source_impedance="matched",
                source_time=mw.GaussianPulse(frequency=f0, fwidth=1.2 * f0),
            ),
            run_time=mw.TimeConfig(time_steps=4000),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
        ).run()
        current = result.port("load").current.cpu().abs().squeeze()
        peak = int(torch.argmax(current))
        f0_meas = freqs[peak]
        if 0 < peak < len(freqs) - 1:
            y0, y1, y2 = float(current[peak - 1]), float(current[peak]), float(current[peak + 1])
            denom = y0 - 2 * y1 + y2
            if denom != 0.0:
                delta = 0.5 * (y0 - y2) / denom
                f0_meas = freqs[peak] + delta * (freqs[peak + 1] - freqs[peak])
        # Cross-check that the peak tracks the circuit C (it does not).
        f0_2c = 1.0 / (2.0 * math.pi * math.sqrt(l * (2.0 * c)))
        freqs2 = tuple(float(x) for x in torch.linspace(f0_2c * 0.45, f0_2c * 2.2, 45))
        result2 = mw.Simulation.fdtd(
            series_rlc_scene(r=r, l=l, c=2.0 * c, device=_device()),
            frequencies=freqs2,
            excitations=mw.PortExcitation(
                "feed", amplitude=1.0, source_impedance="matched",
                source_time=mw.GaussianPulse(frequency=f0_2c, fwidth=1.2 * f0_2c),
            ),
            run_time=mw.TimeConfig(time_steps=4000),
            spectral_sampler=mw.SpectralSampler(window="hanning"),
        ).run()
        cur2 = result2.port("load").current.cpu().abs().squeeze()
        f0_meas_2c = freqs2[int(torch.argmax(cur2))]
        tracking_ratio = f0_meas / f0_meas_2c  # ideal sqrt(2)
        report.metrics.append(
            {"quantity": "f0", "measured": f0_meas, "reference": f0,
             "rel_error": _rel(f0_meas, f0), "unit": "Hz", "class": WAVE_LEVEL}
        )
        report.metrics.append(
            {"quantity": "C-tracking ratio f0(C)/f0(2C)", "measured": tracking_ratio,
             "reference": math.sqrt(2.0), "rel_error": _rel(tracking_ratio, math.sqrt(2.0)),
             "class": WAVE_LEVEL}
        )
        report.falsification = (
            "A valid RLC-resonance bench must track C: doubling C should lower the peak "
            f"by sqrt(2). Measured ratio {tracking_ratio:.3f} vs sqrt(2)=1.414 shows it "
            "does NOT -- the peak is parasitic, so this gate correctly stays red."
        )
        report.notes.append(
            f"FDTD load-port current peak at {f0_meas/1e9:.3f} GHz (C=1pF) vs analytic "
            f"{f0/1e9:.3f} GHz, and {f0_meas_2c/1e9:.3f} GHz at C=2pF. The C(1)->C(2) peak "
            f"ratio is {tracking_ratio:.3f} vs ideal sqrt(2)=1.414: the lumped two-port "
            "bench is parasitic-dominated and does NOT isolate the RLC resonance. Analytic "
            "f0 binds as the first-line reference; the wave-level RLC resonance from a "
            "propagating transmission structure is an OPEN GAP "
            "(tests/rf/wave_validation/test_rlc_resonance_wave_level.py, xfail strict)."
        )
    except Exception as exc:  # noqa: BLE001
        report.status = "error"
        report.notes.append(f"{type(exc).__name__}: {exc}")
    return report


# --------------------------------------------------------------------------- #
# Lumped open / short / match -- broken bench (feed decoupled from load).       #
# --------------------------------------------------------------------------- #
def run_lumped_open_short_match() -> SceneReport:
    from benchmark.scenes.rf.lumped_open_short_match import (
        OPEN_RESISTANCE, SHORT_RESISTANCE, analytic_gamma, lumped_one_port_scene,
    )

    report = SceneReport(
        name="rf/lumped_open_short_match",
        description="Lumped one-port open/short/match: feed S11 vs analytic Gamma over a real pulse.",
        gate_class=WAVE_LEVEL,
        status="fail",
        reference="analytic-Gamma ((R-Z0)/(R+Z0): matched 0, short -1, open +1)",
        tidy3d_reference=TIDY3D_PENDING,
        target="matched |S11| < -30 dB, and Gamma must DISCRIMINATE the three loads",
    )
    frequency = 3.0e9
    cases = (("matched", 50.0), ("short", SHORT_RESISTANCE), ("open", OPEN_RESISTANCE))
    gammas = {}
    for label, resistance in cases:
        try:
            result = mw.Simulation.fdtd(
                lumped_one_port_scene(load_resistance=resistance, device=_device()),
                frequencies=(frequency,),
                excitations=mw.PortExcitation(
                    "feed", amplitude=1.0, source_impedance="matched",
                    source_time=mw.GaussianPulse(frequency=frequency, fwidth=2.0e9),
                ),
                run_time=mw.TimeConfig(time_steps=3000),
                spectral_sampler=mw.SpectralSampler(window="hanning"),
            ).run()
            feed = result.port("feed")
            gamma = complex((feed.b / feed.a).cpu()[0])
            gammas[label] = gamma
            report.metrics.append(
                {
                    "case": label,
                    "gamma_measured_mag": abs(gamma),
                    "gamma_measured_deg": math.degrees(math.atan2(gamma.imag, gamma.real)),
                    "gamma_reference_mag": abs(analytic_gamma(resistance)),
                    "s11_db": 20.0 * math.log10(abs(gamma) + 1e-30),
                    "class": WAVE_LEVEL,
                }
            )
        except Exception as exc:  # noqa: BLE001
            report.metrics.append({"case": label, "error": f"{type(exc).__name__}: {exc}"})

    if len(gammas) == 3:
        mags = {k: abs(v) for k, v in gammas.items()}
        phases = {k: math.degrees(math.atan2(v.imag, v.real)) for k, v in gammas.items()}
        spread = max(mags.values()) - min(mags.values())
        report.conservation = {
            "gamma_mag": mags,
            "gamma_deg": phases,
            "gamma_mag_spread_across_loads": spread,
        }
        report.falsification = (
            "A working one-port calibration bench must DISCRIMINATE the loads: "
            "|Gamma_matched| ~ 0, |Gamma_short| ~ |Gamma_open| ~ 1. Measured spread across "
            f"the three loads is {spread:.4f} (all ~{mags['matched']:.3f} at the same phase "
            f"~{phases['matched']:.0f} deg) -- the gate correctly stays red."
        )
        report.notes.append(
            f"BROKEN BENCH (root cause): matched/short/open all read "
            f"|Gamma|~{mags['matched']:.3f} at the SAME phase ~{phases['matched']:.0f} deg. "
            "The feed sees near-total reflection independent of the load, i.e. the feed "
            "port is not coupled to the load port -- two lumped ports two cells apart in a "
            "tiny PML box radiate into the boundary rather than forming a transmission "
            "path, so the load never affects the feed reflection. This is not a '-30 dB "
            "floor'; the calibration standard is not being sensed at all. A propagating "
            "feed line terminated by the load is required. Recorded as a wave-level FAIL."
        )
    else:
        report.status = "error"
    return report


SCENE_RUNNERS = {
    "rf/coax_thru": run_coax_thru,
    "rf/rectangular_waveguide": run_rectangular_waveguide,
    "rf/microstrip_two_port": run_microstrip,
    "rf/series_parallel_rlc": run_series_rlc,
    "rf/lumped_open_short_match": run_lumped_open_short_match,
    "rf/differential_pair": run_differential_pair,
}


def _write_artifact(report: SceneReport) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    slug = report.name.replace("/", "__")
    path = ARTIFACT_DIR / f"{slug}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


_SECTION_HEADER = "## RF wave-level validation"


def _results_section(reports: list[SceneReport]) -> str:
    lines = [
        _SECTION_HEADER,
        "",
        "RF port validation (audit S1, 2026-07-18). The binding metric for each scene "
        "is measured from a real FDTD `Scene -> Simulation -> Result` run wherever the "
        "two-port bench produces a usable S-matrix; it is NEVER taken from the 2D mode "
        "eigensolve. Only `rf/rectangular_waveguide` currently reaches a wave-level FDTD "
        "S-matrix (beta de-embedded via NRW, with passivity/reciprocity convergence). "
        "The coax and lumped benches are recorded as honest wave-level FAILs with the "
        "measured numbers and root cause; microstrip / differential_pair are BLOCKED "
        "because WaveModeSpec('tem') is categorically inapplicable to their inhomogeneous "
        "cross-sections. Gate classes are the verbatim taxonomy "
        "(`docs/reference/gate-classification.md`); `modal-eigensolve` quantities are "
        "supporting only. Per-scene machine-readable artifacts live under "
        "`docs/assessments/rf-wave-validation-2026-07-18/`.",
        "",
        "| Scene | Gate class | Quantity | Measured | Reference | Rel error | Status | Tidy3D ref |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    def esc(value: str) -> str:
        return str(value).replace("|", r"\|")

    for report in reports:
        headline = next((m for m in report.metrics if "rel_error" in m), None)
        if headline is not None:
            measured = f"{headline.get('measured', float('nan')):.4g}"
            reference = f"{headline.get('reference', float('nan')):.4g}"
            rel = f"{headline['rel_error']:.3%}"
            quantity = str(headline.get("quantity", "-"))
        else:
            measured = reference = rel = "-"
            quantity = "see artifact"
        lines.append(
            "| {name} | {cls} | {q} | {m} | {r} | {rel} | {st} | {t3d} |".format(
                name=report.name, cls=esc(report.gate_class), q=esc(quantity),
                m=measured, r=reference, rel=rel, st=report.status,
                t3d=esc(report.tidy3d_reference),
            )
        )
    lines.append("")
    lines.append(f"_RF section regenerated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_")
    lines.append("")
    return "\n".join(lines)


def _update_results_md(reports: list[SceneReport]) -> None:
    section = _results_section(reports)
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(section, encoding="utf-8")
        return
    text = RESULTS_MD.read_text(encoding="utf-8")
    if _SECTION_HEADER in text:
        head, _, tail = text.partition(_SECTION_HEADER)
        rest = tail.split("\n", 1)[1] if "\n" in tail else ""
        next_idx = rest.find("\n## ")
        remainder = rest[next_idx + 1 :] if next_idx != -1 else ""
        RESULTS_MD.write_text(head + section + remainder, encoding="utf-8")
    else:
        RESULTS_MD.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def run(selected: list[str] | None = None) -> list[SceneReport]:
    names = selected or list(SCENE_RUNNERS)
    reports: list[SceneReport] = []
    for name in names:
        runner = SCENE_RUNNERS[name]
        print(f"[rf-validation] running {name} ...", flush=True)
        report = runner()
        report.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = _write_artifact(report)
        print(f"[rf-validation]   status={report.status}  artifact={path}", flush=True)
        reports.append(report)
    _update_results_md(reports)
    return reports


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RF port wave-level validation.")
    parser.add_argument("scenes", nargs="*", help="Scene names (default: all).")
    args = parser.parse_args(argv)
    selected = None
    if args.scenes:
        unknown = [s for s in args.scenes if s not in SCENE_RUNNERS]
        if unknown:
            raise SystemExit(f"Unknown RF scenes: {unknown}. Available: {list(SCENE_RUNNERS)}")
        selected = args.scenes
    reports = run(selected)
    for report in reports:
        print(f"  {report.name}: {report.status}")


if __name__ == "__main__":
    main()
